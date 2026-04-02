"""
detector.py
===========
FiberYOLODetector: segmentación YOLO con pipeline SAHI integrado.

El modelo fue entrenado con imágenes upscaleadas ×4 y tiles de 640px.
En inferencia se replica el mismo proceso:
  1. Upscale ×SCALE_FACTOR (default 4) con interpolación Lanczos
  2. Slicing con SAHI (tiles 640×640, overlap 50%)
  3. NMS para fusionar detecciones de tiles solapados
  4. Extracción de parámetros por PCA sobre cada máscara
  5. Rescalado de coordenadas ÷ SCALE_FACTOR al espacio original

La interfaz pública (detect) devuelve exactamente lo mismo que antes:
    (list[Detection], next_det_id)
"""
from __future__ import annotations
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

from .models import Detection
from .image_utils import wrap_angle_deg


# ─────────────────────────────────────────────
# PARÁMETROS DE INFERENCIA SAHI
# (deben coincidir con los usados en entrenamiento)
# ─────────────────────────────────────────────
_SCALE_FACTOR  = 4      # mismo que SCALE_FACTOR en train_yolo26.py
_TILE_SIZE     = 640    # mismo que TILE_SIZE en train_yolo26.py
_OVERLAP_RATIO = 0.5    # mismo que OVERLAP en train_yolo26.py
_IOU_THRESHOLD = 0.3    # NMS entre tiles solapados


def _pca_fiber_params(mask_arr: np.ndarray, scale: int) -> dict | None:
    """
    Extrae centroide y ángulo de una fibra desde su máscara binaria
    usando PCA sobre los píxeles activos.

    Usa cos(2θ)/sin(2θ) internamente para debugging, pero devuelve
    angle_deg en [0°, 180°) consistente con la simetría bidireccional
    de las fibras.

    mask_arr : array uint8 en espacio upscaleado
    scale    : factor de upscale para rescalar al espacio original
    """
    ys, xs = np.where(mask_arr > 0)
    if len(xs) < 5:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    cx_up = pts[:, 0].mean()
    cy_up = pts[:, 1].mean()

    pts_c = pts - np.array([cx_up, cy_up])
    cov = np.cov(pts_c.T)
    _, eigvecs = np.linalg.eigh(cov)

    # Eje principal (mayor eigenvalue → último columna de eigh)
    main_axis = eigvecs[:, -1]   # (dx, dy) en coords imagen
    minor_axis = eigvecs[:, 0]

    # Ángulo [0°, 180°) — bidireccional
    angle_deg = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
    if angle_deg < 0:
        angle_deg += 180.0

    # Largo y ancho por proyección sobre los ejes
    proj_main  = pts_c @ main_axis
    proj_minor = pts_c @ minor_axis
    length_up  = float(proj_main.max()  - proj_main.min())
    width_up   = float(proj_minor.max() - proj_minor.min())

    return {
        "cx":        cx_up / scale,
        "cy":        cy_up / scale,
        "angle_deg": float(angle_deg),
        "main_axis": main_axis,
        "length_px": length_up / scale,
        "width_px":  width_up  / scale,
    }


class FiberYOLODetector:
    """
    Detector de fibras con pipeline SAHI integrado.

    Parámetros adicionales (con defaults que coinciden con el entrenamiento):
        scale_factor  : factor de upscale antes de tilear (default 4)
        tile_size     : tamaño del tile en px (default 640)
        overlap_ratio : solapamiento entre tiles (default 0.5)
        iou_threshold : umbral NMS entre tiles (default 0.3)
        device        : "cuda:0" | "cpu" | None (autodetect)
    """

    def __init__(
        self,
        weights_path: Path,
        conf: float,
        device: str | int | None = None,
        scale_factor: int   = _SCALE_FACTOR,
        tile_size: int      = _TILE_SIZE,
        overlap_ratio: float = _OVERLAP_RATIO,
        iou_threshold: float = _IOU_THRESHOLD,
    ):
        if not weights_path.exists():
            raise FileNotFoundError(f"No existe modelo YOLO: {weights_path}")

        self.model        = YOLO(str(weights_path))
        self.conf         = conf
        self.device       = device
        self.scale_factor = scale_factor
        self.tile_size    = tile_size
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold

        # Importar SAHI aquí para no romper si no está instalado en otros módulos
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            self._get_sliced_prediction = get_sliced_prediction
            self._sahi_available = True

            # Crear el modelo SAHI una sola vez (no en cada frame)
            device_str = self._resolve_device_str(device, scale_factor)
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type           = "ultralytics",
                model_path           = str(weights_path),
                confidence_threshold = conf,
                device               = device_str,
            )
            print(f"[DET] SAHI listo — device: {device_str}", flush=True)
        except ImportError:
            self._sahi_available = False
            self._sahi_model     = None
            print("[WARN] sahi no instalado — usando inferencia directa sin tiling", flush=True)

    def _upscale_to_bgr(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        """Upscalea imagen RGB uint8 × scale_factor con Lanczos → BGR."""
        h, w = image_rgb_u8.shape[:2]
        img_up = cv2.resize(
            image_rgb_u8,
            (w * self.scale_factor, h * self.scale_factor),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return cv2.cvtColor(img_up, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _resolve_device_str(device, scale_factor=None) -> str:
        """Resuelve el device como string para SAHI."""
        import torch
        if device is not None:
            d = str(device)
            if d.isdigit():
                return f"cuda:{d}"
            return d
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _resolve_device(self) -> str:
        return self._resolve_device_str(self.device)

    def _detect_sahi(
        self,
        image_rgb_u8: np.ndarray,
        frame_idx: int,
        image_name: str,
        next_det_id: int,
    ) -> tuple[list[Detection], int]:
        """Inferencia con upscale + SAHI slicing. El modelo ya está cargado en __init__."""
        img_bgr_up = self._upscale_to_bgr(image_rgb_u8)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), img_bgr_up, [cv2.IMWRITE_JPEG_QUALITY, 95])

        try:
            result = self._get_sliced_prediction(
                str(tmp_path),
                self._sahi_model,          # ← modelo reutilizado, no recreado
                slice_height                = self.tile_size,
                slice_width                 = self.tile_size,
                overlap_height_ratio        = self.overlap_ratio,
                overlap_width_ratio         = self.overlap_ratio,
                postprocess_type            = "NMS",
                postprocess_match_threshold = self.iou_threshold,
                perform_standard_pred       = False,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        h_orig, w_orig = image_rgb_u8.shape[:2]
        detections: list[Detection] = []

        for pred in result.object_prediction_list:
            if pred.score.value < self.conf or pred.mask is None:
                continue

            mask_arr = pred.mask.bool_mask.astype(np.uint8)
            params   = _pca_fiber_params(mask_arr, self.scale_factor)
            if params is None:
                continue

            # Bounding box rescalada al espacio original
            bbox = pred.bbox
            x1 = bbox.minx / self.scale_factor
            y1 = bbox.miny / self.scale_factor
            x2 = bbox.maxx / self.scale_factor
            y2 = bbox.maxy / self.scale_factor

            detections.append(Detection(
                det_id     = next_det_id,
                frame_idx  = frame_idx,
                image_name = image_name,
                cx         = params["cx"],
                cy         = params["cy"],
                angle_deg  = params["angle_deg"],
                length_px  = params["length_px"],
                width_px   = params["width_px"],
                area_px    = params["length_px"] * params["width_px"],
                score      = float(pred.score.value),
                bbox_xyxy  = [x1, y1, x2, y2],
            ))
            next_det_id += 1

        return detections, next_det_id

    def _detect_direct(
        self,
        image_rgb_u8: np.ndarray,
        frame_idx: int,
        image_name: str,
        next_det_id: int,
    ) -> tuple[list[Detection], int]:
        """Fallback: inferencia directa sin SAHI (imagen completa sin upscale)."""
        pred_kwargs: dict = {
            "source":  image_rgb_u8,
            "conf":    self.conf,
            "verbose": False,
        }
        if self.device is not None:
            pred_kwargs["device"] = self.device

        results = self.model.predict(**pred_kwargs)
        if not results:
            return [], next_det_id

        result = results[0]
        h, w   = image_rgb_u8.shape[:2]

        if result.masks is None or result.masks.xy is None:
            return [], next_det_id

        from .image_utils import polygon_to_mask, contour_geometry_from_mask

        detections: list[Detection] = []
        boxes = result.boxes

        for i, poly_raw in enumerate(result.masks.xy):
            poly = np.array(poly_raw, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue

            mask_u8 = polygon_to_mask(poly, h, w)
            geom    = contour_geometry_from_mask(mask_u8)
            if geom is None:
                continue

            score = 1.0
            if boxes is not None and i < len(boxes):
                try:
                    score = float(boxes.conf[i].item())
                except Exception:
                    pass

            detections.append(Detection(
                det_id     = next_det_id,
                frame_idx  = frame_idx,
                image_name = image_name,
                cx         = geom["cx"],
                cy         = geom["cy"],
                angle_deg  = geom["angle_deg"],
                length_px  = geom["length_px"],
                width_px   = geom["width_px"],
                area_px    = geom["area_px"],
                score      = score,
                bbox_xyxy  = geom["bbox_xyxy"],
            ))
            next_det_id += 1

        return detections, next_det_id

    def detect(
        self,
        image_rgb_u8: np.ndarray,
        frame_idx: int,
        image_name: str,
        next_det_id: int,
    ) -> tuple[list[Detection], int]:
        """
        Detecta fibras en un frame RGB uint8.
        Usa SAHI con upscale ×4 si está disponible, directo si no.
        Retorna (lista de detecciones, próximo det_id).
        """
        if self._sahi_available:
            return self._detect_sahi(image_rgb_u8, frame_idx, image_name, next_det_id)
        else:
            return self._detect_direct(image_rgb_u8, frame_idx, image_name, next_det_id)
