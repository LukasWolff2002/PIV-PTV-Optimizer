"""
detector.py
===========
FiberYOLODetector: segmentación YOLO frame a frame.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .models import Detection
from .image_utils import polygon_to_mask, contour_geometry_from_mask


class FiberYOLODetector:
    """Ejecuta segmentación de instancias YOLO y devuelve Detection por fibra."""

    def __init__(
        self,
        weights_path: Path,
        conf: float,
        device: str | int | None = None,
    ):
        if not weights_path.exists():
            raise FileNotFoundError(f"No existe modelo YOLO: {weights_path}")
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.device = device

    def detect(
        self,
        image_rgb_u8: np.ndarray,
        frame_idx: int,
        image_name: str,
        next_det_id: int,
        return_masks: bool = False,
    ) -> tuple[list[Detection], int] | tuple[list[Detection], int, list]:
        """
        Detecta fibras en un frame RGB uint8.
        Retorna (lista de detecciones, próximo det_id disponible).
        """
        pred_kwargs: dict = {
            "source": image_rgb_u8,
            "conf": self.conf,
            "verbose": False,
        }
        if self.device is not None:
            pred_kwargs["device"] = self.device

        results = self.model.predict(**pred_kwargs)
        if not results:
            return [], next_det_id

        result = results[0]
        h, w = image_rgb_u8.shape[:2]

        if result.masks is None or result.masks.xy is None:
            return [], next_det_id

        detections: list[Detection] = []
        masks_out: list = []
        boxes = result.boxes

        for i, poly_raw in enumerate(result.masks.xy):
            poly = np.array(poly_raw, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue

            mask_u8 = polygon_to_mask(poly, h, w)
            geom = contour_geometry_from_mask(mask_u8)
            if geom is None:
                continue

            score = 1.0
            if boxes is not None and i < len(boxes):
                try:
                    score = float(boxes.conf[i].item())
                except Exception:
                    pass

            detections.append(Detection(
                det_id=next_det_id,
                frame_idx=frame_idx,
                image_name=image_name,
                cx=geom["cx"],
                cy=geom["cy"],
                angle_deg=geom["angle_deg"],
                length_px=geom["length_px"],
                width_px=geom["width_px"],
                area_px=geom["area_px"],
                score=score,
                bbox_xyxy=geom["bbox_xyxy"],
            ))
            masks_out.append(mask_u8)
            next_det_id += 1

        if return_masks:
            return detections, next_det_id, masks_out
        return detections, next_det_id
