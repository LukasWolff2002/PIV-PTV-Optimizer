from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from ultralytics import YOLO


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from PTV.Codes.PreProcessing.filters import apply_preprocessing

# ============================================================
# HELPERS
# ============================================================

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wrap_angle_deg(angle_deg: float) -> float:
    """Normaliza ángulo a [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Diferencia angular mínima en grados."""
    return abs(wrap_angle_deg(a_deg - b_deg))


def np_to_builtin(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Tipo no serializable: {type(obj)}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )


def list_images(images_dir: Path, max_images: int | None = None) -> list[Path]:
    valid_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]
    imgs.sort(key=lambda p: natural_key(p.name))
    if max_images is not None:
        imgs = imgs[:max_images]
    return imgs


def read_image_any(path: Path) -> np.ndarray:
    """
    Retorna imagen como:
    - grayscale: (H, W)
    - color:     (H, W, 3) en RGB
    """
    ext = path.suffix.lower()

    if ext in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"No se pudo leer imagen: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return arr


def normalize_to_uint8_for_yolo(img: np.ndarray) -> np.ndarray:
    """
    Convierte a uint8 RGB o grayscale expandido a RGB para YOLO.
    """
    if img.ndim == 2:
        base = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        base = img[..., :3]
    else:
        raise ValueError(f"Formato de imagen no soportado: shape={img.shape}")

    if base.dtype == np.uint8:
        out = base.copy()
    elif base.dtype == np.uint16:
        out = np.clip(base / 257.0, 0, 255).astype(np.uint8)
    else:
        base_f = base.astype(np.float32)
        mn = float(np.min(base_f))
        mx = float(np.max(base_f))
        if mx <= mn:
            out = np.zeros_like(base_f, dtype=np.uint8)
        else:
            out = ((base_f - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out


def image_to_float01_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Replica la lógica de load_image(), pero sobre una imagen ya cargada en memoria.
    Retorna float64 en rango 0-1, en escala de grises.
    """
    arr = img.copy()

    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[..., 0]

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float64) / 65535.0
    else:
        arr = arr.astype(np.float64)
        mx = arr.max()
        if mx > 1.0:
            arr = arr / mx

    return arr


def preprocess_frame_for_ptv(raw_img: np.ndarray, preprocess_params: dict | None) -> np.ndarray:
    """
    Aplica el mismo preprocesamiento de PIV a una imagen de PTV.
    Salida: imagen RGB uint8 lista para YOLO.
    """
    if not preprocess_params:
        return normalize_to_uint8_for_yolo(raw_img)

    img01 = image_to_float01_grayscale(raw_img)
    img01_proc = apply_preprocessing(img01, preprocess_params)

    img_u8 = np.clip(img01_proc * 255.0, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    return img_rgb


def load_mask_as_bool(mask_path: Path, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    """
    Convención deseada:
    - negro  -> mantener
    - blanco -> eliminar
    """
    mask = read_image_any(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask_bool = mask == 0

    if expected_hw is not None:
        h, w = expected_hw
        if mask_bool.shape != (h, w):
            raise ValueError(
                f"La máscara fija tiene shape {mask_bool.shape}, "
                f"pero se esperaba {(h, w)}"
            )
    return mask_bool


def apply_static_mask_to_rgb(rgb: np.ndarray, static_mask_keep: np.ndarray) -> np.ndarray:
    """
    static_mask_keep=True donde se conserva imagen.
    """
    out = rgb.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError("Se esperaba imagen RGB uint8.")
    out[~static_mask_keep] = 0
    return out


def polygon_to_mask(poly_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def contour_geometry_from_mask(mask_u8: np.ndarray) -> dict | None:
    """
    Extrae geometría de una fibra desde la máscara binaria.
    Retorna:
        cx, cy, angle_deg, length_px, width_px, area_px, bbox_xyxy
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-12:
        return None

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(cnt)
    bbox_xyxy = [float(x), float(y), float(x + w), float(y + h)]

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), angle = rect

    if rw >= rh:
        length_px = float(rw)
        width_px = float(rh)
        angle_deg = float(angle)
    else:
        length_px = float(rh)
        width_px = float(rw)
        angle_deg = float(angle + 90.0)

    angle_deg = wrap_angle_deg(angle_deg)

    return {
        "cx": cx,
        "cy": cy,
        "angle_deg": angle_deg,
        "length_px": length_px,
        "width_px": width_px,
        "area_px": area,
        "bbox_xyxy": bbox_xyxy,
    }


# ============================================================
# CONFIG / MODELS
# ============================================================

@dataclass(frozen=True)
class TrackingConfig:
    images_dir: Path
    out_dir: Path
    weights_path: Path
    runs_segment_dir: Path | None

    fps: float
    px_per_mm: float
    width_px: int
    height_px: int

    apply_dynamic_mask: bool
    apply_static_mask: bool
    fixed_mask_path: Path | None
    preprocess_params: dict | None

    max_images: int | None
    alpha: float
    beta: float
    gamma: float

    gate_x_px: float
    gate_y_px: float
    gate_angle_deg: float

    conf: float
    min_frames_keep: int
    annotate: bool

    device: str | int | None = None
    max_misses: int = 2

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


@dataclass
class Detection:
    det_id: int
    frame_idx: int
    image_name: str
    cx: float
    cy: float
    angle_deg: float
    length_px: float
    width_px: float
    area_px: float
    score: float
    bbox_xyxy: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    angle_deg: float = 0.0
    omega: float = 0.0
    alpha_ang: float = 0.0
    length_px: float = 0.0
    width_px: float = 0.0


@dataclass
class TrackRecord:
    frame_idx: int
    image_name: str
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    angle_deg: float
    omega: float
    alpha_ang: float
    length_px: float
    width_px: float
    det_id: int | None = None


@dataclass
class Track:
    track_id: int
    state: TrackState
    hits: int = 0
    misses: int = 0
    is_active: bool = True
    history: list[TrackRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "hits": self.hits,
            "misses": self.misses,
            "is_active": self.is_active,
            "history": [asdict(h) for h in self.history],
        }


# ============================================================
# ABG FILTER
# ============================================================

def predict_state_abg(state: TrackState, dt: float) -> TrackState:
    return TrackState(
        x=state.x + state.vx * dt + 0.5 * state.ax * dt * dt,
        y=state.y + state.vy * dt + 0.5 * state.ay * dt * dt,
        vx=state.vx + state.ax * dt,
        vy=state.vy + state.ay * dt,
        ax=state.ax,
        ay=state.ay,
        angle_deg=wrap_angle_deg(state.angle_deg + state.omega * dt + 0.5 * state.alpha_ang * dt * dt),
        omega=state.omega + state.alpha_ang * dt,
        alpha_ang=state.alpha_ang,
        length_px=state.length_px,
        width_px=state.width_px,
    )


def update_state_abg(
    pred: TrackState,
    det: Detection,
    alpha: float,
    beta: float,
    gamma: float,
    dt: float,
) -> TrackState:
    rx = det.cx - pred.x
    ry = det.cy - pred.y
    ra = wrap_angle_deg(det.angle_deg - pred.angle_deg)

    x = pred.x + alpha * rx
    y = pred.y + alpha * ry

    vx = pred.vx + (beta / max(dt, 1e-12)) * rx
    vy = pred.vy + (beta / max(dt, 1e-12)) * ry

    ax = pred.ax + (2.0 * gamma / max(dt * dt, 1e-12)) * rx
    ay = pred.ay + (2.0 * gamma / max(dt * dt, 1e-12)) * ry

    angle_deg = wrap_angle_deg(pred.angle_deg + alpha * ra)
    omega = pred.omega + (beta / max(dt, 1e-12)) * ra
    alpha_ang = pred.alpha_ang + (2.0 * gamma / max(dt * dt, 1e-12)) * ra

    return TrackState(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        ax=ax,
        ay=ay,
        angle_deg=angle_deg,
        omega=omega,
        alpha_ang=alpha_ang,
        length_px=det.length_px,
        width_px=det.width_px,
    )


# ============================================================
# DETECTOR
# ============================================================

class FiberYOLODetector:
    def __init__(self, weights_path: Path, conf: float, device: str | int | None = None):
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
    ) -> tuple[list[Detection], int]:
        """
        Ejecuta segmentación YOLO sobre una imagen RGB uint8.
        Requiere modelo de segmentación. Si no hay máscaras, ignora detecciones.
        """
        pred_kwargs = {
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

        detections: list[Detection] = []

        boxes = result.boxes
        masks = result.masks

        if masks is None or masks.xy is None:
            return [], next_det_id

        n_inst = len(masks.xy)
        for i in range(n_inst):
            poly = np.array(masks.xy[i], dtype=np.float32)
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
                    score = 1.0

            det = Detection(
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
            )
            detections.append(det)
            next_det_id += 1

        return detections, next_det_id


# ============================================================
# TRACKER
# ============================================================

class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.active_tracks: list[Track] = []
        self.finished_tracks: list[Track] = []
        self.next_track_id = 1

    def _candidate_cost(self, track: Track, det: Detection) -> float | None:
        pred = track.state

        dx = det.cx - pred.x
        dy = det.cy - pred.y
        da = angle_diff_deg(det.angle_deg, pred.angle_deg)

        if abs(dx) > self.cfg.gate_x_px:
            return None
        if abs(dy) > self.cfg.gate_y_px:
            return None
        if da > self.cfg.gate_angle_deg:
            return None

        sx = abs(dx) / max(self.cfg.gate_x_px, 1e-12)
        sy = abs(dy) / max(self.cfg.gate_y_px, 1e-12)
        sa = da / max(self.cfg.gate_angle_deg, 1e-12)
        cost = sx + sy + sa
        return float(cost)

    def _new_track_from_detection(self, det: Detection, image_name: str) -> Track:
        state = TrackState(
            x=det.cx,
            y=det.cy,
            angle_deg=det.angle_deg,
            length_px=det.length_px,
            width_px=det.width_px,
        )
        tr = Track(track_id=self.next_track_id, state=state, hits=1, misses=0, is_active=True)
        tr.history.append(
            TrackRecord(
                frame_idx=det.frame_idx,
                image_name=image_name,
                x=state.x,
                y=state.y,
                vx=state.vx,
                vy=state.vy,
                ax=state.ax,
                ay=state.ay,
                angle_deg=state.angle_deg,
                omega=state.omega,
                alpha_ang=state.alpha_ang,
                length_px=state.length_px,
                width_px=state.width_px,
                det_id=det.det_id,
            )
        )
        self.next_track_id += 1
        return tr

    def step(self, detections: list[Detection], frame_idx: int, image_name: str) -> None:
        dt = self.cfg.dt

        for tr in self.active_tracks:
            tr.state = predict_state_abg(tr.state, dt)

        candidates: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.active_tracks):
            for di, det in enumerate(detections):
                cost = self._candidate_cost(tr, det)
                if cost is not None:
                    candidates.append((cost, ti, di))

        candidates.sort(key=lambda x: x[0])

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        for _, ti, di in candidates:
            if ti in assigned_tracks or di in assigned_dets:
                continue

            tr = self.active_tracks[ti]
            det = detections[di]

            tr.state = update_state_abg(
                pred=tr.state,
                det=det,
                alpha=self.cfg.alpha,
                beta=self.cfg.beta,
                gamma=self.cfg.gamma,
                dt=dt,
            )
            tr.hits += 1
            tr.misses = 0

            tr.history.append(
                TrackRecord(
                    frame_idx=frame_idx,
                    image_name=image_name,
                    x=tr.state.x,
                    y=tr.state.y,
                    vx=tr.state.vx,
                    vy=tr.state.vy,
                    ax=tr.state.ax,
                    ay=tr.state.ay,
                    angle_deg=tr.state.angle_deg,
                    omega=tr.state.omega,
                    alpha_ang=tr.state.alpha_ang,
                    length_px=tr.state.length_px,
                    width_px=tr.state.width_px,
                    det_id=det.det_id,
                )
            )

            assigned_tracks.add(ti)
            assigned_dets.add(di)

        survivors: list[Track] = []
        for ti, tr in enumerate(self.active_tracks):
            if ti not in assigned_tracks:
                tr.misses += 1
                if tr.misses <= self.cfg.max_misses:
                    survivors.append(tr)
                else:
                    tr.is_active = False
                    self.finished_tracks.append(tr)
            else:
                survivors.append(tr)

        self.active_tracks = survivors

        for di, det in enumerate(detections):
            if di not in assigned_dets:
                self.active_tracks.append(self._new_track_from_detection(det, image_name))

    def close_all(self) -> None:
        for tr in self.active_tracks:
            tr.is_active = False
            self.finished_tracks.append(tr)
        self.active_tracks = []

    def get_all_tracks(self) -> list[Track]:
        return list(self.finished_tracks) + list(self.active_tracks)


# ============================================================
# EXPORTERS
# ============================================================

def export_detections_csv(detections: list[Detection], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "det_id", "frame_idx", "image_name",
            "cx_px", "cy_px", "angle_deg",
            "length_px", "width_px", "area_px", "score",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ])
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            w.writerow([
                d.det_id, d.frame_idx, d.image_name,
                d.cx, d.cy, d.angle_deg,
                d.length_px, d.width_px, d.area_px, d.score,
                x1, y1, x2, y2,
            ])


def export_tracks_csv(tracks: list[Track], px_per_mm: float, fps: float, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "frame_idx", "image_name",
            "x_px", "y_px",
            "x_mm", "y_mm",
            "vx_px_s", "vy_px_s",
            "vx_mm_s", "vy_mm_s",
            "ax_px_s2", "ay_px_s2",
            "ax_mm_s2", "ay_mm_s2",
            "angle_deg", "omega_deg_s", "alpha_ang_deg_s2",
            "length_px", "width_px", "det_id",
        ])

        for tr in tracks:
            for rec in tr.history:
                w.writerow([
                    tr.track_id,
                    rec.frame_idx,
                    rec.image_name,
                    rec.x,
                    rec.y,
                    rec.x / px_per_mm,
                    rec.y / px_per_mm,
                    rec.vx,
                    rec.vy,
                    rec.vx / px_per_mm,
                    rec.vy / px_per_mm,
                    rec.ax,
                    rec.ay,
                    rec.ax / px_per_mm,
                    rec.ay / px_per_mm,
                    rec.angle_deg,
                    rec.omega,
                    rec.alpha_ang,
                    rec.length_px,
                    rec.width_px,
                    rec.det_id,
                ])


def export_tracks_json(tracks: list[Track], path: Path) -> None:
    data = {"tracks": [tr.to_dict() for tr in tracks]}
    save_json(data, path)


# ============================================================
# ANNOTATION
# ============================================================

def annotate_frame(
    image_rgb: np.ndarray,
    detections: list[Detection],
    tracks: list[Track],
    frame_idx: int,
    image_name: str,
    out_path: Path,
    gate_x_px: float,
    gate_y_px: float,
) -> None:
    canvas = image_rgb.copy()
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), (0, 220, 255), 1)
        cv2.circle(canvas_bgr, (int(round(d.cx)), int(round(d.cy))), 2, (0, 220, 255), -1)

        half = max(4, int(round(d.length_px / 2.0)))
        ang = math.radians(d.angle_deg)
        dx = int(round(math.cos(ang) * half))
        dy = int(round(math.sin(ang) * half))
        p1 = (int(round(d.cx - dx)), int(round(d.cy - dy)))
        p2 = (int(round(d.cx + dx)), int(round(d.cy + dy)))
        cv2.line(canvas_bgr, p1, p2, (255, 255, 0), 1)

    for tr in tracks:
        recs = [r for r in tr.history if r.frame_idx == frame_idx and r.image_name == image_name]
        if not recs:
            continue
        rec = recs[-1]

        cx = int(round(rec.x))
        cy = int(round(rec.y))

        cv2.circle(canvas_bgr, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(
            canvas_bgr,
            f"ID {tr.track_id}",
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        gx = int(round(gate_x_px))
        gy = int(round(gate_y_px))
        cv2.rectangle(
            canvas_bgr,
            (cx - gx, cy - gy),
            (cx + gx, cy + gy),
            (255, 0, 255),
            1,
        )

    cv2.imwrite(str(out_path), canvas_bgr)


# ============================================================
# CONFIG BUILD
# ============================================================

def build_tracking_config(cfg: dict) -> TrackingConfig:
    ptv = cfg["ptv"]
    cam = cfg["camera"]

    images_dir = Path(ptv["images_dir"])
    out_dir = Path(ptv["out_dir"])
    weights_path = Path(ptv["weights_path"])
    runs_segment_dir = Path(ptv["runs_segment_dir"]) if ptv.get("runs_segment_dir") else None

    fixed_mask_path = None
    if ptv.get("fixed_mask_path"):
        fixed_mask_path = Path(ptv["fixed_mask_path"])

    preprocess_params = ptv.get("preprocess_params", None)

    return TrackingConfig(
        images_dir=images_dir,
        out_dir=out_dir,
        weights_path=weights_path,
        runs_segment_dir=runs_segment_dir,
        fps=float(ptv["fps"]),
        px_per_mm=float(cam["px_per_mm"]),
        width_px=int(ptv["width_px"]),
        height_px=int(ptv["height_px"]),
        apply_dynamic_mask=bool(ptv.get("apply_dynamic_mask", False)),
        apply_static_mask=bool(ptv.get("apply_static_mask", False)),
        fixed_mask_path=fixed_mask_path,
        preprocess_params=preprocess_params,
        max_images=ptv.get("max_images", None),
        alpha=float(ptv["alpha"]),
        beta=float(ptv["beta"]),
        gamma=float(ptv["gamma"]),
        gate_x_px=float(ptv["gate_x_px"]),
        gate_y_px=float(ptv["gate_y_px"]),
        gate_angle_deg=float(ptv["gate_angle_deg"]),
        conf=float(ptv["conf"]),
        min_frames_keep=int(ptv["min_frames_keep"]),
        annotate=bool(ptv.get("annotate", False)),
        device=None,
        max_misses=2,
    )


def validate_config(run_cfg: TrackingConfig) -> None:
    if not run_cfg.images_dir.exists():
        raise FileNotFoundError(f"No existe images_dir: {run_cfg.images_dir}")
    if not run_cfg.images_dir.is_dir():
        raise NotADirectoryError(f"images_dir no es carpeta: {run_cfg.images_dir}")

    if not run_cfg.weights_path.exists():
        raise FileNotFoundError(f"No existe weights_path: {run_cfg.weights_path}")

    if run_cfg.apply_static_mask:
        if run_cfg.fixed_mask_path is None:
            raise ValueError("apply_static_mask=True pero fixed_mask_path es None")
        if not run_cfg.fixed_mask_path.exists():
            raise FileNotFoundError(f"No existe fixed_mask_path: {run_cfg.fixed_mask_path}")

    if run_cfg.fps <= 0:
        raise ValueError("fps debe ser > 0")
    if run_cfg.px_per_mm <= 0:
        raise ValueError("px_per_mm debe ser > 0")
    if run_cfg.width_px <= 0 or run_cfg.height_px <= 0:
        raise ValueError("width_px y height_px deben ser > 0")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python RunCode/ptv_run.py RunCode/pipeline_config.json")

    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe config JSON: {cfg_path}")

    cfg = load_json(cfg_path)
    run_cfg = build_tracking_config(cfg)
    validate_config(run_cfg)

    ensure_dir(run_cfg.out_dir)
    ann_dir = run_cfg.out_dir / "annotations"
    ensure_dir(ann_dir)

    images = list_images(run_cfg.images_dir, run_cfg.max_images)
    if not images:
        raise RuntimeError(f"No encontré imágenes en: {run_cfg.images_dir}")

    print(f"[PTV] images_dir: {run_cfg.images_dir}", flush=True)
    print(f"[PTV] out_dir: {run_cfg.out_dir}", flush=True)
    print(f"[PTV] weights_path: {run_cfg.weights_path}", flush=True)
    print(f"[PTV] frames: {len(images)}", flush=True)

    static_mask_keep = None

    detector = FiberYOLODetector(
        weights_path=run_cfg.weights_path,
        conf=run_cfg.conf,
        device=run_cfg.device,
    )
    tracker = Tracker(run_cfg)

    all_detections: list[Detection] = []
    next_det_id = 1

    for frame_idx, img_path in enumerate(images):
        print(f"[PTV] frame {frame_idx+1}/{len(images)} -> {img_path.name}", flush=True)

        raw = read_image_any(img_path)

        # 1) Preprocesamiento tipo PIV, pero con params PTV
        rgb_u8 = preprocess_frame_for_ptv(raw, run_cfg.preprocess_params)

        h, w = rgb_u8.shape[:2]
        if (h, w) != (run_cfg.height_px, run_cfg.width_px):
            print(
                f"[WARN] Shape imagen {img_path.name}: {(h, w)} "
                f"!= esperado {(run_cfg.height_px, run_cfg.width_px)}",
                flush=True,
            )

        # 2) Máscara fija invertida: negro mantiene, blanco elimina
        if run_cfg.apply_static_mask:
            if static_mask_keep is None:
                static_mask_keep = load_mask_as_bool(
                    run_cfg.fixed_mask_path,
                    expected_hw=(h, w),
                )
            rgb_u8 = apply_static_mask_to_rgb(rgb_u8, static_mask_keep)

        # 3) Segmentación YOLO
        detections, next_det_id = detector.detect(
            image_rgb_u8=rgb_u8,
            frame_idx=frame_idx,
            image_name=img_path.name,
            next_det_id=next_det_id,
        )

        all_detections.extend(detections)
        tracker.step(detections=detections, frame_idx=frame_idx, image_name=img_path.name)

        if run_cfg.annotate:
            tracks_now = tracker.get_all_tracks()
            annotate_frame(
                image_rgb=rgb_u8,
                detections=detections,
                tracks=tracks_now,
                frame_idx=frame_idx,
                image_name=img_path.name,
                out_path=ann_dir / f"{img_path.stem}.png",
                gate_x_px=run_cfg.gate_x_px,
                gate_y_px=run_cfg.gate_y_px,
            )

    tracker.close_all()
    tracks_all = tracker.get_all_tracks()

    tracks_filtered = [
        tr for tr in tracks_all
        if len(tr.history) >= run_cfg.min_frames_keep
    ]

    export_detections_csv(all_detections, run_cfg.out_dir / "detections.csv")
    export_tracks_csv(
        tracks_filtered,
        px_per_mm=run_cfg.px_per_mm,
        fps=run_cfg.fps,
        path=run_cfg.out_dir / "tracks.csv",
    )
    export_tracks_json(tracks_filtered, run_cfg.out_dir / "tracks.json")

    summary = {
        "meta": cfg.get("meta", {}),
        "camera": cfg.get("camera", {}),
        "ptv": cfg.get("ptv", {}),
        "results": {
            "n_frames": len(images),
            "n_detections": len(all_detections),
            "n_tracks_raw": len(tracks_all),
            "n_tracks_filtered": len(tracks_filtered),
            "min_frames_keep": run_cfg.min_frames_keep,
        },
    }
    save_json(summary, run_cfg.out_dir / "summary.json")

    print("[PTV] listo.", flush=True)
    print(f"[PTV] detections.csv -> {run_cfg.out_dir / 'detections.csv'}", flush=True)
    print(f"[PTV] tracks.csv     -> {run_cfg.out_dir / 'tracks.csv'}", flush=True)
    print(f"[PTV] tracks.json    -> {run_cfg.out_dir / 'tracks.json'}", flush=True)
    print(f"[PTV] summary.json   -> {run_cfg.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()