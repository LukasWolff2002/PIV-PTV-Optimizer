"""
config.py
=========
TrackingConfig: configuración inmutable del pipeline PTV.
Incluye build_tracking_config (desde JSON) y validate_config.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


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


def build_tracking_config(cfg: dict) -> TrackingConfig:
    """Construye TrackingConfig desde el dict del JSON de pipeline."""
    ptv = cfg["ptv"]
    cam = cfg["camera"]

    return TrackingConfig(
        images_dir=Path(ptv["images_dir"]),
        out_dir=Path(ptv["out_dir"]),
        weights_path=Path(ptv["weights_path"]),
        runs_segment_dir=Path(ptv["runs_segment_dir"]) if ptv.get("runs_segment_dir") else None,
        fps=float(ptv["fps"]),
        px_per_mm=float(cam["px_per_mm"]),
        width_px=int(ptv["width_px"]),
        height_px=int(ptv["height_px"]),
        apply_dynamic_mask=bool(ptv.get("apply_dynamic_mask", False)),
        apply_static_mask=bool(ptv.get("apply_static_mask", False)),
        fixed_mask_path=Path(ptv["fixed_mask_path"]) if ptv.get("fixed_mask_path") else None,
        preprocess_params=ptv.get("preprocess_params"),
        max_images=ptv.get("max_images"),
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


def validate_config(cfg: TrackingConfig) -> None:
    """Valida existencia de rutas y valores lógicos."""
    if not cfg.images_dir.exists():
        raise FileNotFoundError(f"images_dir no existe: {cfg.images_dir}")
    if not cfg.images_dir.is_dir():
        raise NotADirectoryError(f"images_dir no es carpeta: {cfg.images_dir}")
    if not cfg.weights_path.exists():
        raise FileNotFoundError(f"weights_path no existe: {cfg.weights_path}")
    if cfg.apply_static_mask:
        if cfg.fixed_mask_path is None:
            raise ValueError("apply_static_mask=True pero fixed_mask_path es None")
        if not cfg.fixed_mask_path.exists():
            raise FileNotFoundError(f"fixed_mask_path no existe: {cfg.fixed_mask_path}")
    if cfg.fps <= 0:
        raise ValueError("fps debe ser > 0")
    if cfg.px_per_mm <= 0:
        raise ValueError("px_per_mm debe ser > 0")
    if cfg.width_px <= 0 or cfg.height_px <= 0:
        raise ValueError("width_px y height_px deben ser > 0")
