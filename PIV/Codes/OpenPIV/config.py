from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PIVConfig:
    # =========================
    # Paths
    # =========================
    images_dir: Path
    masks_dir: Path
    out_dir: Path

    # =========================
    # Timing
    # =========================
    dt_ms: float

    # =========================
    # Calibration
    # =========================
    px_per_mm: float

    # =========================
    # PIV params
    # =========================
    window_sizes: List[int]
    overlaps: List[int]
    search_area_factor: int
    sig2noise_method: str

    # =========================
    # Masking
    # =========================
    # Solo usa máscara dinámica por frame desde masks_dir
    mask_threshold: float
    apply_dynamic_mask: bool = True  # siempre True en este modo

    # =========================
    # Viewer
    # =========================
    default_quiver_scale: float = 8.0
    quiver_width: float = 0.0025

    # =========================
    # Velocity-based validation
    # =========================
    keep_percentile: float = 90.0

    # =========================
    # Local median validation
    # =========================
    lm_kernel: int = 1
    lm_thresh: float = 2.0
    lm_eps: float = 0.1

    # =========================
    # Helpers
    # =========================
    def dt_s(self) -> float:
        return self.dt_ms / 1000.0

    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm