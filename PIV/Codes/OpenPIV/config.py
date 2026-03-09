# config.py
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
    mask_threshold: float
    apply_dynamic_mask: bool = True

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
    lm_thresh: float = 3.0
    lm_eps: float = 0.1

    # =========================
    # MEJORA #5: replace_outliers parametrizable
    # Antes hardcodeados: kernel_size=2, max_iter=3
    # =========================
    replace_outliers_kernel: int = 2
    replace_outliers_max_iter: int = 3

    # =========================
    # MEJORA #3: exportar grilla completa o solo válidos
    # True  => todas las filas (grilla consistente entre momentos)
    # False => solo vectores válidos (comportamiento anterior)
    # =========================
    export_full_grid: bool = True

    # =========================
    # Helpers
    # =========================
    def dt_s(self) -> float:
        return self.dt_ms / 1000.0

    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm