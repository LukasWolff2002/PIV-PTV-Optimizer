# config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrackingConfig:
    # captura
    fps: float = 200.0
    max_images: Optional[int] = None

    # ABG gains (alpha-beta-gamma)
    alpha: float = 0.95
    beta: float = 0.95
    gamma: float = 0.05

    # gating (matching)
    gate_x_px: float = 10.0
    gate_y_px: float = 10.0
    gate_angle_deg: float = 5.0

    # YOLO
    conf: float = 0.25

    # filtro post
    min_frames_keep: int = 20

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)