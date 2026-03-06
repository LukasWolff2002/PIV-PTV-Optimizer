# models.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class Detection:
    cx: float
    cy: float
    angle_deg: float
    length_px: float
    score: float
    box_xyxy: np.ndarray  # shape (4,)


@dataclass
class TrackState:
    # estado dinámico XY
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0

    # estado angular
    angle_deg: float = 0.0
    omega: float = 0.0
    alpha_ang: float = 0.0

    # “atributo” (no dinámico)
    length_px: float = 0.0


@dataclass
class Track:
    track_id: str
    state: TrackState
    history: Dict[str, List]  # centroide, angulo, largo, frame, kalman-like states


@dataclass(frozen=True)
class RunPaths:
    base: Path
    images_dir: Path
    weights_path: Path
    runs_segment_dir: Path