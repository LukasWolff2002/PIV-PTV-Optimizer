# models.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class PairJob:
    pair_id: int
    img_a: Path
    img_b: Path
    mask_a: Path
    mask_b: Path
    name: str
    dt_ms: float  # ← AGREGADO: dt específico para este par

@dataclass(frozen=True)
class PIVResult:
    pair_id: int
    x_mm: np.ndarray
    y_mm: np.ndarray
    u_mms: np.ndarray
    v_mms: np.ndarray
    in_mask: np.ndarray
    bg_display: np.ndarray
    img_a: Path
    img_b: Path
    dt_ms: float  # ← AGREGADO: para trazabilidad

@dataclass(frozen=True)
class PIVResultFinal(PIVResult):
    pass