"""
models.py
=========
Dataclasses del dominio PTV.

Unidades de salida:
- Posición  : mm  (x_mm, y_mm)
- Velocidad : mm/s
- Aceleración: mm/s²
- Ángulo    : grados
- Longitud/ancho: mm

El campo dt_s en TrackRecord almacena el Δt real usado para esa observación,
lo que permite análisis correcto de trayectorias con timestep variable.

Campos DPTV (Depth-from-Defocus PTV):
- defocus_score   : W_aparente / W_ideal  (1.0 = en foco, >1 = desenfocado)
- depth_blur_px   : contribución del blur de desenfoque en píxeles
- depth_blur_mm   : blur de desenfoque en mm (proporcional a Δz)
- depth_mm        : estimación de |Δz| desde el plano focal (None sin calibración)
- depth_confidence: calidad de la estimación [0, 1]
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class Detection:
    det_id: int
    frame_idx: int        # índice en la secuencia ORIGINAL (no seleccionada)
    image_name: str
    cx: float             # px
    cy: float             # px
    angle_deg: float
    length_px: float
    width_px: float
    area_px: float
    score: float
    bbox_xyxy: list[float]

    # DPTV depth-from-defocus fields (populated by DPTVEstimator in detector.detect())
    defocus_score:    float        = 1.0
    depth_blur_px:    float        = 0.0
    depth_blur_mm:    float        = 0.0
    depth_mm:         float | None = None
    depth_confidence: float        = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackState:
    """Estado interno del filtro ABG — trabaja en píxeles para el SSS."""
    x: float          # px
    y: float          # px
    vx: float = 0.0   # px/s
    vy: float = 0.0   # px/s
    ax: float = 0.0   # px/s²
    ay: float = 0.0   # px/s²
    angle_deg: float = 0.0
    omega: float = 0.0         # deg/s
    alpha_ang: float = 0.0     # deg/s²
    length_px: float = 0.0
    width_px: float = 0.0


@dataclass
class TrackRecord:
    """
    Registro de una observación del track.

    Posiciones y velocidades se almacenan en mm para facilitar el análisis.
    dt_s es el Δt real de esta observación (puede variar entre regiones).
    """
    frame_idx: int         # índice en secuencia ORIGINAL
    region_name: str       # nombre de la región temporal ("alta_velocidad", etc.)
    region_idx: int        # índice de región (0-based)
    image_name: str
    timestamp_s: float     # tiempo absoluto = frame_idx / fps

    # Posición en mm
    x_mm: float
    y_mm: float

    # Velocidad en mm/s
    vx_mm_s: float
    vy_mm_s: float

    # Aceleración en mm/s²
    ax_mm_s2: float
    ay_mm_s2: float

    # Orientación
    angle_deg: float
    omega_deg_s: float      # velocidad angular deg/s
    alpha_ang_deg_s2: float # aceleración angular deg/s²

    # Geometría en mm
    length_mm: float
    width_mm: float

    # Timestep efectivo de esta observación
    dt_s: float

    det_id: int | None = None

    # DPTV depth-from-defocus (copied from the matched Detection)
    defocus_score:    float        = 1.0
    depth_blur_px:    float        = 0.0
    depth_blur_mm:    float        = 0.0
    depth_mm:         float | None = None
    depth_confidence: float        = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Track:
    track_id: int
    state: TrackState          # en px — para el SSS
    hits: int = 0
    misses: int = 0
    is_active: bool = True
    history: list[TrackRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        from .dptv import DPTVEstimator
        return {
            "track_id":   self.track_id,
            "hits":       self.hits,
            "misses":     self.misses,
            "is_active":  self.is_active,
            "depth_stats": DPTVEstimator.track_depth_stats(self.history),
            "history":    [r.to_dict() for r in self.history],
        }