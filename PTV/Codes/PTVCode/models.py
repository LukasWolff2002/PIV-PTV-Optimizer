"""
models.py
=========
Dataclasses del dominio PTV: detecciones y tracks.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


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
