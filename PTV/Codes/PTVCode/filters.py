"""
filters.py
==========
Filtro alpha-beta-gamma (ABG) para predicción y actualización de estado.
"""
from __future__ import annotations
from .models import TrackState, Detection
from .image_utils import wrap_angle_deg, wrap_angle_fiber


def predict_state_abg(state: TrackState, dt: float) -> TrackState:
    """Predicción cinemática con modelo de aceleración constante."""
    return TrackState(
        x=state.x + state.vx * dt + 0.5 * state.ax * dt * dt,
        y=state.y + state.vy * dt + 0.5 * state.ay * dt * dt,
        vx=state.vx + state.ax * dt,
        vy=state.vy + state.ay * dt,
        ax=state.ax,
        ay=state.ay,
        angle_deg=wrap_angle_deg(
            state.angle_deg + state.omega * dt + 0.5 * state.alpha_ang * dt * dt
        ),
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
    """Corrección del estado con residuo de la detección."""
    dt = max(dt, 1e-12)

    rx = det.cx - pred.x
    ry = det.cy - pred.y

    # Residuo angular con simetría π: las fibras son indistinguibles a θ y θ+180°.
    # wrap_angle_fiber acota el residuo a [-90°, 90°), evitando que saltos
    # aparentes de ±180° (por ambigüedad de orientación en YOLO) disparen
    # omega a decenas de miles de deg/s y generen rotación espúrea infinita.
    ra = wrap_angle_fiber(det.angle_deg - pred.angle_deg)

    return TrackState(
        x=pred.x + alpha * rx,
        y=pred.y + alpha * ry,
        vx=pred.vx + (beta / dt) * rx,
        vy=pred.vy + (beta / dt) * ry,
        ax=pred.ax + (2.0 * gamma / (dt * dt)) * rx,
        ay=pred.ay + (2.0 * gamma / (dt * dt)) * ry,
        angle_deg=wrap_angle_deg(pred.angle_deg + alpha * ra),
        omega=pred.omega + (beta / dt) * ra,
        alpha_ang=pred.alpha_ang + (2.0 * gamma / (dt * dt)) * ra,
        length_px=det.length_px,
        width_px=det.width_px,
    )