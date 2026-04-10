"""
filters.py
==========
Filtro alpha-beta-gamma (ABG) para predicción y actualización de estado.

Los parámetros angulares (alpha_ang, beta_ang, gamma_ang, omega_decay,
alpha_ang_decay) se reciben como argumentos — sus valores viven en
variables_ptv.py y se propagan a través de TrackingConfig. Esto permite
ajustarlos sin tocar el código del filtro.

Por qué parámetros angulares separados:
  Con dt = 1/220 s, usar BETA=0.95 en ángulo produce
    delta_omega = (0.95 / 0.00455) * residuo_deg ≈ 209 × residuo
  El ruido PCA del detector para fibras de aspect ~65 es ≈ 5-10°.
  → delta_omega ≈ 1460 deg/s por frame de ruido, cuando la fibra
    físicamente rota a ≤ 100 deg/s (ecuación de Jeffery, flujo laminar).
  Con BETA_ANG=0.05: delta_omega ≈ 77 deg/s — plausible.
"""
from __future__ import annotations
from .models import TrackState, Detection
from .image_utils import wrap_angle_deg, wrap_angle_fiber


def predict_state_abg(
    state: TrackState,
    dt: float,
    omega_decay: float     = 0.92,
    alpha_ang_decay: float = 0.80,
) -> TrackState:
    """
    Predicción cinemática con modelo de aceleración constante.

    Cinemática lineal  : estándar ABG sin cambios.
    Cinemática angular : aplica omega_decay y alpha_ang_decay para amortiguar
                         valores espúreos acumulados por ruido del detector.
                         Una fibra que realmente rota renueva su omega en cada
                         update y apenas nota el decay.

    Args:
        omega_decay     : fracción de omega que se retiene por frame [0, 1].
                          0.92 → omega espúreo cae al ~29% en 15 frames.
        alpha_ang_decay : fracción de alpha_ang retenida por frame [0, 1].
                          Más agresivo que omega: aceleración angular raramente
                          es física en flujo laminar.
    """
    omega_pred     = state.omega     * omega_decay     + state.alpha_ang * dt
    alpha_ang_pred = state.alpha_ang * alpha_ang_decay

    return TrackState(
        x=state.x  + state.vx * dt + 0.5 * state.ax * dt * dt,
        y=state.y  + state.vy * dt + 0.5 * state.ay * dt * dt,
        vx=state.vx + state.ax * dt,
        vy=state.vy + state.ay * dt,
        ax=state.ax,
        ay=state.ay,
        angle_deg=wrap_angle_deg(
            state.angle_deg + omega_pred * dt + 0.5 * alpha_ang_pred * dt * dt
        ),
        omega=omega_pred,
        alpha_ang=alpha_ang_pred,
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
    alpha_ang: float   = 0.95,
    beta_ang: float    = 0.05,
    gamma_ang: float   = 0.001,
) -> TrackState:
    """
    Corrección del estado con residuo de la detección.

    Cinemática lineal  : usa alpha, beta, gamma (sin cambios respecto
                         a la versión original).
    Cinemática angular : usa alpha_ang, beta_ang, gamma_ang separados
                         y más conservadores.

    Args:
        alpha, beta, gamma         : ganancias para x, y, vx, vy, ax, ay.
        alpha_ang, beta_ang, gamma_ang : ganancias para angle, omega, alpha_ang.
                         Defaults = valores de variables_ptv.py.
    """
    dt = max(dt, 1e-12)

    rx = det.cx  - pred.x
    ry = det.cy  - pred.y

    # Residuo angular con simetría π → acota a [-90°, 90°).
    # Evita que la ambigüedad θ vs θ+180° del eigenvector PCA genere
    # residuos de ±180° que dispararían omega con cualquier beta.
    ra = wrap_angle_fiber(det.angle_deg - pred.angle_deg)

    return TrackState(
        # ── Cinemática lineal ────────────────────────────────────────
        x=pred.x   + alpha * rx,
        y=pred.y   + alpha * ry,
        vx=pred.vx + (beta  / dt) * rx,
        vy=pred.vy + (beta  / dt) * ry,
        ax=pred.ax + (2.0 * gamma  / (dt * dt)) * rx,
        ay=pred.ay + (2.0 * gamma  / (dt * dt)) * ry,

        # ── Cinemática angular ───────────────────────────────────────
        # alpha_ang igual a alpha: corrige posición angular sin amplificar por 1/dt.
        # beta_ang << beta: evita omega espúreo por ruido PCA del detector.
        # gamma_ang ≈ 0: aceleración angular irrelevante en flujo laminar.
        angle_deg=wrap_angle_deg(pred.angle_deg + alpha_ang * ra),
        omega=pred.omega      + (beta_ang  / dt)              * ra,
        alpha_ang=pred.alpha_ang + (2.0 * gamma_ang / (dt * dt)) * ra,

        length_px=det.length_px,
        width_px=det.width_px,
    )