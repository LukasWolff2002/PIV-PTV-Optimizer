# abg_filter.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
from config import TrackingConfig
from models import TrackState, Detection


def normalize_angle_deg(a: float) -> float:
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


def shortest_angle_diff_deg(measured: float, filtered: float) -> float:
    diff = measured - filtered
    while diff > 180:
        diff -= 360
    while diff <= -180:
        diff += 360
    return diff


@dataclass(frozen=True)
class ABGFilter:
    cfg: TrackingConfig

    # --------------------
    # Predicción (sin medición)
    # --------------------
    def predict(self, s: TrackState, dt: float) -> TrackState:
        x = s.x + s.vx * dt + 0.5 * s.ax * dt * dt
        y = s.y + s.vy * dt + 0.5 * s.ay * dt * dt
        vx = s.vx + s.ax * dt
        vy = s.vy + s.ay * dt

        ang = normalize_angle_deg(s.angle_deg + s.omega * dt + 0.5 * s.alpha_ang * dt * dt)
        omega = s.omega + s.alpha_ang * dt

        return TrackState(
            x=x, y=y, vx=vx, vy=vy, ax=s.ax, ay=s.ay,
            angle_deg=ang, omega=omega, alpha_ang=s.alpha_ang,
            length_px=s.length_px
        )

    # --------------------
    # Corrección (usa medición)
    # --------------------
    def update(self, pred: TrackState, det: Detection, dt: float) -> TrackState:
        a = self.cfg.alpha
        b = self.cfg.beta
        g = self.cfg.gamma

        # posición
        rx = det.cx - pred.x
        ry = det.cy - pred.y

        x = pred.x + a * rx
        y = pred.y + a * ry

        vx = pred.vx + b * (rx / dt)
        vy = pred.vy + b * (ry / dt)

        ax = pred.ax + g * (2.0 * rx / (dt * dt))
        ay = pred.ay + g * (2.0 * ry / (dt * dt))

        # ángulo
        dtheta = shortest_angle_diff_deg(det.angle_deg, pred.angle_deg)
        ang = normalize_angle_deg(pred.angle_deg + a * dtheta)
        omega = pred.omega + b * (dtheta / dt)

        # tu fórmula original tenía factor raro (0.5); dejo versión consistente con posición:
        alpha_ang = pred.alpha_ang + g * (2.0 * dtheta / (dt * dt))

        # largo: lo tomamos como atributo medido (puedes suavizarlo si quieres)
        length_px = det.length_px

        return TrackState(
            x=x, y=y, vx=vx, vy=vy, ax=ax, ay=ay,
            angle_deg=ang, omega=omega, alpha_ang=alpha_ang,
            length_px=length_px
        )