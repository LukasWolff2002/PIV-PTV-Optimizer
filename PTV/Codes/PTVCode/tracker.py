"""
tracker.py
==========
Tracker multi-objeto con filtro ABG y gate espacial + angular.
"""
from __future__ import annotations

from .config import TrackingConfig
from .models import Detection, Track, TrackRecord, TrackState
from .filters import predict_state_abg, update_state_abg
from .image_utils import angle_diff_deg


class Tracker:
    """
    Asigna detecciones a tracks existentes usando cost mínimo con gate.
    Crea nuevos tracks para detecciones no asignadas.
    Elimina tracks con demasiados misses.
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.active_tracks: list[Track] = []
        self.finished_tracks: list[Track] = []
        self.next_track_id = 1

    def _candidate_cost(self, track: Track, det: Detection) -> float | None:
        """Retorna costo de asignación o None si está fuera del gate."""
        pred = track.state
        dx = det.cx - pred.x
        dy = det.cy - pred.y
        da = angle_diff_deg(det.angle_deg, pred.angle_deg)

        if abs(dx) > self.cfg.gate_x_px:
            return None
        if abs(dy) > self.cfg.gate_y_px:
            return None
        if da > self.cfg.gate_angle_deg:
            return None

        sx = abs(dx) / max(self.cfg.gate_x_px, 1e-12)
        sy = abs(dy) / max(self.cfg.gate_y_px, 1e-12)
        sa = da / max(self.cfg.gate_angle_deg, 1e-12)
        return float(sx + sy + sa)

    def _new_track(self, det: Detection, image_name: str) -> Track:
        state = TrackState(
            x=det.cx, y=det.cy,
            angle_deg=det.angle_deg,
            length_px=det.length_px,
            width_px=det.width_px,
        )
        tr = Track(track_id=self.next_track_id, state=state, hits=1)
        tr.history.append(TrackRecord(
            frame_idx=det.frame_idx, image_name=image_name,
            x=state.x, y=state.y,
            vx=state.vx, vy=state.vy,
            ax=state.ax, ay=state.ay,
            angle_deg=state.angle_deg,
            omega=state.omega, alpha_ang=state.alpha_ang,
            length_px=state.length_px, width_px=state.width_px,
            det_id=det.det_id,
        ))
        self.next_track_id += 1
        return tr

    def step(
        self,
        detections: list[Detection],
        frame_idx: int,
        image_name: str,
    ) -> None:
        dt = self.cfg.dt

        # Predicción
        for tr in self.active_tracks:
            tr.state = predict_state_abg(tr.state, dt)

        # Construir lista de candidatos ordenada por costo
        candidates: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.active_tracks):
            for di, det in enumerate(detections):
                cost = self._candidate_cost(tr, det)
                if cost is not None:
                    candidates.append((cost, ti, di))
        candidates.sort(key=lambda x: x[0])

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        for _, ti, di in candidates:
            if ti in assigned_tracks or di in assigned_dets:
                continue

            tr = self.active_tracks[ti]
            det = detections[di]

            tr.state = update_state_abg(
                pred=tr.state, det=det,
                alpha=self.cfg.alpha, beta=self.cfg.beta,
                gamma=self.cfg.gamma, dt=dt,
            )
            tr.hits += 1
            tr.misses = 0
            tr.history.append(TrackRecord(
                frame_idx=frame_idx, image_name=image_name,
                x=tr.state.x, y=tr.state.y,
                vx=tr.state.vx, vy=tr.state.vy,
                ax=tr.state.ax, ay=tr.state.ay,
                angle_deg=tr.state.angle_deg,
                omega=tr.state.omega, alpha_ang=tr.state.alpha_ang,
                length_px=tr.state.length_px, width_px=tr.state.width_px,
                det_id=det.det_id,
            ))
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        # Gestión de misses
        survivors: list[Track] = []
        for ti, tr in enumerate(self.active_tracks):
            if ti not in assigned_tracks:
                tr.misses += 1
                if tr.misses <= self.cfg.max_misses:
                    survivors.append(tr)
                else:
                    tr.is_active = False
                    self.finished_tracks.append(tr)
            else:
                survivors.append(tr)
        self.active_tracks = survivors

        # Nuevos tracks para detecciones no asignadas
        for di, det in enumerate(detections):
            if di not in assigned_dets:
                self.active_tracks.append(self._new_track(det, image_name))

    def close_all(self) -> None:
        for tr in self.active_tracks:
            tr.is_active = False
            self.finished_tracks.append(tr)
        self.active_tracks = []

    def get_all_tracks(self) -> list[Track]:
        return list(self.finished_tracks) + list(self.active_tracks)
