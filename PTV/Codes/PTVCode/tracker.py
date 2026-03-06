# tracker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

from config import TrackingConfig
from models import Detection, Track, TrackState
from abg_filter import ABGFilter, normalize_angle_deg, shortest_angle_diff_deg


@dataclass
class Tracker:
    cfg: TrackingConfig

    def __post_init__(self) -> None:
        self.filt = ABGFilter(self.cfg)
        self.tracks: Dict[str, Track] = {}
        self.next_id: int = 0

    def _new_id(self) -> str:
        self.next_id += 1
        return str(self.next_id)

    def _init_track(self, det: Detection, frame_idx: int) -> Track:
        tid = self._new_id()
        s = TrackState(x=det.cx, y=det.cy, angle_deg=normalize_angle_deg(det.angle_deg), length_px=det.length_px)
        tr = Track(track_id=tid, state=s, history={
            "centroide": [[det.cx, det.cy]],
            "largo_maximo": [[det.length_px]],
            "angulo": [[normalize_angle_deg(det.angle_deg)]],
            "frame": [[frame_idx]],
            "estado": [self._state_to_list(s)],
        })
        return tr

    def _state_to_list(self, s: TrackState) -> list:
        # para JSON “compatible” con tu formato (similar a kalman list)
        return [
            [s.x, s.y],
            [s.vx, s.vy],
            [s.ax, s.ay],
            [s.angle_deg],
            [s.omega],
            [s.alpha_ang],
            [s.length_px],
        ]

    def _gating_ok(self, pred: TrackState, det: Detection) -> bool:
        dx = abs(pred.x - det.cx)
        dy = abs(pred.y - det.cy)
        if dx > self.cfg.gate_x_px or dy > self.cfg.gate_y_px:
            return False

        dtheta = abs(shortest_angle_diff_deg(det.angle_deg, pred.angle_deg))
        if dtheta > self.cfg.gate_angle_deg:
            return False

        return True

    def _cost(self, pred: TrackState, det: Detection) -> float:
        # costo simple: distancia + peso angular
        dx = pred.x - det.cx
        dy = pred.y - det.cy
        d = math.sqrt(dx*dx + dy*dy)
        dtheta = abs(shortest_angle_diff_deg(det.angle_deg, pred.angle_deg))
        return d + 0.3 * dtheta

    def step(self, detections: List[Detection], frame_idx: int, dt: float) -> Dict[int, str]:
        """
        Procesa un frame.
        Retorna mapping: index_det -> track_id asignado
        """
        assigned: Dict[int, str] = {}
        used_tracks: set[str] = set()

        # 1) predecir todos los tracks existentes
        preds: Dict[str, TrackState] = {}
        for tid, tr in self.tracks.items():
            preds[tid] = self.filt.predict(tr.state, dt)

        # 2) construir todas las parejas (det, track) válidas por gating
        candidates: List[Tuple[float, int, str]] = []
        for i, det in enumerate(detections):
            for tid, pred in preds.items():
                if self._gating_ok(pred, det):
                    candidates.append((self._cost(pred, det), i, tid))

        # 3) asignación greedy por menor costo
        candidates.sort(key=lambda x: x[0])
        for cost, i, tid in candidates:
            if i in assigned:
                continue
            if tid in used_tracks:
                continue
            assigned[i] = tid
            used_tracks.add(tid)

        # 4) actualizar tracks asignados; crear tracks nuevos para detecciones no asignadas
        for i, det in enumerate(detections):
            if i in assigned:
                tid = assigned[i]
                pred = preds[tid]
                new_state = self.filt.update(pred, det, dt)
                tr = self.tracks[tid]
                tr.state = new_state
                tr.history["centroide"].append([det.cx, det.cy])
                tr.history["largo_maximo"].append([det.length_px])
                tr.history["angulo"].append([normalize_angle_deg(det.angle_deg)])
                tr.history["frame"].append([frame_idx])
                tr.history["estado"].append(self._state_to_list(new_state))
            else:
                tr = self._init_track(det, frame_idx)
                self.tracks[tr.track_id] = tr
                assigned[i] = tr.track_id

        return assigned

    def export_dict(self) -> dict:
        out = {}
        for tid, tr in self.tracks.items():
            out[tid] = tr.history
        return out