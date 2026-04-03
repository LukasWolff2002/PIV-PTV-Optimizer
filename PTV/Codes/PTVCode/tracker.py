"""
tracker.py
==========
Tracker multi-objeto con Similarity Search Scheme vectorizado.

En vez del gate secuencial (dx, dy, dangle por separado), construye
un vector de 5 características por fibra y usa similitud coseno para
correlacionar tracks con detecciones en una sola operación matricial.

Vector de características (normalizado L2):
    [ w1·cos(2θ),  w2·sin(2θ),  w3·L/L_ref,  w4·cx/W,  w5·cy/H ]

Usar cos(2θ)/sin(2θ) en vez de cos(θ)/sin(θ) resuelve la simetría:
una fibra es bidireccional → θ y θ+180° deben dar el mismo vector.
Con la doble frecuencia, cos(2·0°)=cos(2·180°)=1 ✓

El match se acepta si la similitud coseno supera sim_threshold.
Un gate espacial duro (max_dist_px) actúa como salvaguarda adicional.
"""
from __future__ import annotations

import math
import numpy as np

from .config import TrackingConfig
from .models import Detection, Track, TrackRecord, TrackState
from .filters import predict_state_abg, update_state_abg


# ─────────────────────────────────────────────
# FEATURE VECTOR
# ─────────────────────────────────────────────

def _fiber_feature_vector(
    cx: float,
    cy: float,
    angle_deg: float,
    length_px: float,
    width_px: float,
    height_px: float,
    l_ref: float,
    weights: tuple[float, float, float, float, float],
) -> np.ndarray:
    """
    Construye vector de características de una fibra.

    Parámetros
    ----------
    cx, cy       : centroide en píxeles
    angle_deg    : ángulo en grados (la fibra es bidireccional)
    length_px    : largo en píxeles
    width_px     : ancho de imagen (para normalizar cx)
    height_px    : alto de imagen (para normalizar cy)
    l_ref        : largo de referencia para normalizar length_px
    weights      : (w_cos, w_sin, w_len, w_cx, w_cy)

    Retorna
    -------
    np.ndarray shape (5,)  sin normalizar L2 (la normalización se hace fuera)
    """
    rad2 = math.radians(2.0 * angle_deg)   # doble frecuencia → simetría 180°
    w_cos, w_sin, w_len, w_cx, w_cy = weights

    v = np.array([
        w_cos * math.cos(rad2),
        w_sin * math.sin(rad2),
        w_len * (length_px / max(l_ref, 1e-6)),
        w_cx  * (cx / max(width_px, 1e-6)),
        w_cy  * (cy / max(height_px, 1e-6)),
    ], dtype=np.float64)
    return v


def _build_feature_matrix(
    items: list,
    get_cx: callable,
    get_cy: callable,
    get_angle: callable,
    get_length: callable,
    width_px: float,
    height_px: float,
    l_ref: float,
    weights: tuple,
) -> np.ndarray:
    """
    Construye matriz de features F ∈ ℝ^(N×5) con filas normalizadas L2.
    Cada fila representa un track predicho o una detección.
    """
    n = len(items)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float64)

    F = np.zeros((n, 5), dtype=np.float64)
    for i, item in enumerate(items):
        v = _fiber_feature_vector(
            cx=get_cx(item),
            cy=get_cy(item),
            angle_deg=get_angle(item),
            length_px=get_length(item),
            width_px=width_px,
            height_px=height_px,
            l_ref=l_ref,
            weights=weights,
        )
        norm = np.linalg.norm(v)
        F[i] = v / norm if norm > 1e-9 else v
    return F


# ─────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────

class Tracker:
    """
    Tracker multi-objeto con Similarity Search Scheme.

    Parámetros adicionales en TrackingConfig (leídos con getattr + default):
        sim_threshold  : float = 0.85  — similitud coseno mínima para aceptar match
        max_dist_px    : float = 80.0  — gate espacial duro (Euclidean, píxeles)
        feat_weights   : tuple = (1.0, 1.0, 0.5, 1.5, 1.5)
                         pesos de (cos2θ, sin2θ, largo, cx, cy)
        l_ref_px       : float = 101.4 — largo de referencia para normalizar
                         (default: 13 mm × 7.8 px/mm)
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.active_tracks: list[Track] = []
        self.finished_tracks: list[Track] = []
        self.next_track_id = 1

        # Parámetros del similarity search
        self.sim_threshold: float = getattr(cfg, "sim_threshold", 0.85)
        self.max_dist_px: float   = getattr(cfg, "max_dist_px", 80.0)
        self.feat_weights: tuple  = getattr(cfg, "feat_weights", (1.0, 1.0, 0.5, 1.5, 1.5))
        self.l_ref_px: float      = getattr(cfg, "l_ref_px", 13.0 * 7.8)  # 101.4 px

    # ── helpers para los lambdas de build_feature_matrix ──────────

    def _track_feats(self, tracks: list[Track]) -> np.ndarray:
        return _build_feature_matrix(
            items=tracks,
            get_cx=lambda t: t.state.x,
            get_cy=lambda t: t.state.y,
            get_angle=lambda t: t.state.angle_deg,
            get_length=lambda t: t.state.length_px,
            width_px=float(self.cfg.width_px),
            height_px=float(self.cfg.height_px),
            l_ref=self.l_ref_px,
            weights=self.feat_weights,
        )

    def _det_feats(self, dets: list[Detection]) -> np.ndarray:
        return _build_feature_matrix(
            items=dets,
            get_cx=lambda d: d.cx,
            get_cy=lambda d: d.cy,
            get_angle=lambda d: d.angle_deg,
            get_length=lambda d: d.length_px,
            width_px=float(self.cfg.width_px),
            height_px=float(self.cfg.height_px),
            l_ref=self.l_ref_px,
            weights=self.feat_weights,
        )

    # ── asignación ────────────────────────────────────────────────

    def _similarity_matrix(
        self,
        Q: np.ndarray,   # (T, 5) tracks predichos
        D: np.ndarray,   # (N, 5) detecciones
    ) -> np.ndarray:
        """
        Calcula matriz de similitud coseno S ∈ ℝ^(T×N).
        S[i,j] = similitud entre track i y detección j.
        Rango [-1, 1]; mayor = más similar.
        """
        # Q y D ya vienen normalizados L2 fila a fila
        return Q @ D.T   # (T, N)

    def _spatial_gate(
        self,
        tracks: list[Track],
        dets: list[Detection],
    ) -> np.ndarray:
        """
        Máscara booleana (T×N): True si la distancia Euclidiana
        entre track i y detección j es ≤ max_dist_px.

        Vectorizado con broadcasting NumPy — O(T×N) sin loop Python.
        """
        T, N = len(tracks), len(dets)
        if T == 0 or N == 0:
            return np.zeros((T, N), dtype=bool)

        # Posiciones de tracks (T, 2) y detecciones (N, 2)
        trk_xy = np.array([[tr.state.x, tr.state.y] for tr in tracks], dtype=np.float64)
        det_xy = np.array([[d.cx, d.cy] for d in dets], dtype=np.float64)

        # Broadcasting: (T, 1, 2) - (1, N, 2) → (T, N, 2) → dist (T, N)
        diff = trk_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))

        return dist <= self.max_dist_px

    def _assign(
        self,
        tracks: list[Track],
        dets: list[Detection],
    ) -> list[tuple[int, int]]:
        """
        Asigna tracks a detecciones usando similitud coseno vectorizada.

        Retorna lista de pares (track_idx, det_idx) asignados,
        en orden descendente de similitud (greedy one-to-one).
        """
        T, N = len(tracks), len(dets)
        if T == 0 or N == 0:
            return []

        Q = self._track_feats(tracks)   # (T, 5)
        D = self._det_feats(dets)       # (N, 5)

        S = self._similarity_matrix(Q, D)   # (T, N)

        # Aplicar gate espacial: similaridad fuera del gate → -inf
        gate_mask = self._spatial_gate(tracks, dets)
        S[~gate_mask] = -np.inf

        # Aplicar umbral de similitud
        S[S < self.sim_threshold] = -np.inf

        # Greedy assignment: iterar por orden de similitud descendente
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        assignments: list[tuple[int, int]] = []

        # Aplanar y ordenar
        valid = np.argwhere(np.isfinite(S))
        if len(valid) == 0:
            return []

        scores = S[valid[:, 0], valid[:, 1]]
        order = np.argsort(-scores)   # descendente

        for idx in order:
            ti, di = int(valid[idx, 0]), int(valid[idx, 1])
            if ti in assigned_tracks or di in assigned_dets:
                continue
            assignments.append((ti, di))
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        return assignments

    # ── nuevo track ───────────────────────────────────────────────

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

    # ── step principal ────────────────────────────────────────────

    def step(
        self,
        detections: list[Detection],
        frame_idx: int,
        image_name: str,
    ) -> None:
        dt = self.cfg.dt

        # 1) Predicción ABG para todos los tracks activos
        for tr in self.active_tracks:
            tr.state = predict_state_abg(tr.state, dt)

        # 2) Asignación global por similitud coseno
        #    Se calcula la matriz completa S(T×N) antes de asignar cualquier par,
        #    garantizando que cada track compite por la mejor detección disponible
        #    considerando a TODAS las fibras simultáneamente.
        assignments = self._assign(self.active_tracks, detections)

        assigned_tracks: set[int] = {ti for ti, _ in assignments}
        assigned_dets:   set[int] = {di for _, di in assignments}

        # 3) Corrección ABG para tracks asignados
        for ti, di in assignments:
            tr  = self.active_tracks[ti]
            det = detections[di]

            tr.state = update_state_abg(
                pred=tr.state, det=det,
                alpha=self.cfg.alpha, beta=self.cfg.beta,
                gamma=self.cfg.gamma, dt=dt,
            )
            tr.hits  += 1
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

        # 4) Tracks no asignados: miss
        #    Si max_misses == 0 (default), el track termina inmediatamente
        #    cuando no es detectado — no se extrapola posición.
        #    Si max_misses > 0, se toleran N frames sin detección.
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

        # 5) Nuevos tracks para detecciones no asignadas a ningún track existente
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