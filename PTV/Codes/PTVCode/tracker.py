"""
tracker.py
==========
Tracker multi-objeto con Similarity Search Scheme vectorizado.

Convención de máscaras:
  blanco (valor alto) = zona a IGNORAR (no analizar)
  negro  (valor bajo) = zona a ANALIZAR

Cambios respecto a la versión anterior:
- _load_static_mask: eliminado ImageOps.invert que invertía la convención.
  Ahora blanco=ignorar se respeta directamente sin transformación.
- _load_mask_for_image: threshold adaptativo según rango real del archivo
  (funciona con uint8, uint16 y float [0,1] sin configuración adicional).
- step() recibe max_dist_px explícito por frame (gate por región temporal).
- step() recibe dt_s explícito → soporta timestep variable entre regiones.
- TrackRecord se guarda en mm (requiere px_per_mm en config).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import TrackingConfig
from .models import Detection, Track, TrackRecord, TrackState
from .filters import predict_state_abg, update_state_abg


# ─────────────────────────────────────────────
# THRESHOLD ADAPTATIVO
# ─────────────────────────────────────────────

def _auto_threshold(m: np.ndarray) -> float:
    """
    Threshold al 50% del rango máximo del array.
    Funciona con cualquier tipo de máscara binaria:
      float [0, 1]      → 0.5
      uint8 [0, 255]    → 127.0
      uint16 [0, 65535] → 32767.0
    """
    vmax = float(m.max())
    if vmax <= 1.0:
        return 0.5
    elif vmax <= 255.0:
        return 127.0
    else:
        return 32767.0


# ─────────────────────────────────────────────
# CARGA DE MÁSCARAS
# ─────────────────────────────────────────────

def _natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _build_mask_map(masks_dir: Path) -> dict[str, Path]:
    """Construye mapa stem → path para máscaras ya generadas."""
    out: dict[str, Path] = {}
    for p in masks_dir.glob("*.tif*"):
        stem = p.stem
        if stem.endswith("_mask"):
            img_stem = stem[:-5]
            if img_stem not in out:
                out[img_stem] = p
    return out


def _load_mask_for_image(
    mask_map: dict[str, Path],
    img_name: str,
) -> Optional[np.ndarray]:
    """
    Carga máscara dinámica para una imagen.
    Retorna bool array (H, W) o None.

    Convención: True = enmascarado (ignorar).
      blanco (valor alto) → True  → descartar detección
      negro  (valor bajo) → False → conservar detección

    Threshold adaptativo: funciona con uint8, uint16 y float [0,1].
    """
    stem = Path(img_name).stem
    mask_path = mask_map.get(stem)
    if mask_path is None:
        return None
    try:
        import tifffile
        m = tifffile.imread(str(mask_path)).astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        return m > _auto_threshold(m)
    except Exception as e:
        print(f"[WARN] No se pudo cargar máscara {mask_path}: {e}", flush=True)
        return None


# ─────────────────────────────────────────────
# FEATURE VECTOR
# ─────────────────────────────────────────────

def _fiber_feature_vector(
    cx: float, cy: float,
    angle_deg: float, length_px: float,
    width_px: float, height_px: float,
    l_ref: float,
    weights: tuple[float, float, float, float, float],
) -> np.ndarray:
    rad2 = math.radians(2.0 * angle_deg)
    w_cos, w_sin, w_len, w_cx, w_cy = weights
    return np.array([
        w_cos * math.cos(rad2),
        w_sin * math.sin(rad2),
        w_len * (length_px / max(l_ref, 1e-6)),
        w_cx  * (cx / max(width_px, 1e-6)),
        w_cy  * (cy / max(height_px, 1e-6)),
    ], dtype=np.float64)


def _build_feature_matrix(items, get_cx, get_cy, get_angle, get_length,
                           width_px, height_px, l_ref, weights) -> np.ndarray:
    n = len(items)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float64)
    F = np.zeros((n, 5), dtype=np.float64)
    for i, item in enumerate(items):
        v = _fiber_feature_vector(
            get_cx(item), get_cy(item), get_angle(item), get_length(item),
            width_px, height_px, l_ref, weights,
        )
        norm = np.linalg.norm(v)
        F[i] = v / norm if norm > 1e-9 else v
    return F


# ─────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────

class Tracker:
    """
    Tracker multi-objeto con Similarity Search Scheme y filtro ABG.

    Convención de máscaras:
      blanco (valor alto) = ignorar  → True  → detección descartada
      negro  (valor bajo) = analizar → False → detección conservada

    El gate espacial (max_dist_px) se recibe en cada step() desde el runner,
    calculado a partir del max_dist_mm definido por región temporal en variables_ptv.py.
    """

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.active_tracks:   list[Track] = []
        self.finished_tracks: list[Track] = []
        self.next_track_id = 1

        self.sim_threshold: float    = getattr(cfg, "sim_threshold", 0.85)
        self._default_max_dist_px: float = getattr(cfg, "max_dist_px", 80.0)
        self.feat_weights: tuple     = getattr(cfg, "feat_weights", (1.0, 1.0, 0.5, 1.5, 1.5))
        self.l_ref_px: float         = getattr(cfg, "l_ref_px", 13.0 * 7.8)
        self.px_per_mm: float        = cfg.px_per_mm

        # Mapa de máscaras dinámicas (precargado una vez)
        self._mask_map: dict[str, Path] = {}
        if cfg.apply_dynamic_mask and cfg.masks_dir and cfg.masks_dir.exists():
            self._mask_map = _build_mask_map(cfg.masks_dir)
            print(
                f"[TRACKER] Máscaras dinámicas: {len(self._mask_map)} archivos "
                f"en {cfg.masks_dir}",
                flush=True,
            )
        else:
            print("[TRACKER] Sin máscaras dinámicas.", flush=True)

        # Máscara fija (estática) — se carga una vez
        self._static_mask: Optional[np.ndarray] = None
        if cfg.apply_static_mask and cfg.fixed_mask_path and cfg.fixed_mask_path.exists():
            self._static_mask = self._load_static_mask(cfg.fixed_mask_path)
            if self._static_mask is not None:
                n_ignore = int(self._static_mask.sum())
                n_total  = self._static_mask.size
                print(
                    f"[TRACKER] Máscara fija: {n_ignore}/{n_total} px ignorados "
                    f"({100 * n_ignore / n_total:.1f}%)",
                    flush=True,
                )
        else:
            print("[TRACKER] Sin máscara fija.", flush=True)

    def _load_static_mask(self, path: Path) -> Optional[np.ndarray]:
        """
        Carga máscara fija. Retorna bool array (H, W) o None.

        Convención: True = enmascarado (ignorar).
          blanco (valor alto) → True  → ignorar
          negro  (valor bajo) → False → analizar

        Threshold adaptativo: funciona con uint8, uint16 y float [0,1].
        NO invierte la imagen — la convención blanco=ignorar se aplica directamente.
        """
        try:
            import tifffile
            raw = tifffile.imread(str(path))
            m   = raw.astype(np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            thr  = _auto_threshold(m)
            vmin = float(m.min())
            vmax = float(m.max())
            print(
                f"[TRACKER] Máscara fija: dtype={raw.dtype} "
                f"rango=[{vmin:.1f}, {vmax:.1f}] → threshold={thr:.1f}",
                flush=True,
            )
            return m > thr
        except Exception as e:
            print(f"[WARN] No se pudo cargar máscara fija {path}: {e}", flush=True)
            return None

    def _get_mask_for_frame(self, img_name: str) -> Optional[np.ndarray]:
        """
        Retorna máscara combinada (dinámica | estática).
        True = enmascarado (ignorar).
        """
        dyn = None
        if self._mask_map:
            dyn = _load_mask_for_image(self._mask_map, img_name)

        if dyn is None and self._static_mask is None:
            return None
        if dyn is None:
            return self._static_mask
        if self._static_mask is None:
            return dyn
        return dyn | self._static_mask

    def _det_in_mask(self, det: Detection, mask: Optional[np.ndarray]) -> bool:
        """True si el centroide de la detección cae en zona enmascarada."""
        if mask is None:
            return False
        cx = max(0, min(int(round(det.cx)), mask.shape[1] - 1))
        cy = max(0, min(int(round(det.cy)), mask.shape[0] - 1))
        return bool(mask[cy, cx])

    # ── Feature matrices ──────────────────────────────────────────

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

    # ── Asignación ────────────────────────────────────────────────

    def _spatial_gate(
        self,
        tracks: list[Track],
        dets: list[Detection],
        max_dist_px: float,
    ) -> np.ndarray:
        """Máscara booleana (T×N): True si dist ≤ max_dist_px."""
        T, N = len(tracks), len(dets)
        if T == 0 or N == 0:
            return np.zeros((T, N), dtype=bool)
        trk_xy = np.array([[tr.state.x, tr.state.y] for tr in tracks], dtype=np.float64)
        det_xy  = np.array([[d.cx, d.cy] for d in dets],               dtype=np.float64)
        diff    = trk_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :]
        dist    = np.sqrt((diff ** 2).sum(axis=2))
        return dist <= max_dist_px

    def _assign(
        self,
        tracks: list[Track],
        dets: list[Detection],
        max_dist_px: float,
    ) -> list[tuple[int, int]]:
        """Asignación SSS con gate espacial por región."""
        T, N = len(tracks), len(dets)
        if T == 0 or N == 0:
            return []

        Q = self._track_feats(tracks)
        D = self._det_feats(dets)
        S = Q @ D.T

        gate_mask = self._spatial_gate(tracks, dets, max_dist_px)
        S[~gate_mask]             = -np.inf
        S[S < self.sim_threshold] = -np.inf

        valid = np.argwhere(np.isfinite(S))
        if len(valid) == 0:
            return []

        scores = S[valid[:, 0], valid[:, 1]]
        order  = np.argsort(-scores)

        assigned_tracks: set[int] = set()
        assigned_dets:   set[int] = set()
        assignments: list[tuple[int, int]] = []

        for idx in order:
            ti, di = int(valid[idx, 0]), int(valid[idx, 1])
            if ti in assigned_tracks or di in assigned_dets:
                continue
            assignments.append((ti, di))
            assigned_tracks.add(ti)
            assigned_dets.add(di)

        return assignments

    # ── Helpers de registro ───────────────────────────────────────

    def _make_record(
        self,
        state: TrackState,
        det: Detection,
        frame_idx_original: int,
        image_name: str,
        timestamp_s: float,
        region_name: str,
        region_idx: int,
        dt_s: float,
    ) -> TrackRecord:
        mm = self.px_per_mm
        return TrackRecord(
            frame_idx=frame_idx_original,
            region_name=region_name,
            region_idx=region_idx,
            image_name=image_name,
            timestamp_s=timestamp_s,
            x_mm=state.x / mm,
            y_mm=state.y / mm,
            vx_mm_s=state.vx / mm,
            vy_mm_s=state.vy / mm,
            ax_mm_s2=state.ax / mm,
            ay_mm_s2=state.ay / mm,
            angle_deg=state.angle_deg,
            omega_deg_s=state.omega,
            alpha_ang_deg_s2=state.alpha_ang,
            length_mm=state.length_px / mm,
            width_mm=state.width_px / mm,
            dt_s=dt_s,
            det_id=det.det_id,
            defocus_score=det.defocus_score,
            depth_blur_px=det.depth_blur_px,
            depth_blur_mm=det.depth_blur_mm,
            depth_mm=det.depth_mm,
            depth_confidence=det.depth_confidence,
        )

    def _new_track(
        self,
        det: Detection,
        frame_idx_original: int,
        image_name: str,
        timestamp_s: float,
        region_name: str,
        region_idx: int,
        dt_s: float,
    ) -> Track:
        state = TrackState(
            x=det.cx, y=det.cy,
            angle_deg=det.angle_deg,
            length_px=det.length_px,
            width_px=det.width_px,
        )
        tr = Track(track_id=self.next_track_id, state=state, hits=1)
        tr.history.append(self._make_record(
            state=state, det=det,
            frame_idx_original=frame_idx_original,
            image_name=image_name,
            timestamp_s=timestamp_s,
            region_name=region_name,
            region_idx=region_idx,
            dt_s=dt_s,
        ))
        self.next_track_id += 1
        return tr

    # ── Step principal ────────────────────────────────────────────

    def step(
        self,
        detections: list[Detection],
        frame_idx_original: int,
        image_name: str,
        dt_s: float,
        max_dist_px: float,
        timestamp_s: float,
        region_name: str,
        region_idx: int,
    ) -> None:
        """
        Procesa un frame.

        dt_s y max_dist_px pueden cambiar entre llamadas al cambiar de región.
        Convención de máscara: blanco=ignorar, negro=analizar.
        """
        # ── 0) Filtrar detecciones en zona enmascarada ───────────
        mask = self._get_mask_for_frame(image_name)
        valid_dets = [d for d in detections if not self._det_in_mask(d, mask)]
        n_filtered = len(detections) - len(valid_dets)
        if n_filtered > 0:
            print(
                f"[TRACKER] frame {frame_idx_original}: "
                f"{n_filtered} det(s) en zona enmascarada → descartadas "
                f"({len(valid_dets)} restantes)",
                flush=True,
            )

        # ── 1) Predicción ABG ────────────────────────────────────
        for tr in self.active_tracks:
            tr.state = predict_state_abg(
                tr.state, dt_s,
                omega_decay=getattr(self.cfg, "omega_decay", 0.92),
                alpha_ang_decay=getattr(self.cfg, "alpha_ang_decay", 0.80),
            )

        # ── 2) Asignación SSS con gate de región ─────────────────
        assignments = self._assign(self.active_tracks, valid_dets, max_dist_px)

        assigned_tracks: set[int] = {ti for ti, _ in assignments}
        assigned_dets:   set[int] = {di for _, di in assignments}

        # ── 3) Corrección ABG y registro en mm ───────────────────
        for ti, di in assignments:
            tr  = self.active_tracks[ti]
            det = valid_dets[di]
            tr.state = update_state_abg(
                pred=tr.state, det=det,
                alpha=self.cfg.alpha, beta=self.cfg.beta,
                gamma=self.cfg.gamma, dt=dt_s,
                alpha_ang=getattr(self.cfg, "alpha_ang", 0.95),
                beta_ang=getattr(self.cfg,  "beta_ang",  0.05),
                gamma_ang=getattr(self.cfg, "gamma_ang", 0.001),
            )
            tr.hits   += 1
            tr.misses  = 0
            tr.history.append(self._make_record(
                state=tr.state, det=det,
                frame_idx_original=frame_idx_original,
                image_name=image_name,
                timestamp_s=timestamp_s,
                region_name=region_name,
                region_idx=region_idx,
                dt_s=dt_s,
            ))

        # ── 4) Tracks no asignados ───────────────────────────────
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

        # ── 5) Nuevos tracks ─────────────────────────────────────
        for di, det in enumerate(valid_dets):
            if di not in assigned_dets:
                self.active_tracks.append(self._new_track(
                    det=det,
                    frame_idx_original=frame_idx_original,
                    image_name=image_name,
                    timestamp_s=timestamp_s,
                    region_name=region_name,
                    region_idx=region_idx,
                    dt_s=dt_s,
                ))

    def close_all(self) -> None:
        for tr in self.active_tracks:
            tr.is_active = False
            self.finished_tracks.append(tr)
        self.active_tracks = []

    def get_all_tracks(self) -> list[Track]:
        return list(self.finished_tracks) + list(self.active_tracks)