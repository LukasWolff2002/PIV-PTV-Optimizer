"""
analisis_ptv_3d.py
==================
Animación 3D de fibras PTV con estimación de profundidad DPTV.

Sistema de coordenadas:
  X, Y : posición en el dominio (mm), igual que la imagen
  Z    : distancia al plano focal (PIV laser sheet)
         depth_blur_mm (siempre disponible) o depth_mm (si k_blur calibrado)

El plano PIV se muestra en Z=0. Las fibras aparecen "suspendidas" a distintas
alturas según su distancia al plano focal. Las más cercanas a Z=0 son las que
mejor correlacionan con la medición PIV.

Salidas por carpeta de experimento:
  ptv_3d_interactivo.html  — animación 3D interactiva con plotly (recomendado)
  ptv_3d_animacion.mp4     — video 3D con matplotlib + ffmpeg (GEN_MP4=True)

Uso:
  python analisis_ptv_3d.py

Requiere: numpy, matplotlib, scipy, plotly
  pip install plotly
"""
from __future__ import annotations

import csv
import json
import colorsys
import multiprocessing as mp
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[3D] WARN: plotly no disponible → instalar con: pip install plotly")


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

BASE_DIR = Path("ResultadosPTV")

GEN_PLOTLY_HTML = True    # animación interactiva HTML (recomendado)
GEN_MP4         = False   # video MP4 3D (lento; requiere ffmpeg)

# Frames en la animación plotly (más = archivo más grande y carga más lenta)
MAX_PLOTLY_FRAMES = 120
# Submuestreo para MP4
ANIM_FRAME_STEP_3D = 4

# Velocidad de reproducción por defecto relativa al tiempo real.
# 1.0 = tiempo real, 2.0 = el doble de rápido, 0.5 = la mitad de rápido.
# Los botones de velocidad en el HTML permiten cambiarla en tiempo de ejecución.
PLAYBACK_SPEED = 1.0

# Segundos de trayectoria pasada mostrada como estela
TRAIL_S = 0.4

# Suavizado del eje Z (profundidad) por track.
# depth_blur_mm es ruidoso frame a frame porque el ancho de la fibra medido
# varía por ruido de imagen, no por cambio real de profundidad.
# DEPTH_SMOOTH_HALF_WIN = mitad de la ventana de media móvil en pasos de la grilla.
# Window total = 2*N+1 pasos. A 220 fps con skip=4, cada paso ≈ 18 ms.
#   10 → ventana ~360 ms   (suavizado ligero)
#   20 → ventana ~720 ms   (recomendado — elimina saltos sin borrar tendencia)
#   40 → ventana ~1.4 s    (muy agresivo — solo tendencia lenta)
# Poner en 0 para deshabilitar el suavizado.
DEPTH_SMOOTH_HALF_WIN = 20

# Qué usar como eje Z: "blur" (depth_blur_mm) o "depth" (depth_mm calibrado)
# Si usas "depth" pero el pipeline no está calibrado, se cae a "blur" automáticamente
Z_MODE = "blur"

# Colorear fibras por: "velocity" | "defocus_score" | "track_id"
COLOR_BY = "velocity"

AUTO_EXTENT   = True
EXTENT_MM_FIXED = [0, 130, 0, 130]

N_WORKERS = max(1, mp.cpu_count() - 2)


# ══════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════

def load_schedule(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "sched_idx":       int(row["sched_idx"]),
                "frame_idx":       int(row["frame_idx"]),
                "region_name":     row["region_name"],
                "region_idx":      int(row["region_idx"]),
                "timestamp_s":     float(row["timestamp_s"]),
                "dt_s":            float(row["dt_s"]),
            })
    return rows


def load_tracks(path: Path) -> tuple[list[dict], dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tracks", []), data.get("metadata", {})


def get_cam_params(metadata: dict, folder: Path) -> tuple[float, float]:
    fps = metadata.get("fps")
    if not fps:
        fps = 660.0 if "cam-4" in folder.name.lower() else 220.0
    ptv = metadata.get("ptv", {}) or {}
    cam = metadata.get("camera", {}) or {}
    px_per_mm = ptv.get("px_per_mm") or cam.get("px_per_mm")
    if not px_per_mm:
        for key, val in {"cam-1": 8.0, "cam-2": 7.8, "cam-3": 7.8, "cam-4": 10.7}.items():
            if key in folder.name.lower():
                px_per_mm = val
                break
        else:
            px_per_mm = 7.8
    return float(fps), float(px_per_mm)


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN E INTERPOLACIÓN
# ══════════════════════════════════════════════════════════════════

def extract_per_track(tracks: list[dict]) -> dict[int, list[dict]]:
    per_track: dict[int, list[dict]] = {}
    for tr in tracks:
        tid = tr.get("track_id")
        if tid is None:
            continue
        tid = int(tid)
        records = []
        for rec in tr.get("history", []):
            ts = rec.get("timestamp_s")
            x  = rec.get("x_mm")
            y  = rec.get("y_mm")
            if ts is None or x is None or y is None:
                continue
            dm_raw = rec.get("depth_mm")
            records.append({
                "timestamp_s":      float(ts),
                "x_mm":             float(x),
                "y_mm":             float(y),
                "vx_mm_s":          float(rec.get("vx_mm_s") or 0.0),
                "vy_mm_s":          float(rec.get("vy_mm_s") or 0.0),
                "angle_deg":        float(rec.get("angle_deg") or 0.0),
                "length_mm":        float(rec.get("length_mm") or 1.0),
                "defocus_score":    float(rec.get("defocus_score") or 1.0),
                "depth_blur_mm":    float(rec.get("depth_blur_mm") or 0.0),
                "depth_mm":         float(dm_raw) if dm_raw is not None else None,
                "depth_confidence": float(rec.get("depth_confidence") or 0.0),
            })
        if records:
            records.sort(key=lambda r: r["timestamp_s"])
            per_track[tid] = records
    return per_track


def _interp_field(t_obs: np.ndarray, values: np.ndarray,
                  t_grid: np.ndarray) -> np.ndarray:
    n = len(t_obs)
    in_range = (t_grid >= t_obs[0]) & (t_grid <= t_obs[-1])
    out = np.full(len(t_grid), np.nan)
    if not in_range.any():
        return out
    t_q = t_grid[in_range]
    if n == 1:
        out[in_range] = values[0]
    elif n <= 3:
        out[in_range] = np.interp(t_q, t_obs, values)
    else:
        try:
            cs = CubicSpline(t_obs, values, bc_type="not-a-knot")
            out[in_range] = cs(t_q)
        except Exception:
            out[in_range] = np.interp(t_q, t_obs, values)
    return out


def interpolate_tracks_3d(
    per_track: dict[int, list[dict]],
    schedule: list[dict],
    z_mode: str = "blur",
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    """
    Construye grilla temporal uniforme e interpola todos los tracks.
    Retorna (t_grid, {track_id: {field: array}}).
    """
    t_sched = np.array([e["timestamp_s"] for e in schedule])
    t_min, t_max = float(t_sched.min()), float(t_sched.max())
    dts = np.array([e["dt_s"] for e in schedule])
    dt  = float(dts.min())
    t_grid = np.arange(t_min, t_max + dt * 0.5, dt)

    interp: dict[int, dict[str, np.ndarray]] = {}
    for tid, records in per_track.items():
        t_obs = np.array([r["timestamp_s"] for r in records])

        x_mm    = np.array([r["x_mm"]          for r in records])
        y_mm    = np.array([r["y_mm"]           for r in records])
        vx      = np.array([r["vx_mm_s"]        for r in records])
        vy      = np.array([r["vy_mm_s"]        for r in records])
        angle   = np.array([r["angle_deg"]      for r in records])
        length  = np.array([r["length_mm"]      for r in records])
        blur    = np.array([r["depth_blur_mm"]  for r in records])
        ds      = np.array([r["defocus_score"]  for r in records])
        conf    = np.array([r["depth_confidence"] for r in records])
        dm_raw  = [r["depth_mm"] for r in records]
        dm      = np.array([v if v is not None else np.nan for v in dm_raw])

        # Z axis
        if z_mode == "depth" and np.isfinite(dm).any():
            z_arr = dm
        else:
            z_arr = blur

        fields = {
            "x_mm": x_mm, "y_mm": y_mm,
            "vx_mm_s": vx, "vy_mm_s": vy,
            "angle_deg": angle, "length_mm": length,
            "depth_blur_mm": blur, "depth_mm": dm,
            "defocus_score": ds, "depth_confidence": conf,
            "z": z_arr,
        }
        interp[tid] = {name: _interp_field(t_obs, vals, t_grid)
                       for name, vals in fields.items()}
    return t_grid, interp


def compute_extent(per_track: dict[int, list[dict]]) -> list[float]:
    all_x = [r["x_mm"] for recs in per_track.values() for r in recs]
    all_y = [r["y_mm"] for recs in per_track.values() for r in recs]
    if not all_x:
        return EXTENT_MM_FIXED
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    mx = (xmax - xmin) * 0.03
    my = (ymax - ymin) * 0.03
    return [xmin - mx, xmax + mx, ymin - my, ymax + my]


def smooth_depth_z(
    interp: dict[int, dict[str, np.ndarray]],
    half_win: int,
) -> None:
    """
    Suaviza in-place el campo 'z' (y 'depth_blur_mm') de cada track con
    una media móvil de ventana (2*half_win + 1) pasos.

    depth_blur_mm es inherentemente ruidoso: el ancho medido de una fibra
    varía frame a frame por ruido de imagen aunque la fibra no se mueva en
    profundidad.  Las fibras en fluidos viscosos cambian de profundidad
    lentamente (inercia alta), así que este suavizado es físicamente válido.
    """
    if half_win <= 0:
        return
    n_kernel = 2 * half_win + 1
    kernel   = np.ones(n_kernel) / n_kernel

    for arrays in interp.values():
        for field in ("z", "depth_blur_mm"):
            if field not in arrays:
                continue
            arr  = arrays[field]
            mask = np.isfinite(arr)
            if not mask.any():
                continue
            idx     = np.where(mask)[0]
            segment = arr[idx].copy()
            seg_len = len(segment)

            if seg_len < 3:
                continue

            if seg_len < n_kernel:
                hw   = (seg_len - 1) // 2
                kern = np.ones(2 * hw + 1) / (2 * hw + 1)
            else:
                hw   = half_win
                kern = kernel

            smoothed = np.convolve(segment, kern, mode="same")
            k_eff = len(kern)
            for i in range(min(hw, len(smoothed))):
                smoothed[i] *= k_eff / (i + hw + 1)
            for i in range(1, min(hw + 1, len(smoothed))):
                smoothed[-i] *= k_eff / (i + hw)

            arr[idx] = smoothed


# ══════════════════════════════════════════════════════════════════
# UTILIDADES DE COLOR
# ══════════════════════════════════════════════════════════════════

def track_color_hex(tid: int) -> str:
    hue = (int(tid) * 137.508) % 360.0
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.75, 0.90)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def precompute_color_range(
    interp: dict[int, dict[str, np.ndarray]],
    color_by: str,
) -> tuple[float, float]:
    vals = []
    for arrays in interp.values():
        if color_by == "velocity":
            vx = arrays["vx_mm_s"]; vy = arrays["vy_mm_s"]
            sp = np.sqrt(vx**2 + vy**2)
            vals.append(sp[np.isfinite(sp)])
        elif color_by == "defocus_score":
            ds = arrays["defocus_score"]
            vals.append(ds[np.isfinite(ds)])
        elif color_by == "depth_blur_mm":
            bm = arrays["depth_blur_mm"]
            vals.append(bm[np.isfinite(bm)])
    if not vals:
        return 0.0, 1.0
    all_v = np.concatenate(vals)
    return float(np.nanpercentile(all_v, 5)), float(np.nanpercentile(all_v, 95))


def get_scalar_value(arrays: dict[str, np.ndarray], fi: int, color_by: str) -> float:
    if color_by == "velocity":
        vx = arrays["vx_mm_s"][fi]; vy = arrays["vy_mm_s"][fi]
        return float(np.sqrt(vx**2 + vy**2)) if np.isfinite(vx) and np.isfinite(vy) else np.nan
    elif color_by == "defocus_score":
        v = arrays["defocus_score"][fi]
        return float(v) if np.isfinite(v) else np.nan
    return np.nan


# ══════════════════════════════════════════════════════════════════
# PLOTLY — ANIMACIÓN 3D INTERACTIVA
# ══════════════════════════════════════════════════════════════════

def _piv_plane_trace(extent_mm: list[float], z_range: list[float]) -> "go.Surface":
    """Superficie semitransparente en Z=0 representando el plano PIV (laser sheet)."""
    xmin, xmax, ymin, ymax = extent_mm
    xs = np.array([[xmin, xmax], [xmin, xmax]])
    ys = np.array([[ymin, ymin], [ymax, ymax]])
    zs = np.zeros_like(xs)
    return go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.15,
        colorscale=[[0, "cyan"], [1, "cyan"]],
        showscale=False,
        name="Plano PIV (Z=0)",
        hoverinfo="skip",
    )


def _build_frame_data(
    fi: int,
    t_val: float,
    t_grid: np.ndarray,
    interp: dict[int, dict[str, np.ndarray]],
    trail_steps: int,
    color_by: str,
    vmin: float,
    vmax: float,
    extent_mm: list[float],
) -> tuple[list, list[str]]:
    """
    Construye los trazos de fibras y estelas para un instante de tiempo.
    Retorna (traces_data_dicts, labels) listos para go.Frame(data=...).
    """
    # ── Fibras: rods en el instante actual ─────────────────────────
    rod_x, rod_y, rod_z, rod_color = [], [], [], []
    cx_list, cy_list, cz_list = [], [], []
    color_list, hover_list = [], []

    for tid, arrays in interp.items():
        cx = arrays["x_mm"][fi]; cy = arrays["y_mm"][fi]; cz = arrays["z"][fi]
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)):
            continue

        angle  = arrays["angle_deg"][fi]
        length = arrays["length_mm"][fi]
        scalar = get_scalar_value(arrays, fi, color_by)

        ang_rad = np.deg2rad(angle) if np.isfinite(angle) else 0.0
        half    = (float(length) / 2.0) if np.isfinite(length) else 3.0
        dx = np.cos(ang_rad) * half
        dy = np.sin(ang_rad) * half

        c_val = float(scalar) if np.isfinite(scalar) else float(vmin)
        rod_x.extend([cx - dx, cx + dx, None])
        rod_y.extend([cy - dy, cy + dy, None])
        rod_z.extend([cz, cz, None])
        rod_color.extend([c_val, c_val, None])

        cx_list.append(cx); cy_list.append(cy); cz_list.append(cz)
        color_list.append(c_val)
        blur = float(arrays["depth_blur_mm"][fi]) if np.isfinite(arrays["depth_blur_mm"][fi]) else 0.0
        vx = arrays["vx_mm_s"][fi]; vy = arrays["vy_mm_s"][fi]
        sp = float(np.sqrt(vx**2 + vy**2)) if np.isfinite(vx) and np.isfinite(vy) else 0.0
        hover_list.append(
            f"Track {tid}<br>|v|={sp:.1f} mm/s<br>blur={blur:.3f} mm<br>"
            f"z={cz:.3f} mm<br>t={t_val:.4f}s"
        )

    # Trace 0: rod lines
    rod_trace = dict(
        type="scatter3d",
        x=rod_x, y=rod_y, z=rod_z,
        mode="lines",
        line=dict(
            color=[c if c is not None else float(vmin) for c in rod_color],
            colorscale="Turbo",
            cmin=vmin, cmax=vmax,
            width=5,
        ),
        showlegend=False,
        name="fibras",
        hoverinfo="skip",
    )

    # Trace 1: centro de fibras (markers, para hover)
    center_trace = dict(
        type="scatter3d",
        x=cx_list, y=cy_list, z=cz_list,
        mode="markers",
        marker=dict(
            size=4,
            color=color_list,
            colorscale="Turbo",
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=color_by.replace("_", " ")),
                thickness=12, len=0.5, x=1.02,
            ),
        ),
        text=hover_list,
        hoverinfo="text",
        showlegend=False,
        name="centros",
    )

    # ── Estelas ─────────────────────────────────────────────────────
    trail_x, trail_y, trail_z, trail_c = [], [], [], []
    fi_start = max(0, fi - trail_steps)

    for tid, arrays in interp.items():
        past_x = arrays["x_mm"][fi_start:fi + 1]
        past_y = arrays["y_mm"][fi_start:fi + 1]
        past_z = arrays["z"][fi_start:fi + 1]
        mask   = np.isfinite(past_x) & np.isfinite(past_y) & np.isfinite(past_z)
        px = past_x[mask]; py = past_y[mask]; pz = past_z[mask]
        if len(px) < 2:
            continue
        trail_x.extend(list(px) + [None])
        trail_y.extend(list(py) + [None])
        trail_z.extend(list(pz) + [None])

    trail_trace = dict(
        type="scatter3d",
        x=trail_x, y=trail_y, z=trail_z,
        mode="lines",
        line=dict(color="rgba(180,180,180,0.4)", width=2),
        showlegend=False,
        name="estelas",
        hoverinfo="skip",
    )

    return [rod_trace, center_trace, trail_trace]


def build_plotly_animation(
    folder: Path,
    t_grid: np.ndarray,
    interp: dict[int, dict[str, np.ndarray]],
    per_track: dict[int, list[dict]],
    extent_mm: list[float],
    z_mode: str,
    schedule: list[dict],
    metadata: dict,
) -> None:
    if not HAS_PLOTLY:
        print("[3D] Plotly no disponible — omitiendo HTML.")
        return
    if not interp:
        print("[3D] Sin tracks — omitiendo animación.")
        return

    # Rango Z
    all_z = np.concatenate([
        a["z"][np.isfinite(a["z"])] for a in interp.values()
    ])
    z_max  = float(np.nanpercentile(all_z, 97)) if all_z.size > 0 else 1.0
    z_label = "depth_mm (calibrado)" if z_mode == "depth" else "blur de profundidad (mm)"

    vmin, vmax = precompute_color_range(interp, COLOR_BY)

    # Tiempo de la grilla
    dt_grid = float(np.diff(t_grid[:2])[0]) if len(t_grid) > 1 else 0.001
    trail_steps = max(1, int(TRAIL_S / dt_grid))

    # Subsamplear frames
    n_frames = min(MAX_PLOTLY_FRAMES, len(t_grid))
    frame_indices = np.linspace(0, len(t_grid) - 1, n_frames, dtype=int)

    # Duración base por frame (ms) para reproducción a tiempo real.
    # Cada frame plotly cubre (t_max-t_min)/(n_frames-1) segundos.
    t_span = float(t_grid[-1] - t_grid[0])
    frame_real_ms = (t_span / max(1, n_frames - 1)) * 1000.0
    # Duración inicial según PLAYBACK_SPEED; mínimo 16 ms (≈60 fps browser limit)
    default_dur_ms = max(16, int(frame_real_ms / PLAYBACK_SPEED))

    print(f"[3D] Construyendo {n_frames} frames plotly...", flush=True)

    # Plano PIV estático
    piv_plane = _piv_plane_trace(extent_mm, [0.0, z_max])

    # Primer frame para la figura inicial
    init_data = _build_frame_data(
        int(frame_indices[0]), float(t_grid[frame_indices[0]]),
        t_grid, interp, trail_steps, COLOR_BY, vmin, vmax, extent_mm,
    )

    # Construir todos los frames
    frames = []
    for i, fi in enumerate(frame_indices):
        fi = int(fi); t_val = float(t_grid[fi])
        frame_traces = _build_frame_data(
            fi, t_val, t_grid, interp, trail_steps, COLOR_BY, vmin, vmax, extent_mm,
        )
        frames.append(go.Frame(
            data=[go.Scatter3d(**tr) for tr in frame_traces],
            name=f"{t_val:.4f}",
            traces=[0, 1, 2],   # actualizar solo estos traces (no el plano PIV)
        ))
        if (i + 1) % max(1, n_frames // 10) == 0 or i == n_frames - 1:
            print(f"  → {i+1}/{n_frames}", flush=True)

    # Layout 3D
    xmin, xmax, ymin, ymax = extent_mm
    t_min = float(t_grid[0]); t_max = float(t_grid[-1])
    n_tracks = len([t for t in interp if
                    np.isfinite(interp[t]["x_mm"]).any()])

    layout = go.Layout(
        title=dict(
            text=(
                f"Animación 3D PTV — {folder.name}<br>"
                f"<sub>N_tracks={n_tracks} | Z = {z_label} | "
                f"color = {COLOR_BY} | "
                f"t=[{t_min:.3f}s, {t_max:.3f}s]</sub>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="X (mm)", range=[xmin, xmax]),
            yaxis=dict(title="Y (mm)", range=[ymin, ymax]),
            zaxis=dict(title=f"Z: {z_label}", range=[-z_max * 0.05, z_max * 1.1]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(
                x=(xmax - xmin) / max(xmax - xmin, ymax - ymin),
                y=(ymax - ymin) / max(xmax - xmin, ymax - ymin),
                z=0.5,
            ),
        ),
        updatemenus=[
            # ── Fila 1: Play / Pausa ─────────────────────────────────
            dict(
                type="buttons",
                direction="left",
                x=0.0, xanchor="left",
                y=-0.06, yanchor="top",
                pad=dict(r=10, t=5),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": default_dur_ms, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        }],
                    ),
                    dict(
                        label="⏸ Pausa",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                    ),
                ],
                showactive=True,
            ),
            # ── Fila 2: control de velocidad ─────────────────────────
            # Cada botón reproduce desde el frame actual a esa velocidad.
            # frame_real_ms es la duración de un frame a tiempo real.
            dict(
                type="buttons",
                direction="left",
                x=0.0, xanchor="left",
                y=-0.14, yanchor="top",
                pad=dict(r=6, t=5),
                buttons=[
                    dict(
                        label=label,
                        method="animate",
                        args=[None, {
                            "frame": {
                                "duration": max(16, int(frame_real_ms / mult)),
                                "redraw": True,
                            },
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        }],
                    )
                    for label, mult in [
                        ("×¼",  0.25),
                        ("×½",  0.5),
                        ("×1",  1.0),
                        ("×2",  2.0),
                        ("×4",  4.0),
                        ("×8",  8.0),
                    ]
                ],
                showactive=True,
            ),
        ],
        sliders=[dict(
            active=0,
            yanchor="top", xanchor="left",
            x=0.0, y=0.0,
            pad=dict(b=10, t=50),
            len=1.0,
            currentvalue=dict(
                prefix="t = ",
                suffix=" s",
                visible=True,
                xanchor="right",
                font=dict(size=12),
            ),
            transition=dict(duration=0),
            steps=[
                dict(
                    method="animate",
                    args=[[f.name], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                    label=f"{float(f.name):.3f}s",
                )
                for f in frames
            ],
        )],
        margin=dict(l=0, r=0, t=80, b=180),
        height=700,
    )

    # Combinar datos iniciales con el plano PIV
    init_traces = [go.Scatter3d(**tr) for tr in init_data] + [piv_plane]

    fig = go.Figure(data=init_traces, frames=frames, layout=layout)

    # Añadir anotación del plano PIV en la leyenda
    fig.add_annotation(
        text=(
            "Plano azul semitransparente = plano focal (PIV laser sheet, Z=0).<br>"
            "Fibras en Z≈0 están en el plano de medición PIV y correlacionan mejor<br>"
            "con la velocidad del fluido medida por PIV."
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10),
        showarrow=False,
    )

    out_html = folder / "ptv_3d_interactivo.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[3D] HTML interactivo → {out_html}", flush=True)


# ══════════════════════════════════════════════════════════════════
# MATPLOTLIB — VIDEO MP4 3D
# ══════════════════════════════════════════════════════════════════

def _render_frame_3d(args: tuple) -> str:
    """Renderiza un frame 3D con matplotlib. Top-level para multiprocessing."""
    (video_i, fi, t_val, n_tracks_total,
     track_data, extent_mm, z_max, z_label,
     color_by, vmin, vmax, folder_name, out_png) = args

    fig = plt.figure(figsize=(13, 8))
    ax  = fig.add_subplot(111, projection="3d")

    xmin, xmax, ymin, ymax = extent_mm
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-z_max * 0.02, z_max * 1.05)
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)
    ax.set_zlabel(f"Z: {z_label}", fontsize=9)
    ax.set_title(
        f"{folder_name}   t = {t_val:.4f} s   "
        f"[{sum(1 for d in track_data if d['active'])} fibras activas / {n_tracks_total} total]",
        fontsize=10,
    )
    ax.invert_yaxis()  # imagen: y crece hacia abajo

    # Plano PIV semitransparente en Z=0
    xs_p = np.array([[xmin, xmax], [xmin, xmax]])
    ys_p = np.array([[ymin, ymin], [ymax, ymax]])
    zs_p = np.zeros_like(xs_p)
    ax.plot_surface(xs_p, ys_p, zs_p, alpha=0.12, color="cyan", linewidth=0)
    ax.text2D(0.02, 0.02, "Plano PIV (Z=0)", transform=ax.transAxes,
              fontsize=7, color="teal")

    # Colormap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.turbo

    for td in track_data:
        if not td["active"]:
            continue
        cx, cy, cz = td["cx"], td["cy"], td["cz"]
        angle = td["angle"]; length = td["length"]
        scalar = td["scalar"]
        trail_x, trail_y, trail_z = td["trail_x"], td["trail_y"], td["trail_z"]

        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)):
            continue

        color = cmap(norm(scalar)) if np.isfinite(scalar) else (0.5, 0.5, 0.5, 0.8)

        # Rod
        ang_rad = np.deg2rad(angle) if np.isfinite(angle) else 0.0
        half = (length / 2.0) if np.isfinite(length) else 3.0
        dx = np.cos(ang_rad) * half
        dy = np.sin(ang_rad) * half
        ax.plot(
            [cx - dx, cx + dx], [cy - dy, cy + dy], [cz, cz],
            color=color, lw=2.5, alpha=0.92,
        )
        ax.scatter([cx], [cy], [cz], color=color, s=18, zorder=5, depthshade=True)

        # Estela
        if len(trail_x) >= 2:
            ax.plot(trail_x, trail_y, trail_z,
                    color=(*color[:3], 0.35), lw=1.2)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.55,
                 label=color_by.replace("_", " "))

    plt.tight_layout()
    plt.savefig(out_png, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_matplotlib_mp4(
    folder: Path,
    t_grid: np.ndarray,
    interp: dict[int, dict[str, np.ndarray]],
    extent_mm: list[float],
    z_mode: str,
) -> None:
    if not interp:
        return

    all_z = np.concatenate([a["z"][np.isfinite(a["z"])] for a in interp.values()])
    z_max  = float(np.nanpercentile(all_z, 97)) if all_z.size > 0 else 1.0
    z_label = "depth_mm" if z_mode == "depth" else "blur profundidad (mm)"

    vmin, vmax = precompute_color_range(interp, COLOR_BY)

    dt_grid = float(np.diff(t_grid[:2])[0]) if len(t_grid) > 1 else 0.001
    trail_steps = max(1, int(TRAIL_S / dt_grid))
    fps_video   = max(1.0, min(1.0 / (dt_grid * ANIM_FRAME_STEP_3D), 120.0))

    frame_indices = list(range(0, len(t_grid), ANIM_FRAME_STEP_3D))
    n_tracks_total = len(interp)

    tmp_dir = tempfile.mkdtemp(prefix="ptv3d_")
    args_list = []

    for video_i, fi in enumerate(frame_indices):
        t_val  = float(t_grid[fi])
        out_png = str(Path(tmp_dir) / f"frame_{video_i:06d}.png")
        fi_start = max(0, fi - trail_steps)

        track_data = []
        for tid, arrays in interp.items():
            cx = arrays["x_mm"][fi]; cy = arrays["y_mm"][fi]; cz = arrays["z"][fi]
            active = np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)
            angle  = float(arrays["angle_deg"][fi]) if np.isfinite(arrays["angle_deg"][fi]) else 0.0
            length = float(arrays["length_mm"][fi]) if np.isfinite(arrays["length_mm"][fi]) else 3.0
            scalar = get_scalar_value(arrays, fi, COLOR_BY)

            past_x = arrays["x_mm"][fi_start:fi + 1]
            past_y = arrays["y_mm"][fi_start:fi + 1]
            past_z = arrays["z"][fi_start:fi + 1]
            mask   = np.isfinite(past_x) & np.isfinite(past_y) & np.isfinite(past_z)

            track_data.append({
                "active": bool(active),
                "cx": float(cx), "cy": float(cy), "cz": float(cz),
                "angle": angle, "length": length, "scalar": float(scalar),
                "trail_x": list(past_x[mask]),
                "trail_y": list(past_y[mask]),
                "trail_z": list(past_z[mask]),
            })

        args_list.append((
            video_i, fi, t_val, n_tracks_total,
            track_data, extent_mm, z_max, z_label,
            COLOR_BY, vmin, vmax, folder.name, out_png,
        ))

    print(f"[3D] Renderizando {len(args_list)} frames MP4 ({N_WORKERS} workers)...", flush=True)
    with mp.Pool(processes=N_WORKERS) as pool:
        for done, _ in enumerate(pool.imap_unordered(_render_frame_3d, args_list), 1):
            if done % max(1, len(args_list) // 10) == 0 or done == len(args_list):
                print(f"  → {done}/{len(args_list)}", flush=True)

    out_mp4 = folder / "ptv_3d_animacion.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps_video:.4f}",
        "-i", str(Path(tmp_dir) / "frame_%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[3D] ERROR ffmpeg:\n{result.stderr}")
        else:
            print(f"[3D] Video 3D → {out_mp4}", flush=True)
    except FileNotFoundError:
        print("[3D] ERROR: ffmpeg no está instalado.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    for tracks_path in sorted(BASE_DIR.rglob("tracks.json")):
        folder        = tracks_path.parent
        schedule_path = folder / "schedule.csv"

        print(f"\n[3D] ═══ {folder} ═══", flush=True)
        if not schedule_path.exists():
            print(f"[3D] WARN: No hay schedule.csv — omitido.")
            continue

        try:
            with tracks_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            tracks   = data.get("tracks", [])
            metadata = data.get("metadata", {})

            schedule = load_schedule(schedule_path)
            fps, px_per_mm = get_cam_params(metadata, folder)

            print(f"[3D] fps={fps}  px_per_mm={px_per_mm}  tracks={len(tracks)}", flush=True)
            if not tracks:
                print("[3D] Sin tracks — omitido."); continue

            # Verificar campos DPTV
            sample_rec = (tracks[0].get("history") or [{}])[0]
            has_dptv = "defocus_score" in sample_rec
            if not has_dptv:
                print("[3D] WARN: Sin campos DPTV — Z=0 para todas las fibras.")

            per_track = extract_per_track(tracks)
            extent_mm = compute_extent(per_track) if AUTO_EXTENT else EXTENT_MM_FIXED

            # Determinar modo Z: si no hay calibración, forzar blur
            z_mode_eff = Z_MODE
            if Z_MODE == "depth":
                all_dm = [r["depth_mm"] for recs in per_track.values()
                          for r in recs if r["depth_mm"] is not None]
                if not all_dm:
                    print("[3D] depth_mm no disponible (k_blur sin calibrar) → usando blur")
                    z_mode_eff = "blur"

            print(
                f"[3D] Z={z_mode_eff}  "
                f"dominio x=[{extent_mm[0]:.1f},{extent_mm[1]:.1f}]  "
                f"y=[{extent_mm[2]:.1f},{extent_mm[3]:.1f}] mm",
                flush=True,
            )

            # Grilla temporal e interpolación
            t_grid, interp = interpolate_tracks_3d(per_track, schedule, z_mode_eff)

            # Suavizar eje Z para eliminar ruido de detección frame a frame
            if DEPTH_SMOOTH_HALF_WIN > 0:
                smooth_depth_z(interp, DEPTH_SMOOTH_HALF_WIN)
                dt_grid = float(np.diff(t_grid[:2])[0]) if len(t_grid) > 1 else 0.001
                win_s = DEPTH_SMOOTH_HALF_WIN * 2 * dt_grid
                print(f"[3D] Suavizado Z: ventana={2*DEPTH_SMOOTH_HALF_WIN+1} pasos "
                      f"(≈{win_s*1000:.0f} ms)", flush=True)

            if GEN_PLOTLY_HTML:
                build_plotly_animation(
                    folder, t_grid, interp, per_track, extent_mm,
                    z_mode_eff, schedule, metadata,
                )

            if GEN_MP4:
                build_matplotlib_mp4(folder, t_grid, interp, extent_mm, z_mode_eff)

            print(f"[3D] OK → {folder}", flush=True)

        except Exception as e:
            import traceback
            print(f"[3D] ERROR en {folder}: {e}"); traceback.print_exc()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
