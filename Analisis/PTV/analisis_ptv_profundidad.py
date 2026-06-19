"""
analisis_ptv_profundidad.py
===========================
Análisis PTV estándar + estimación de profundidad DPTV.

Genera por cada carpeta de experimento:
  ptv_heatmaps.png              — heatmaps de velocidad lineal y angular
  ptv_velocidad_tiempo.png      — velocidad escalar vs tiempo por track
  ptv_profundidad_heatmap.png   — distribución espacial de profundidad
  ptv_profundidad_dist.png      — histogramas de defocus_score y blur
  ptv_profundidad_vs_vel.png    — correlación profundidad–velocidad (plot clave PIV)
  ptv_profundidad_tiempo.png    — blur/profundidad vs tiempo por track
  ptv_resumen_tracks.png        — resumen por track (velocidad media vs profundidad media)

DPTV usa el ancho aparente de las fibras para inferir distancia al plano focal:
  defocus_score  — W_medida / W_ideal  (1.0 = en foco, >1 = desenfocada)
  depth_blur_mm  — contribución del blur [mm], proporcional a |Δz|, SIN calibración
  depth_mm       — |Δz| estimado [mm], SOLO si k_blur fue calibrado (puede ser None)
  depth_confidence — calidad [0,1]: 0=en foco/indistinguible, 1=claramente desenfocada

Uso:
  python analisis_ptv_profundidad.py

Configura BASE_DIR y los toggles de sección según necesites.
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import CubicSpline


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

BASE_DIR = Path("ResultadosPTV")

# Qué análisis generar
GEN_HEATMAPS_VEL        = True   # heatmaps de velocidad (análisis original)
GEN_VEL_TIEMPO          = True   # velocidad vs tiempo (análisis original)
GEN_ANIMACION           = False  # animación MP4 (lenta; activar si se necesita)
GEN_DEPTH_HEATMAP       = True   # distribución espacial de profundidad
GEN_DEPTH_DIST          = True   # histogramas de defocus_score y blur
GEN_DEPTH_VS_VEL        = True   # correlación profundidad–velocidad
GEN_DEPTH_TIEMPO        = True   # profundidad vs tiempo
GEN_RESUMEN_TRACKS      = True   # velocidad media vs profundidad media por track

# Umbral para considerar una fibra "cerca del plano focal"
NEAR_FOCUS_DEFOCUS_THR = 1.5     # defocus_score < umbral → fibra cerca del foco
NEAR_FOCUS_CONF_THR    = 0.5     # depth_confidence < umbral → no claramente desenfocada

# Parámetros de animación y heatmap (igual que en el script original)
TARGET_DT_S      = None
ANIM_FRAME_STEP  = 3
ANIM_FILENAME    = "ptv_animacion.mp4"
HEATMAP_FILENAME = "ptv_heatmaps.png"
GRID_SIZE        = (60, 60)
HEATMAP_COVERAGE = 0.95
ANGULAR_SYMMETRIC = True
AUTO_EXTENT      = True
EXTENT_MM_FIXED  = [0, 130, 0, 130]

N_WORKERS = max(1, mp.cpu_count() - 2)

plt.rcParams.update({
    "font.family":      "Times New Roman",
    "font.size":        13,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "figure.titlesize": 13,
    "lines.linewidth":  1.8,
})


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
                "image_name":      row["image_name"],
                "region_name":     row["region_name"],
                "region_idx":      int(row["region_idx"]),
                "timestamp_s":     float(row["timestamp_s"]),
                "dt_s":            float(row["dt_s"]),
                "dt_ms":           float(row["dt_ms"]),
                "n_detections":    int(row["n_detections"]),
                "n_tracks_active": int(row["n_tracks_active"]),
                "track_ids":       [int(x) for x in row["track_ids"].split("|") if x],
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


def get_dptv_config(metadata: dict) -> dict:
    """Extrae la configuración DPTV del metadata del JSON."""
    return metadata.get("dptv", {}) or {}


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE TRAYECTORIAS (con campos DPTV)
# ══════════════════════════════════════════════════════════════════

def extract_per_track(tracks: list[dict]) -> dict[int, list[dict]]:
    """
    Extrae trayectorias con campos cinemáticos Y de profundidad DPTV.
    Retorna {track_id: [records ordenados por timestamp_s]}.
    """
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
            # Profundidad: usar get() con defaults para compatibilidad
            # con archivos generados antes de la implementación DPTV
            depth_mm_raw = rec.get("depth_mm")
            records.append({
                "timestamp_s":     float(ts),
                "x_mm":            float(x),
                "y_mm":            float(y),
                "vx_mm_s":         float(rec.get("vx_mm_s") or 0.0),
                "vy_mm_s":         float(rec.get("vy_mm_s") or 0.0),
                "omega_deg_s":     float(rec.get("omega_deg_s") or 0.0),
                "angle_deg":       float(rec.get("angle_deg") or 0.0),
                "length_mm":       float(rec.get("length_mm") or 1.0),
                "region_name":     str(rec.get("region_name") or ""),
                "dt_s":            float(rec.get("dt_s") or 0.0),
                # DPTV
                "defocus_score":   float(rec.get("defocus_score") or 1.0),
                "depth_blur_mm":   float(rec.get("depth_blur_mm") or 0.0),
                "depth_blur_px":   float(rec.get("depth_blur_px") or 0.0),
                "depth_mm":        float(depth_mm_raw) if depth_mm_raw is not None else None,
                "depth_confidence":float(rec.get("depth_confidence") or 0.0),
            })
        if records:
            records.sort(key=lambda r: r["timestamp_s"])
            per_track[tid] = records
    return per_track


def extract_track_depth_stats(tracks: list[dict]) -> dict[int, dict]:
    """
    Extrae el bloque depth_stats por track del JSON (resumen pre-calculado).
    Retorna {track_id: depth_stats_dict}.
    """
    out = {}
    for tr in tracks:
        tid = tr.get("track_id")
        if tid is None:
            continue
        out[int(tid)] = tr.get("depth_stats", {}) or {}
    return out


# ══════════════════════════════════════════════════════════════════
# INTERPOLACIÓN A GRILLA UNIFORME
# ══════════════════════════════════════════════════════════════════

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


def interpolate_track(records: list[dict], t_grid: np.ndarray) -> dict[str, np.ndarray]:
    t_obs = np.array([r["timestamp_s"] for r in records])
    fields = {
        "x_mm":           np.array([r["x_mm"]           for r in records]),
        "y_mm":           np.array([r["y_mm"]            for r in records]),
        "vx_mm_s":        np.array([r["vx_mm_s"]         for r in records]),
        "vy_mm_s":        np.array([r["vy_mm_s"]         for r in records]),
        "omega_deg_s":    np.array([r["omega_deg_s"]     for r in records]),
        "angle_deg":      np.array([r["angle_deg"]       for r in records]),
        "length_mm":      np.array([r["length_mm"]       for r in records]),
        "defocus_score":  np.array([r["defocus_score"]   for r in records]),
        "depth_blur_mm":  np.array([r["depth_blur_mm"]   for r in records]),
        "depth_confidence": np.array([r["depth_confidence"] for r in records]),
    }
    # depth_mm puede tener None — rellenar con NaN
    dm_raw = [r["depth_mm"] for r in records]
    fields["depth_mm"] = np.array(
        [v if v is not None else np.nan for v in dm_raw], dtype=float
    )
    return {name: _interp_field(t_obs, vals, t_grid)
            for name, vals in fields.items()}


def build_uniform_grid(
    per_track: dict[int, list[dict]],
    schedule: list[dict],
    target_dt_s: float | None = None,
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    t_sched = np.array([e["timestamp_s"] for e in schedule])
    t_min, t_max = float(t_sched.min()), float(t_sched.max())
    dts = np.array([e["dt_s"] for e in schedule])
    dt  = float(target_dt_s) if target_dt_s is not None else float(dts.min())
    t_grid = np.arange(t_min, t_max + dt * 0.5, dt)

    regions_seen = {}
    for e in schedule:
        if e["region_name"] not in regions_seen:
            regions_seen[e["region_name"]] = e["dt_ms"]
    region_str = "  ".join(f"{r}({d:.2f}ms)" for r, d in regions_seen.items())
    print(
        f"[VIZ] Grilla: [{t_min:.3f}s, {t_max:.3f}s]  dt={dt*1000:.3f}ms  "
        f"N={len(t_grid)}  Regiones: {region_str}", flush=True,
    )

    interp: dict[int, dict[str, np.ndarray]] = {}
    for tid, records in per_track.items():
        interp[tid] = interpolate_track(records, t_grid)
    return t_grid, interp


def smooth_velocities(
    interpolated_tracks: dict[int, dict[str, np.ndarray]],
    half_window: int = 2,
) -> None:
    n = 2 * half_window + 1
    kernel = np.ones(n) / n
    for arrays in interpolated_tracks.values():
        for field in ("vx_mm_s", "vy_mm_s", "omega_deg_s"):
            arr = arrays[field]
            finite_mask = np.isfinite(arr)
            if not finite_mask.any():
                continue
            idx = np.where(finite_mask)[0]
            segment = arr[idx]
            seg_len = len(segment)
            if seg_len < n:
                hw = (seg_len - 1) // 2
                if hw == 0:
                    continue
                k_eff = 2 * hw + 1
                kern  = np.ones(k_eff) / k_eff
            else:
                hw, k_eff, kern = half_window, n, kernel
            smoothed = np.convolve(segment, kern, mode="same")
            for i in range(min(hw, len(smoothed))):
                smoothed[i] *= k_eff / (i + hw + 1)
            for i in range(1, min(hw + 1, len(smoothed))):
                smoothed[-i] *= k_eff / (i + hw)
            arr[idx] = smoothed


# ══════════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════

def compute_extent(per_track: dict[int, list[dict]],
                   margin_frac: float = 0.03) -> list[float]:
    all_x = [r["x_mm"] for recs in per_track.values() for r in recs]
    all_y = [r["y_mm"] for recs in per_track.values() for r in recs]
    if not all_x:
        return EXTENT_MM_FIXED
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    mx = (xmax - xmin) * margin_frac
    my = (ymax - ymin) * margin_frac
    return [xmin - mx, xmax + mx, ymin - my, ymax + my]


def track_color(tid: int) -> tuple:
    hue = (int(tid) * 137.508) % 360.0
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.80, 0.92)
    return (r, g, b)


def setup_ax(ax, extent_mm: list[float]) -> None:
    xmin, xmax, ymin, ymax = extent_mm
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.18, linewidth=0.6)


def robust_clim(values: np.ndarray, coverage: float = 0.90,
                symmetric: bool = False) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = float(np.quantile(np.abs(v), coverage))
        return -vmax, vmax
    lo = (1.0 - coverage) / 2.0
    return float(np.quantile(v, lo)), float(np.quantile(v, 1.0 - lo))


def _bin_avg(xs: np.ndarray, ys: np.ndarray, vals: np.ndarray,
             grid_size: tuple, extent_mm: list[float]):
    nx, ny = grid_size
    xmin, xmax, ymin, ymax = extent_mm
    xb = np.linspace(xmin, xmax, nx + 1)
    yb = np.linspace(ymin, ymax, ny + 1)
    mask = np.isfinite(vals) & np.isfinite(xs) & np.isfinite(ys)
    xi = np.clip(np.digitize(xs[mask], xb) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(ys[mask], yb) - 1, 0, ny - 1)
    s = np.zeros((nx, ny))
    c = np.zeros((nx, ny))
    for i, j, v in zip(xi, yi, vals[mask]):
        s[i, j] += v
        c[i, j] += 1
    avg = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    return avg, c, xb, yb


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS ORIGINAL — HEATMAPS VELOCIDAD
# ══════════════════════════════════════════════════════════════════

def plot_heatmaps(
    folder: Path,
    t_grid: np.ndarray,
    interpolated_tracks: dict[int, dict[str, np.ndarray]],
    schedule: list[dict],
    extent_mm: list[float],
    metadata: dict,
) -> None:
    xs_l, ys_l, vx_l, vy_l, om_l = [], [], [], [], []
    for arr in interpolated_tracks.values():
        xs_l.append(arr["x_mm"]); ys_l.append(arr["y_mm"])
        vx_l.append(arr["vx_mm_s"]); vy_l.append(arr["vy_mm_s"])
        om_l.append(arr["omega_deg_s"])

    xs_all = np.concatenate(xs_l); ys_all = np.concatenate(ys_l)
    vx_all = np.concatenate(vx_l); vy_all = np.concatenate(vy_l)
    om_all = np.concatenate(om_l)
    sp_all = np.sqrt(vx_all**2 + vy_all**2)

    sp_grid, sp_cnt, _, _ = _bin_avg(xs_all, ys_all, sp_all, GRID_SIZE, extent_mm)
    om_grid, om_cnt, _, _ = _bin_avg(xs_all, ys_all, om_all, GRID_SIZE, extent_mm)

    sp_valid = sp_all[np.isfinite(sp_all)]
    om_valid = om_all[np.isfinite(om_all)]
    vmin_sp, vmax_sp = robust_clim(sp_valid, HEATMAP_COVERAGE)
    vmin_om, vmax_om = robust_clim(om_valid, HEATMAP_COVERAGE, symmetric=ANGULAR_SYMMETRIC)

    tr_info = ""
    tr_regions = metadata.get("temporal_regions") or []
    if tr_regions:
        tr_info = f" | {len(tr_regions)} regiones"

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(
        f"Heatmaps PTV — {folder.name}\n"
        f"N_tracks={len(interpolated_tracks)} | dt={np.diff(t_grid[:2])[0]*1000:.2f}ms{tr_info}"
    )
    panels = [
        ("turbo",    axes[0], sp_grid, sp_cnt, sp_valid, "Velocidad lineal promedio (mm/s)", vmin_sp, vmax_sp),
        ("coolwarm", axes[1], om_grid, om_cnt, om_valid, "Velocidad angular promedio (°/s)",  vmin_om, vmax_om),
    ]
    for cmap_name, ax, grid, cnt, valid, label, vmin, vmax in panels:
        cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad(alpha=0.0)
        masked = np.ma.masked_where(cnt == 0, grid)
        ax.imshow(masked.T, origin="lower", extent=extent_mm, cmap=cmap,
                  interpolation="bilinear", vmin=vmin, vmax=vmax, aspect="equal")
        setup_ax(ax, extent_mm)
        cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.82)
        cb.set_label(f"{label}\nmedia={np.nanmean(valid):.2f}  N={np.isfinite(valid).sum()}")

    axes[0].set_title("Velocidad lineal |v| (mm/s)")
    axes[1].set_title("Velocidad angular ω (°/s)")
    plt.tight_layout()
    out = folder / HEATMAP_FILENAME
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Heatmap velocidad → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS ORIGINAL — VELOCIDAD VS TIEMPO
# ══════════════════════════════════════════════════════════════════

def plot_velocity_vs_time(
    folder: Path,
    per_track: dict[int, list[dict]],
    schedule: list[dict],
) -> None:
    region_bounds: dict[str, tuple[float, float]] = {}
    for e in schedule:
        rname = e["region_name"]; t = e["timestamp_s"]
        if rname not in region_bounds:
            region_bounds[rname] = (t, t)
        else:
            lo, hi = region_bounds[rname]
            region_bounds[rname] = (min(lo, t), max(hi, t))

    REGION_COLORS = ["#d4e8f7","#fde8c8","#d4f0d4","#f7d4d4","#e8d4f7","#f7f0d4","#d4f7f0"]
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (rname, (t0, t1)) in enumerate(region_bounds.items()):
        ax.axvspan(t0, t1, alpha=0.35, color=REGION_COLORS[i % len(REGION_COLORS)],
                   label=rname, zorder=0)
        ax.text((t0 + t1) / 2, 0, rname.replace("_", "\n"),
                ha="center", va="bottom", fontsize=8, color="gray")
    for tid, records in per_track.items():
        ts = np.array([r["timestamp_s"] for r in records])
        sp = np.sqrt(np.array([r["vx_mm_s"] for r in records])**2 +
                     np.array([r["vy_mm_s"] for r in records])**2)
        ax.plot(ts, sp, "-", color=track_color(tid), lw=0.8, alpha=0.7)
    seen_t: set[str] = set()
    for e in schedule:
        if e["region_idx"] > 0 and e["region_name"] not in seen_t:
            ax.axvline(e["timestamp_s"], color="gray", lw=0.8, linestyle="--", alpha=0.6)
            seen_t.add(e["region_name"])
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Velocidad escalar |v| (mm/s)")
    ax.set_title(f"Velocidad vs tiempo — {folder.name}")
    plt.tight_layout()
    out = folder / "ptv_velocidad_tiempo.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Velocidad vs tiempo → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS DPTV 1 — HEATMAP ESPACIAL DE PROFUNDIDAD
# ══════════════════════════════════════════════════════════════════

def plot_depth_heatmap(
    folder: Path,
    per_track: dict[int, list[dict]],
    extent_mm: list[float],
    dptv_cfg: dict,
) -> None:
    """
    Heatmap 2D de defocus_score y depth_blur_mm sobre el dominio.
    Revela si la profundidad varía sistemáticamente con la posición.
    """
    xs = np.array([r["x_mm"]          for recs in per_track.values() for r in recs])
    ys = np.array([r["y_mm"]          for recs in per_track.values() for r in recs])
    ds = np.array([r["defocus_score"]  for recs in per_track.values() for r in recs])
    bm = np.array([r["depth_blur_mm"] for recs in per_track.values() for r in recs])
    dm_raw = [r["depth_mm"]           for recs in per_track.values() for r in recs]
    dm = np.array([v if v is not None else np.nan for v in dm_raw], dtype=float)
    has_depth_mm = np.isfinite(dm).any()

    ncols = 3 if has_depth_mm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 2:
        axes = list(axes)
    fig.suptitle(f"Distribución espacial de profundidad DPTV — {folder.name}")

    panels = [
        (axes[0], ds, "viridis",   "Defocus score medio (W_meas/W_ideal)",
         "defocus_score [1=en foco]", False),
        (axes[1], bm, "plasma",    "Blur de profundidad medio (mm)",
         "depth_blur_mm [mm]", False),
    ]
    if has_depth_mm:
        panels.append(
            (axes[2], dm, "inferno", "Profundidad media |Δz| (mm)",
             "depth_mm [mm]", False)
        )

    for ax, vals, cmap_name, title, label, symmetric in panels:
        grid, cnt, xb, yb = _bin_avg(xs, ys, vals, GRID_SIZE, extent_mm)
        vmin, vmax = robust_clim(vals[np.isfinite(vals)], HEATMAP_COVERAGE, symmetric)
        cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad(alpha=0.0)
        masked = np.ma.masked_where(cnt == 0, grid)
        im = ax.imshow(masked.T, origin="lower", extent=extent_mm, cmap=cmap,
                       interpolation="bilinear", vmin=vmin, vmax=vmax, aspect="equal")
        setup_ax(ax, extent_mm)
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, shrink=0.82)
        cb.set_label(label)

    calibrated = dptv_cfg.get("depth_calibrated", False)
    calib_note = (f"k_blur={dptv_cfg.get('k_blur_px_per_mm'):.3f} px/mm"
                  if calibrated else "sin calibrar (depth_mm=None)")
    fig.text(0.5, 0.01, f"DPTV: fibra {dptv_cfg.get('fiber_width_mm',0.2)}mm × "
             f"{dptv_cfg.get('fiber_length_mm',13.0)}mm | {calib_note}",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    out = folder / "ptv_profundidad_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Heatmap profundidad → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS DPTV 2 — DISTRIBUCIÓN DE PROFUNDIDAD
# ══════════════════════════════════════════════════════════════════

def plot_depth_distribution(
    folder: Path,
    per_track: dict[int, list[dict]],
    dptv_cfg: dict,
) -> None:
    """
    Histogramas de defocus_score, depth_blur_mm y depth_confidence.
    Muestra la distribución de profundidades en el experimento completo.
    """
    all_records = [r for recs in per_track.values() for r in recs]
    ds   = np.array([r["defocus_score"]   for r in all_records])
    bm   = np.array([r["depth_blur_mm"]   for r in all_records])
    conf = np.array([r["depth_confidence"] for r in all_records])
    dm_raw = [r["depth_mm"] for r in all_records]
    dm   = np.array([v if v is not None else np.nan for v in dm_raw])
    has_dm = np.isfinite(dm).any()

    near_focus_pct = 100.0 * np.mean(ds < NEAR_FOCUS_DEFOCUS_THR)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4 if has_dm else 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4 if has_dm else 3)]
    fig.suptitle(
        f"Distribución de profundidad DPTV — {folder.name}\n"
        f"N={len(all_records)} observaciones | "
        f"{near_focus_pct:.1f}% cerca del plano focal (defocus<{NEAR_FOCUS_DEFOCUS_THR})"
    )

    # ── Defocus score ──────────────────────────────────────────────
    ax = axes[0]
    ax.hist(ds[np.isfinite(ds)], bins=50, color="#4878D0", edgecolor="none", alpha=0.85)
    ax.axvline(NEAR_FOCUS_DEFOCUS_THR, color="red", lw=1.5, linestyle="--",
               label=f"umbral={NEAR_FOCUS_DEFOCUS_THR}")
    ax.axvline(1.0, color="green", lw=1.2, linestyle=":", label="en foco perfecto")
    ax.set_xlabel("defocus_score")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Defocus score\nmedia={np.nanmean(ds):.2f}  σ={np.nanstd(ds):.2f}")
    ax.legend(fontsize=8)

    # ── Blur de profundidad ────────────────────────────────────────
    ax = axes[1]
    ax.hist(bm[np.isfinite(bm)], bins=50, color="#6ACC65", edgecolor="none", alpha=0.85)
    ax.set_xlabel("depth_blur_mm")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Blur de profundidad\nmedia={np.nanmean(bm):.3f} mm  "
                 f"p95={np.nanpercentile(bm,95):.3f} mm")

    # ── Confianza ─────────────────────────────────────────────────
    ax = axes[2]
    ax.hist(conf[np.isfinite(conf)], bins=40, color="#EE854A", edgecolor="none", alpha=0.85)
    ax.axvline(NEAR_FOCUS_CONF_THR, color="red", lw=1.5, linestyle="--",
               label=f"conf<{NEAR_FOCUS_CONF_THR}\n→ cerca del foco")
    ax.set_xlabel("depth_confidence")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Confianza de profundidad\n(0=en foco, 1=claramente desenfocada)")
    ax.legend(fontsize=8)

    # ── Profundidad absoluta (si calibrado) ───────────────────────
    if has_dm:
        ax = axes[3]
        dm_finite = dm[np.isfinite(dm)]
        ax.hist(dm_finite, bins=50, color="#956CB4", edgecolor="none", alpha=0.85)
        ax.set_xlabel("depth_mm")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Profundidad |Δz| (calibrada)\n"
                     f"media={np.nanmean(dm):.2f} mm  "
                     f"p95={np.nanpercentile(dm_finite,95):.2f} mm")

    plt.tight_layout()
    out = folder / "ptv_profundidad_dist.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Distribución profundidad → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS DPTV 3 — PROFUNDIDAD VS VELOCIDAD (plot clave para PIV)
# ══════════════════════════════════════════════════════════════════

def plot_depth_vs_velocity(
    folder: Path,
    per_track: dict[int, list[dict]],
    dptv_cfg: dict,
) -> None:
    """
    Correlación entre profundidad y velocidad de la fibra.

    Plot principal para la publicación: muestra si las fibras más
    desenfocadas (lejos del plano PIV) tienen velocidades distintas
    a las fibras cerca del plano focal.

    También muestra la velocidad ponderada por cercanía al foco
    (el estimador más comparable con el PIV).
    """
    all_records = [r for recs in per_track.values() for r in recs]
    bm   = np.array([r["depth_blur_mm"]   for r in all_records])
    conf = np.array([r["depth_confidence"] for r in all_records])
    vx   = np.array([r["vx_mm_s"]         for r in all_records])
    vy   = np.array([r["vy_mm_s"]         for r in all_records])
    sp   = np.sqrt(vx**2 + vy**2)
    ds   = np.array([r["defocus_score"]   for r in all_records])

    dm_raw = [r["depth_mm"] for r in all_records]
    dm = np.array([v if v is not None else np.nan for v in dm_raw])
    use_dm = np.isfinite(dm).any()

    # Eje de profundidad: depth_mm si calibrado, depth_blur_mm si no
    depth_axis = dm if use_dm else bm
    depth_label = "Profundidad |Δz| (mm)" if use_dm else "Blur de profundidad (mm)"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Correlación profundidad–velocidad DPTV — {folder.name}")

    # ── Panel 1: scatter profundidad vs velocidad ──────────────────
    ax = axes[0]
    mask = np.isfinite(depth_axis) & np.isfinite(sp)
    sc = ax.scatter(
        depth_axis[mask], sp[mask],
        c=conf[mask], cmap="YlOrRd",
        s=6, alpha=0.5, linewidths=0,
        vmin=0, vmax=1,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label("depth_confidence\n(0=en foco, 1=desenfocada)")

    # Línea de tendencia
    if mask.sum() > 10:
        from numpy.polynomial.polynomial import polyfit
        try:
            x_clean = depth_axis[mask]; y_clean = sp[mask]
            ok = np.isfinite(x_clean) & np.isfinite(y_clean)
            if ok.sum() > 5:
                coeffs = polyfit(x_clean[ok], y_clean[ok], 1)
                xf = np.array([x_clean[ok].min(), x_clean[ok].max()])
                ax.plot(xf, coeffs[0] + coeffs[1] * xf, "r--", lw=1.5,
                        label=f"tendencia m={coeffs[1]:.3f}")
                ax.legend(fontsize=9)
        except Exception:
            pass

    ax.set_xlabel(depth_label)
    ax.set_ylabel("Velocidad escalar |v| (mm/s)")
    ax.set_title("Velocidad vs profundidad\n(coloreado por confianza)")

    # ── Panel 2: velocidad media por bin de profundidad ───────────
    ax = axes[1]
    x_vals = depth_axis[mask]
    y_vals = sp[mask]
    n_bins = 15
    if np.isfinite(x_vals).sum() > n_bins:
        bins = np.percentile(x_vals[np.isfinite(x_vals)],
                             np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) > 2:
            idxs = np.digitize(x_vals, bins) - 1
            bin_means_x, bin_means_y, bin_stds_y = [], [], []
            for b in range(len(bins) - 1):
                mask_b = idxs == b
                if mask_b.sum() > 2:
                    bin_means_x.append(float(np.nanmedian(x_vals[mask_b])))
                    bin_means_y.append(float(np.nanmean(y_vals[mask_b])))
                    bin_stds_y.append(float(np.nanstd(y_vals[mask_b])))
            if bin_means_x:
                bx = np.array(bin_means_x)
                by = np.array(bin_means_y)
                bs = np.array(bin_stds_y)
                ax.fill_between(bx, by - bs, by + bs,
                                alpha=0.2, color="#4878D0")
                ax.plot(bx, by, "o-", color="#4878D0", lw=2, ms=6)

    ax.set_xlabel(depth_label)
    ax.set_ylabel("Velocidad media ± σ (mm/s)")
    ax.set_title("Perfil de velocidad vs profundidad\n(bins de percentil)")

    # ── Panel 3: velocidad ponderada por cercanía al foco ─────────
    ax = axes[2]
    # w_focus = 1 - depth_confidence  (1 = en foco, 0 = desenfocada)
    w = 1.0 - conf
    w_norm = w / (w.sum() if w.sum() > 0 else 1.0)

    all_sp  = sp[np.isfinite(sp)]
    w_sp    = w[np.isfinite(sp)]
    w_sp_n  = w_sp / (w_sp.sum() if w_sp.sum() > 0 else 1.0)

    v_mean_simple   = float(np.nanmean(sp))
    v_mean_weighted = float(np.average(all_sp, weights=w_sp_n))
    pct_near = 100.0 * np.mean(ds < NEAR_FOCUS_DEFOCUS_THR)

    categories = ["Velocidad media\n(todas las fibras)",
                  "Velocidad ponderada\n(peso=cercanía al foco)"]
    values     = [v_mean_simple, v_mean_weighted]
    colors     = ["#6ACC65", "#4878D0"]
    bars = ax.bar(categories, values, color=colors, edgecolor="k",
                  linewidth=0.8, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * max(values),
                f"{val:.2f} mm/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Velocidad escalar media (mm/s)")
    ax.set_title(
        f"Estimador de velocidad comparable con PIV\n"
        f"({pct_near:.1f}% fibras cerca del plano focal, defocus<{NEAR_FOCUS_DEFOCUS_THR})"
    )
    ax.set_ylim(0, max(values) * 1.25)

    note = ("Velocidad ponderada: cada fibra contribuye\nproporcionalmente "
            "a su cercanía al plano focal.\nMás comparable con medición PIV.")
    ax.text(0.98, 0.97, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.8))

    plt.tight_layout()
    out = folder / "ptv_profundidad_vs_vel.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Profundidad vs velocidad → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS DPTV 4 — PROFUNDIDAD VS TIEMPO
# ══════════════════════════════════════════════════════════════════

def plot_depth_vs_time(
    folder: Path,
    per_track: dict[int, list[dict]],
    schedule: list[dict],
) -> None:
    """
    Evolución temporal de la profundidad (depth_blur_mm) por track.
    Permite ver si las fibras cambian de profundidad durante el flujo.
    """
    region_bounds: dict[str, tuple[float, float]] = {}
    for e in schedule:
        rname = e["region_name"]; t = e["timestamp_s"]
        if rname not in region_bounds:
            region_bounds[rname] = (t, t)
        else:
            lo, hi = region_bounds[rname]
            region_bounds[rname] = (min(lo, t), max(hi, t))

    REGION_COLORS = ["#d4e8f7","#fde8c8","#d4f0d4","#f7d4d4","#e8d4f7","#f7f0d4","#d4f7f0"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Profundidad vs tiempo — {folder.name}")

    for ax in (ax1, ax2):
        for i, (rname, (t0, t1)) in enumerate(region_bounds.items()):
            ax.axvspan(t0, t1, alpha=0.20, color=REGION_COLORS[i % len(REGION_COLORS)], zorder=0)

    # Panel 1: defocus_score vs tiempo
    all_ts_sorted = sorted({r["timestamp_s"] for recs in per_track.values() for r in recs})

    for tid, records in per_track.items():
        ts = np.array([r["timestamp_s"] for r in records])
        ds = np.array([r["defocus_score"] for r in records])
        ax1.plot(ts, ds, "-", color=track_color(tid), lw=0.9, alpha=0.65)

    ax1.axhline(NEAR_FOCUS_DEFOCUS_THR, color="red", lw=1.5, linestyle="--",
                label=f"umbral foco={NEAR_FOCUS_DEFOCUS_THR}")
    ax1.axhline(1.0, color="green", lw=1.2, linestyle=":", label="foco perfecto")
    ax1.set_ylabel("Defocus score")
    ax1.set_title("Defocus score por track vs tiempo")
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: blur (mm) vs tiempo + fracción de fibras cerca del foco (eje secundario)
    for tid, records in per_track.items():
        ts = np.array([r["timestamp_s"] for r in records])
        bm = np.array([r["depth_blur_mm"] for r in records])
        ax2.plot(ts, bm, "-", color=track_color(tid), lw=0.9, alpha=0.65)

    if all_ts_sorted and len(all_ts_sorted) > 10:
        t_bins = np.linspace(all_ts_sorted[0], all_ts_sorted[-1], 30)
        frac_vals, t_mid_vals = [], []
        for i in range(len(t_bins) - 1):
            window = [r for recs in per_track.values() for r in recs
                      if t_bins[i] <= r["timestamp_s"] < t_bins[i + 1]]
            if window:
                frac_vals.append(float(np.mean(
                    [r["defocus_score"] < NEAR_FOCUS_DEFOCUS_THR for r in window]
                )))
                t_mid_vals.append((t_bins[i] + t_bins[i + 1]) / 2)

        if frac_vals:
            ax2b = ax2.twinx()
            ax2b.plot(t_mid_vals, [f * 100 for f in frac_vals],
                      "k--", lw=1.5, label="% fibras cerca del foco")
            ax2b.set_ylabel("% cerca del plano focal", color="black")
            ax2b.set_ylim(0, 110)
            ax2b.legend(fontsize=9, loc="upper left")

    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Blur de profundidad (mm)")
    ax2.set_title("Blur de profundidad por track vs tiempo")

    seen_t: set[str] = set()
    for e in schedule:
        if e["region_idx"] > 0 and e["region_name"] not in seen_t:
            for ax in (ax1, ax2):
                ax.axvline(e["timestamp_s"], color="gray", lw=0.8, linestyle="--", alpha=0.5)
            seen_t.add(e["region_name"])

    plt.tight_layout()
    out = folder / "ptv_profundidad_tiempo.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Profundidad vs tiempo → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANÁLISIS DPTV 5 — RESUMEN POR TRACK
# ══════════════════════════════════════════════════════════════════

def plot_track_summary(
    folder: Path,
    per_track: dict[int, list[dict]],
    track_depth_stats: dict[int, dict],
) -> None:
    """
    Scatter: velocidad media del track vs profundidad media del track.
    Cada punto es un track, coloreado por duración.
    Este es el plot más directo para la correlación PIV-PTV.
    """
    tids, v_means, blur_means, dm_means, durations, n_obs = [], [], [], [], [], []

    for tid, records in per_track.items():
        if not records:
            continue
        sp_arr = np.sqrt(np.array([r["vx_mm_s"] for r in records])**2 +
                         np.array([r["vy_mm_s"] for r in records])**2)
        v_mean = float(np.nanmean(sp_arr))
        bm_arr = np.array([r["depth_blur_mm"] for r in records])
        bm_mean = float(np.nanmean(bm_arr))

        stats  = track_depth_stats.get(tid, {})
        dm_val = stats.get("mean_depth_mm")
        duration = records[-1]["timestamp_s"] - records[0]["timestamp_s"]

        tids.append(tid); v_means.append(v_mean)
        blur_means.append(bm_mean); dm_means.append(dm_val)
        durations.append(duration); n_obs.append(len(records))

    if not tids:
        return

    has_dm = any(v is not None for v in dm_means)
    ncols = 3 if has_dm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 2:
        axes = list(axes)
    fig.suptitle(f"Resumen por track — {folder.name}  (N={len(tids)} tracks)")

    dur_arr = np.array(durations)
    norm = Normalize(vmin=dur_arr.min(), vmax=dur_arr.max())
    cmap = plt.get_cmap("viridis")

    # Panel 1: velocidad media vs blur medio
    ax = axes[0]
    sc = ax.scatter(blur_means, v_means,
                    c=dur_arr, cmap=cmap, norm=norm, s=60, edgecolors="k", lw=0.5)
    ax.set_xlabel("Blur medio del track (mm)")
    ax.set_ylabel("Velocidad media |v| (mm/s)")
    ax.set_title("Velocidad vs profundidad\npor track")
    fig.colorbar(sc, ax=ax, label="Duración del track (s)", shrink=0.82)

    # Correlación
    bm_arr = np.array(blur_means); v_arr = np.array(v_means)
    ok = np.isfinite(bm_arr) & np.isfinite(v_arr)
    if ok.sum() > 3:
        corr = float(np.corrcoef(bm_arr[ok], v_arr[ok])[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.3f}",
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    # Panel 2: distribución de velocidad media por track
    ax = axes[1]
    ax.hist(v_means, bins=20, color="#4878D0", edgecolor="k", lw=0.5, alpha=0.85)
    ax.axvline(np.nanmean(v_means), color="red", lw=1.5, linestyle="--",
               label=f"media={np.nanmean(v_means):.2f} mm/s")
    ax.set_xlabel("Velocidad media |v| (mm/s)")
    ax.set_ylabel("N tracks")
    ax.set_title("Distribución de velocidades medias\npor track")
    ax.legend(fontsize=9)

    # Panel 3: velocidad vs profundidad calibrada (si disponible)
    if has_dm:
        ax = axes[2]
        dm_arr = np.array([v if v is not None else np.nan for v in dm_means])
        sc2 = ax.scatter(dm_arr, v_arr,
                         c=dur_arr, cmap=cmap, norm=norm, s=60, edgecolors="k", lw=0.5)
        ax.set_xlabel("Profundidad media |Δz| (mm)")
        ax.set_ylabel("Velocidad media |v| (mm/s)")
        ax.set_title("Velocidad vs profundidad calibrada\npor track")
        fig.colorbar(sc2, ax=ax, label="Duración del track (s)", shrink=0.82)
        ok2 = np.isfinite(dm_arr) & np.isfinite(v_arr)
        if ok2.sum() > 3:
            corr2 = float(np.corrcoef(dm_arr[ok2], v_arr[ok2])[0, 1])
            ax.text(0.05, 0.95, f"r = {corr2:.3f}",
                    transform=ax.transAxes, fontsize=11, va="top",
                    bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    plt.tight_layout()
    out = folder / "ptv_resumen_tracks.png"
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[VIZ] Resumen por track → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ANIMACIÓN (igual que el original)
# ══════════════════════════════════════════════════════════════════

def _render_frame(args: tuple) -> str:
    (video_i, t_val, frame_data, folder_name,
     extent_mm, out_png_str, region_name) = args
    fig, (ax_pos, ax_trk) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{folder_name}  —  t = {t_val:.4f} s  [{region_name}]", fontsize=12)
    for ax in (ax_pos, ax_trk):
        xmin, xmax, ymin, ymax = extent_mm
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.invert_yaxis(); ax.grid(True, alpha=0.18, linewidth=0.6)
    ax_pos.set_title("Posición actual"); ax_trk.set_title("Trayectorias acumuladas")
    n_vis = 0
    for td in frame_data:
        cur_x, cur_y = td["cur_x"], td["cur_y"]
        if not np.isfinite(cur_x) or not np.isfinite(cur_y):
            continue
        n_vis += 1
        hue = (int(td["tid"]) * 137.508) % 360.0
        import colorsys as _cs
        color = _cs.hsv_to_rgb(hue / 360.0, 0.80, 0.92)
        half  = max(td["cur_l"] / 2.0, 0.5)
        ang   = np.deg2rad(td["cur_a"])
        dx, dy = np.cos(ang) * half, np.sin(ang) * half
        ax_pos.plot([cur_x - dx, cur_x + dx], [cur_y - dy, cur_y + dy],
                    color=color, lw=2.2, alpha=0.9)
        ax_pos.plot(cur_x, cur_y, "o", color=color, ms=3)
        if len(td["hist_x"]) >= 2:
            ax_trk.plot(td["hist_x"], td["hist_y"], "-", color=color, lw=1.4, alpha=0.80)
        if td["hist_x"]:
            ax_trk.plot(td["hist_x"][-1], td["hist_y"][-1], "o", color=color, ms=3.5)
    ax_pos.text(0.02, 0.98, f"t={t_val:.4f}s  |  n={n_vis}",
                transform=ax_pos.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
    plt.tight_layout()
    plt.savefig(out_png_str, dpi=100, bbox_inches="tight"); plt.close(fig)
    return out_png_str


def _region_at_time(t: float, schedule: list[dict]) -> str:
    return min(schedule, key=lambda e: abs(e["timestamp_s"] - t))["region_name"]


def build_animation(
    folder: Path,
    t_grid: np.ndarray,
    interpolated_tracks: dict[int, dict[str, np.ndarray]],
    schedule: list[dict],
    extent_mm: list[float],
) -> None:
    if not interpolated_tracks:
        return
    frame_indices = list(range(0, len(t_grid), ANIM_FRAME_STEP))
    if not frame_indices:
        return
    dt_grid  = float(np.diff(t_grid[:2])[0]) if len(t_grid) > 1 else 1.0
    fps_video = max(1.0, min(1.0 / (dt_grid * ANIM_FRAME_STEP), 120.0))
    arrs_by_tid = {tid: {"x": a["x_mm"], "y": a["y_mm"],
                          "a": a["angle_deg"], "l": a["length_mm"]}
                   for tid, a in interpolated_tracks.items()}
    tmp_dir   = tempfile.mkdtemp(prefix="ptv_anim_")
    args_list = []
    for video_i, grid_i in enumerate(frame_indices):
        t_val = float(t_grid[grid_i])
        out_png = str(Path(tmp_dir) / f"frame_{video_i:06d}.png")
        frame_data = []
        for tid, arrs in arrs_by_tid.items():
            cur_x = float(arrs["x"][grid_i]); cur_y = float(arrs["y"][grid_i])
            cur_a = float(arrs["a"][grid_i]) if np.isfinite(arrs["a"][grid_i]) else 0.0
            cur_l = float(arrs["l"][grid_i]) if np.isfinite(arrs["l"][grid_i]) else 1.0
            mask  = np.isfinite(arrs["x"][:grid_i+1]) & np.isfinite(arrs["y"][:grid_i+1])
            frame_data.append({
                "tid": tid, "cur_x": cur_x, "cur_y": cur_y,
                "cur_a": cur_a, "cur_l": cur_l,
                "hist_x": arrs["x"][:grid_i+1][mask].tolist(),
                "hist_y": arrs["y"][:grid_i+1][mask].tolist(),
            })
        args_list.append((video_i, t_val, frame_data,
                          folder.name, extent_mm, out_png,
                          _region_at_time(t_val, schedule)))
    print(f"[VIZ] Renderizando {len(args_list)} frames...", flush=True)
    with mp.Pool(processes=N_WORKERS) as pool:
        for done, _ in enumerate(pool.imap_unordered(_render_frame, args_list), 1):
            if done % max(1, len(args_list) // 10) == 0 or done == len(args_list):
                print(f"  → {done}/{len(args_list)}", flush=True)
    out_mp4 = folder / ANIM_FILENAME
    cmd = ["ffmpeg", "-y", "-framerate", f"{fps_video:.4f}",
           "-i", str(Path(tmp_dir) / "frame_%06d.png"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(out_mp4)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[VIZ] ERROR ffmpeg:\n{result.stderr}")
        else:
            print(f"[VIZ] Video → {out_mp4}", flush=True)
    except FileNotFoundError:
        print("[VIZ] ERROR: ffmpeg no está instalado.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    for tracks_path in sorted(BASE_DIR.rglob("tracks.json")):
        folder        = tracks_path.parent
        schedule_path = folder / "schedule.csv"

        print(f"\n[VIZ] ═══ {folder} ═══", flush=True)
        if not schedule_path.exists():
            print(f"[VIZ] WARN: No hay schedule.csv en {folder} — omitido.")
            continue

        try:
            schedule, (tracks, metadata) = load_schedule(schedule_path), load_tracks(tracks_path)
            fps, px_per_mm = get_cam_params(metadata, folder)
            dptv_cfg       = get_dptv_config(metadata)

            print(f"[VIZ] fps={fps}  px_per_mm={px_per_mm}  tracks={len(tracks)}"
                  f"  DPTV={'sí' if dptv_cfg.get('enabled') else 'no (datos viejos)'}", flush=True)

            if not tracks:
                print("[VIZ] Sin tracks — omitido."); continue

            # Verificar si los datos tienen campos DPTV
            sample_rec = (tracks[0].get("history") or [{}])[0]
            has_dptv = "defocus_score" in sample_rec
            if not has_dptv:
                print("[VIZ] WARN: tracks.json no tiene campos DPTV. "
                      "Regenera los resultados con el pipeline actualizado.")

            per_track        = extract_per_track(tracks)
            track_dep_stats  = extract_track_depth_stats(tracks)
            extent_mm        = compute_extent(per_track) if AUTO_EXTENT else EXTENT_MM_FIXED

            print(f"[VIZ] Tracks con datos: {len(per_track)}  "
                  f"Dominio: x=[{extent_mm[0]:.1f},{extent_mm[1]:.1f}]  "
                  f"y=[{extent_mm[2]:.1f},{extent_mm[3]:.1f}] mm", flush=True)

            # Grilla uniforme (necesaria para heatmaps y animación)
            if GEN_HEATMAPS_VEL or GEN_ANIMACION:
                t_grid, interpolated = build_uniform_grid(per_track, schedule, TARGET_DT_S)
                smooth_velocities(interpolated, half_window=2)
            else:
                t_grid, interpolated = None, {}

            # ── Análisis originales ──────────────────────────────────
            if GEN_HEATMAPS_VEL and t_grid is not None:
                plot_heatmaps(folder, t_grid, interpolated, schedule, extent_mm, metadata)

            if GEN_VEL_TIEMPO:
                plot_velocity_vs_time(folder, per_track, schedule)

            if GEN_ANIMACION and t_grid is not None:
                build_animation(folder, t_grid, interpolated, schedule, extent_mm)

            # ── Análisis DPTV (solo si hay datos de profundidad) ────
            if has_dptv:
                if GEN_DEPTH_HEATMAP:
                    plot_depth_heatmap(folder, per_track, extent_mm, dptv_cfg)

                if GEN_DEPTH_DIST:
                    plot_depth_distribution(folder, per_track, dptv_cfg)

                if GEN_DEPTH_VS_VEL:
                    plot_depth_vs_velocity(folder, per_track, dptv_cfg)

                if GEN_DEPTH_TIEMPO:
                    plot_depth_vs_time(folder, per_track, schedule)

                if GEN_RESUMEN_TRACKS:
                    plot_track_summary(folder, per_track, track_dep_stats)

            print(f"[VIZ] OK → {folder}", flush=True)

        except Exception as e:
            import traceback
            print(f"[VIZ] ERROR en {folder}: {e}"); traceback.print_exc()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
