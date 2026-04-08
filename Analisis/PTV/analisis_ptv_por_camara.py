"""
ptv_visualizer.py
=================
Visualizador PTV — usa tracks.json + schedule.csv.

El schedule.csv es la fuente de verdad temporal:
  - Define exactamente qué frames se analizaron
  - Provee timestamp_s y dt_s reales por frame (variables entre regiones)
  - Permite construir ejes temporales correctos para animaciones y heatmaps

Flujo:
  1. Cargar schedule.csv → grilla temporal real (irregular)
  2. Cargar tracks.json → trayectorias con timestamp_s ya correcto
  3. Interpolar tracks a grilla uniforme entre los timestamps del schedule
     (necesario para animación fluida cuando hay pocos frames por región)
  4. Generar: heatmaps de velocidad + animación MP4

Interpolación:
  - Dentro del rango de vida de un track: interpolación cúbica (spline)
    si tiene ≥4 puntos, lineal si tiene 2-3, constante si tiene 1.
  - Fuera del rango: NaN (el track no existe en ese instante).
  - La grilla de animación usa el TARGET_DT_S más fino entre todas las
    regiones para no perder detalle en alta velocidad.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import colorsys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

BASE_DIR = Path("ResultadosPTV")

# Grilla temporal para animación y heatmaps.
# None → usar el dt mínimo encontrado en schedule.csv (máxima resolución).
# Valor explícito (ej. 1/220) → forzar ese paso independientemente del schedule.
TARGET_DT_S: float | None = None

# Submuestreo para el video: tomar 1 de cada N frames de la grilla uniforme.
# Sube este valor si la animación es muy larga o el render es lento.
ANIM_FRAME_STEP = 3
ANIM_FPS        = 20
ANIM_FILENAME   = "ptv_animacion.mp4"
HEATMAP_FILENAME = "ptv_heatmaps.png"

GRID_SIZE         = (60, 60)    # resolución del heatmap (celdas)
HEATMAP_COVERAGE  = 0.95        # percentil para escala de colores
ANGULAR_SYMMETRIC = True        # escala simétrica para velocidad angular

AUTO_EXTENT    = True
EXTENT_MM_FIXED = [0, 130, 0, 130]  # [xmin, xmax, ymin, ymax] mm, si AUTO_EXTENT=False

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


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

def load_schedule(path: Path) -> list[dict]:
    """
    Carga schedule.csv.
    Cada fila es un frame analizado con su contexto temporal.

    Columnas relevantes:
        sched_idx, frame_idx, timestamp_s, dt_s, dt_ms,
        region_name, region_idx, n_detections, n_tracks_active
    """
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    """
    Carga tracks.json nuevo formato.
    Returns (tracks, metadata).
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    tracks   = data.get("tracks", [])
    return tracks, metadata


def get_cam_params(metadata: dict, folder: Path) -> tuple[float, float]:
    """Extrae fps y px_per_mm del JSON o infiere del nombre de carpeta."""
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


# ─────────────────────────────────────────────
# EXTRACCIÓN DE TRAYECTORIAS POR TRACK
# ─────────────────────────────────────────────

def extract_per_track(tracks: list[dict]) -> dict[int, list[dict]]:
    """
    Extrae un dict {track_id: [records ordenados por timestamp_s]}.
    Cada record tiene: timestamp_s, x_mm, y_mm, vx_mm_s, vy_mm_s,
                       omega_deg_s, angle_deg, length_mm, region_name, dt_s
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
            records.append({
                "timestamp_s":  float(ts),
                "x_mm":         float(x),
                "y_mm":         float(y),
                "vx_mm_s":      float(rec.get("vx_mm_s") or 0.0),
                "vy_mm_s":      float(rec.get("vy_mm_s") or 0.0),
                "omega_deg_s":  float(rec.get("omega_deg_s") or 0.0),
                "angle_deg":    float(rec.get("angle_deg") or 0.0),
                "length_mm":    float(rec.get("length_mm") or 1.0),
                "region_name":  str(rec.get("region_name") or ""),
                "dt_s":         float(rec.get("dt_s") or 0.0),
            })

        if records:
            records.sort(key=lambda r: r["timestamp_s"])
            per_track[tid] = records

    return per_track


# ─────────────────────────────────────────────
# INTERPOLACIÓN A GRILLA UNIFORME
# ─────────────────────────────────────────────

def _interp_field(t_obs: np.ndarray, values: np.ndarray,
                  t_grid: np.ndarray) -> np.ndarray:
    """
    Interpola `values` observados en `t_obs` sobre `t_grid`.

    Estrategia por número de puntos:
      1 punto  → constante en el rango
      2-3 pts  → lineal
      ≥4 pts   → cúbica natural (CubicSpline, no extrapola)

    Fuera del rango [t_obs[0], t_obs[-1]] → NaN.
    """
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
    """
    Interpola todas las variables de un track sobre t_grid.
    Retorna dict {campo: array(len(t_grid))}.
    """
    t_obs = np.array([r["timestamp_s"] for r in records])
    fields = {
        "x_mm":        np.array([r["x_mm"]        for r in records]),
        "y_mm":        np.array([r["y_mm"]         for r in records]),
        "vx_mm_s":     np.array([r["vx_mm_s"]      for r in records]),
        "vy_mm_s":     np.array([r["vy_mm_s"]      for r in records]),
        "omega_deg_s": np.array([r["omega_deg_s"]  for r in records]),
        "angle_deg":   np.array([r["angle_deg"]    for r in records]),
        "length_mm":   np.array([r["length_mm"]    for r in records]),
    }

    return {name: _interp_field(t_obs, vals, t_grid)
            for name, vals in fields.items()}

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

            # ── FIX: si el segmento es más corto que la ventana,
            #    reducir el kernel al tamaño real del segmento (siempre impar)
            seg_len = len(segment)
            if seg_len < n:
                hw = (seg_len - 1) // 2      # half_window efectivo
                if hw == 0:                  # segmento de 1 punto → nada que suavizar
                    continue
                k_eff = 2 * hw + 1
                kern  = np.ones(k_eff) / k_eff
            else:
                hw    = half_window
                k_eff = n
                kern  = kernel

            smoothed = np.convolve(segment, kern, mode="same")

            # Corregir los extremos del padding implícito de 'same'
            for i in range(min(hw, len(smoothed))):
                w = i + hw + 1
                smoothed[i] *= k_eff / w
            for i in range(1, min(hw + 1, len(smoothed))):
                w = i + hw
                smoothed[-i] *= k_eff / w

            arr[idx] = smoothed
            
def build_uniform_grid(
    per_track: dict[int, list[dict]],
    schedule: list[dict],
    target_dt_s: float | None = None,
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    """
    Construye grilla temporal uniforme e interpola todos los tracks sobre ella.

    La grilla usa el dt mínimo del schedule (entre todas las regiones) para
    no perder resolución temporal en zonas de alta velocidad.

    El schedule garantiza que la grilla cubre exactamente el rango temporal
    de los datos analizados.

    Args:
        per_track:   {track_id: records}
        schedule:    filas del schedule.csv
        target_dt_s: paso forzado (None = usar mínimo del schedule)

    Returns:
        t_grid:              array 1D de tiempos uniformes
        interpolated_tracks: {track_id: {campo: array(len(t_grid))}}
    """
    # Rango temporal del schedule
    t_sched = np.array([e["timestamp_s"] for e in schedule])
    t_min   = float(t_sched.min())
    t_max   = float(t_sched.max())

    # Paso de tiempo
    if target_dt_s is not None:
        dt = float(target_dt_s)
    else:
        # Usar el dt mínimo de entre todas las regiones del schedule
        dts = np.array([e["dt_s"] for e in schedule])
        dt  = float(dts.min())

    t_grid = np.arange(t_min, t_max + dt * 0.5, dt)

    # Estadísticas de regiones para el log
    regions_seen = {}
    for e in schedule:
        rname = e["region_name"]
        if rname not in regions_seen:
            regions_seen[rname] = e["dt_ms"]

    region_str = "  ".join(f"{r}({dt:.2f}ms)" for r, dt in regions_seen.items())
    print(
        f"[VIZ] Grilla temporal: [{t_min:.3f}s, {t_max:.3f}s]  "
        f"dt={dt*1000:.3f}ms  N={len(t_grid)} pasos\n"
        f"      Regiones: {region_str}",
        flush=True,
    )

    interpolated: dict[int, dict[str, np.ndarray]] = {}
    for tid, records in per_track.items():
        interpolated[tid] = interpolate_track(records, t_grid)

    return t_grid, interpolated


# ─────────────────────────────────────────────
# EXTENT DEL DOMINIO
# ─────────────────────────────────────────────

def compute_extent(per_track: dict[int, list[dict]],
                   margin_frac: float = 0.03) -> list[float]:
    """Extent en mm con margen, calculado sobre todas las observaciones."""
    all_x = [r["x_mm"] for recs in per_track.values() for r in recs]
    all_y = [r["y_mm"] for recs in per_track.values() for r in recs]
    if not all_x:
        return EXTENT_MM_FIXED
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    mx = (xmax - xmin) * margin_frac
    my = (ymax - ymin) * margin_frac
    return [xmin - mx, xmax + mx, ymin - my, ymax + my]


# ─────────────────────────────────────────────
# UTILIDADES GRÁFICAS
# ─────────────────────────────────────────────

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
    ax.invert_yaxis()   # coordenadas imagen: y crece hacia abajo
    ax.grid(True, alpha=0.18, linewidth=0.6)


def rod_endpoints(x: float, y: float, angle_deg: float,
                  length_mm: float) -> tuple:
    half = length_mm / 2.0
    ang  = np.deg2rad(angle_deg)
    dx   = np.cos(ang) * half
    dy   = np.sin(ang) * half
    return (x - dx, x + dx), (y - dy, y + dy)


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


# ─────────────────────────────────────────────
# HEATMAPS
# ─────────────────────────────────────────────

def _bin_avg(xs: np.ndarray, ys: np.ndarray, vals: np.ndarray,
             grid_size: tuple, extent_mm: list[float]):
    nx, ny = grid_size
    xmin, xmax, ymin, ymax = extent_mm
    xb = np.linspace(xmin, xmax, nx + 1)
    yb = np.linspace(ymin, ymax, ny + 1)

    mask  = np.isfinite(vals) & np.isfinite(xs) & np.isfinite(ys)
    xi    = np.clip(np.digitize(xs[mask], xb) - 1, 0, nx - 1)
    yi    = np.clip(np.digitize(ys[mask], yb) - 1, 0, ny - 1)

    s = np.zeros((nx, ny))
    c = np.zeros((nx, ny))
    for i, j, v in zip(xi, yi, vals[mask]):
        s[i, j] += v
        c[i, j] += 1

    avg = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    return avg, c, xb, yb


def plot_heatmaps(
    folder: Path,
    t_grid: np.ndarray,
    interpolated_tracks: dict[int, dict[str, np.ndarray]],
    schedule: list[dict],
    extent_mm: list[float],
    metadata: dict,
) -> None:
    """
    Genera heatmaps de velocidad lineal y angular sobre la grilla uniforme.

    Dibuja también líneas verticales marcando los cambios de región temporal.
    """
    # Recopilar todas las muestras interpoladas
    xs_l, ys_l, vx_l, vy_l, om_l = [], [], [], [], []
    for arr in interpolated_tracks.values():
        xs_l.append(arr["x_mm"])
        ys_l.append(arr["y_mm"])
        vx_l.append(arr["vx_mm_s"])
        vy_l.append(arr["vy_mm_s"])
        om_l.append(arr["omega_deg_s"])

    xs_all = np.concatenate(xs_l)
    ys_all = np.concatenate(ys_l)
    vx_all = np.concatenate(vx_l)
    vy_all = np.concatenate(vy_l)
    om_all = np.concatenate(om_l)
    sp_all = np.sqrt(vx_all**2 + vy_all**2)

    sp_grid, sp_cnt, _, _ = _bin_avg(xs_all, ys_all, sp_all, GRID_SIZE, extent_mm)
    om_grid, om_cnt, _, _ = _bin_avg(xs_all, ys_all, om_all, GRID_SIZE, extent_mm)

    sp_valid = sp_all[np.isfinite(sp_all)]
    om_valid = om_all[np.isfinite(om_all)]

    vmin_sp, vmax_sp = robust_clim(sp_valid, HEATMAP_COVERAGE, symmetric=False)
    vmin_om, vmax_om = robust_clim(om_valid, HEATMAP_COVERAGE, symmetric=ANGULAR_SYMMETRIC)

    # Info de regiones para el título
    tr_info = ""
    tr_regions = metadata.get("temporal_regions") or []
    if tr_regions:
        tr_info = f" | {len(tr_regions)} regiones temporales"

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(
        f"Heatmaps PTV — {folder.name}\n"
        f"N_tracks={len(interpolated_tracks)} | "
        f"grilla dt={np.diff(t_grid[:2])[0]*1000:.2f}ms{tr_info}"
    )

    panels = [
        ("turbo",    axes[0], sp_grid, sp_cnt, sp_valid,
         "Velocidad lineal promedio (mm/s)", vmin_sp, vmax_sp),
        ("coolwarm", axes[1], om_grid, om_cnt, om_valid,
         "Velocidad angular promedio (°/s)",  vmin_om, vmax_om),
    ]

    for cmap_name, ax, grid, cnt, valid, label, vmin, vmax in panels:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)
        masked = np.ma.masked_where(cnt == 0, grid)

        ax.imshow(
            masked.T, origin="lower",
            extent=extent_mm, cmap=cmap,
            interpolation="bilinear",
            vmin=vmin, vmax=vmax, aspect="equal",
        )
        setup_ax(ax, extent_mm)

        n_valid = int(np.isfinite(valid).sum())
        cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.82)
        cb.set_label(
            f"{label}\n"
            f"media={np.nanmean(valid):.2f}  "
            f"p{int(HEATMAP_COVERAGE*100)}=[{vmin:.2f},{vmax:.2f}]  "
            f"N={n_valid}"
        )

    axes[0].set_title("Velocidad lineal |v| (mm/s)")
    axes[1].set_title("Velocidad angular ω (°/s)")

    plt.tight_layout()
    out = folder / HEATMAP_FILENAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Heatmap → {out}", flush=True)


# ─────────────────────────────────────────────
# GRÁFICO DE VELOCIDAD vs TIEMPO (por región)
# ─────────────────────────────────────────────

def plot_velocity_vs_time(
    folder: Path,
    per_track: dict[int, list[dict]],
    schedule: list[dict],
) -> None:
    """
    Gráfico de velocidad escalar vs tiempo para todos los tracks.
    Sombreado por región temporal. Usa los timestamps reales (irregulares).
    """
    # Límites de cada región desde el schedule
    region_bounds: dict[str, tuple[float, float]] = {}
    for e in schedule:
        rname = e["region_name"]
        t     = e["timestamp_s"]
        if rname not in region_bounds:
            region_bounds[rname] = (t, t)
        else:
            lo, hi = region_bounds[rname]
            region_bounds[rname] = (min(lo, t), max(hi, t))

    REGION_COLORS = [
        "#d4e8f7", "#fde8c8", "#d4f0d4", "#f7d4d4",
        "#e8d4f7", "#f7f0d4", "#d4f7f0",
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Sombrear regiones
    for i, (rname, (t0, t1)) in enumerate(region_bounds.items()):
        color = REGION_COLORS[i % len(REGION_COLORS)]
        ax.axvspan(t0, t1, alpha=0.35, color=color, label=rname, zorder=0)
        ax.text(
            (t0 + t1) / 2, ax.get_ylim()[1] if i == 0 else 0,
            rname.replace("_", "\n"), ha="center", va="bottom",
            fontsize=8, color="gray",
        )

    # Una línea por track
    for tid, records in per_track.items():
        ts = np.array([r["timestamp_s"] for r in records])
        vx = np.array([r["vx_mm_s"]     for r in records])
        vy = np.array([r["vy_mm_s"]      for r in records])
        sp = np.sqrt(vx**2 + vy**2)
        color = track_color(tid)
        ax.plot(ts, sp, "-", color=color, lw=0.8, alpha=0.7, label=f"T{tid}")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Velocidad escalar |v| (mm/s)")
    ax.set_title(f"Velocidad vs tiempo — {folder.name}")

    # Marcar cambios de región con línea vertical
    seen_t = set()
    for e in schedule:
        if e["region_idx"] > 0:
            t = e["timestamp_s"]
            # Primera aparición de esta región
            if e["region_name"] not in seen_t:
                ax.axvline(t, color="gray", lw=0.8, linestyle="--", alpha=0.6)
                seen_t.add(e["region_name"])

    plt.tight_layout()
    out = folder / "ptv_velocidad_tiempo.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Velocidad vs tiempo → {out}", flush=True)


# ─────────────────────────────────────────────
# RENDER DE UN FRAME (top-level para multiprocessing)
# ─────────────────────────────────────────────

def _render_frame(args: tuple) -> str:
    """Renderiza un frame del video. Top-level para pickle."""
    (video_i, t_val, frame_data, folder_name,
     extent_mm, out_png_str, region_name) = args

    fig, (ax_pos, ax_trk) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{folder_name}  —  t = {t_val:.4f} s  [{region_name}]",
        fontsize=12,
    )

    for ax in (ax_pos, ax_trk):
        xmin, xmax, ymin, ymax = extent_mm
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.18, linewidth=0.6)

    ax_pos.set_title("Posición actual")
    ax_trk.set_title("Trayectorias acumuladas")

    n_vis = 0
    for td in frame_data:
        tid    = td["tid"]
        cur_x  = td["cur_x"]
        cur_y  = td["cur_y"]
        cur_a  = td["cur_a"]
        cur_l  = td["cur_l"]
        hist_x = td["hist_x"]
        hist_y = td["hist_y"]

        if not np.isfinite(cur_x) or not np.isfinite(cur_y):
            continue
        n_vis += 1
        color = tuple((int(tid) * 137.508 * i) % 1.0
                      for i in [1, 2, 3])
        # Reconstruir color de forma reproducible
        hue = (int(tid) * 137.508) % 360.0
        import colorsys as _cs
        color = _cs.hsv_to_rgb(hue / 360.0, 0.80, 0.92)

        # Segmento de fibra
        half = max(cur_l / 2.0, 0.5)
        ang  = np.deg2rad(cur_a)
        dx, dy = np.cos(ang) * half, np.sin(ang) * half
        ax_pos.plot(
            [cur_x - dx, cur_x + dx],
            [cur_y - dy, cur_y + dy],
            color=color, lw=2.2, alpha=0.9,
        )
        ax_pos.plot(cur_x, cur_y, "o", color=color, ms=3)

        # Trayectoria acumulada
        if len(hist_x) >= 2:
            ax_trk.plot(hist_x, hist_y, "-", color=color, lw=1.4, alpha=0.80)
        if hist_x:
            ax_trk.plot(hist_x[-1], hist_y[-1], "o", color=color, ms=3.5)

    ax_pos.text(
        0.02, 0.98,
        f"t={t_val:.4f}s  |  n={n_vis}",
        transform=ax_pos.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_png_str, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_png_str


# ─────────────────────────────────────────────
# ANIMACIÓN
# ─────────────────────────────────────────────

def _region_at_time(t: float, schedule: list[dict]) -> str:
    """Retorna el nombre de la región en el instante t."""
    # Buscar el entry del schedule más cercano a t
    closest = min(schedule, key=lambda e: abs(e["timestamp_s"] - t))
    return closest["region_name"]


def build_animation(
    folder: Path,
    t_grid: np.ndarray,
    interpolated_tracks: dict[int, dict[str, np.ndarray]],
    schedule: list[dict],
    extent_mm: list[float],
) -> None:
    if not interpolated_tracks:
        print(f"[VIZ] Sin tracks — sin animación.")
        return

    frame_indices = list(range(0, len(t_grid), ANIM_FRAME_STEP))
    if not frame_indices:
        return

    dt_grid = float(np.diff(t_grid[:2])[0]) if len(t_grid) > 1 else 1.0

    # ── CAMBIO: FPS calculado para velocidad real ──────────────────────────
    # Cada frame del video representa ANIM_FRAME_STEP pasos de la grilla,
    # es decir, un intervalo de dt_grid * ANIM_FRAME_STEP segundos.
    # Para que el video corra a velocidad real: fps_real = 1 / ese intervalo.
    real_fps = 1.0 / (dt_grid * ANIM_FRAME_STEP)
    # Limitar a un máximo razonable (evita videos inutilizablemente rápidos
    # si la grilla es muy fina) y a un mínimo (evita videos de 1 fps).
    fps_video = max(1.0, min(real_fps, 120.0))
    # ──────────────────────────────────────────────────────────────────────

    duracion_real_s = len(frame_indices) * dt_grid * ANIM_FRAME_STEP
    print(
        f"[VIZ] Animación: {len(frame_indices)} frames del video  "
        f"(grilla={len(t_grid)}, step={ANIM_FRAME_STEP})  "
        f"fps_video={fps_video:.2f}  "          # ← antes era ANIM_FPS fijo
        f"duración_video={len(frame_indices)/fps_video:.1f}s  "
        f"duración_real={duracion_real_s:.2f}s",
        flush=True,
    )

    # Pre-extraer arrays por track para slicing rápido
    arrs_by_tid = {
        tid: {
            "x": a["x_mm"], "y": a["y_mm"],
            "a": a["angle_deg"], "l": a["length_mm"],
        }
        for tid, a in interpolated_tracks.items()
    }

    tmp_dir = tempfile.mkdtemp(prefix="ptv_anim_")
    args_list = []

    for video_i, grid_i in enumerate(frame_indices):
        t_val      = float(t_grid[grid_i])
        region_now = _region_at_time(t_val, schedule)
        out_png    = str(Path(tmp_dir) / f"frame_{video_i:06d}.png")

        frame_data = []
        for tid, arrs in arrs_by_tid.items():
            cur_x = float(arrs["x"][grid_i])
            cur_y = float(arrs["y"][grid_i])
            cur_a = float(arrs["a"][grid_i]) if np.isfinite(arrs["a"][grid_i]) else 0.0
            cur_l = float(arrs["l"][grid_i]) if np.isfinite(arrs["l"][grid_i]) else 1.0

            # Historia: posiciones finitas desde t=0 hasta el frame actual
            mask  = (np.isfinite(arrs["x"][:grid_i + 1]) &
                     np.isfinite(arrs["y"][:grid_i + 1]))
            hist_x = arrs["x"][:grid_i + 1][mask].tolist()
            hist_y = arrs["y"][:grid_i + 1][mask].tolist()

            frame_data.append({
                "tid":    tid,
                "cur_x":  cur_x,  "cur_y": cur_y,
                "cur_a":  cur_a,  "cur_l": cur_l,
                "hist_x": hist_x, "hist_y": hist_y,
            })

        args_list.append((
            video_i, t_val, frame_data,
            folder.name, extent_mm, out_png, region_now,
        ))

    print(f"[VIZ] Renderizando {len(args_list)} frames ({N_WORKERS} workers)...", flush=True)
    with mp.Pool(processes=N_WORKERS) as pool:
        for done, _ in enumerate(pool.imap_unordered(_render_frame, args_list), 1):
            if done % max(1, len(args_list) // 10) == 0 or done == len(args_list):
                print(f"  → {done}/{len(args_list)}", flush=True)

    # Ensamblar con ffmpeg
    out_mp4 = folder / ANIM_FILENAME
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps_video:.4f}",       # ← antes era str(ANIM_FPS)
        "-i", str(Path(tmp_dir) / "frame_%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
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


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def main() -> None:
    # Buscar todas las carpetas con tracks.json Y schedule.csv
    for tracks_path in sorted(BASE_DIR.rglob("tracks.json")):
        folder        = tracks_path.parent
        schedule_path = folder / "schedule.csv"

        print(f"\n[VIZ] ═══ {folder} ═══", flush=True)

        # ── Verificar que existe schedule.csv ─────────────────────
        if not schedule_path.exists():
            print(
                f"[VIZ] WARN: No se encontró schedule.csv en {folder}.\n"
                f"      Regenera los resultados con la versión actualizada del runner.",
                flush=True,
            )
            continue

        try:
            # ── Cargar ────────────────────────────────────────────
            schedule         = load_schedule(schedule_path)
            tracks, metadata = load_tracks(tracks_path)
            fps, px_per_mm   = get_cam_params(metadata, folder)

            print(
                f"[VIZ] fps={fps}  px_per_mm={px_per_mm}  "
                f"tracks={len(tracks)}  schedule_frames={len(schedule)}",
                flush=True,
            )

            if not tracks:
                print(f"[VIZ] Sin tracks — omitido.")
                continue

            # ── Extraer trayectorias ───────────────────────────────
            per_track = extract_per_track(tracks)
            print(
                f"[VIZ] Tracks con datos: {len(per_track)}  "
                f"| Rango temporal schedule: "
                f"{schedule[0]['timestamp_s']:.3f}s → "
                f"{schedule[-1]['timestamp_s']:.3f}s",
                flush=True,
            )

            # ── Extent del dominio ─────────────────────────────────
            extent_mm = compute_extent(per_track) if AUTO_EXTENT else EXTENT_MM_FIXED
            print(
                f"[VIZ] Dominio: x=[{extent_mm[0]:.1f},{extent_mm[1]:.1f}]mm  "
                f"y=[{extent_mm[2]:.1f},{extent_mm[3]:.1f}]mm",
                flush=True,
            )

            # ── Grilla uniforme e interpolación ────────────────────
            # El schedule define los timestamps reales (irregulares por región).
            # Interpolamos a una grilla uniforme con el dt más fino del schedule
            # para que la animación y los heatmaps tengan densidad temporal
            # consistente en todo el rango, incluyendo zonas de baja velocidad
            # donde hay muy pocos frames observados.
            t_grid, interpolated = build_uniform_grid(
                per_track, schedule, target_dt_s=TARGET_DT_S
            )

            # ── NUEVO: suavizar velocidades antes de analizar ──────────────
            smooth_velocities(interpolated, half_window=2)
            # half_window=2 → ventana de 5 puntos (t±2 pasos de la grilla)
            # Aumentar si la grilla es muy fina o hay mucho ruido de tracking
            # ─────────────────────────────────────────────────────────────

            # ── Heatmaps ──────────────────────────────────────────
            plot_heatmaps(folder, t_grid, interpolated,
                          schedule, extent_mm, metadata)

            # ── Velocidad vs tiempo ───────────────────────────────
            plot_velocity_vs_time(folder, per_track, schedule)

            # ── Animación ─────────────────────────────────────────
            build_animation(folder, t_grid, interpolated,
                            schedule, extent_mm)

            print(f"[VIZ] OK → {folder}", flush=True)

        except Exception as e:
            import traceback
            print(f"[VIZ] ERROR en {folder}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()