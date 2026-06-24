"""
Análisis del perfil de flujo 3D mediante DPTV
==============================================

Usa los resultados PTV (tracks.json) para analizar cómo varía la velocidad
y la dirección del flujo en función de la profundidad estimada por DPTV.

Hallazgos principales (cam-1, m70-toma-1, carbopol):
  - La dirección del flujo rota ~45° con la profundidad: de ~20° (zona horizontal
    del dispositivo L) a ~65° (zona vertical, flujo gravitacional).
  - La velocidad aumenta con la profundidad porque la zona vertical está acelerada
    por la gravedad respecto a la zona horizontal ya frenada.
  - La asimetría espacial 2D (lado derecho 50% más rápido) refleja la posición
    del outlet del L y el canal principal de flujo.

Uso:
    python analisis_flujo_dptv.py
    python analisis_flujo_dptv.py --folder ResultadosPTV/ResultadosPTV/m70-toma-1-cam-1-n-3000-car-02-ptv
    python analisis_flujo_dptv.py --save-figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "ResultadosPTV" / "ResultadosPTV"
OUTPUT_DIR   = PROJECT_ROOT / "Calibracion" / "DPTV"


# ── carga de datos ─────────────────────────────────────────────────────────────

def load_tracks(folder: Path) -> tuple[list[dict], dict]:
    tf = folder / "tracks.json"
    sf = folder / "summary.json"
    with open(tf, encoding="utf-8") as f:
        tracks = json.load(f)["tracks"]
    with open(sf, encoding="utf-8") as f:
        summary = json.load(f)
    return tracks, summary


def extract_records(tracks: list[dict], min_track_len: int = 5) -> list[dict]:
    records = []
    for t in tracks:
        hist = t.get("history", [])
        if len(hist) < min_track_len:
            continue
        for h in hist:
            dm   = h.get("depth_mm")
            dc   = h.get("depth_confidence", 0)
            vx   = h.get("vx_mm_s", 0) or 0
            vy   = h.get("vy_mm_s", 0) or 0
            spd  = math.sqrt(vx**2 + vy**2)
            if dm is None or dm <= 0 or spd <= 0:
                continue
            records.append({
                "depth":      dm,
                "confidence": dc,
                "speed":      min(spd, 300),
                "vx":         vx,
                "vy":         vy,
                "x":          h.get("x_mm", 0),
                "y":          h.get("y_mm", 0),
                "track_id":   t["track_id"],
            })
    return records


# ── análisis por bins de profundidad ──────────────────────────────────────────

DEPTH_BINS = [0, 5, 10, 15, 20, 25, 30, 50]
BIN_LABELS = ["0–5", "5–10", "10–15", "15–20", "20–25", "25–30", "30+"]


def profile_by_depth(records: list[dict]) -> list[dict]:
    out = []
    for lo, hi, label in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:], BIN_LABELS):
        recs = [r for r in records if lo <= r["depth"] < hi]
        if len(recs) < 5:
            continue
        speeds = np.array([r["speed"] for r in recs])
        vxs    = np.array([r["vx"]    for r in recs])
        vys    = np.array([r["vy"]    for r in recs])
        confs  = np.array([r["confidence"] for r in recs])
        vx_m, vy_m = vxs.mean(), vys.mean()
        angle = math.degrees(math.atan2(vy_m, vx_m))
        out.append({
            "bin":        label,
            "lo":         lo,
            "hi":         hi,
            "mid":        (lo + hi) / 2,
            "n":          len(recs),
            "speed_mean": float(speeds.mean()),
            "speed_p25":  float(np.percentile(speeds, 25)),
            "speed_p50":  float(np.median(speeds)),
            "speed_p75":  float(np.percentile(speeds, 75)),
            "vx_mean":    float(vx_m),
            "vy_mean":    float(vy_m),
            "angle_deg":  angle,
            "conf_mean":  float(confs.mean()),
        })
    return out


# ── gráficos ──────────────────────────────────────────────────────────────────

def depth_color(d: float, max_d: float = 32.0) -> tuple:
    t = min(1.0, d / max_d)
    return (0.31 + 0.65 * t, 0.56 - 0.22 * t, 0.97 - 0.66 * t)


def plot_analysis(records: list[dict], profile: list[dict],
                  folder_name: str, out_dir: Path, save: bool) -> None:

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Análisis de flujo 3D — DPTV  ({folder_name})", fontsize=13)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    depths = np.array([r["depth"] for r in records])
    speeds = np.array([r["speed"] for r in records])
    confs  = np.array([r["confidence"] for r in records])
    xs     = np.array([r["x"] for r in records])
    ys     = np.array([r["y"] for r in records])

    mids   = [p["mid"]        for p in profile]
    smeans = [p["speed_mean"] for p in profile]
    sp25   = [p["speed_p25"]  for p in profile]
    sp75   = [p["speed_p75"]  for p in profile]
    angles = [p["angle_deg"]  for p in profile]
    cmeans = [p["conf_mean"]  for p in profile]
    colors = [depth_color(m)  for m in mids]

    # ── 1. Histograma de profundidad ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(depths, bins=40, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(31, color="grey", lw=1.2, ls="--", label="pared (31 mm)")
    ax.set_xlabel("|Δz| desde plano focal  [mm]")
    ax.set_ylabel("Observaciones")
    ax.set_title("Distribución de profundidad estimada")
    ax.legend(fontsize=8)

    # ── 2. Velocidad vs profundidad ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(mids, sp25, sp75, alpha=0.18, color="steelblue")
    ax.plot(mids, smeans, "o-", lw=2, ms=6, color="steelblue", label="media")
    ax.plot(mids, [p["speed_p50"] for p in profile], "s--",
            lw=1.2, ms=4, color="steelblue", alpha=0.6, label="mediana")
    for m, s, c in zip(mids, smeans, colors):
        ax.scatter(m, s, color=c, s=60, zorder=5)
    ax.set_xlabel("|Δz|  [mm]")
    ax.set_ylabel("Velocidad  [mm/s]")
    ax.set_title("Velocidad vs profundidad")
    ax.legend(fontsize=8)

    # Línea Poiseuille esperado (referencia)
    ax.plot([0, 31], [195, 100], "r--", lw=1, alpha=0.5, label="Poiseuille ref.")
    ax.legend(fontsize=8)

    # ── 3. Dirección del flujo (rosa) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    for p, c in zip(profile, colors):
        mag = math.sqrt(p["vx_mean"]**2 + p["vy_mean"]**2)
        th  = math.radians(p["angle_deg"])
        ax.annotate("", xy=(th, mag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
        ax.text(th, mag * 1.12, p["bin"] + "mm", fontsize=7,
                ha="center", va="center", color=c)
    ax.set_title("Dirección media del flujo por profundidad", pad=12, fontsize=10)
    ax.set_xlabel("Eje X = derecha, Eje Y = arriba (mm/s)")

    # ── 4. Confianza de profundidad vs profundidad ────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(depths, confs, s=2, alpha=0.15, color="steelblue")
    ax.plot(mids, cmeans, "o-", color="steelblue", lw=2, ms=5)
    ax.set_xlabel("|Δz|  [mm]")
    ax.set_ylabel("depth_confidence")
    ax.set_title("Confianza de profundidad\n(noise_px = 6.8, corregido)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="conf=0.5")
    ax.legend(fontsize=8)

    # ── 5. Mapa 2D de velocidad media ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    nc = 8
    grid_v = np.full((nc, nc), np.nan)
    grid_c = np.zeros((nc, nc), dtype=int)
    cs = 128.0 / nc
    for r in records:
        xi = min(int(r["x"] / cs), nc - 1)
        yi = min(int(r["y"] / cs), nc - 1)
        if np.isnan(grid_v[yi, xi]):
            grid_v[yi, xi] = r["speed"]
            grid_c[yi, xi] = 1
        else:
            grid_v[yi, xi] += r["speed"]
            grid_c[yi, xi] += 1
    with np.errstate(invalid="ignore"):
        grid_mean = grid_v / np.where(grid_c > 0, grid_c, 1)
        grid_mean[grid_c == 0] = np.nan
    im = ax.imshow(grid_mean, origin="lower", extent=[0, 128, 0, 128],
                   cmap="jet", vmin=80, vmax=280, aspect="auto")
    plt.colorbar(im, ax=ax, label="mm/s", shrink=0.8)
    ax.set_xlabel("X  [mm]")
    ax.set_ylabel("Y  [mm]")
    ax.set_title("Mapa 2D de velocidad media")

    # ── 6. Ángulo de flujo vs profundidad ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for p, c in zip(profile, colors):
        ax.scatter(p["mid"], p["angle_deg"], color=c, s=80, zorder=5)
    ax.plot(mids, angles, "o-", color="grey", lw=1.5, ms=0, alpha=0.5)
    ax.axhline(45, color="red", lw=0.8, ls="--", alpha=0.5, label="45°")
    ax.set_xlabel("|Δz|  [mm]")
    ax.set_ylabel("Ángulo del flujo  [°]")
    ax.set_title("Rotación de dirección con profundidad\n(20° horizontal → 65° vertical)")
    ax.set_ylim(0, 90)
    ax.legend(fontsize=8)
    ax.text(16, 22, "← zona horizontal\n   del L", fontsize=8, color="steelblue")
    ax.text(22, 68, "← zona vertical\n   del L", fontsize=8, color="#c47a00")

    if save:
        out_path = out_dir / f"analisis_flujo_{folder_name}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Figura guardada: {out_path.relative_to(PROJECT_ROOT)}")
    else:
        plt.show()
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Análisis flujo 3D desde DPTV")
    ap.add_argument("--folder", default=None,
                    help="Carpeta de resultados PTV (default: primera encontrada)")
    ap.add_argument("--save-figures", action="store_true",
                    help="Guardar figuras en Calibracion/DPTV/ en vez de mostrarlas")
    ap.add_argument("--min-track-len", type=int, default=5)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.folder:
        folders = [Path(args.folder)]
    else:
        folders = sorted(RESULTS_DIR.glob("*-ptv")) if RESULTS_DIR.exists() else []

    if not folders:
        print(f"No se encontraron carpetas PTV en {RESULTS_DIR}")
        return

    for folder in folders:
        print(f"\n-- {folder.name} --")
        tracks, summary = load_tracks(folder)
        records = extract_records(tracks, args.min_track_len)
        if not records:
            print("  Sin registros con depth_mm + speed.")
            continue

        profile = profile_by_depth(records)

        print(f"  Registros: {len(records)}, tracks: {len(tracks)}")
        print(f"  Profundidad: mean={np.mean([r['depth'] for r in records]):.1f} mm")
        print(f"\n  {'Bin':8s} {'N':>6} {'speed':>8} {'angle':>7} {'conf':>6}")
        for p in profile:
            print(f"  {p['bin']:8s} {p['n']:>6} {p['speed_mean']:>8.1f}"
                  f" {p['angle_deg']:>7.1f} {p['conf_mean']:>6.3f}")

        plot_analysis(records, profile, folder.name, OUTPUT_DIR, args.save_figures)


if __name__ == "__main__":
    main()
