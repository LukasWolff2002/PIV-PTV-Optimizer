"""
Calibración empírica de DPTV_K_BLUR_PX_PER_MM
==============================================

Este script estima el parámetro k_blur (px/mm) del modelo DPTV a partir de:
1. La distribución de anchos detectados por YOLO+PCA en los resultados PTV.
2. La geometría conocida del contenedor (viga 62 mm → profundidad máxima 31 mm).

Modelo físico
-------------
W_apparent² = W_ideal² + (k_blur × Δz)²
  W_ideal  ≈ FIBER_WIDTH_MM × px_per_mm  (ancho de fibra en foco perfecto)
  blur_px  = sqrt(W_apparent² - W_ideal²)
  depth_mm = blur_px / k_blur

Metodología de calibración geométrica
--------------------------------------
Se asume que las fibras se distribuyen uniformemente en la profundidad del
contenedor (0 … max_depth_mm), con el plano focal en el centro:
    E[depth] = max_depth / 2  (valor esperado de |Δz|)
    k_blur   = E[blur_px] / E[depth]

Para la viga rectangular (cámaras 2 y 3):
    ancho interno ≈ 62 mm → max_depth = 31 mm

Para el dispositivo L (cámaras 1 y 4):
    Las dimensiones internas del canal deben medirse con calibre.
    Mientras tanto se usa la misma geometría que la viga como estimación.

Cómo mejorar la calibración
----------------------------
1. Colocar una fibra a profundidades conocidas (z = 0, 5, 10, 20, 30 mm)
   frente a cada cámara y medir width_px en las detecciones.
2. Ajustar k_blur = Δ(blur_px) / Δz con regresión lineal.
3. Actualizar DPTV_K_BLUR_PX_PER_MM en RunCode/variables_ptv.py.

Valor estimado actualmente: 2.5 px/mm
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Configuración ──────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR    = PROJECT_ROOT / "ResultadosPTV" / "ResultadosPTV"

# Geometría del contenedor  [mm]
# Cambiar si se conocen las dimensiones reales del dispositivo L
BEAM_HALF_DEPTH_MM   = 31.0   # beam formwork: ancho 62 mm → half-depth 31 mm
L_DEVICE_HALF_DEPTH_MM = 31.0  # estimación; actualizar con medición real

HALF_DEPTH_BY_CAM = {1: L_DEVICE_HALF_DEPTH_MM,
                      2: BEAM_HALF_DEPTH_MM,
                      3: BEAM_HALF_DEPTH_MM,
                      4: L_DEVICE_HALF_DEPTH_MM}

OUTPUT_DIR = PROJECT_ROOT / "Calibracion" / "DPTV"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIBER_WIDTH_MM = 0.2   # ancho físico de la fibra [mm]


# ── Lectura de datos ───────────────────────────────────────────────────────────

def load_detections(folder: Path) -> tuple[int, float, list[float]]:
    """Lee detections.csv y devuelve (cam, px_per_mm, [width_px, ...])."""
    det_file = folder / "detections.csv"
    sum_file = folder / "summary.json"
    if not det_file.exists():
        return None, None, []

    with open(sum_file) as f:
        meta = json.load(f)
    cam        = meta["camera"]["cam"]
    px_per_mm  = meta["camera"]["px_per_mm"]

    widths = []
    with open(det_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                widths.append(float(row["width_px"]))
            except (KeyError, ValueError):
                pass
    return cam, px_per_mm, widths


def compute_k_blur(widths_px: list[float], px_per_mm: float,
                   half_depth_mm: float) -> dict:
    """Estima k_blur con el método de media geométrica."""
    w_ideal = FIBER_WIDTH_MM * px_per_mm
    blurs   = [math.sqrt(max(0.0, w**2 - w_ideal**2)) for w in widths_px]
    blurs_arr = np.array(blurs)

    mean_blur  = float(np.mean(blurs_arr))
    p50_blur   = float(np.median(blurs_arr))
    p95_blur   = float(np.percentile(blurs_arr, 95))
    p99_blur   = float(np.percentile(blurs_arr, 99))
    mean_depth = half_depth_mm / 2.0  # E[|Δz|] para distribución uniforme

    k_blur_mean = mean_blur / mean_depth if mean_depth > 0 else None
    k_blur_p95  = p95_blur  / half_depth_mm if half_depth_mm > 0 else None

    return {
        "n":            len(widths_px),
        "w_ideal_px":   w_ideal,
        "mean_blur_px": mean_blur,
        "p50_blur_px":  p50_blur,
        "p95_blur_px":  p95_blur,
        "p99_blur_px":  p99_blur,
        "half_depth_mm": half_depth_mm,
        "k_blur_mean":  k_blur_mean,
        "k_blur_p95":   k_blur_p95,
        "k_blur_recommended": k_blur_mean,
        "blurs": blurs_arr,
        "widths": np.array(widths_px),
    }


# ── Visualización ──────────────────────────────────────────────────────────────

def plot_calibration(cam: int, px_per_mm: float,
                     stats: dict, out_dir: Path) -> None:
    half_d = stats["half_depth_mm"]
    k      = stats["k_blur_recommended"]
    blurs  = stats["blurs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"DPTV — calibración geométrica  (cam-{cam},  {px_per_mm} px/mm)",
                 fontsize=13)

    # Histograma de blur_px
    ax = axes[0]
    ax.hist(blurs, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(stats["mean_blur_px"], color="red",   lw=1.5, label=f"media={stats['mean_blur_px']:.1f} px")
    ax.axvline(stats["p95_blur_px"],  color="orange", lw=1.5, label=f"p95={stats['p95_blur_px']:.1f} px")
    ax.set_xlabel("blur_px  (=√(width² − w_ideal²))  [px]")
    ax.set_ylabel("Detecciones")
    ax.set_title("Distribución de blur_px")
    ax.legend(fontsize=9)

    # Curva de calibración: blur_px vs depth_mm
    ax2 = axes[1]
    depth_range = np.linspace(0, half_d * 1.1, 200)
    if k:
        blur_curve  = k * depth_range
        ax2.plot(depth_range, blur_curve, "k-", lw=2, label=f"k_blur={k:.2f} px/mm")

    # Mapeo de percentiles observados a profundidad
    for label, bp, color in [
        ("media",  stats["mean_blur_px"], "red"),
        ("p50",    stats["p50_blur_px"],  "steelblue"),
        ("p95",    stats["p95_blur_px"],  "orange"),
    ]:
        if k:
            d = bp / k
            ax2.axhline(bp, color=color, lw=0.8, ls="--")
            ax2.axvline(d,  color=color, lw=0.8, ls="--")
            ax2.scatter([d], [bp], color=color, zorder=5, label=f"{label}: {d:.1f} mm")

    ax2.axvline(half_d, color="grey", lw=1.2, ls=":", label=f"pared ({half_d:.0f} mm)")
    ax2.set_xlabel("depth_mm  (|Δz| desde plano focal)  [mm]")
    ax2.set_ylabel("blur_px")
    ax2.set_title("Curva de calibración blur_px vs depth_mm")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / f"calibracion_dptv_cam{cam}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Figura guardada: {out_path.relative_to(PROJECT_ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    folders = sorted(RESULTS_DIR.glob("*-ptv")) if RESULTS_DIR.exists() else []
    if not folders:
        print(f"No se encontraron carpetas PTV en {RESULTS_DIR}")
        return

    summary_all = {}

    for folder in folders:
        cam, px_per_mm, widths = load_detections(folder)
        if not widths:
            print(f"[skip] {folder.name} — sin detections.csv o vacío")
            continue

        half_d = HALF_DEPTH_BY_CAM.get(cam, BEAM_HALF_DEPTH_MM)
        stats  = compute_k_blur(widths, px_per_mm, half_d)

        print(f"\n── cam-{cam}  ({folder.name}) ──")
        print(f"  N detecciones   : {stats['n']}")
        print(f"  w_ideal_px      : {stats['w_ideal_px']:.2f} px")
        print(f"  mean_blur_px    : {stats['mean_blur_px']:.1f} px")
        print(f"  p95_blur_px     : {stats['p95_blur_px']:.1f} px")
        print(f"  half_depth_mm   : {half_d:.1f} mm")
        print(f"  k_blur (media)  : {stats['k_blur_mean']:.3f} px/mm")
        print(f"  k_blur (p95)    : {stats['k_blur_p95']:.3f} px/mm")
        print(f"  → RECOMENDADO   : {stats['k_blur_recommended']:.3f} px/mm")

        # Verificación: depth estimada con k_blur recomendado
        k = stats["k_blur_recommended"]
        if k:
            print("\n  Verificación de profundidades estimadas:")
            for label, wp in [("p1",    np.percentile(stats['widths'], 1)),
                               ("p10",   np.percentile(stats['widths'], 10)),
                               ("media", np.mean(stats['widths'])),
                               ("p90",   np.percentile(stats['widths'], 90)),
                               ("p99",   np.percentile(stats['widths'], 99))]:
                w_ideal = stats["w_ideal_px"]
                blur    = math.sqrt(max(0, wp**2 - w_ideal**2))
                depth   = blur / k
                print(f"    width_{label}={wp:.1f}px → blur={blur:.1f}px → depth={depth:.1f}mm")

        plot_calibration(cam, px_per_mm, stats, OUTPUT_DIR)
        summary_all[str(folder.name)] = {
            "cam":        cam,
            "px_per_mm":  px_per_mm,
            "half_depth_mm": half_d,
            "k_blur_mean": round(stats["k_blur_mean"], 4) if stats["k_blur_mean"] else None,
            "k_blur_p95":  round(stats["k_blur_p95"], 4)  if stats["k_blur_p95"]  else None,
            "k_blur_recommended": round(stats["k_blur_recommended"], 4) if stats["k_blur_recommended"] else None,
        }

    # Guardar resumen JSON
    out_json = OUTPUT_DIR / "calibracion_dptv_resumen.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"\nResumen guardado: {out_json.relative_to(PROJECT_ROOT)}")

    print("\n── Valor sugerido para variables_ptv.py ──")
    vals = [v["k_blur_recommended"] for v in summary_all.values()
            if v["k_blur_recommended"]]
    if vals:
        suggested = round(sum(vals) / len(vals), 2)
        print(f"  DPTV_K_BLUR_PX_PER_MM = {suggested}  (promedio entre cámaras)")
    print("\n  ⚠ Estimación geométrica — para calibración precisa colocar fibras")
    print("    a profundidades conocidas y medir width_px con el pipeline.")


if __name__ == "__main__":
    main()
