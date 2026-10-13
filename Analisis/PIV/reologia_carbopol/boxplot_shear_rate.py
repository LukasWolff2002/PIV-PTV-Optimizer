"""
Boxplot de tasa de corte gamma_dot (1/s) para Carbopol 0.2 % y 0.5 %
usando los pares (tau, gamma_dot) extraídos de la zona Z1 por extraer_reologia_piv.py.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PIV_ANALISIS = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers"
sys.path.insert(0, PIV_ANALISIS)
os.chdir(PIV_ANALISIS)          # para que REPO_DIR = Path(__file__).parent funcione

from extraer_reologia_piv import extraer_nube_run, CONCENTRACIONES

# ─── 1. Recolectar gamma_dot por concentración ──────────────────────────────
data_by_conc = {}   # {"02": array, "05": array}

SUBSET = {"02": ["m71-toma-1-n-0000-car-02-piv"]}   # one run only

for conc_key, params in CONCENTRACIONES.items():
    if conc_key not in SUBSET:
        continue
    gd_all = []
    print(f"\n{'='*55}")
    print(f"  Concentración: {params['label']}")
    for run_name in SUBSET[conc_key]:
        df = extraer_nube_run(run_name, tau0=params["tau0"],
                              rho=params["rho"], verbose=True)
        if df.empty:
            print(f"  SKIP {run_name}: sin datos")
            continue
        gd_all.append(df["gamma_dot"].values)
        print(f"  {run_name}: {len(df):,} pts  gd=[{df['gamma_dot'].min():.3f}, "
              f"{df['gamma_dot'].max():.2f}] s⁻¹")
    if gd_all:
        data_by_conc[conc_key] = np.concatenate(gd_all)
    else:
        data_by_conc[conc_key] = np.array([])

# ─── 2. Estadísticas rápidas ─────────────────────────────────────────────────
for k, arr in data_by_conc.items():
    label = CONCENTRACIONES[k]["label"]
    if len(arr) == 0:
        print(f"\n{label}: sin datos")
        continue
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"\n{label}  (N={len(arr):,})")
    print(f"  p5={p[0]:.3f}  Q1={p[1]:.3f}  median={p[2]:.3f}  "
          f"Q3={p[3]:.3f}  p95={p[4]:.3f}  max={arr.max():.3f}  s⁻¹")

# ─── 3. Figura ───────────────────────────────────────────────────────────────
COLORS = {"02": "#4a7ab5", "05": "#c45022"}
LABELS = {k: CONCENTRACIONES[k]["label"] for k in CONCENTRACIONES}

conc_keys   = [k for k in ["02", "05"] if len(data_by_conc.get(k, [])) > 0]
plot_data   = [data_by_conc[k] for k in conc_keys]
plot_labels = [LABELS[k] for k in conc_keys]
plot_colors = [COLORS[k] for k in conc_keys]

fig, ax = plt.subplots(figsize=(5.5, 5.0))

bp = ax.boxplot(
    plot_data,
    notch=True,
    patch_artist=True,
    widths=0.4,
    whis=(5, 95),          # whiskers at 5th / 95th percentile
    showfliers=False,
    medianprops=dict(color="white", linewidth=2.0),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    boxprops=dict(linewidth=1.2),
)

for patch, col in zip(bp["boxes"], plot_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.80)

# Overlay a strip of individual points (thin out if too many)
rng = np.random.default_rng(42)
for i, (arr, col) in enumerate(zip(plot_data, plot_colors)):
    # Subsample at most 3000 points for clarity
    sub = arr if len(arr) <= 3000 else rng.choice(arr, 3000, replace=False)
    x_jitter = rng.uniform(-0.15, 0.15, len(sub))
    ax.scatter(i + 1 + x_jitter, sub,
               s=2, alpha=0.15, color=col, linewidths=0, zorder=2)

ax.set_ylabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=12)
ax.set_xticks(list(range(1, len(plot_labels) + 1)))
ax.set_xticklabels(plot_labels, fontsize=11)
ax.set_title(
    "Distribución de tasa de corte — Zona Z1 (brazo inclinado 60°)\nPIV · Carbopol 0,2 %",
    fontsize=10, pad=8
)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
ax.tick_params(axis="both", which="major", labelsize=10)

# Annotation with N and key percentiles
for i, (k, arr) in enumerate(zip(conc_keys, plot_data)):
    p50 = np.median(arr)
    p95 = np.percentile(arr, 95)
    ymax = ax.get_ylim()[1]
    ax.text(i + 1, ymax * 0.97,
            f"N={len(arr):,}\nmediana={p50:.2f} s⁻¹\np95={p95:.2f} s⁻¹",
            ha="center", va="top", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))

plt.tight_layout()
out_path = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\boxplot_shear_rate_z1.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nFigura guardada: {out_path}")
plt.close()
