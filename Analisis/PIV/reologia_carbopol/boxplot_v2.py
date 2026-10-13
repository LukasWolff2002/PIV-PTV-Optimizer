"""
Boxplot mejorado: mitad violin / mitad box, eje lineal cortado en p99,
puntos más visibles, anotación max=23 s⁻¹.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

PIV_ANALISIS = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers"
sys.path.insert(0, PIV_ANALISIS)
os.chdir(PIV_ANALISIS)

from extraer_reologia_piv import extraer_nube_run, CONCENTRACIONES

SUBSET = {"02": ["m71-toma-1-n-0000-car-02-piv"]}

data_by_conc = {}
for conc_key, params in CONCENTRACIONES.items():
    if conc_key not in SUBSET:
        continue
    gd_all = []
    for run_name in SUBSET[conc_key]:
        df = extraer_nube_run(run_name, tau0=params["tau0"],
                              rho=params["rho"], verbose=True)
        if not df.empty:
            gd_all.append(df["gamma_dot"].values)
    data_by_conc[conc_key] = np.concatenate(gd_all) if gd_all else np.array([])

# ── figura ───────────────────────────────────────────────────────────────────
COLOR  = "#4a7ab5"
rng    = np.random.default_rng(42)
arr    = data_by_conc["02"]

p1, p25, p50, p75, p95, p99 = np.percentile(arr, [1, 25, 50, 75, 95, 99])
y_max_plot = p99 * 1.08          # eje cortado en p99 + 8 % de margen
x_ctr      = 1.0                 # centro del gráfico

fig, ax = plt.subplots(figsize=(5.0, 6.0))

# ── 1. VIOLIN (mitad izquierda) ──────────────────────────────────────────────
kde      = gaussian_kde(arr[arr <= y_max_plot * 2], bw_method=0.18)
y_kde    = np.linspace(0, y_max_plot, 400)
kde_vals = kde(y_kde)
kde_norm = kde_vals / kde_vals.max() * 0.38   # ancho máximo normalizado

ax.fill_betweenx(y_kde,
                 x_ctr - kde_norm, x_ctr,
                 color=COLOR, alpha=0.55, linewidth=0)
ax.plot(x_ctr - kde_norm, y_kde,
        color=COLOR, linewidth=1.2, alpha=0.85)

# ── 2. BOX (mitad derecha) ───────────────────────────────────────────────────
bw = 0.18   # semiancho de caja
# caja IQR
ax.add_patch(mpatches.FancyBboxPatch(
    (x_ctr, p25), bw, p75 - p25,
    boxstyle="square,pad=0", linewidth=1.4,
    edgecolor=COLOR, facecolor=COLOR, alpha=0.55))
# mediana
ax.plot([x_ctr, x_ctr + bw], [p50, p50],
        color="white", linewidth=2.5, zorder=5)
# bigotes p5–p95
ax.plot([x_ctr + bw / 2, x_ctr + bw / 2], [np.percentile(arr, 5), p25],
        color=COLOR, linewidth=1.4, zorder=4)
ax.plot([x_ctr + bw / 2, x_ctr + bw / 2], [p75, p95],
        color=COLOR, linewidth=1.4, zorder=4)
# topes de bigote
for y_cap in [np.percentile(arr, 5), p95]:
    ax.plot([x_ctr + bw * 0.15, x_ctr + bw * 0.85], [y_cap, y_cap],
            color=COLOR, linewidth=1.4, zorder=4)

# ── 3. PUNTOS (franja central, más visibles) ─────────────────────────────────
n_pts   = 2500
sub     = rng.choice(arr[arr <= y_max_plot * 1.5], min(n_pts, len(arr)), replace=False)
sub     = sub[sub <= y_max_plot]
x_jit   = rng.uniform(-0.06, 0.06, len(sub))
ax.scatter(x_ctr + x_jit, sub,
           s=5, alpha=0.35, color=COLOR,
           linewidths=0, zorder=3)

# ── 4. ANOTACIONES ───────────────────────────────────────────────────────────
ax.annotate(f"máx = {arr.max():.1f} s⁻¹\n(fuera de escala)",
            xy=(x_ctr + bw / 2, y_max_plot),
            xytext=(x_ctr + bw / 2 + 0.28, y_max_plot * 0.90),
            fontsize=8.5, color="#333",
            arrowprops=dict(arrowstyle="-|>", color="#777",
                            lw=1.0, mutation_scale=10),
            ha="left")

# estadísticas en caja de texto
stats_txt = (f"mediana = {p50:.2f} s⁻¹\n"
             f"Q1 = {p25:.2f}  Q3 = {p75:.2f}\n"
             f"p95 = {p95:.2f} s⁻¹\n"
             f"N = {len(arr):,}")
ax.text(0.03, 0.97, stats_txt,
        transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

# ── 5. ETIQUETAS Y ESTILO ────────────────────────────────────────────────────
ax.set_xlim(0.40, 1.70)
ax.set_ylim(-0.3, y_max_plot)
ax.set_xticks([x_ctr])
ax.set_xticklabels(["Carbopol 0,2 %"], fontsize=11)
ax.set_ylabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=12)
ax.set_title("Distribución de tasa de corte\nZona Z1 — brazo inclinado 60°",
             fontsize=11, pad=8)
ax.axhline(0, color="#bbb", linewidth=0.6, linestyle=":")
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.tick_params(axis="both", which="major", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# leyenda mitad izquierda / mitad derecha
leg = [mpatches.Patch(color=COLOR, alpha=0.55, label="Violin (densidad)"),
       mpatches.Patch(color=COLOR, alpha=0.55, label="Box (IQR)")]
ax.legend(handles=leg, loc="upper right", fontsize=8, framealpha=0.85)

plt.tight_layout()
OUT = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\boxplot_shear_rate_v2.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Guardado: {OUT}")
plt.close()
