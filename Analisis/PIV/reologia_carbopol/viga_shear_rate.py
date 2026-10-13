"""
Shear rate en la viga (Vf1c1 ... Vf2c3) — Carbopol 0.2 %
Flujo horizontal: T=(1,0), N=(0,1)  -> gd = |du_x/dy|
Corte temporal: t > 1.8 s
"""
import sys, os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from pathlib import Path

PIV_ANALISIS = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers"
PIV_DIR      = Path(r"C:\Users\patri\OneDrive\Desktop\PIV-PTV-Optimizer"
                    r"\PIV\PIV_INTERPOLADO\PIV_INTERPOLADO")
sys.path.insert(0, PIV_ANALISIS)
os.chdir(PIV_ANALISIS)

from definir_zonas import asignar_zona, POLIGONOS_ZONA, n_filas_viga, n_columnas_viga
from extraer_reologia_piv import (cargar_frame, extraer_timestamp,
                                   natural_sort_key, ETAPAS_JSON)

T_MIN   = 1.8
RUNS_02 = ["m71-toma-1-n-0000-car-02-piv"]
K_VEC, DIST_MAX, MIN_VEC, COND_MAX = 6, 5.0, 4, 1e4

# Coordenadas fijas de la viga: flujo en x, normal en y
T_V = np.array([1.0, 0.0])
N_V = np.array([0.0, 1.0])

def gd_fixed(x, y, u, v, T, N):
    n_pts = len(x); gd = np.full(n_pts, np.nan)
    s  = x*T[0] + y*T[1]; n_ = x*N[0] + y*N[1]; us = u*T[0] + v*T[1]
    pts = np.column_stack([x, y]); tree = cKDTree(pts)
    for i in range(n_pts):
        dists, idx = tree.query(pts[i], k=K_VEC+1,
                                distance_upper_bound=DIST_MAX+1e-6, workers=1)
        mask = (dists > 0) & (dists <= DIST_MAX); idx = idx[mask]
        if len(idx) < MIN_VEC: continue
        A = np.column_stack([np.ones(len(idx)), s[idx]-s[i], n_[idx]-n_[i]])
        try:
            if np.linalg.cond(A) > COND_MAX: continue
            c_, _, _, _ = np.linalg.lstsq(A, us[idx], rcond=None)
            val = abs(c_[2])
            if np.isfinite(val) and val >= 0: gd[i] = val
        except: pass
    return gd

# Nombres de celdas ordenados
CELDAS = [f"Vf{f+1}c{c+1}" for f in range(n_filas_viga) for c in range(n_columnas_viga)]

with open(ETAPAS_JSON) as f:
    etapas_data = json.load(f)

gd_celda = {cel: [] for cel in CELDAS}
frame_counts = {cel: 0 for cel in CELDAS}

for run_name in RUNS_02:
    carpeta = PIV_DIR / run_name
    files   = sorted(glob.glob(str(carpeta / "frame_*.txt")),
                     key=lambda f: natural_sort_key(os.path.basename(f)))
    t_peak  = etapas_data[f"{run_name}_Z1"]["t_peak"]
    print(f"Run: {run_name}  frames={len(files)}")

    for i, fpath in enumerate(files):
        t_val = extraer_timestamp(fpath)
        if t_val <= T_MIN:
            continue
        df = cargar_frame(fpath)
        if df.empty: continue
        x, y, u, v = df["x"].values, df["y"].values, df["u"].values, df["v"].values
        zonas = asignar_zona(x, y)

        for cel in CELDAS:
            mask = zonas == cel
            if mask.sum() < MIN_VEC + 1: continue
            g = gd_fixed(x[mask], y[mask], u[mask], v[mask], T_V, N_V)
            vals = g[np.isfinite(g) & (g >= 0)]
            if len(vals) > 0:
                gd_celda[cel].extend(vals.tolist())
                frame_counts[cel] += 1

        if i % 100 == 0:
            print(f"  frame {i}/{len(files)}  t={t_val:.2f}s", end="\r")

print()

# ── Estadísticas por celda ───────────────────────────────────────────────────
print("\nEstadisticas por celda (t > 1.8s):")
print(f"{'Celda':<10} {'N':>8} {'med':>7} {'Q1':>7} {'Q3':>7} {'p95':>8} {'max':>8}")
for cel in CELDAS:
    arr = np.array(gd_celda[cel])
    if len(arr) == 0:
        print(f"{cel:<10}  sin datos"); continue
    p = np.percentile(arr, [25, 50, 75, 95])
    print(f"{cel:<10} {len(arr):>8,} {p[1]:>7.3f} {p[0]:>7.3f} {p[2]:>7.3f} {p[3]:>8.3f} {arr.max():>8.3f}")

np.save(os.path.join(PIV_ANALISIS, "gd_viga_por_celda.npy"),
        {cel: np.array(v) for cel, v in gd_celda.items()})
print("\nArray guardado: gd_viga_por_celda.npy")

# ── Figura: grilla 2x3 de violin+box ─────────────────────────────────────────
# Color gradiente por columna (indica posición a lo largo del beam)
COL_COLORS = ["#2b6ba8", "#5b9bd5", "#91c4e8"]  # c1=oscuro, c3=claro
rng = np.random.default_rng(42)

fig, axes = plt.subplots(n_filas_viga, n_columnas_viga, figsize=(11, 7),
                         sharey=False, sharex=False)
fig.suptitle("Tasa de corte gamma_dot en la Viga horizontal\n"
             "Carbopol 0.2 % · m71-toma-1 · t > 1.8 s", fontsize=12)

for fila in range(n_filas_viga):
    for col in range(n_columnas_viga):
        cel  = f"Vf{fila+1}c{col+1}"
        ax   = axes[fila][col]
        arr  = np.array(gd_celda[cel])
        color = COL_COLORS[col]

        ax.set_title(f"{cel}  (fila {fila+1}, col {col+1})", fontsize=9)

        if len(arr) < 10:
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); continue

        p     = np.percentile(arr, [5, 25, 50, 75, 95, 99])
        y_max = p[5] * 1.08
        x_ctr = 1.0; bw = 0.18

        # Violin (izquierda)
        kde      = gaussian_kde(arr[arr <= y_max*2], bw_method=0.18)
        y_kde    = np.linspace(0, y_max, 300)
        kde_vals = kde(y_kde); kde_norm = kde_vals / kde_vals.max() * 0.35
        ax.fill_betweenx(y_kde, x_ctr - kde_norm, x_ctr,
                         color=color, alpha=0.50, linewidth=0)
        ax.plot(x_ctr - kde_norm, y_kde, color=color, lw=1.2)

        # Box (derecha)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_ctr, p[1]), bw, p[3]-p[1],
            boxstyle="square,pad=0", lw=1.3,
            edgecolor=color, facecolor=color, alpha=0.55))
        ax.plot([x_ctr, x_ctr+bw], [p[2]]*2, color="white", lw=2.0, zorder=5)
        ax.plot([x_ctr+bw/2]*2, [p[0], p[1]], color=color, lw=1.2, zorder=4)
        ax.plot([x_ctr+bw/2]*2, [p[3], p[4]], color=color, lw=1.2, zorder=4)
        for yc in [p[0], p[4]]:
            ax.plot([x_ctr+bw*0.2, x_ctr+bw*0.8], [yc]*2, color=color, lw=1.2)

        # Puntos
        sub   = rng.choice(arr[arr <= y_max], min(1500, len(arr[arr <= y_max])), replace=False)
        x_jit = rng.uniform(-0.05, 0.05, len(sub))
        ax.scatter(x_ctr + x_jit, sub, s=4, alpha=0.30, color=color, linewidths=0, zorder=3)

        stats = f"med={p[2]:.2f}\np95={p[4]:.2f}\nN={len(arr)//1000}k"
        ax.text(0.04, 0.97, stats, transform=ax.transAxes, fontsize=7.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", alpha=0.9))

        ax.set_xlim(0.40, 1.72); ax.set_ylim(-0.2, y_max)
        ax.set_xticks([])
        ax.set_ylabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=9)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Anotaciones de columna
for col, label in enumerate(["Columna 1\n(entrada)", "Columna 2\n(centro)", "Columna 3\n(salida)"]):
    axes[0][col].set_title(f"{label}\n{axes[0][col].get_title()}", fontsize=8)

plt.tight_layout()
OUT = os.path.join(PIV_ANALISIS, "viga_shear_rate_2x3.png")
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Figura guardada: {OUT}")
plt.close()

# ── Figura resumen: medianas por celda ───────────────────────────────────────
medianas = np.zeros((n_filas_viga, n_columnas_viga))
p95s     = np.zeros_like(medianas)
for fila in range(n_filas_viga):
    for col in range(n_columnas_viga):
        arr = np.array(gd_celda[f"Vf{fila+1}c{col+1}"])
        if len(arr) > 0:
            medianas[fila, col] = np.median(arr)
            p95s[fila, col]     = np.percentile(arr, 95)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.5))
for ax2, data, title in zip(axes2,
                             [medianas, p95s],
                             ["Mediana gamma_dot (s-1)", "p95 gamma_dot (s-1)"]):
    im = ax2.imshow(data, aspect="auto", cmap="Blues",
                    vmin=0, vmax=data.max()*1.05)
    plt.colorbar(im, ax=ax2, label="s-1")
    ax2.set_title(title, fontsize=10)
    ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["c1\n(entrada)","c2","c3\n(salida)"])
    ax2.set_yticks([0,1]);   ax2.set_yticklabels(["f1\n(sup)","f2\n(inf)"])
    for fila in range(n_filas_viga):
        for col in range(n_columnas_viga):
            ax2.text(col, fila, f"{data[fila,col]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if data[fila,col] > data.max()*0.6 else "black")

fig2.suptitle("Distribucion espacial gamma_dot en la Viga\n"
              "Carbopol 0.2 % · t > 1.8 s", fontsize=11)
plt.tight_layout()
OUT2 = os.path.join(PIV_ANALISIS, "viga_mapa_gamma_dot.png")
plt.savefig(OUT2, dpi=180, bbox_inches="tight")
print(f"Mapa guardado: {OUT2}")
plt.close()
