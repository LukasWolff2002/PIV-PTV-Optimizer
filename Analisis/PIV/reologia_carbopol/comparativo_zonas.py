"""
Una sola pasada sobre los frames: calcula γ̇ en Z1, Z2 y Z3 simultáneamente.
Luego genera figura comparativa de las tres distribuciones.
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

from definir_zonas import asignar_zona
from extraer_reologia_piv import (cargar_frame, extraer_timestamp,
                                   natural_sort_key, ETAPAS_JSON,
                                   calcular_gamma_dot_arm, a_coordenadas_brazo)

RUNS_02  = ["m71-toma-1-n-0000-car-02-piv"]
K_VEC, DIST_MAX, MIN_VEC, COND_MAX = 6, 5.0, 4, 1e4

# Z3 coordenadas fijas
T_Z3 = np.array([ 0.866, -0.500])
N_Z3 = np.array([ 0.500,  0.866])

def gd_fixed_coords(x, y, u, v, T, N):
    """KNN γ̇ con ejes fijos (s,n)."""
    n_pts = len(x)
    gd    = np.full(n_pts, np.nan)
    s     = x*T[0] + y*T[1]
    n     = x*N[0] + y*N[1]
    us    = u*T[0] + v*T[1]
    pts   = np.column_stack([x, y])
    tree  = cKDTree(pts)
    for i in range(n_pts):
        dists, idx = tree.query(pts[i], k=K_VEC+1,
                                 distance_upper_bound=DIST_MAX+1e-6, workers=1)
        mask = (dists > 0) & (dists <= DIST_MAX)
        idx  = idx[mask]
        if len(idx) < MIN_VEC:
            continue
        A = np.column_stack([np.ones(len(idx)), s[idx]-s[i], n[idx]-n[i]])
        try:
            if np.linalg.cond(A) > COND_MAX:
                continue
            c, _, _, _ = np.linalg.lstsq(A, us[idx], rcond=None)
            val = abs(c[2])
            if np.isfinite(val) and val >= 0:
                gd[i] = val
        except Exception:
            continue
    return gd

def gd_local_vel(x, y, u, v):
    """KNN γ̇ con dirección local de velocidad (para Z2 codo)."""
    n_pts = len(x)
    gd    = np.full(n_pts, np.nan)
    pts   = np.column_stack([x, y])
    tree  = cKDTree(pts)
    for i in range(n_pts):
        dists, idx = tree.query(pts[i], k=K_VEC+1,
                                 distance_upper_bound=DIST_MAX+1e-6, workers=1)
        mask = (dists > 0) & (dists <= DIST_MAX)
        idx  = idx[mask]
        if len(idx) < MIN_VEC:
            continue
        all_i  = np.append(idx, i)
        u_loc  = np.mean(u[all_i]);  v_loc = np.mean(v[all_i])
        spd    = np.hypot(u_loc, v_loc)
        if spd < 1e-6:
            continue
        T = np.array([u_loc, v_loc]) / spd
        N = np.array([-T[1], T[0]])
        dx = x[idx]-x[i];  dy = y[idx]-y[i]
        ds = dx*T[0]+dy*T[1];  dn = dx*N[0]+dy*N[1]
        us = u[idx]*T[0]+v[idx]*T[1]
        A  = np.column_stack([np.ones(len(idx)), ds, dn])
        try:
            if np.linalg.cond(A) > COND_MAX:
                continue
            c, _, _, _ = np.linalg.lstsq(A, us, rcond=None)
            val = abs(c[2])
            if np.isfinite(val) and val >= 0:
                gd[i] = val
        except Exception:
            continue
    return gd

# ── Pasada única sobre frames ─────────────────────────────────────────────────
with open(ETAPAS_JSON) as f:
    etapas_data = json.load(f)

gd = {"Z1": [], "Z2": [], "Z3": []}

for run_name in RUNS_02:
    carpeta = PIV_DIR / run_name
    files   = sorted(glob.glob(str(carpeta / "frame_*.txt")),
                     key=lambda f: natural_sort_key(os.path.basename(f)))
    key_z1  = f"{run_name}_Z1"
    t_peak  = etapas_data[key_z1]["t_peak"]
    print(f"Run: {run_name}  frames={len(files)}  t_peak={t_peak:.2f}s")

    for i, fpath in enumerate(files):
        t_val = extraer_timestamp(fpath)
        if t_val <= t_peak:
            continue
        df = cargar_frame(fpath)
        if df.empty:
            continue

        x, y = df["x"].values, df["y"].values
        u, v = df["u"].values, df["v"].values
        zonas = asignar_zona(x, y)

        # Z1
        mz1 = zonas == "Z1"
        if mz1.sum() > MIN_VEC:
            s_, n_, us_, _ = a_coordenadas_brazo(df[mz1].reset_index(drop=True))
            g = calcular_gamma_dot_arm(s_, n_, us_)
            gd["Z1"].extend(g[np.isfinite(g) & (g >= 0)].tolist())

        # Z2
        mz2 = zonas == "Z2"
        if mz2.sum() > MIN_VEC:
            g = gd_local_vel(x[mz2], y[mz2], u[mz2], v[mz2])
            gd["Z2"].extend(g[np.isfinite(g) & (g >= 0)].tolist())

        # Z3
        mz3 = zonas == "Z3"
        if mz3.sum() > MIN_VEC:
            g = gd_fixed_coords(x[mz3], y[mz3], u[mz3], v[mz3], T_Z3, N_Z3)
            gd["Z3"].extend(g[np.isfinite(g) & (g >= 0)].tolist())

        if i % 100 == 0:
            print(f"  frame {i}/{len(files)}  t={t_val:.2f}s  "
                  f"Z1={mz1.sum()} Z2={mz2.sum()} Z3={mz3.sum()}", end="\r")

print()
for z, lst in gd.items():
    arr = np.array(lst)
    np.save(f"gd_{z.lower()}_combined.npy", arr)
    p = np.percentile(arr, [5,25,50,75,95,99]) if len(arr) > 0 else [0]*6
    print(f"{z}: N={len(arr):,}  median={p[2]:.3f}  p95={p[4]:.3f}  max={arr.max():.3f} s⁻¹")

# ── Figura comparativa ────────────────────────────────────────────────────────
ZONES  = ["Z1", "Z2", "Z3"]
COLORS = {"Z1": "#4a7ab5", "Z2": "#e8a020", "Z3": "#c45022"}
LABELS = {"Z1": "Z1\nBrazo\ninclinado",
          "Z2": "Z2\nCodo",
          "Z3": "Z3\nSalida"}

fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=False)
fig.suptitle("Distribución de tasa de corte γ̇ — Carbopol 0,2 %\n"
             "PIV · run m71-toma-1 · un frame de referencia por zona",
             fontsize=12, y=1.01)

rng = np.random.default_rng(42)

for ax, zone in zip(axes, ZONES):
    arr   = np.array(gd[zone])
    color = COLORS[zone]
    if len(arr) == 0:
        ax.set_title(f"{zone}\nSin datos"); continue

    p       = np.percentile(arr, [5, 25, 50, 75, 95, 99])
    y_ceil  = p[5] * 1.08
    x_ctr   = 1.0

    # Violin izquierdo
    kde      = gaussian_kde(arr[arr <= y_ceil * 2], bw_method=0.18)
    y_kde    = np.linspace(0, y_ceil, 400)
    kde_vals = kde(y_kde)
    kde_norm = kde_vals / kde_vals.max() * 0.38
    ax.fill_betweenx(y_kde, x_ctr - kde_norm, x_ctr,
                     color=color, alpha=0.50, linewidth=0)
    ax.plot(x_ctr - kde_norm, y_kde, color=color, linewidth=1.2, alpha=0.85)

    # Box derecho
    bw = 0.18
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_ctr, p[1]), bw, p[3]-p[1],
        boxstyle="square,pad=0", lw=1.4,
        edgecolor=color, facecolor=color, alpha=0.55))
    ax.plot([x_ctr, x_ctr+bw], [p[2]]*2, color="white", lw=2.5, zorder=5)
    ax.plot([x_ctr+bw/2]*2, [p[0], p[1]], color=color, lw=1.4, zorder=4)
    ax.plot([x_ctr+bw/2]*2, [p[3], p[4]], color=color, lw=1.4, zorder=4)
    for yc in [p[0], p[4]]:
        ax.plot([x_ctr+bw*0.15, x_ctr+bw*0.85], [yc]*2, color=color, lw=1.4, zorder=4)

    # Puntos
    sub   = rng.choice(arr[arr <= y_ceil], min(2000, len(arr[arr <= y_ceil])), replace=False)
    x_jit = rng.uniform(-0.06, 0.06, len(sub))
    ax.scatter(x_ctr + x_jit, sub, s=5, alpha=0.35, color=color, linewidths=0, zorder=3)

    # Anotación max
    ax.annotate(f"máx={arr.max():.1f}", xy=(x_ctr+bw/2, y_ceil),
                xytext=(x_ctr+bw/2+0.25, y_ceil*0.88),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.9, mutation_scale=9),
                ha="left")

    # Stats
    stats = (f"med={p[2]:.2f}\nQ3={p[3]:.2f}\np95={p[4]:.2f}\nN={len(arr)//1000}k")
    ax.text(0.04, 0.97, stats, transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    ax.set_xlim(0.40, 1.72); ax.set_ylim(-0.5, y_ceil)
    ax.set_xticks([x_ctr])
    ax.set_xticklabels([LABELS[zone]], fontsize=10)
    ax.set_ylabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=11)
    ax.axhline(0, color="#bbb", lw=0.6, ls=":")
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\comparativo_z1_z2_z3.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"\nFigura guardada: {OUT}")
plt.close()
