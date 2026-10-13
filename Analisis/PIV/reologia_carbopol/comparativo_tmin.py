"""
Compara distribuciones γ̇ con t > 0.5s (actual) vs t > 1.8s (sin transitorio inicial).
Una sola pasada sobre los frames.
"""
import sys, os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

T_MIN_OLD = 0.5
T_MIN_NEW = 1.8
RUNS_02   = ["m71-toma-1-n-0000-car-02-piv"]
K_VEC, DIST_MAX, MIN_VEC, COND_MAX = 6, 5.0, 4, 1e4
T_Z3 = np.array([ 0.866, -0.500])
N_Z3 = np.array([ 0.500,  0.866])

def gd_fixed(x, y, u, v, T, N):
    n_pts = len(x); gd = np.full(n_pts, np.nan)
    s = x*T[0]+y*T[1]; n = x*N[0]+y*N[1]; us = u*T[0]+v*T[1]
    pts = np.column_stack([x, y]); tree = cKDTree(pts)
    for i in range(n_pts):
        dists, idx = tree.query(pts[i], k=K_VEC+1, distance_upper_bound=DIST_MAX+1e-6, workers=1)
        mask = (dists > 0) & (dists <= DIST_MAX); idx = idx[mask]
        if len(idx) < MIN_VEC: continue
        A = np.column_stack([np.ones(len(idx)), s[idx]-s[i], n[idx]-n[i]])
        try:
            if np.linalg.cond(A) > COND_MAX: continue
            c, _, _, _ = np.linalg.lstsq(A, us[idx], rcond=None)
            val = abs(c[2])
            if np.isfinite(val) and val >= 0: gd[i] = val
        except: pass
    return gd

def gd_local(x, y, u, v):
    n_pts = len(x); gd = np.full(n_pts, np.nan)
    pts = np.column_stack([x, y]); tree = cKDTree(pts)
    for i in range(n_pts):
        dists, idx = tree.query(pts[i], k=K_VEC+1, distance_upper_bound=DIST_MAX+1e-6, workers=1)
        mask = (dists > 0) & (dists <= DIST_MAX); idx = idx[mask]
        if len(idx) < MIN_VEC: continue
        all_i = np.append(idx, i)
        ul = np.mean(u[all_i]); vl = np.mean(v[all_i])
        spd = np.hypot(ul, vl)
        if spd < 1e-6: continue
        Tv = np.array([ul, vl])/spd; Nv = np.array([-Tv[1], Tv[0]])
        dx = x[idx]-x[i]; dy = y[idx]-y[i]
        ds = dx*Tv[0]+dy*Tv[1]; dn = dx*Nv[0]+dy*Nv[1]
        us = u[idx]*Tv[0]+v[idx]*Tv[1]
        A = np.column_stack([np.ones(len(idx)), ds, dn])
        try:
            if np.linalg.cond(A) > COND_MAX: continue
            c, _, _, _ = np.linalg.lstsq(A, us, rcond=None)
            val = abs(c[2])
            if np.isfinite(val) and val >= 0: gd[i] = val
        except: pass
    return gd

with open(ETAPAS_JSON) as f:
    etapas_data = json.load(f)

# solo frames en (T_MIN_OLD, T_MIN_NEW] para cuantificar la diferencia
# + frames > T_MIN_NEW para el nuevo set
gd_old_only = {"Z1": [], "Z2": [], "Z3": []}   # t in (0.5, 1.8]
gd_new      = {"Z1": [], "Z2": [], "Z3": []}   # t > 1.8

for run_name in RUNS_02:
    carpeta = PIV_DIR / run_name
    files   = sorted(glob.glob(str(carpeta / "frame_*.txt")),
                     key=lambda f: natural_sort_key(os.path.basename(f)))
    t_peak = etapas_data[f"{run_name}_Z1"]["t_peak"]
    print(f"Run: {run_name}  frames={len(files)}")

    for i, fpath in enumerate(files):
        t_val = extraer_timestamp(fpath)
        if t_val <= t_peak:
            continue
        in_transient = (t_val <= T_MIN_NEW)

        df = cargar_frame(fpath)
        if df.empty: continue
        x, y, u, v = df["x"].values, df["y"].values, df["u"].values, df["v"].values
        zonas = asignar_zona(x, y)

        # Z1
        mz1 = zonas == "Z1"
        if mz1.sum() > MIN_VEC:
            s_, n_, us_, _ = a_coordenadas_brazo(df[mz1].reset_index(drop=True))
            g = calcular_gamma_dot_arm(s_, n_, us_)
            vals = g[np.isfinite(g) & (g >= 0)].tolist()
            if in_transient:
                gd_old_only["Z1"].extend(vals)
            else:
                gd_new["Z1"].extend(vals)

        # Z2
        mz2 = zonas == "Z2"
        if mz2.sum() > MIN_VEC:
            g = gd_local(x[mz2], y[mz2], u[mz2], v[mz2])
            vals = g[np.isfinite(g) & (g >= 0)].tolist()
            if in_transient: gd_old_only["Z2"].extend(vals)
            else:             gd_new["Z2"].extend(vals)

        # Z3
        mz3 = zonas == "Z3"
        if mz3.sum() > MIN_VEC:
            g = gd_fixed(x[mz3], y[mz3], u[mz3], v[mz3], T_Z3, N_Z3)
            vals = g[np.isfinite(g) & (g >= 0)].tolist()
            if in_transient: gd_old_only["Z3"].extend(vals)
            else:             gd_new["Z3"].extend(vals)

        if i % 100 == 0:
            print(f"  frame {i}/{len(files)}  t={t_val:.2f}s", end="\r")

print()

# Reconstruir arrays completos
gd_full = {z: np.array(gd_old_only[z] + gd_new[z]) for z in ["Z1","Z2","Z3"]}
gd_filt = {z: np.array(gd_new[z]) for z in ["Z1","Z2","Z3"]}

for z in ["Z1","Z2","Z3"]:
    af = gd_full[z]; an = gd_filt[z]
    pf = np.percentile(af, [50,95]) if len(af) else [0,0]
    pn = np.percentile(an, [50,95]) if len(an) else [0,0]
    print(f"{z}: t>0.5s  N={len(af):,} med={pf[0]:.3f} p95={pf[1]:.3f} | "
          f"t>1.8s  N={len(an):,} med={pn[0]:.3f} p95={pn[1]:.3f}  "
          f"dmed={pn[0]-pf[0]:+.3f}")

# ── Figura: KDE overlay para cada zona ───────────────────────────────────────
ZONES  = ["Z1","Z2","Z3"]
COLORS = {"Z1":"#4a7ab5","Z2":"#e8a020","Z3":"#c45022"}
LABELS = {"Z1":"Z1 — Brazo inclinado","Z2":"Z2 — Codo","Z3":"Z3 — Salida"}

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Efecto del corte temporal: t > 0.5s vs t > 1.8s\n"
             "Carbopol 0,2 % · run m71-toma-1", fontsize=12)

for ax, z in zip(axes, ZONES):
    af = gd_full[z]; an = gd_filt[z]
    col = COLORS[z]
    # límite del eje: p99 del set completo
    p99 = np.percentile(af, 99) if len(af) else 10
    x_grid = np.linspace(0, p99, 400)

    if len(af) > 10:
        kde_f = gaussian_kde(af[af <= p99*2], bw_method=0.18)
        y_f   = kde_f(x_grid); y_f /= y_f.max()
        ax.fill_between(x_grid, y_f, alpha=0.30, color=col, label="t > 0.5s (actual)")
        ax.plot(x_grid, y_f, color=col, lw=1.5)
    if len(an) > 10:
        kde_n = gaussian_kde(an[an <= p99*2], bw_method=0.18)
        y_n   = kde_n(x_grid); y_n /= y_n.max()
        ax.fill_between(x_grid, y_n, alpha=0.45, color="black",
                        hatch="////", label="t > 1.8s (sin transitorio)")
        ax.plot(x_grid, y_n, color="black", lw=1.5, ls="--")

    # líneas de mediana
    for arr, ls, lbl in [(af,"--",f"med={np.median(af):.2f}"),
                         (an,"-", f"med={np.median(an):.2f}")]:
        if len(arr) > 0:
            ax.axvline(np.median(arr), color=col if ls=="--" else "black",
                       lw=1.2, ls=ls, alpha=0.8)

    ax.set_xlabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=11)
    ax.set_ylabel("Densidad (normalizada)", fontsize=10)
    ax.set_title(LABELS[z], fontsize=10)
    ax.set_xlim(0, p99); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(ls=":", lw=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\comparativo_tmin_z1z2z3.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Figura guardada: {OUT}")
plt.close()
