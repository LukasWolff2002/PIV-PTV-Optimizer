"""
Boxplot γ̇ Zona Z3 (salida horizontal) — Carbopol 0.2 %
Coordenadas fijas Z3: flow_hat=(0.866,-0.500), n_hat=(0.500,0.866)
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
from extraer_reologia_piv import cargar_frame, extraer_timestamp, natural_sort_key, ETAPAS_JSON

RUNS_02  = ["m71-toma-1-n-0000-car-02-piv"]
K_VEC, DIST_MAX, MIN_VEC, COND_MAX = 6, 5.0, 4, 1e4

# Z3: flujo en dirección s4→s1, ángulo -30° desde +x
T_Z3 = np.array([ 0.866, -0.500])   # streamwise
N_Z3 = np.array([ 0.500,  0.866])   # cross-stream (sustrato→superficie)

def calcular_gamma_dot_z3(x_arr, y_arr, u_arr, v_arr):
    N    = len(x_arr)
    gd   = np.full(N, np.nan)
    pts  = np.column_stack([x_arr, y_arr])
    tree = cKDTree(pts)
    s_arr = x_arr * T_Z3[0] + y_arr * T_Z3[1]
    n_arr = x_arr * N_Z3[0] + y_arr * N_Z3[1]
    u_s   = u_arr * T_Z3[0] + v_arr * T_Z3[1]

    for i in range(N):
        dists, idxs = tree.query(pts[i], k=K_VEC+1,
                                  distance_upper_bound=DIST_MAX+1e-6, workers=1)
        mask = (dists > 0) & (dists <= DIST_MAX)
        idxs = idxs[mask]
        if len(idxs) < MIN_VEC:
            continue
        ds = s_arr[idxs] - s_arr[i]
        dn = n_arr[idxs] - n_arr[i]
        du = u_s[idxs]
        A  = np.column_stack([np.ones(len(idxs)), ds, dn])
        try:
            if np.linalg.cond(A) > COND_MAX:
                continue
            c, _, _, _ = np.linalg.lstsq(A, du, rcond=None)
            val = abs(c[2])
            if np.isfinite(val) and val >= 0:
                gd[i] = val
        except Exception:
            continue
    return gd

with open(ETAPAS_JSON) as f:
    etapas_data = json.load(f)

gd_z3 = []
for run_name in RUNS_02:
    carpeta = PIV_DIR / run_name
    files   = sorted(glob.glob(str(carpeta / "frame_*.txt")),
                     key=lambda f: natural_sort_key(os.path.basename(f)))
    key_z1  = f"{run_name}_Z1"
    if key_z1 not in etapas_data:
        print(f"AVISO: {key_z1} no en etapas"); continue
    t_peak = etapas_data[key_z1]["t_peak"]
    print(f"\nRun: {run_name}  t_peak={t_peak:.2f}s  frames={len(files)}")

    for i, fpath in enumerate(files):
        t_val = extraer_timestamp(fpath)
        if t_val <= t_peak:
            continue
        df = cargar_frame(fpath)
        if df.empty:
            continue
        zonas   = asignar_zona(df["x"].values, df["y"].values)
        mask_z3 = zonas == "Z3"
        df_z3   = df[mask_z3].reset_index(drop=True)
        if len(df_z3) < MIN_VEC + 1:
            continue
        gd = calcular_gamma_dot_z3(df_z3["x"].values, df_z3["y"].values,
                                    df_z3["u"].values, df_z3["v"].values)
        valid = np.isfinite(gd) & (gd >= 0)
        gd_z3.extend(gd[valid].tolist())
        if i % 100 == 0:
            print(f"  frame {i}/{len(files)}  t={t_val:.2f}s  "
                  f"Z3_pts={mask_z3.sum()}  valid={valid.sum()}", end="\r")

print(f"\n\nTotal pares válidos Z3: {len(gd_z3):,}")
arr = np.array(gd_z3)
if len(arr) == 0:
    print("Sin datos."); sys.exit(1)

p = np.percentile(arr, [5, 25, 50, 75, 95, 99])
print(f"p5={p[0]:.3f}  Q1={p[1]:.3f}  median={p[2]:.3f}  "
      f"Q3={p[3]:.3f}  p95={p[4]:.3f}  p99={p[5]:.3f}  max={arr.max():.3f}  s⁻¹")

# Guardar array para figura comparativa
np.save(r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\gd_z3.npy", arr)
print("Array guardado: gd_z3.npy")

# Figura
COLOR      = "#c45022"
rng        = np.random.default_rng(42)
y_max_plot = p[5] * 1.08

fig, ax = plt.subplots(figsize=(5.0, 6.0))
x_ctr = 1.0
kde      = gaussian_kde(arr[arr <= y_max_plot * 2], bw_method=0.18)
y_kde    = np.linspace(0, y_max_plot, 400)
kde_vals = kde(y_kde)
kde_norm = kde_vals / kde_vals.max() * 0.38
ax.fill_betweenx(y_kde, x_ctr - kde_norm, x_ctr,
                 color=COLOR, alpha=0.55, linewidth=0)
ax.plot(x_ctr - kde_norm, y_kde, color=COLOR, linewidth=1.2, alpha=0.85)

bw   = 0.18
p25, p50, p75 = p[1], p[2], p[3]
p05, p95_     = np.percentile(arr, 5), p[4]
ax.add_patch(mpatches.FancyBboxPatch((x_ctr, p25), bw, p75-p25,
    boxstyle="square,pad=0", linewidth=1.4,
    edgecolor=COLOR, facecolor=COLOR, alpha=0.55))
ax.plot([x_ctr, x_ctr+bw], [p50]*2, color="white", linewidth=2.5, zorder=5)
ax.plot([x_ctr+bw/2]*2, [p05, p25], color=COLOR, linewidth=1.4, zorder=4)
ax.plot([x_ctr+bw/2]*2, [p75, p95_], color=COLOR, linewidth=1.4, zorder=4)
for y_cap in [p05, p95_]:
    ax.plot([x_ctr+bw*0.15, x_ctr+bw*0.85], [y_cap]*2,
            color=COLOR, linewidth=1.4, zorder=4)

sub   = rng.choice(arr[arr <= y_max_plot], min(2500, len(arr[arr <= y_max_plot])), replace=False)
x_jit = rng.uniform(-0.06, 0.06, len(sub))
ax.scatter(x_ctr + x_jit, sub, s=6, alpha=0.40, color=COLOR, linewidths=0, zorder=3)

ax.annotate(f"máx = {arr.max():.1f} s⁻¹\n(fuera de escala)",
            xy=(x_ctr+bw/2, y_max_plot),
            xytext=(x_ctr+bw/2+0.28, y_max_plot*0.90),
            fontsize=8.5, color="#333",
            arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.0, mutation_scale=10),
            ha="left")

stats_txt = (f"mediana = {p50:.2f} s⁻¹\nQ1={p25:.2f}  Q3={p75:.2f}\n"
             f"p95 = {p95_:.2f} s⁻¹\nN = {len(arr):,}")
ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))
ax.set_xlim(0.40, 1.70); ax.set_ylim(-0.3, y_max_plot)
ax.set_xticks([x_ctr]); ax.set_xticklabels(["Carbopol 0,2 %"], fontsize=11)
ax.set_ylabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=12)
ax.set_title("Distribución de tasa de corte\nZona Z3 — salida (30° horizontal)", fontsize=11, pad=8)
ax.axhline(0, color="#bbb", linewidth=0.6, linestyle=":")
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
plt.tight_layout()
OUT = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\boxplot_shear_rate_z3.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Figura guardada: {OUT}")
plt.close()
