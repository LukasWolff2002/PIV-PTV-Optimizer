"""
Boxplot de tasa de corte γ̇ en Zona Z2 (codo) — Carbopol 0.2 %
γ̇ calculado con KNN usando dirección local de velocidad en cada punto.
"""
import sys, os, glob, json, re
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

CONC_LABEL = "Carbopol 0,2 %"
RUNS_02    = ["m71-toma-1-n-0000-car-02-piv"]

K_VEC      = 6
DIST_MAX   = 5.0
MIN_VEC    = 4
COND_MAX   = 1e4

# ── γ̇ local por KNN con dirección de velocidad local ─────────────────────────
def calcular_gamma_dot_local(x_arr, y_arr, u_arr, v_arr,
                             k=K_VEC, dist_max=DIST_MAX,
                             min_vec=MIN_VEC, cond_max=COND_MAX):
    N   = len(x_arr)
    gd  = np.full(N, np.nan)
    pts = np.column_stack([x_arr, y_arr])
    tree = cKDTree(pts)

    for i in range(N):
        dists, idxs = tree.query(pts[i], k=k + 1,
                                  distance_upper_bound=dist_max + 1e-6,
                                  workers=1)
        mask = (dists > 0) & (dists <= dist_max)
        idxs = idxs[mask]
        if len(idxs) < min_vec:
            continue

        # Dirección local del flujo (media de vecinos + punto central)
        all_idx = np.append(idxs, i)
        u_loc = np.mean(u_arr[all_idx])
        v_loc = np.mean(v_arr[all_idx])
        spd   = np.hypot(u_loc, v_loc)
        if spd < 1e-6:
            continue
        T = np.array([u_loc, v_loc]) / spd        # streamwise
        N_hat = np.array([-T[1], T[0]])            # cross-stream (perpendicular)

        dx  = x_arr[idxs] - x_arr[i]
        dy  = y_arr[idxs] - y_arr[i]
        ds  = dx * T[0]     + dy * T[1]
        dn  = dx * N_hat[0] + dy * N_hat[1]
        u_s = u_arr[idxs] * T[0] + v_arr[idxs] * T[1]

        A = np.column_stack([np.ones(len(idxs)), ds, dn])
        try:
            if np.linalg.cond(A) > cond_max:
                continue
            coeffs, _, _, _ = np.linalg.lstsq(A, u_s, rcond=None)
            val = abs(coeffs[2])   # |du_s / dn|
            if np.isfinite(val) and val >= 0:
                gd[i] = val
        except Exception:
            continue
    return gd

# ── Cargar etapas desde Z1 key (mismo experimento) ────────────────────────────
with open(ETAPAS_JSON) as f:
    etapas_data = json.load(f)

# ── Loop principal ─────────────────────────────────────────────────────────────
gd_z2 = []

for run_name in RUNS_02:
    carpeta = PIV_DIR / run_name
    files   = sorted(glob.glob(str(carpeta / "frame_*.txt")),
                     key=lambda f: natural_sort_key(os.path.basename(f)))
    if not files:
        print(f"ERROR: no frames en {carpeta}"); continue

    key_z1 = f"{run_name}_Z1"
    if key_z1 not in etapas_data:
        print(f"AVISO: {key_z1} no en etapas_zonas.json"); continue

    e       = etapas_data[key_z1]
    t_peak  = e["t_peak"]
    t_quasi = e["t_quasi"]
    print(f"\nRun: {run_name}  t_peak={t_peak:.2f}s  t_quasi={t_quasi:.2f}s")
    print(f"Total frames: {len(files)}")

    for i, fpath in enumerate(files):
        t_val = extraer_timestamp(fpath)
        if t_val <= t_peak:
            continue

        df = cargar_frame(fpath)
        if df.empty:
            continue

        zonas    = asignar_zona(df["x"].values, df["y"].values)
        mask_z2  = zonas == "Z2"
        df_z2    = df[mask_z2].reset_index(drop=True)

        if len(df_z2) < MIN_VEC + 1:
            continue

        gd = calcular_gamma_dot_local(
            df_z2["x"].values, df_z2["y"].values,
            df_z2["u"].values, df_z2["v"].values)

        valid = np.isfinite(gd) & (gd >= 0)
        gd_z2.extend(gd[valid].tolist())

        if i % 100 == 0:
            print(f"  frame {i}/{len(files)}  t={t_val:.2f}s  "
                  f"Z2_pts={mask_z2.sum()}  valid={valid.sum()}", end="\r")

print(f"\n\nTotal pares válidos Z2: {len(gd_z2):,}")

arr = np.array(gd_z2)
if len(arr) == 0:
    print("Sin datos — saliendo."); sys.exit(1)

p = np.percentile(arr, [5, 25, 50, 75, 95, 99])
print(f"p5={p[0]:.3f}  Q1={p[1]:.3f}  median={p[2]:.3f}  "
      f"Q3={p[3]:.3f}  p95={p[4]:.3f}  p99={p[5]:.3f}  max={arr.max():.3f}  s⁻¹")

# ── Figura mitad violin / mitad box ──────────────────────────────────────────
COLOR     = "#e8a020"    # color Z2 (amarillo/naranja)
rng       = np.random.default_rng(42)
y_max_plot = p[5] * 1.08   # eje hasta p99

fig, ax = plt.subplots(figsize=(5.0, 6.0))
x_ctr   = 1.0

# Violin izquierdo
kde      = gaussian_kde(arr[arr <= y_max_plot * 2], bw_method=0.18)
y_kde    = np.linspace(0, y_max_plot, 400)
kde_vals = kde(y_kde)
kde_norm = kde_vals / kde_vals.max() * 0.38

ax.fill_betweenx(y_kde, x_ctr - kde_norm, x_ctr,
                 color=COLOR, alpha=0.55, linewidth=0)
ax.plot(x_ctr - kde_norm, y_kde, color=COLOR, linewidth=1.2, alpha=0.85)

# Box derecho
bw  = 0.18
p25, p50, p75 = p[1], p[2], p[3]
p05, p95_     = np.percentile(arr, 5), p[4]
ax.add_patch(mpatches.FancyBboxPatch(
    (x_ctr, p25), bw, p75 - p25,
    boxstyle="square,pad=0", linewidth=1.4,
    edgecolor=COLOR, facecolor=COLOR, alpha=0.55))
ax.plot([x_ctr, x_ctr + bw], [p50, p50],
        color="white", linewidth=2.5, zorder=5)
ax.plot([x_ctr + bw/2]*2, [p05, p25], color=COLOR, linewidth=1.4, zorder=4)
ax.plot([x_ctr + bw/2]*2, [p75, p95_], color=COLOR, linewidth=1.4, zorder=4)
for y_cap in [p05, p95_]:
    ax.plot([x_ctr + bw*0.15, x_ctr + bw*0.85], [y_cap]*2,
            color=COLOR, linewidth=1.4, zorder=4)

# Puntos
sub   = rng.choice(arr[arr <= y_max_plot], min(2500, len(arr[arr <= y_max_plot])),
                   replace=False)
x_jit = rng.uniform(-0.06, 0.06, len(sub))
ax.scatter(x_ctr + x_jit, sub, s=6, alpha=0.40,
           color=COLOR, linewidths=0, zorder=3)

# Anotación max
ax.annotate(f"máx = {arr.max():.1f} s⁻¹\n(fuera de escala)",
            xy=(x_ctr + bw/2, y_max_plot),
            xytext=(x_ctr + bw/2 + 0.28, y_max_plot * 0.90),
            fontsize=8.5, color="#333",
            arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.0,
                            mutation_scale=10),
            ha="left")

# Stats box
stats_txt = (f"mediana = {p50:.2f} s⁻¹\n"
             f"Q1 = {p25:.2f}  Q3 = {p75:.2f}\n"
             f"p95 = {p95_:.2f} s⁻¹\n"
             f"N = {len(arr):,}")
ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

ax.set_xlim(0.40, 1.70)
ax.set_ylim(-0.3, y_max_plot)
ax.set_xticks([x_ctr])
ax.set_xticklabels([CONC_LABEL], fontsize=11)
ax.set_ylabel(r"$\dot{\gamma}$ (s$^{-1}$)", fontsize=12)
ax.set_title("Distribución de tasa de corte\nZona Z2 — codo",
             fontsize=11, pad=8)
ax.axhline(0, color="#bbb", linewidth=0.6, linestyle=":")
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)

legend_items = [
    mpatches.Patch(color=COLOR, alpha=0.55, label="Violin (densidad)"),
    mpatches.Patch(color=COLOR, alpha=0.55, label="Box (IQR · p5–p95)"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.85)

plt.tight_layout()
OUT = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\boxplot_shear_rate_z2.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"\nFigura guardada: {OUT}")
plt.close()
