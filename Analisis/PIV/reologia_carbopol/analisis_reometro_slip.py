"""
Análisis reometro HR30 — Carbopol 0.2 % — 3 gaps (0.5 / 1.0 / 2.0 mm)
Geometría M105 = placa paralela 40 mm

1. Curva up (Sweeps 1+2+3) para cada gap
2. Corrección de slip Mooney: γ̇_app(h) = γ̇_true + 2·v_slip(τ)/h
3. Ajuste Herschel-Bulkley a datos corregidos
4. Power-law para v_slip(τ)
5. Comparación con rango γ̇ del PIV
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

PIV_ANALISIS = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers"

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
with open(fr"{PIV_ANALISIS}\datos_reometro\data_parsed.json") as f:
    raw = json.load(f)

GAPS_MM = [0.5, 1.0, 2.0]  # mm
GAPS    = {str(g): g for g in GAPS_MM}

# Up-curve: sweeps 1, 2, 3
def get_upcurve(gap_str):
    parts = (raw[gap_str]["Flow sweep - 1"] +
             raw[gap_str]["Flow sweep - 2"] +
             raw[gap_str]["Flow sweep - 3"])
    arr = np.array(parts)          # cols: tau, gdot, visc, t, T, Nsigma
    # sort by gdot
    idx = np.argsort(arr[:, 1])
    return arr[idx, 0], arr[idx, 1]   # tau, gdot_app

tau_up  = {}
gdot_up = {}
for gs in ["0.5", "1.0", "2.0"]:
    tau_up[gs], gdot_up[gs] = get_upcurve(gs)

# ── 2. Mooney slip correction ─────────────────────────────────────────────────
# At each stress τ_i, interpolate γ̇_app for each gap, then fit linear
# γ̇_app vs 2/h → slope = v_slip(τ_i), intercept = γ̇_true(τ_i)

# Common τ range in the FLOWING regime (above yield): τ ∈ [18, 80] Pa
tau_grid = np.logspace(np.log10(18), np.log10(80), 40)

# Monotone interpolators (PCHIP) in log-log for each gap
# x = log10(tau), y = log10(gdot_app)  — sort by tau, remove non-monotone kinks
interp = {}
for gs in ["0.5", "1.0", "2.0"]:
    t = tau_up[gs]; g = gdot_up[gs]
    mask = (g > 0) & (t > 0)
    t_m, g_m = t[mask], g[mask]
    # sort by tau
    idx = np.argsort(t_m)
    t_s, g_s = t_m[idx], g_m[idx]
    # enforce strictly increasing tau
    t_out, g_out = [t_s[0]], [g_s[0]]
    for i in range(1, len(t_s)):
        if t_s[i] > t_out[-1]:
            t_out.append(t_s[i]); g_out.append(g_s[i])
    t_s, g_s = np.array(t_out), np.array(g_out)
    interp[gs] = PchipInterpolator(np.log10(t_s), np.log10(g_s), extrapolate=False)

inv_h = np.array([2/0.5e-3, 2/1.0e-3, 2/2.0e-3])  # 2/h in m⁻¹ (h in meters)

gdot_true_list = []
tau_common     = []
vslip_list     = []

for tau_i in tau_grid:
    logt = np.log10(tau_i)
    gdot_i = []
    ok = True
    for gs in ["0.5", "1.0", "2.0"]:
        v = interp[gs](logt)
        if np.isnan(v):
            ok = False; break
        gdot_i.append(10**v)
    if not ok: continue
    gdot_i = np.array(gdot_i)

    # linear fit: gdot_app = gdot_true + v_slip * (2/h)
    try:
        c = np.polyfit(inv_h, gdot_i, 1)   # c[0]=v_slip, c[1]=gdot_true
        if c[0] < 0 or c[1] < 0: continue  # unphysical
        vslip_list.append(c[0])
        gdot_true_list.append(c[1])
        tau_common.append(tau_i)
    except:
        continue

tau_corr  = np.array(tau_common)
gdot_corr = np.array(gdot_true_list)
vslip_arr = np.array(vslip_list)

print(f"\n=== Mooney correction: {len(tau_corr)} puntos en tau=[{tau_corr[0]:.1f}..{tau_corr[-1]:.1f}] Pa ===")
print(f"gdot_true: [{gdot_corr.min():.4f}..{gdot_corr.max():.3f}] s-1")
print(f"v_slip:    [{vslip_arr.min()*1e3:.4f}..{vslip_arr.max()*1e3:.3f}] mm/s")

# ── 3. Ajuste Herschel-Bulkley (curva corregida) ──────────────────────────────
def hb(gdot, tau0, K, n):
    return tau0 + K * gdot**n

# Ajuste uncorrected (gap 1mm up-curve)
tau1 = tau_up["1.0"]; g1 = gdot_up["1.0"]
mask1 = (g1 > 0.05) & (tau1 > 15)  # above yield region
p0 = [30, 20, 0.5]
try:
    popt_raw, _ = curve_fit(hb, g1[mask1], tau1[mask1], p0=p0,
                             bounds=([0,0,0.1],[150,500,1.0]), maxfev=5000)
    print(f"\nHB raw (gap 1mm, no slip):  tau0={popt_raw[0]:.2f} K={popt_raw[1]:.3f} n={popt_raw[2]:.3f}")
except Exception as e:
    popt_raw = None; print("HB raw fit failed:", e)

# Ajuste corregido
mask_c = gdot_corr > 0.02
try:
    popt_cor, _ = curve_fit(hb, gdot_corr[mask_c], tau_corr[mask_c], p0=p0,
                             bounds=([0,0,0.1],[150,500,1.0]), maxfev=5000)
    print(f"HB corregido (Mooney):      tau0={popt_cor[0]:.2f} K={popt_cor[1]:.3f} n={popt_cor[2]:.3f}")
except Exception as e:
    popt_cor = None; print("HB corrected fit failed:", e)

# PIV yield stress reference
tau0_piv = 40.6

# ── 4. Power-law v_slip(τ) ────────────────────────────────────────────────────
def vslip_pl(tau, alpha, m):
    return alpha * tau**m

mask_vs = vslip_arr > 0
try:
    popt_vs, _ = curve_fit(vslip_pl, tau_corr[mask_vs], vslip_arr[mask_vs],
                            p0=[1e-6, 1.0], bounds=([0,0],[1,5]), maxfev=5000)
    print(f"v_slip power-law:           alpha={popt_vs[0]:.3e} m={popt_vs[1]:.3f}")
    print(f"  (alpha en m/s, tau en Pa)")
except Exception as e:
    popt_vs = None; print("v_slip fit failed:", e)

# ── 5. Figuras ────────────────────────────────────────────────────────────────
COLORS_GAP = {"0.5": "#c45022", "1.0": "#e8a020", "2.0": "#4a7ab5"}
LABELS_GAP = {"0.5": "h = 0.5 mm", "1.0": "h = 1.0 mm", "2.0": "h = 2.0 mm"}

fig = plt.figure(figsize=(15, 10))
gs_fig = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.38)
ax1 = fig.add_subplot(gs_fig[0, 0])   # tau vs gdot — 3 gaps, up-curve raw
ax2 = fig.add_subplot(gs_fig[0, 1])   # gdot_app vs 2/h @ varios tau
ax3 = fig.add_subplot(gs_fig[0, 2])   # v_slip vs tau
ax4 = fig.add_subplot(gs_fig[1, 0])   # HB corrected + raw + PIV range
ax5 = fig.add_subplot(gs_fig[1, 1])   # viscosidad vs gdot
ax6 = fig.add_subplot(gs_fig[1, 2])   # tau vs gdot — down-curve

fig.suptitle("Carbopol 0.2 % — HR30 — Corrección de slip (Mooney) — 3 gaps",
             fontsize=12)

# --- ax1: up-curve raw ---
for gs in ["0.5", "1.0", "2.0"]:
    ax1.scatter(gdot_up[gs], tau_up[gs], s=20, color=COLORS_GAP[gs],
                label=LABELS_GAP[gs], zorder=5)
    # spline
    g_plot = np.logspace(np.log10(gdot_up[gs].min()*0.9),
                          np.log10(gdot_up[gs].max()*1.1), 200)
    logt_i = interp[gs](np.log10(g_plot))
    mask_p = ~np.isnan(logt_i)
    ax1.plot(g_plot[mask_p], 10**logt_i[mask_p],
             color=COLORS_GAP[gs], lw=1.0, alpha=0.6)
ax1.axhline(tau0_piv, color="gray", ls="--", lw=1.0, label=f"tau0 PIV={tau0_piv} Pa")
ax1.set_xlabel(r"$\dot{\gamma}_{app}$ (s$^{-1}$)")
ax1.set_ylabel(r"$\tau$ (Pa)")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_title("Up-curve — 3 gaps (sin corregir)")
ax1.legend(fontsize=7); ax1.grid(ls=":", lw=0.5, alpha=0.5)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# --- ax2: Mooney verification @ 3 tau levels ---
tau_check = [20, 40, 65]
colors_check = ["#7b2d8b", "#2b6ba8", "#c45022"]
inv_h_plot = np.array([2/0.5e-3, 2/1.0e-3, 2/2.0e-3])
for tc, cc in zip(tau_check, colors_check):
    logt = np.log10(tc)
    gi = []
    for gs in ["0.5", "1.0", "2.0"]:
        v = interp[gs](logt)
        gi.append(10**v if not np.isnan(v) else np.nan)
    gi = np.array(gi)
    ok = ~np.isnan(gi)
    if ok.sum() >= 2:
        ax2.scatter(inv_h_plot[ok], gi[ok], s=40, color=cc, zorder=5)
        c_fit = np.polyfit(inv_h_plot[ok], gi[ok], 1)
        x_p = np.array([0, inv_h_plot.max()*1.1])
        ax2.plot(x_p, np.polyval(c_fit, x_p), color=cc, lw=1.5,
                 label=f"tau={tc:.0f} Pa  vs={c_fit[0]*1e3:.3f} mm/s")
ax2.set_xlabel("2/h (m$^{-1}$)")
ax2.set_ylabel(r"$\dot{\gamma}_{app}$ (s$^{-1}$)")
ax2.set_title("Verificación Mooney\n(linea = γ̇_true en y-intercept)")
ax2.legend(fontsize=7); ax2.grid(ls=":", lw=0.5, alpha=0.5)
ax2.set_xlim(-100, inv_h_plot.max()*1.15)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# --- ax3: v_slip vs tau ---
ax3.scatter(tau_corr, vslip_arr*1e3, s=20, color="#2b6ba8", zorder=5, label="datos Mooney")
if popt_vs is not None:
    tau_p = np.logspace(np.log10(tau_corr.min()*0.9), np.log10(tau_corr.max()*1.1), 200)
    ax3.plot(tau_p, vslip_pl(tau_p, *popt_vs)*1e3, color="#c45022", lw=2,
             label=f"v_s=alpha*tau^m\nalpha={popt_vs[0]:.2e} m/s\nm={popt_vs[1]:.3f}")
ax3.axvline(tau0_piv, color="gray", ls="--", lw=1, label=f"tau0 PIV={tau0_piv} Pa")
ax3.set_xlabel(r"$\tau$ (Pa)")
ax3.set_ylabel(r"$v_{slip}$ (mm/s)")
ax3.set_title("Velocidad de slip vs estrés")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.legend(fontsize=7); ax3.grid(ls=":", lw=0.5, alpha=0.5)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# --- ax4: HB curves + PIV ranges ---
g_plot = np.logspace(-3, 2, 300)
if popt_raw is not None:
    ax4.plot(g_plot, hb(g_plot, *popt_raw), color=COLORS_GAP["1.0"], lw=2,
             ls="--", label=f"HB raw h=1mm: tau0={popt_raw[0]:.1f} K={popt_raw[1]:.2f} n={popt_raw[2]:.3f}")
if popt_cor is not None:
    ax4.plot(g_plot, hb(g_plot, *popt_cor), color="#c45022", lw=2.5,
             label=f"HB Mooney: tau0={popt_cor[0]:.1f} K={popt_cor[1]:.2f} n={popt_cor[2]:.3f}")
ax4.scatter(gdot_corr, tau_corr, s=18, color="#c45022", zorder=5, alpha=0.7)
ax4.axhline(tau0_piv, color="gray", ls=":", lw=1.2, alpha=0.8)
ax4.text(0.003, tau0_piv+1, f"tau0_PIV={tau0_piv} Pa", fontsize=7, color="gray")

# PIV shear rate ranges (Z1,Z2,Z3 — t>1.8s)
piv_med = {"Z1":0.402, "Z2":0.500, "Z3":0.627}
piv_p95 = {"Z1":3.145, "Z2":5.621, "Z3":10.330}
piv_col = {"Z1":"#4a7ab5","Z2":"#e8a020","Z3":"#c45022"}
for iz, z in enumerate(["Z1","Z2","Z3"]):
    ax4.axvspan(piv_med[z], piv_p95[z], alpha=0.12, color=piv_col[z],
                label=f"PIV {z} med-p95")
    ax4.axvline(piv_med[z], color=piv_col[z], lw=0.8, ls=":", alpha=0.7)

ax4.set_xlabel(r"$\dot{\gamma}$ (s$^{-1}$)")
ax4.set_ylabel(r"$\tau$ (Pa)")
ax4.set_xscale("log"); ax4.set_yscale("log")
ax4.set_xlim(1e-3, 100); ax4.set_ylim(5, 120)
ax4.set_title("HB corregido vs PIV rango gamma_dot")
ax4.legend(fontsize=6.5, loc="lower right"); ax4.grid(ls=":", lw=0.5, alpha=0.5)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

# --- ax5: viscosidad vs gdot_true (corrected) ---
eta_corr = tau_corr / gdot_corr
ax5.scatter(gdot_corr, eta_corr, s=20, color="#c45022", zorder=5, label="datos Mooney")
for gs in ["0.5", "1.0", "2.0"]:
    t = tau_up[gs]; g = gdot_up[gs]
    ax5.scatter(g, t/g, s=12, color=COLORS_GAP[gs], alpha=0.5, label=LABELS_GAP[gs])
ax5.set_xlabel(r"$\dot{\gamma}$ (s$^{-1}$)")
ax5.set_ylabel(r"$\eta$ (Pa·s)")
ax5.set_xscale("log"); ax5.set_yscale("log")
ax5.set_title("Viscosidad aparente vs gamma_dot")
ax5.legend(fontsize=7); ax5.grid(ls=":", lw=0.5, alpha=0.5)
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

# --- ax6: histeresis (up vs down, gap 1mm) ---
def get_downcurve(gap_str):
    parts = (raw[gap_str]["Flow sweep - 4"] +
             raw[gap_str]["Flow sweep - 5"] +
             raw[gap_str]["Flow sweep - 6"])
    arr = np.array(parts)
    idx = np.argsort(arr[:, 1])
    return arr[idx, 0], arr[idx, 1]

for gs in ["0.5", "1.0", "2.0"]:
    tau_d, gdot_d = get_downcurve(gs)
    ax6.scatter(gdot_up[gs], tau_up[gs], s=15, color=COLORS_GAP[gs],
                marker="^", label=f"up {LABELS_GAP[gs]}", zorder=5)
    ax6.scatter(gdot_d, tau_d, s=15, color=COLORS_GAP[gs],
                marker="v", alpha=0.5)
ax6.axhline(tau0_piv, color="gray", ls="--", lw=1.0, label=f"tau0 PIV={tau0_piv} Pa")
ax6.set_xlabel(r"$\dot{\gamma}_{app}$ (s$^{-1}$)")
ax6.set_ylabel(r"$\tau$ (Pa)")
ax6.set_xscale("log"); ax6.set_yscale("log")
ax6.set_title("Histéresis up (triangulo) vs down (triangulo inv)")
ax6.legend(fontsize=6.5, ncol=2); ax6.grid(ls=":", lw=0.5, alpha=0.5)
ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

OUT = fr"{PIV_ANALISIS}\reometro_slip_hb.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"\nFigura guardada: {OUT}")
plt.close()

# ── Resumen numérico ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN — Carbopol 0.2 % — HR30 — 3 gaps")
print("="*60)
print(f"\nParametros HB (up-curve, SIN correccion, gap 1mm):")
if popt_raw is not None:
    print(f"  tau0 = {popt_raw[0]:.2f} Pa   K = {popt_raw[1]:.3f} Pa.s^n   n = {popt_raw[2]:.3f}")
print(f"\nParametros HB (up-curve, CON correccion Mooney):")
if popt_cor is not None:
    print(f"  tau0 = {popt_cor[0]:.2f} Pa   K = {popt_cor[1]:.3f} Pa.s^n   n = {popt_cor[2]:.3f}")
print(f"\nReferencia PIV (Balmforth geometric):")
print(f"  tau0 = {tau0_piv:.2f} Pa")
print(f"\nSlip power-law: v_slip = alpha * tau^m")
if popt_vs is not None:
    print(f"  alpha = {popt_vs[0]:.4e} m/s   m = {popt_vs[1]:.3f}")
    tau_test = np.array([20, 40, 60, 80])
    vs_test  = vslip_pl(tau_test, *popt_vs) * 1e3
    print(f"  v_slip @ tau=[20,40,60,80] Pa: {vs_test} mm/s")
print(f"\nRango gamma_dot PIV (t>1.8s, p50..p95):")
print(f"  Z1:   {piv_med['Z1']:.3f} .. {piv_p95['Z1']:.2f} s-1")
print(f"  Z2:   {piv_med['Z2']:.3f} .. {piv_p95['Z2']:.2f} s-1")
print(f"  Z3:   {piv_med['Z3']:.3f} .. {piv_p95['Z3']:.2f} s-1")
print(f"  Viga: ~0.07..1.4 s-1")

# Guardar resultados
np.save(fr"{PIV_ANALISIS}\hb_params.npy", {
    "tau0_piv": tau0_piv,
    "tau0_raw": popt_raw[0] if popt_raw is not None else None,
    "K_raw": popt_raw[1] if popt_raw is not None else None,
    "n_raw": popt_raw[2] if popt_raw is not None else None,
    "tau0_mooney": popt_cor[0] if popt_cor is not None else None,
    "K_mooney": popt_cor[1] if popt_cor is not None else None,
    "n_mooney": popt_cor[2] if popt_cor is not None else None,
    "vslip_alpha": popt_vs[0] if popt_vs is not None else None,
    "vslip_m": popt_vs[1] if popt_vs is not None else None,
})
print("\nParametros guardados: hb_params.npy")
