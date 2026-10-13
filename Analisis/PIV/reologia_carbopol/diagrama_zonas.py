"""
Diagrama de zonas Z1, Z2, Z3 de la L-beam.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe

PIV_ANALISIS = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers"
sys.path.insert(0, PIV_ANALISIS)
os.chdir(PIV_ANALISIS)

from definir_zonas import (l1, l2, l3, l4, l5, l6,
                            b1, b2, b3, b4,
                            v1, v2, v3, v4,   # Z1
                            c1, c2, c3, c4,   # Z2
                            s1, s2, s3, s4)   # Z3

fig, ax = plt.subplots(figsize=(8, 9))
ax.set_aspect("equal")
ax.set_facecolor("#f8f9fb")
fig.patch.set_facecolor("#f8f9fb")

# ── Colores ──────────────────────────────────────────────────────────────────
C_STRUCT  = "#d4dce8"   # estructura L-beam
C_VIGA    = "#e8eef5"   # viga
C_Z1      = "#4a7ab5"   # Z1 brazo inclinado
C_Z2      = "#e8a020"   # Z2 codo
C_Z3      = "#c45022"   # Z3 salida
ALPHA_Z   = 0.55
C_EDGE    = "#1a3060"

# ── Estructura L-beam (fondo) ─────────────────────────────────────────────────
l_beam_patch = plt.Polygon([l1, l2, l3, l4, l5, l6], closed=True,
    facecolor=C_STRUCT, edgecolor=C_EDGE, linewidth=1.8, zorder=1)
ax.add_patch(l_beam_patch)

# ── Viga ──────────────────────────────────────────────────────────────────────
viga_patch = plt.Polygon([b1, b2, b3, b4], closed=True,
    facecolor=C_VIGA, edgecolor=C_EDGE, linewidth=1.8, zorder=1)
ax.add_patch(viga_patch)

# ── Zonas ────────────────────────────────────────────────────────────────────
for verts, color, label, zlab in [
    ([v1, v2, v3, v4], C_Z1, "Z1", "Z1"),
    ([c1, c2, c3, c4], C_Z2, "Z2", "Z2"),
    ([s1, s2, s3, s4], C_Z3, "Z3", "Z3"),
]:
    ax.add_patch(plt.Polygon(verts, closed=True,
        facecolor=color, edgecolor=C_EDGE,
        linewidth=1.5, alpha=ALPHA_Z, zorder=2))

# ── Etiquetas de zonas ────────────────────────────────────────────────────────
def centroid(pts):
    return np.mean(pts, axis=0)

label_cfg = [
    (centroid([v1, v2, v3, v4]), C_Z1, "Z1\nBrazo\ninclinado\n(60°)"),
    (centroid([c1, c2, c3, c4]), C_Z2, "Z2\nCodo"),
    (centroid([s1, s2, s3, s4]), C_Z3, "Z3\nSalida"),
]
for pos, col, txt in label_cfg:
    ax.text(pos[0], pos[1], txt,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground=col)])

# ── Flechas de flujo ─────────────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="-|>", color="#1a3060",
                lw=1.8, mutation_scale=14, zorder=6)

# Z1: flujo descendente a lo largo del brazo (dirección t̂ = (0.5, 0.866))
c_z1 = centroid([v1, v2, v3, v4])
d = 10.0
for offset in [-5, 0, 5]:
    p_start = c_z1 + np.array([-offset * 0.866, offset * 0.5]) - np.array([0.5, 0.866]) * d
    p_end   = p_start + np.array([0.5, 0.866]) * 2 * d
    ax.annotate("", xy=p_end, xytext=p_start,
                arrowprops=dict(**arrow_kw, alpha=0.7))

# Z2: flujo curvo en el codo (de vertical a horizontal)
c_z2 = centroid([c1, c2, c3, c4])
ax.annotate("", xy=c_z2 + np.array([8, -3]),
            xytext=c_z2 + np.array([-3, 8]),
            arrowprops=dict(**arrow_kw))

# Z3: flujo horizontal hacia la derecha (dirección ĝ ≈ (0.866, -0.5))
c_z3 = centroid([s1, s2, s3, s4])
for offset in [-3, 0, 3]:
    p_start = c_z3 + np.array([offset * 0.5, offset * 0.866]) - np.array([0.866, -0.5]) * 6
    p_end   = p_start + np.array([0.866, -0.5]) * 12
    ax.annotate("", xy=p_end, xytext=p_start,
                arrowprops=dict(**arrow_kw, alpha=0.7))

# ── Leyenda ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_Z1, edgecolor=C_EDGE, alpha=ALPHA_Z,
                   label="Z1 — Brazo inclinado (PIV, cam1–3)"),
    mpatches.Patch(facecolor=C_Z2, edgecolor=C_EDGE, alpha=ALPHA_Z,
                   label="Z2 — Codo"),
    mpatches.Patch(facecolor=C_Z3, edgecolor=C_EDGE, alpha=ALPHA_Z,
                   label="Z3 — Salida (PIV, cam4)"),
]
ax.legend(handles=legend_items, loc="upper left",
          fontsize=9, framealpha=0.9, edgecolor="#ccc")

# ── Dimensiones clave ─────────────────────────────────────────────────────────
# Separación Z3–viga
ax.annotate("", xy=s1, xytext=b1,
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
mid = (np.array(s1) + np.array(b1)) / 2
ax.text(mid[0] + 2, mid[1], "6 mm", fontsize=8, color="#555", va="center")

# ── Ejes ──────────────────────────────────────────────────────────────────────
all_pts = np.vstack([l1, l2, l3, l4, l5, l6, b1, b2, b3, b4])
xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
ax.set_xlim(xmin - 15, xmax + 30)
ax.set_ylim(ymin - 10, ymax + 20)
ax.set_xlabel("x (mm)", fontsize=11)
ax.set_ylabel("y (mm)", fontsize=11)
ax.set_title("Zonas de análisis — L-beam 30°\n"
             "Z1: brazo inclinado · Z2: codo · Z3: salida horizontal",
             fontsize=11, pad=10)
ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUT = r"C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\diagrama_zonas_lbeam.png"
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Guardado: {OUT}")
plt.close()
