#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenPIV viewer estilo PIVlab + export TXT por momento (como PIVlab)
+ MEJORAS: Local Median Validation (tipo PIVlab) + pipeline de validación más robusto

Cambios principales vs tu versión:
1) DT se ingresa en ms (DT_MS) -> se convierte a segundos (DT)
2) OUT_DIR: si ya existen *.txt, se borran antes de exportar
3) Validación al cerrar (multi-core) ahora hace:
   A) Velocity-based validation (tu “región verde/naranjo” en u-v)
   B) Local median validation (similar a PIVlab; detecta vectores puntuales chuecos)
   C) Reemplazo de outliers con replace_outliers (localmean)
   D) Reimpone máscara (no se rellena dentro de máscara)
4) Viewer final: u-v scatter sin contorno (como pediste)

Requisitos:
    pip install openpiv imageio matplotlib numpy tqdm
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple
import os

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.path import Path as MplPath
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# =========================
# CONFIG
# =========================
IMAGES_DIR = Path(r"TomasProcesadas\m72-toma-1-cam-3-n-0-car-02-piv")

MASKS_DIR  = Path(r"Masks\m72-toma-1-cam-3-n-0-car-02-piv")

OUT_DIR    = Path(r"Resultados_openPIV\Cam3")

DT_MS = 1.0              # <-- DT en milisegundos
DT = DT_MS / 1000.0      # segundos (OpenPIV usa segundos)

PX_PER_MM = 20.0
MM_PER_PX = 1.0 / PX_PER_MM

WINDOW_SIZES = [64, 32, 16]
OVERLAPS     = [32, 16, 8]
SEARCH_AREA_FACTOR = 1
SIG2NOISE_METHOD = "peak2peak"

MASK_THRESHOLD = 0

DEFAULT_QUIVER_SCALE = 8.0
QUIVER_WIDTH = 0.0025

# --- Velocity-based validation (región u-v) ---
KEEP_PERCENTILE = 90     # mientras más alto, más grande la región “válida”

# --- Local Median Validation (similar PIVlab) ---
LM_KERNEL = 1            # vecindario 3x3 (1) o 5x5 (2)
LM_THRESH = 2.0          # umbral robusto (2~4 típico)
LM_EPS = 0.1             # evita división por cero en zonas de velocidad ~0


# =========================
# Utils
# =========================
def ensure_folder(path: Path, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"No existe la carpeta {name}: {path}")
    if not path.is_dir():
        raise RuntimeError(f"No es carpeta {name}: {path}")


def clear_txt_in_out_dir(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for p in out_dir.glob("*.txt"):
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] No pude borrar {p}: {e}")
    return deleted


def read_gray(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def pair_indices(n: int) -> Iterator[Tuple[int, int, int]]:
    pid = 0
    m = (n // 2) * 2
    for i in range(0, m, 2):
        yield pid, i, i + 1
        pid += 1


def whiten_masked_background(frame: np.ndarray, mask_union: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32, copy=True)
    if np.nanmax(np.abs(out)) < 1e-6:
        return np.ones_like(out, dtype=np.float32)

    lo, hi = np.percentile(out, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(out)), float(np.nanmax(out))
        if hi <= lo:
            return np.ones_like(out, dtype=np.float32)

    out = (out - lo) / (hi - lo + 1e-12)
    out = np.clip(out, 0.0, 1.0)
    out[mask_union > MASK_THRESHOLD] = 1.0
    return out


# =========================
# Convex hull (sin SciPy)
# =========================
def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def velocity_region_mask(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray]:
    pts = np.column_stack((u, v))
    if pts.shape[0] < 10:
        return None, np.ones((pts.shape[0],), dtype=bool)

    mu = np.median(u)
    mv = np.median(v)
    dist = np.sqrt((u - mu) ** 2 + (v - mv) ** 2)

    cut = np.percentile(dist, KEEP_PERCENTILE)
    core = pts[dist <= cut]
    if core.shape[0] < 3:
        return None, np.ones((pts.shape[0],), dtype=bool)

    hull = convex_hull(core)
    if hull.shape[0] < 3:
        return None, np.ones((pts.shape[0],), dtype=bool)

    poly = MplPath(hull)
    inside = poly.contains_points(pts)
    hull_closed = np.vstack([hull, hull[0]])
    return hull_closed, inside


# =========================
# Local Median Validation (tipo PIVlab)
# =========================
def local_median_flags(u: np.ndarray, v: np.ndarray,
                       kernel: int = 1,
                       thresh: float = 3.0,
                       eps: float = 0.1) -> np.ndarray:
    """
    Devuelve flags True donde el vector es outlier según criterio robusto local.

    Idea (robusta):
      r_u = |u - med_u| / (mad_u + eps)
      r_v = |v - med_v| / (mad_v + eps)
      outlier si max(r_u, r_v) > thresh

    - kernel=1 => ventana 3x3 (vecinos alrededor)
    - kernel=2 => ventana 5x5
    - ignora NaN en median/mad
    """
    u0 = u.copy()
    v0 = v.copy()

    h, w = u0.shape
    flags = np.zeros((h, w), dtype=bool)

    for i in range(h):
        i0 = max(0, i - kernel)
        i1 = min(h, i + kernel + 1)
        for j in range(w):
            j0 = max(0, j - kernel)
            j1 = min(w, j + kernel + 1)

            # vecinos (incluye centro; lo excluimos después)
            uu = u0[i0:i1, j0:j1].ravel()
            vv = v0[i0:i1, j0:j1].ravel()

            # quitar centro específico si está dentro del corte
            ci = (i - i0) * (j1 - j0) + (j - j0)
            if 0 <= ci < uu.size:
                uu = np.delete(uu, ci)
                vv = np.delete(vv, ci)

            # si el vector central no es finito, lo dejamos para replace_outliers (no lo validamos acá)
            if not (np.isfinite(u0[i, j]) and np.isfinite(v0[i, j])):
                continue

            # vecinos finitos
            m = np.isfinite(uu) & np.isfinite(vv)
            uu = uu[m]
            vv = vv[m]

            if uu.size < 4:
                continue

            med_u = np.median(uu)
            med_v = np.median(vv)

            mad_u = np.median(np.abs(uu - med_u))
            mad_v = np.median(np.abs(vv - med_v))

            ru = np.abs(u0[i, j] - med_u) / (mad_u + eps)
            rv = np.abs(v0[i, j] - med_v) / (mad_v + eps)

            if max(ru, rv) > thresh:
                flags[i, j] = True

    return flags


# =========================
# Data structures
# =========================
@dataclass(frozen=True)
class PIVResult:
    pair_id: int
    x_mm: np.ndarray
    y_mm: np.ndarray
    u_mms: np.ndarray
    v_mms: np.ndarray
    in_mask: np.ndarray
    bg_display: np.ndarray
    img_a: Path
    img_b: Path


@dataclass(frozen=True)
class PIVResultFinal:
    pair_id: int
    x_mm: np.ndarray
    y_mm: np.ndarray
    u_mms: np.ndarray
    v_mms: np.ndarray
    in_mask: np.ndarray
    bg_display: np.ndarray
    img_a: Path
    img_b: Path


# =========================
# Worker: PIV
# =========================
def compute_pair(
    pair_id: int,
    img_a_path: str,
    img_b_path: str,
    mask_a_path: str,
    mask_b_path: str,
    dt: float,
    window_sizes: List[int],
    overlaps: List[int],
    search_area_factor: int,
    sig2noise_method: str,
    mm_per_px: float,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    import openpiv.pyprocess as pyprocess
    import openpiv.filters as filters

    frame_a = read_gray(Path(img_a_path))
    frame_b = read_gray(Path(img_b_path))
    mask_a  = read_gray(Path(mask_a_path))
    mask_b  = read_gray(Path(mask_b_path))

    mask_union = np.maximum(mask_a, mask_b)
    mask_bool = mask_union > MASK_THRESHOLD

    bg_display = whiten_masked_background(frame_a, mask_union)

    fa = frame_a.copy()
    fb = frame_b.copy()
    fa[mask_a > MASK_THRESHOLD] = 0.0
    fb[mask_b > MASK_THRESHOLD] = 0.0

    u_last = v_last = None
    x_px_last = y_px_last = None
    in_mask_last = None

    for ws, ol in zip(window_sizes, overlaps):
        search_area_size = int(ws * search_area_factor)

        u, v, _ = pyprocess.extended_search_area_piv(
            fa, fb,
            window_size=int(ws),
            overlap=int(ol),
            dt=dt,
            search_area_size=search_area_size,
            sig2noise_method=sig2noise_method,
        )

        x_px, y_px = pyprocess.get_coordinates(frame_a.shape, int(ws), int(ol))

        xi = np.clip(np.round(x_px).astype(int), 0, mask_bool.shape[1] - 1)
        yi = np.clip(np.round(y_px).astype(int), 0, mask_bool.shape[0] - 1)
        in_mask = mask_bool[yi, xi]

        u[in_mask] = np.nan
        v[in_mask] = np.nan

        flags = (~np.isfinite(u)) | (~np.isfinite(v))
        u, v = filters.replace_outliers(
            u, v, flags,
            method="localmean",
            max_iter=3,
            kernel_size=2
        )

        u[in_mask] = np.nan
        v[in_mask] = np.nan

        u_last, v_last = u, v
        x_px_last, y_px_last = x_px, y_px
        in_mask_last = in_mask

    x_mm = x_px_last * mm_per_px
    y_mm = y_px_last * mm_per_px
    u_mms = u_last * mm_per_px
    v_mms = v_last * mm_per_px

    return pair_id, x_mm, y_mm, u_mms, v_mms, in_mask_last, bg_display, img_a_path, img_b_path


# =========================
# Worker: Validation (multi-core) con mejoras
# =========================
def validate_one(
    pair_id: int,
    x_mm: np.ndarray, y_mm: np.ndarray,
    u: np.ndarray, v: np.ndarray,
    in_mask: np.ndarray,
    bg_display: np.ndarray,
    img_a_path: str, img_b_path: str
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    import openpiv.filters as filters

    u2 = u.copy()
    v2 = v.copy()

    # (A) Velocity-based: detecta outliers globales por nube u-v
    valid = np.isfinite(u2) & np.isfinite(v2) & (~in_mask)
    if valid.sum() >= 10:
        uvals = u2[valid]
        vvals = v2[valid]
        _hull, inside = velocity_region_mask(uvals, vvals)
        out = ~inside
        flags_vel = np.zeros_like(u2, dtype=bool)
        flags_vel[valid] = out
        u2[flags_vel] = np.nan
        v2[flags_vel] = np.nan

    # (B) Local Median Validation: detecta vectores puntuales chuecos
    #     Solo lo aplicamos donde hay datos finitos y fuera de máscara.
    #     Los marcados se ponen NaN.
    u_tmp = u2.copy()
    v_tmp = v2.copy()
    u_tmp[in_mask] = np.nan
    v_tmp[in_mask] = np.nan

    flags_lm = local_median_flags(u_tmp, v_tmp, kernel=LM_KERNEL, thresh=LM_THRESH, eps=LM_EPS)
    flags_lm = flags_lm & (~in_mask)
    u2[flags_lm] = np.nan
    v2[flags_lm] = np.nan

    # (C) Replace outliers / huecos (como PIVlab cuando “replace vectors”)
    flags2 = (~np.isfinite(u2)) | (~np.isfinite(v2))
    u3, v3 = filters.replace_outliers(
        u2, v2, flags2,
        method="localmean",
        max_iter=3,
        kernel_size=2
    )

    # (D) Re-imponer máscara (NO rellenar dentro)
    u3[in_mask] = np.nan
    v3[in_mask] = np.nan

    return pair_id, x_mm, y_mm, u3, v3, in_mask, bg_display, img_a_path, img_b_path


def apply_velocity_based_validation_multicore(results: List[PIVResult], max_workers: int) -> List[PIVResultFinal]:
    futures = []
    finals: List[PIVResultFinal] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in results:
            futures.append(ex.submit(
                validate_one,
                r.pair_id,
                r.x_mm, r.y_mm,
                r.u_mms, r.v_mms,
                r.in_mask,
                r.bg_display,
                str(r.img_a), str(r.img_b)
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Aplicando validation", unit="par"):
            pair_id, x_mm, y_mm, u3, v3, in_mask, bg, a, b = fut.result()
            finals.append(PIVResultFinal(
                pair_id=pair_id,
                x_mm=x_mm, y_mm=y_mm,
                u_mms=u3, v_mms=v3,
                in_mask=in_mask,
                bg_display=bg,
                img_a=Path(a), img_b=Path(b)
            ))

    finals.sort(key=lambda r: r.pair_id)
    return finals


# =========================
# Export TXT (1 archivo por momento)
# =========================
def export_txt(finals: List[PIVResultFinal], names: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(tqdm(finals, desc="Exportando TXT", unit="archivo")):
        txt_path = out_dir / f"momento_{i:04d}.txt"

        valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)

        x = r.x_mm[valid].ravel()
        y = r.y_mm[valid].ravel()
        u = r.u_mms[valid].ravel()
        v = r.v_mms[valid].ravel()
        speed = np.sqrt(u*u + v*v)
        flag = np.ones_like(u, dtype=int)

        header = (
            f"# OpenPIV export (PIVlab-like)\n"
            f"# pair_id: {r.pair_id}\n"
            f"# source: {names[i]}\n"
            f"# DT_ms: {DT_MS}\n"
            f"# LM_KERNEL: {LM_KERNEL}, LM_THRESH: {LM_THRESH}, LM_EPS: {LM_EPS}\n"
            f"# KEEP_PERCENTILE: {KEEP_PERCENTILE}\n"
            f"# columns: x_mm y_mm u_mm_per_s v_mm_per_s speed_mm_per_s valid\n"
        )

        data = np.column_stack([x, y, u, v, speed, flag])

        np.savetxt(
            txt_path,
            data,
            fmt="%.6f %.6f %.6f %.6f %.6f %d",
            header=header,
            comments=""
        )


# =========================
# Viewers
# =========================
def show_viewer_initial(results: List[PIVResult], names: List[str]) -> None:
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_uv = fig.add_subplot(gs[0, 1])
    plt.subplots_adjust(bottom=0.18, wspace=0.25)

    ax_momento = plt.axes([0.15, 0.08, 0.62, 0.03])
    s_momento = Slider(ax_momento, "Momento", 0, len(results) - 1, valinit=0, valstep=1)

    ax_scale = plt.axes([0.15, 0.03, 0.62, 0.03])
    s_scale = Slider(ax_scale, "Escala", 0.5, 80.0, valinit=DEFAULT_QUIVER_SCALE, valstep=0.5)

    def draw(idx: int, scale: float) -> None:
        r = results[idx]
        ax.clear()
        ax_uv.clear()

        bg = r.bg_display
        h_px, w_px = bg.shape
        ax.imshow(bg, cmap="gray", origin="upper",
                  extent=[0, w_px * MM_PER_PX, h_px * MM_PER_PX, 0])

        valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
        uvals = r.u_mms[valid]
        vvals = r.v_mms[valid]

        if uvals.size < 10:
            ax.quiver(r.x_mm[valid], r.y_mm[valid], r.u_mms[valid], r.v_mms[valid],
                      color="orange", angles="xy", scale_units="xy", scale=scale, width=QUIVER_WIDTH)
            ax.set_title(f"PIV (pocos datos): {names[idx]}")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")
            ax_uv.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
            fig.canvas.draw_idle()
            return

        hull_closed, inside = velocity_region_mask(uvals, vvals)

        inside_grid = np.zeros_like(valid, dtype=bool)
        inside_grid[valid] = inside

        ok = inside_grid
        bad = valid & (~inside_grid)

        ax.quiver(r.x_mm[ok],  r.y_mm[ok],  r.u_mms[ok],  r.v_mms[ok],
                  color="limegreen", angles="xy", scale_units="xy", scale=scale, width=QUIVER_WIDTH)
        ax.quiver(r.x_mm[bad], r.y_mm[bad], r.u_mms[bad], r.v_mms[bad],
                  color="orange", angles="xy", scale_units="xy", scale=scale, width=QUIVER_WIDTH)

        ax.set_title(f"PIV: {names[idx]}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")

        ax_uv.scatter(uvals[inside], vvals[inside], s=6, alpha=0.45, color="limegreen")
        ax_uv.scatter(uvals[~inside], vvals[~inside], s=6, alpha=0.45, color="orange")

        if hull_closed is not None:
            ax_uv.plot(hull_closed[:, 0], hull_closed[:, 1], color="black", linewidth=2)
            ax_uv.scatter(hull_closed[:-1, 0], hull_closed[:-1, 1], s=30, color="limegreen", zorder=5)

        ax_uv.set_title("Velocity-based validation (región)")
        ax_uv.set_xlabel("u [mm/s]")
        ax_uv.set_ylabel("v [mm/s]")
        ax_uv.grid(True, alpha=0.2)

        umax = np.percentile(np.abs(uvals), 99)
        vmax = np.percentile(np.abs(vvals), 99)
        lim = max(umax, vmax, 1e-6)
        ax_uv.set_xlim(-lim, lim)
        ax_uv.set_ylim(-lim, lim)

        fig.canvas.draw_idle()

    def update(_val=None) -> None:
        draw(int(s_momento.val), float(s_scale.val))

    s_momento.on_changed(update)
    s_scale.on_changed(update)
    update()

    plt.show()


def show_viewer_final(finals: List[PIVResultFinal], names: List[str]) -> None:
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_uv = fig.add_subplot(gs[0, 1])
    plt.subplots_adjust(bottom=0.18, wspace=0.25)

    ax_momento = plt.axes([0.15, 0.08, 0.62, 0.03])
    s_momento = Slider(ax_momento, "Momento", 0, len(finals) - 1, valinit=0, valstep=1)

    ax_scale = plt.axes([0.15, 0.03, 0.62, 0.03])
    s_scale = Slider(ax_scale, "Escala", 0.5, 80.0, valinit=DEFAULT_QUIVER_SCALE, valstep=0.5)

    def draw(idx: int, scale: float) -> None:
        r = finals[idx]
        ax.clear()
        ax_uv.clear()

        bg = r.bg_display
        h_px, w_px = bg.shape
        ax.imshow(bg, cmap="gray", origin="upper",
                  extent=[0, w_px * MM_PER_PX, h_px * MM_PER_PX, 0])

        valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)

        ax.quiver(r.x_mm[valid], r.y_mm[valid], r.u_mms[valid], r.v_mms[valid],
                  color="limegreen", angles="xy", scale_units="xy", scale=scale, width=QUIVER_WIDTH)

        ax.set_title(f"FINAL (validado): {names[idx]}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")

        uvals = r.u_mms[valid]
        vvals = r.v_mms[valid]

        if uvals.size > 0:
            ax_uv.scatter(uvals, vvals, s=6, alpha=0.45, color="limegreen")
            ax_uv.set_title("u-v final (scatter)")
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")
            ax_uv.grid(True, alpha=0.2)

            umax = np.percentile(np.abs(uvals), 99)
            vmax = np.percentile(np.abs(vvals), 99)
            lim = max(umax, vmax, 1e-6)
            ax_uv.set_xlim(-lim, lim)
            ax_uv.set_ylim(-lim, lim)
        else:
            ax_uv.text(0.5, 0.5, "Sin datos", ha="center", va="center")

        fig.canvas.draw_idle()

    def update(_val=None) -> None:
        draw(int(s_momento.val), float(s_scale.val))

    s_momento.on_changed(update)
    s_scale.on_changed(update)
    update()

    plt.show()


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_folder(IMAGES_DIR, "IMAGES_DIR")
    ensure_folder(MASKS_DIR, "MASKS_DIR")

    if DT <= 0:
        raise RuntimeError(f"DT_MS debe ser > 0. Valor actual: {DT_MS}")

    images = sorted(IMAGES_DIR.glob("*.tif*"))
    masks  = sorted(MASKS_DIR.glob("*.tif*"))

    if len(images) < 2:
        raise RuntimeError(f"Se necesitan al menos 2 imágenes en {IMAGES_DIR}")
    if len(masks) < len(images):
        raise RuntimeError(f"Se esperaban >= {len(images)} máscaras en {MASKS_DIR}, encontré {len(masks)}")
    if len(WINDOW_SIZES) != len(OVERLAPS):
        raise RuntimeError("WINDOW_SIZES y OVERLAPS deben tener el mismo largo.")

    pairs = list(pair_indices(len(images)))
    if not pairs:
        raise RuntimeError("No hay pares A-B (necesitas al menos 2 imágenes).")

    max_workers = max(1, (os.cpu_count() or 2) - 1)

    futures = []
    results: List[PIVResult] = []
    names: List[str] = [""] * len(pairs)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for pair_id, ia, ib in pairs:
            img_a = images[ia]
            img_b = images[ib]
            m_a = masks[ia]
            m_b = masks[ib]

            names[pair_id] = f"{img_a.name} - {img_b.name}"

            futures.append(ex.submit(
                compute_pair,
                pair_id,
                str(img_a), str(img_b),
                str(m_a), str(m_b),
                DT,
                WINDOW_SIZES,
                OVERLAPS,
                SEARCH_AREA_FACTOR,
                SIG2NOISE_METHOD,
                MM_PER_PX,
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Procesando PIV", unit="par"):
            pair_id, x_mm, y_mm, u_mms, v_mms, in_mask, bg_disp, a, b = fut.result()
            results.append(PIVResult(
                pair_id=pair_id,
                x_mm=x_mm, y_mm=y_mm,
                u_mms=u_mms, v_mms=v_mms,
                in_mask=in_mask,
                bg_display=bg_disp,
                img_a=Path(a), img_b=Path(b),
            ))

    results.sort(key=lambda r: r.pair_id)

    print(f"[OK] DT_MS={DT_MS} ms (DT={DT:.6f} s)")
    print(f"[OK] Pares procesados: {len(results)} | workers: {max_workers}")
    print(f"[INFO] Validación: KEEP_PERCENTILE={KEEP_PERCENTILE} | LocalMedian kernel={LM_KERNEL}, thresh={LM_THRESH}, eps={LM_EPS}")

    # Viewer inicial (cierra para aplicar validación)
    show_viewer_initial(results, names)

    # Validación mejorada (multi-core)
    finals = apply_velocity_based_validation_multicore(results, max_workers=max_workers)

    # Viewer final
    show_viewer_final(finals, names)

    # Limpiar TXT y exportar
    deleted = clear_txt_in_out_dir(OUT_DIR)
    if deleted:
        print(f"[OK] Borrados {deleted} TXT antiguos en: {OUT_DIR.resolve()}")

    export_txt(finals, names, OUT_DIR)
    print(f"[OK] Exportados {len(finals)} archivos TXT en: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()