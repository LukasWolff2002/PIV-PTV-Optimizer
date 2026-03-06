# workers.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .utils import read_gray, whiten_masked_background
from .validation import velocity_region_mask, local_median_flags


def _load_static_mask_if_needed(
    fixed_mask_path: str,
    shape_hw: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Carga máscara fija si existe. Devuelve float32 (H,W) o None.
    """
    if not fixed_mask_path:
        return None

    p = Path(fixed_mask_path)
    if not p.exists():
        return None

    m = read_gray(p)
    if m.ndim == 3:
        m = m[..., 0]

    if m.shape != shape_hw:
        try:
            import cv2
            m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None

    return m


def compute_pair_worker(
    pair_id: int,
    img_a_path: str,
    img_b_path: str,
    mask_a_path: str,
    mask_b_path: str,
    dt_s: float,
    window_sizes: List[int],
    overlaps: List[int],
    search_area_factor: int,
    sig2noise_method: str,
    mm_per_px: float,
    mask_threshold: float,
    apply_dynamic_mask: bool = True,
    apply_static_mask: bool = False,
    fixed_mask_path: str = "",
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    import openpiv.pyprocess as pyprocess
    import openpiv.filters as filters

    frame_a = read_gray(Path(img_a_path))
    frame_b = read_gray(Path(img_b_path))

    # -------------------------
    # 1) Máscara dinámica (por frame, independiente para A y B)
    # -------------------------
    if apply_dynamic_mask:
        mask_a = read_gray(Path(mask_a_path))
        mask_b = read_gray(Path(mask_b_path))
        mask_a_bool = mask_a > mask_threshold
        mask_b_bool = mask_b > mask_threshold
        mask_union  = np.maximum(mask_a, mask_b)
    else:
        mask_a_bool = np.zeros_like(frame_a, dtype=bool)
        mask_b_bool = np.zeros_like(frame_b, dtype=bool)
        mask_union  = np.zeros_like(frame_a, dtype=np.float32)

    # -------------------------
    # 2) Máscara fija (por cámara)
    # -------------------------
    fix_bool = None
    if apply_static_mask:
        mask_fix = _load_static_mask_if_needed(
            fixed_mask_path=fixed_mask_path,
            shape_hw=frame_a.shape[:2],
        )
        if mask_fix is not None:
            fix_bool = mask_fix > mask_threshold

    # -------------------------
    # 3) Máscara final para in_mask y display (unión de todo)
    #    Solo se usa para marcar vectores NaN y display,
    #    NO para enmascarar los frames antes del PIV.
    # -------------------------
    if fix_bool is not None:
        final_mask_bool = (mask_a_bool | mask_b_bool) | fix_bool
        mask_for_display = (final_mask_bool.astype(np.uint8) * 255).astype(np.float32)
    elif apply_dynamic_mask:
        final_mask_bool = mask_a_bool | mask_b_bool
        mask_for_display = mask_union
    else:
        final_mask_bool = np.zeros_like(frame_a, dtype=bool)
        mask_for_display = np.zeros_like(frame_a, dtype=np.float32)

    # -------------------------
    # 4) Display background (whiten masked)
    # -------------------------
    bg_display = whiten_masked_background(frame_a, mask_for_display, mask_threshold)

    # -------------------------
    # 5) Aplicar máscara a frames para PIV
    #    CORRECCIÓN: cada frame se enmascara con su propia máscara dinámica
    #    independientemente (igual que el código original que funcionaba),
    #    más la fija si existe.
    # -------------------------
    fa = frame_a.copy()
    fb = frame_b.copy()

    if fix_bool is not None:
        fa[mask_a_bool | fix_bool] = 0.0
        fb[mask_b_bool | fix_bool] = 0.0
    else:
        fa[mask_a_bool] = 0.0
        fb[mask_b_bool] = 0.0

    # -------------------------
    # 6) Multi-pass PIV
    # -------------------------
    u_last = v_last = None
    x_px_last = y_px_last = None
    in_mask_last = None

    for ws, ol in zip(window_sizes, overlaps):
        search_area_size = int(ws * search_area_factor)

        u, v, _ = pyprocess.extended_search_area_piv(
            fa, fb,
            window_size=int(ws),
            overlap=int(ol),
            dt=dt_s,
            search_area_size=search_area_size,
            sig2noise_method=sig2noise_method,
        )

        x_px, y_px = pyprocess.get_coordinates(frame_a.shape, int(ws), int(ol))

        xi = np.clip(np.round(x_px).astype(int), 0, final_mask_bool.shape[1] - 1)
        yi = np.clip(np.round(y_px).astype(int), 0, final_mask_bool.shape[0] - 1)
        in_mask = final_mask_bool[yi, xi]

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


def validate_pair_worker(
    pair_id: int,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    u_mms: np.ndarray,
    v_mms: np.ndarray,
    in_mask: np.ndarray,
    bg_display: np.ndarray,
    img_a_path: str,
    img_b_path: str,
    keep_percentile: float,
    lm_kernel: int,
    lm_thresh: float,
    lm_eps: float,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    import openpiv.filters as filters

    u2 = u_mms.copy()
    v2 = v_mms.copy()

    # A) Velocity-based global (u-v region)
    valid = np.isfinite(u2) & np.isfinite(v2) & (~in_mask)
    if valid.sum() >= 10:
        uvals = u2[valid]
        vvals = v2[valid]
        _, inside = velocity_region_mask(uvals, vvals, keep_percentile=keep_percentile)

        flags_vel = np.zeros_like(u2, dtype=bool)
        flags_vel[valid] = ~inside
        u2[flags_vel] = np.nan
        v2[flags_vel] = np.nan

    # B) Local median (vectores puntuales outliers)
    u_tmp = u2.copy()
    v_tmp = v2.copy()
    u_tmp[in_mask] = np.nan
    v_tmp[in_mask] = np.nan

    flags_lm = local_median_flags(u_tmp, v_tmp, kernel=lm_kernel, thresh=lm_thresh, eps=lm_eps)
    flags_lm = flags_lm & (~in_mask)
    u2[flags_lm] = np.nan
    v2[flags_lm] = np.nan

    # C) Replace outliers / huecos
    flags2 = (~np.isfinite(u2)) | (~np.isfinite(v2))
    u3, v3 = filters.replace_outliers(
        u2, v2, flags2,
        method="localmean",
        max_iter=3,
        kernel_size=2
    )

    # D) Reimpose mask (no rellenar dentro)
    u3[in_mask] = np.nan
    v3[in_mask] = np.nan

    return pair_id, x_mm, y_mm, u3, v3, in_mask, bg_display, img_a_path, img_b_path