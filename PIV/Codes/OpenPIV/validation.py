# validation.py
from __future__ import annotations
import numpy as np
from matplotlib.path import Path as MplPath
from typing import Tuple


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


def velocity_region_mask(u: np.ndarray, v: np.ndarray, keep_percentile: float) -> Tuple[np.ndarray | None, np.ndarray]:
    pts = np.column_stack((u, v))
    if pts.shape[0] < 10:
        return None, np.ones((pts.shape[0],), dtype=bool)

    mu = np.median(u)
    mv = np.median(v)
    dist = np.sqrt((u - mu) ** 2 + (v - mv) ** 2)

    cut = np.percentile(dist, keep_percentile)
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


def local_median_flags(
    u: np.ndarray, v: np.ndarray,
    kernel: int, thresh: float, eps: float
) -> np.ndarray:
    """
    flags True donde el vector es outlier local (robusto).
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

            uu = u0[i0:i1, j0:j1].ravel()
            vv = v0[i0:i1, j0:j1].ravel()

            ci = (i - i0) * (j1 - j0) + (j - j0)
            if 0 <= ci < uu.size:
                uu = np.delete(uu, ci)
                vv = np.delete(vv, ci)

            if not (np.isfinite(u0[i, j]) and np.isfinite(v0[i, j])):
                continue

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