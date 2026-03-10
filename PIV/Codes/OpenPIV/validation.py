# validation.py
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import generic_filter
from typing import Tuple


# =============================================================
# Convex Hull (sin SciPy)
# =============================================================

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


# =============================================================
# Velocity-based global validation
# =============================================================

def velocity_region_mask(
    u: np.ndarray,
    v: np.ndarray,
    keep_percentile: float,
    method: str = "mahalanobis",  # "mahalanobis" o "circular"
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Validación basada en región de velocidades.
    
    Args:
        u, v: componentes de velocidad
        keep_percentile: percentil a mantener (80-99)
        method: "mahalanobis" (elipse adaptativa) o "circular" (método original)
        
    Returns:
        hull_closed: contorno de la región
        inside: máscara booleana de puntos dentro
    """
    pts = np.column_stack((u, v))
    if pts.shape[0] < 10:
        return None, np.ones((pts.shape[0],), dtype=bool)

    if method == "mahalanobis":
        return _mahalanobis_region(u, v, keep_percentile)
    else:
        return _circular_region(u, v, keep_percentile)


def _circular_region(
    u: np.ndarray,
    v: np.ndarray,
    keep_percentile: float,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Método ORIGINAL (circular) - mantener para comparación.
    """
    pts = np.column_stack((u, v))
    
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


def _mahalanobis_region(
    u: np.ndarray,
    v: np.ndarray,
    chi2_percentile: float,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Método MAHALANOBIS - captura correlación entre u y v.
    Crea elipses orientadas que siguen la dispersión real de datos.
    
    Ventajas:
    - Detecta correlación entre u y v
    - Se adapta a forma elíptica orientada
    - Usa estimadores robustos (MAD)
    """
    # Convertir a arrays normales (no masked)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Filtrar NaN/Inf
    valid_mask = np.isfinite(u) & np.isfinite(v)
    if np.sum(valid_mask) < 10:
        return None, np.ones(u.shape[0], dtype=bool)
    
    u_clean = u[valid_mask]
    v_clean = v[valid_mask]
    pts_clean = np.column_stack((u_clean, v_clean))
    
    # Estimadores robustos del centro
    center = np.array([np.median(u_clean), np.median(v_clean)])
    
    # Desviaciones desde el centro
    dev = pts_clean - center
    
    # MAD robusto para cada componente
    mad_u = np.median(np.abs(dev[:, 0]))
    mad_v = np.median(np.abs(dev[:, 1]))
    
    # Factor de escala: MAD → std (1.4826 para distribución normal)
    scale = 1.4826
    std_u = mad_u * scale
    std_v = mad_v * scale
    
    # Evitar divisiones por cero
    if std_u < 1e-6 or std_v < 1e-6:
        # Fallback a método circular
        return _circular_region(u, v, chi2_percentile)
    
    # Covarianza robusta usando correlación de Spearman-like
    # (producto de desviaciones normalizadas)
    cov_uv = np.median((dev[:, 0] / std_u) * (dev[:, 1] / std_v)) * std_u * std_v
    
    # Matriz de covarianza
    cov = np.array([[std_u**2, cov_uv],
                    [cov_uv, std_v**2]], dtype=np.float64)
    
    # Regularización para evitar singularidad
    cov += np.eye(2) * 1e-6
    
    # Invertir covarianza
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Si falla, usar circular
        return _circular_region(u, v, chi2_percentile)
    
    # Distancia de Mahalanobis al cuadrado
    # CORRECCIÓN: hacer la multiplicación elemento por elemento correctamente
    mahal_sq_clean = np.einsum('ij,ji->i', dev, cov_inv @ dev.T)
    
    # Umbral basado en percentil
    threshold = np.percentile(mahal_sq_clean, chi2_percentile)
    inside_clean = mahal_sq_clean <= threshold
    
    # Mapear de vuelta a todos los puntos (incluyendo NaN originales)
    inside = np.zeros(u.shape[0], dtype=bool)
    inside[valid_mask] = inside_clean
    
    # Generar puntos de la elipse para visualización
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    
    # Eigenvalues y eigenvectors de la covarianza
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Semi-ejes de la elipse
    a = np.sqrt(threshold * eigvals[0])
    b = np.sqrt(threshold * eigvals[1])
    
    # Puntos de la elipse en coordenadas propias
    ellipse_std = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    
    # Rotar según eigenvectors y trasladar al centro
    ellipse_points = (eigvecs @ ellipse_std.T).T + center
    ellipse_closed = np.vstack([ellipse_points, ellipse_points[0]])
    
    return ellipse_closed, inside


# =============================================================
# Local Median Validation — vectorizado con scipy.ndimage
# MEJORA #1: reemplaza el doble loop Python puro (~50-100x más rápido)
# =============================================================

def local_median_flags(
    u: np.ndarray,
    v: np.ndarray,
    kernel: int = 1,
    thresh: float = 2.0,
    eps: float = 0.1,
) -> np.ndarray:
    """
    Detecta outliers locales robustos usando scipy.ndimage.generic_filter.
    Equivalente al doble loop original pero vectorizado.

    r = |val - median(neighbors)| / (MAD(neighbors) + eps)
    outlier si max(r_u, r_v) > thresh

    kernel=1 => ventana 3x3
    kernel=2 => ventana 5x5
    """
    size = 2 * kernel + 1
    ci = size * size // 2  # índice del centro en la ventana aplanada

    def _robust_residual(vals: np.ndarray) -> float:
        center = vals[ci]
        if not np.isfinite(center):
            return 0.0
        neighbors = np.concatenate([vals[:ci], vals[ci + 1:]])
        finite = neighbors[np.isfinite(neighbors)]
        if finite.size < 4:
            return 0.0
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        return float(np.abs(center - med) / (mad + eps))

    ru = generic_filter(u, _robust_residual, size=size, mode="nearest")
    rv = generic_filter(v, _robust_residual, size=size, mode="nearest")

    flags = (np.maximum(ru, rv) > thresh) & np.isfinite(u) & np.isfinite(v)
    return flags