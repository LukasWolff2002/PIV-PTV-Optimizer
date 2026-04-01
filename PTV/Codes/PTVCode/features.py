"""
features.py
===========
Vector de features para Similarity Search Scheme (SSS) de fibras.

Diseño del vector (17 dimensiones):
  Bloque posición    [0:4]  : cx, cy, sin(dir_vx_vy), cos(dir_vx_vy)
  Bloque geometría   [4:8]  : length_px, width_px, aspect_ratio, area_px
  Bloque orientación [8:12] : sin(2θ), cos(2θ), sin(θ), cos(θ)
  Bloque forma       [12:15]: solidity, extent, hu0
  Bloque textura     [15:17]: mean_intensity, std_intensity

Notas de diseño:
- Ángulo codificado con simetría π: sin(2θ)/cos(2θ) hacen que 0°≡180°
- Posición normalizada por tamaño de imagen para invarianza de escala
- Pesos W permiten ajustar importancia relativa de cada bloque
- Score de similitud: similitud coseno tras normalización L2 por dimensión
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass

from .models import Detection, TrackState


# ─────────────────────────────────────────────
# PESOS POR BLOQUE (ajustables en config)
# ─────────────────────────────────────────────

DEFAULT_FEATURE_WEIGHTS = np.array([
    # posición (cx, cy, dir_vx_sin, dir_vx_cos)
    1.0, 1.0, 0.3, 0.3,
    # geometría (length, width, aspect, area)
    0.5, 0.3, 0.4, 0.2,
    # orientación (sin2θ, cos2θ, sinθ, cosθ)
    0.8, 0.8, 0.5, 0.5,
    # forma (solidity, extent, hu0)
    0.3, 0.2, 0.2,
    # textura (mean_int, std_int)
    0.15, 0.15,
], dtype=np.float32)

FEATURE_DIM = len(DEFAULT_FEATURE_WEIGHTS)  # 17


# ─────────────────────────────────────────────
# ESTADÍSTICAS DE NORMALIZACIÓN
# ─────────────────────────────────────────────

@dataclass
class FeatureScaler:
    """
    Z-score scaler con medias y desviaciones fijas por dimensión.
    Fijadas a valores típicos de fibras (13mm @ 7.8px/mm, imagen 1024px).
    Se pueden actualizar online con update() si se desea adaptativo.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def default(cls, img_w: int = 1024, img_h: int = 1024) -> "FeatureScaler":
        """Scaler con valores típicos para fibras 13mm en imagen 1024px."""
        mean = np.array([
            img_w / 2, img_h / 2,   # cx, cy
            0.0, 0.0,                # dir velocidad (sin, cos)
            60.0, 4.0,               # length_px, width_px (proyección media)
            15.0, 240.0,             # aspect_ratio, area_px
            0.0, 0.0,                # sin(2θ), cos(2θ)
            0.0, 0.0,                # sin(θ), cos(θ)
            0.85, 0.70, 0.35,        # solidity, extent, hu0
            0.45, 0.15,              # mean_int, std_int
        ], dtype=np.float32)

        std = np.array([
            img_w * 0.35, img_h * 0.35,  # cx, cy
            0.7, 0.7,                    # dir velocidad
            35.0, 2.0,                   # length, width
            10.0, 150.0,                 # aspect, area
            0.7, 0.7,                    # sin/cos 2θ
            0.7, 0.7,                    # sin/cos θ
            0.10, 0.15, 0.15,            # solidity, extent, hu0
            0.20, 0.10,                  # intensidades
        ], dtype=np.float32)

        return cls(mean=mean, std=std)

    def transform(self, v: np.ndarray) -> np.ndarray:
        """Normaliza vector o matriz de features."""
        return (v - self.mean) / (self.std + 1e-8)


# ─────────────────────────────────────────────
# EXTRACCIÓN DE FEATURES DESDE DETECCIÓN
# ─────────────────────────────────────────────

def extract_features_from_detection(
    det: Detection,
    mask_u8: np.ndarray | None = None,
    image_gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construye vector de features desde una detección YOLO.

    Args:
        det       : Detection con cx, cy, angle_deg, length_px, width_px, area_px
        mask_u8   : máscara binaria uint8 (H×W) — necesaria para features de forma
        image_gray: imagen grayscale float [0,1] — necesaria para features de textura

    Returns:
        vector float32 de FEATURE_DIM dimensiones
    """
    theta = np.radians(det.angle_deg)
    aspect = det.length_px / (det.width_px + 1e-6)

    # Sin velocidad en la detección inicial, dirección indefinida
    v = np.zeros(FEATURE_DIM, dtype=np.float32)

    # [0:4] Posición
    v[0] = det.cx
    v[1] = det.cy
    v[2] = 0.0    # sin(dir_v) — desconocido en detección aislada
    v[3] = 0.0    # cos(dir_v)

    # [4:8] Geometría
    v[4] = det.length_px
    v[5] = det.width_px
    v[6] = aspect
    v[7] = det.area_px

    # [8:12] Orientación con simetría π
    v[8]  = np.sin(2 * theta)
    v[9]  = np.cos(2 * theta)
    v[10] = np.sin(theta)
    v[11] = np.cos(theta)

    # [12:15] Forma — desde máscara si disponible
    if mask_u8 is not None and mask_u8.sum() > 0:
        solidity, extent, hu0 = _shape_features(mask_u8, det.area_px)
    else:
        solidity, extent, hu0 = 0.85, 0.70, 0.35  # valores típicos de fibra

    v[12] = solidity
    v[13] = extent
    v[14] = hu0

    # [15:17] Textura — desde imagen si disponible
    if image_gray is not None and mask_u8 is not None and mask_u8.sum() > 0:
        mean_i, std_i = _texture_features(image_gray, mask_u8)
    else:
        mean_i, std_i = 0.45, 0.15  # valores típicos

    v[15] = mean_i
    v[16] = std_i

    return v


def extract_features_from_state(
    state: TrackState,
    mask_u8: np.ndarray | None = None,
    image_gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construye vector de features desde el estado predicho del filtro de Kalman.
    Incluye dirección de velocidad como feature adicional.
    """
    theta = np.radians(state.angle_deg)
    aspect = state.length_px / (state.width_px + 1e-6)
    area = state.length_px * state.width_px * 0.7  # área elíptica aproximada

    # Dirección de velocidad (feature de movimiento)
    v_mag = np.sqrt(state.vx ** 2 + state.vy ** 2) + 1e-8
    dir_sin = state.vy / v_mag
    dir_cos = state.vx / v_mag

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    vec[0] = state.x
    vec[1] = state.y
    vec[2] = float(dir_sin)
    vec[3] = float(dir_cos)

    vec[4] = state.length_px
    vec[5] = state.width_px
    vec[6] = float(aspect)
    vec[7] = float(area)

    vec[8]  = float(np.sin(2 * theta))
    vec[9]  = float(np.cos(2 * theta))
    vec[10] = float(np.sin(theta))
    vec[11] = float(np.cos(theta))

    # Forma y textura del estado: usamos la detección anterior si hay máscara
    if mask_u8 is not None and mask_u8.sum() > 0:
        solidity, extent, hu0 = _shape_features(mask_u8, float(area))
    else:
        solidity, extent, hu0 = 0.85, 0.70, 0.35

    vec[12] = float(solidity)
    vec[13] = float(extent)
    vec[14] = float(hu0)

    if image_gray is not None and mask_u8 is not None:
        mean_i, std_i = _texture_features(image_gray, mask_u8)
    else:
        mean_i, std_i = 0.45, 0.15

    vec[15] = float(mean_i)
    vec[16] = float(std_i)

    return vec


# ─────────────────────────────────────────────
# FEATURES DE FORMA
# ─────────────────────────────────────────────

def _shape_features(mask_u8: np.ndarray, area_px: float) -> tuple[float, float, float]:
    """
    Extrae solidity, extent y primer momento de Hu desde la máscara.

    - solidity  = area / convex_hull_area → mide cuán "sólida" es la forma
                  fibra recta ~0.9, fibra curva ~0.7
    - extent    = area / bbox_area → mide cuán compacta es en su bbox
    - hu0       = momento de Hu invariante [0], normalizado
                  codifica la distribución de masa de la forma
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.85, 0.70, 0.35

    cnt = max(contours, key=cv2.contourArea)

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area_px / (hull_area + 1e-6))
    solidity = float(np.clip(solidity, 0.0, 1.0))

    # Extent
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(w * h)
    extent = float(area_px / (bbox_area + 1e-6))
    extent = float(np.clip(extent, 0.0, 1.0))

    # Hu moments
    M = cv2.moments(cnt)
    hu = cv2.HuMoments(M).flatten()
    # Normalizar con log para estabilidad numérica
    hu0 = -np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-10)
    hu0 = float(np.clip(hu0 / 10.0, -1.0, 1.0))  # normalizar a [-1, 1]

    return solidity, extent, hu0


# ─────────────────────────────────────────────
# FEATURES DE TEXTURA
# ─────────────────────────────────────────────

def _texture_features(
    image_gray: np.ndarray,
    mask_u8: np.ndarray,
) -> tuple[float, float]:
    """
    Extrae intensidad media y desviación estándar dentro de la máscara.
    Requiere imagen grayscale float [0, 1].
    """
    pixels = image_gray[mask_u8 > 0]
    if len(pixels) == 0:
        return 0.45, 0.15
    return float(pixels.mean()), float(pixels.std())


# ─────────────────────────────────────────────
# SIMILARITY SEARCH SCHEME
# ─────────────────────────────────────────────

def build_detection_matrix(
    detections: list[Detection],
    scaler: FeatureScaler,
    weights: np.ndarray,
    masks: list[np.ndarray | None] | None = None,
    image_gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construye la matriz D de shape (N_det, FEATURE_DIM) con features
    normalizadas y ponderadas para todas las detecciones del frame actual.

    Args:
        detections : lista de detecciones del frame
        scaler     : normalizador z-score
        weights    : pesos por dimensión
        masks      : lista de máscaras uint8 (una por detección), opcional
        image_gray : imagen grayscale [0,1] para features de textura, opcional

    Returns:
        D : ndarray (N_det, FEATURE_DIM) listo para producto matricial
    """
    if not detections:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    rows = []
    for i, det in enumerate(detections):
        mask = masks[i] if masks else None
        feat = extract_features_from_detection(det, mask_u8=mask, image_gray=image_gray)
        rows.append(feat)

    D_raw = np.stack(rows, axis=0)          # (N_det, F)
    D_norm = scaler.transform(D_raw)        # z-score
    D_weighted = D_norm * weights           # ponderar
    return D_weighted.astype(np.float32)


def compute_similarity_scores(
    track_state: TrackState,
    D: np.ndarray,
    scaler: FeatureScaler,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
    image_gray: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calcula el score de similitud coseno entre el vector del track predicho
    y cada fila de la matriz de detecciones D.

    Score ∈ [-1, 1] → 1 = idéntico, -1 = opuesto.
    En la práctica valores > 0.5 indican buena coincidencia.

    Args:
        track_state : estado predicho del filtro de Kalman
        D           : matriz de detecciones (N_det, F) ya normalizada y ponderada
        scaler      : mismo scaler usado para D
        weights     : mismos pesos usados para D
        mask        : máscara del track (última detección asociada)
        image_gray  : imagen para features de textura

    Returns:
        scores : ndarray (N_det,) de similitudes coseno
    """
    if D.shape[0] == 0:
        return np.array([], dtype=np.float32)

    q_raw = extract_features_from_state(track_state, mask_u8=mask, image_gray=image_gray)
    q_norm = scaler.transform(q_raw)
    q_weighted = q_norm * weights  # (F,)

    # Similitud coseno: (D · q) / (||D|| · ||q||)
    D_norms = np.linalg.norm(D, axis=1) + 1e-8      # (N_det,)
    q_norm_val = float(np.linalg.norm(q_weighted)) + 1e-8

    scores = (D @ q_weighted) / (D_norms * q_norm_val)  # (N_det,)
    return scores.astype(np.float32)


def scores_to_cost_matrix(
    similarity_scores: np.ndarray,
) -> np.ndarray:
    """
    Convierte scores de similitud en costos para Hungarian algorithm.
    cost = 1 - similarity  →  similitud alta = costo bajo.
    """
    return (1.0 - similarity_scores).astype(np.float32)
