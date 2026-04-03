from __future__ import annotations
from pathlib import Path

# ============================================================
# VARIABLES Y PARÁMETROS PTV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- DIRECTORIOS PTV ----------
PTV_BASE_DIR     = PROJECT_ROOT / "PTV" / "Tomas"
RESULTS_PTV_ROOT = PROJECT_ROOT / "ResultadosPTV"
RUNS_SEGMENT_DIR = PROJECT_ROOT / "runs" / "segment"

# ---------- FILTRO PARA SELECCIONAR CARPETAS ----------
PTV_METODO = "ptv"

# ---------- PREPROCESAMIENTO PTV ----------
CAM_PREPROCESS_PARAMS_PTV = {
    'cam1': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 175,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 1.4737,
        'highpass_enabled': False, 'highpass_size': 28,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1316, 'max_intensity': 0.6447,
    },
    'cam2': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 155,
        'clahe_clip_limit': 0.0010,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False, 'highpass_size': 14,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1974, 'max_intensity': 0.8421,
    },
    'cam3': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 159,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False, 'highpass_size': 15,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1184, 'max_intensity': 0.7763,
    },
    'cam4': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 200,
        'clahe_clip_limit': 0.0635,
        'intensity_capping': True,
        'capping_n_std': 3.1053,
        'highpass_enabled': False, 'highpass_size': 15,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1053, 'max_intensity': 0.8289,
    },
}

# ---------- MODELO YOLO TRACKING ----------
YOLO_TRACK_MODEL = PROJECT_ROOT / "PTV" / "Codes" / "Segmentation-Models" / "best.pt"
DEVICE_PTV = 0   # 0 = cuda:0 | "cpu" = sin GPU

# ---------- PARÁMETROS PTV ----------
MAX_IMAGES      = 2200
ALPHA           = 0.95   # corrección de posición (filtro ABG)
BETA            = 0.95   # corrección de velocidad
GAMMA           = 0.05   # corrección de aceleración
CONF_TRACK      = 0.1    # confianza mínima del detector YOLO
MIN_FRAMES_KEEP = 5      # mínimo de frames para exportar un track
ANNOTATE        = True
MAX_MISSES      = 2      # 0 = track termina inmediatamente si no se detecta
                         # 1-3 = tolera N frames sin detección (oclusiones)

# ---------- GUARDAR IMÁGENES ----------
# True  → guarda carpetas annotations/ (detecciones) y tracks/ (trayectorias)
# False → no guarda ninguna imagen, solo CSV/JSON y el gráfico interactivo
SAVE_IMAGES = True

# ---------- GATE ESPACIAL (referencia para annotate_frame) ----------
GATE_X     = 10
GATE_Y     = 10
GATE_ANGLE = 5

# ---------- DIMENSIONES DE LA FIBRA ----------
# Usadas para normalizar el vector de similitud.
# px_per_mm viene de CAM_PROFILES en pipeline_global.py según la cámara.
FIBER_LENGTH_MM = 13.0   # largo real de la fibra en mm
FIBER_WIDTH_MM  = 0.2    # ancho real de la fibra en mm

# ---------- PESOS DEL VECTOR DE SIMILITUD ----------
# Vector: [w1·cos(2θ), w2·sin(2θ), w3·L/L_ref, w4·cx/W, w5·cy/H]
# Usar cos(2θ)/sin(2θ) resuelve la simetría bidireccional:
#   fibra a 0° y la misma a 180° → mismo vector.
#
# Pesos recomendados: posición > ángulo > largo
#   - (cx, cy) tienen el doble de peso → la posición es el criterio principal
#   - cos2θ, sin2θ tienen peso 1.0    → ángulo secundario
#   - largo tiene poco peso (0.3)     → varía según perspectiva de la fibra
FEAT_WEIGHTS = (1.0, 1.0, 0.3, 2.0, 2.0)
# (w_cos2θ, w_sin2θ, w_largo, w_cx, w_cy)

# ---------- PARÁMETROS DE SIMILITUD Y GATE — POR CÁMARA ----------
# sim_threshold : similitud coseno mínima para aceptar un match [0, 1]
#   Bajar si hay pocos tracks (0.65-0.75), subir para matches más estrictos.
#
# max_dist_mm   : distancia máxima entre centroides EN MM para considerar match.
#   Se convierte a píxeles automáticamente en pipeline_global usando px_per_mm.
#   Ajustar según desplazamiento máximo esperado de la fibra entre frames.
#   A 220 fps las fibras se mueven poco — 2-5 mm suele ser suficiente.
#   A 660 fps (cam4) el movimiento es aún menor.

CAM_TRACKING_PARAMS = {
    1: dict(sim_threshold=0.99, max_dist_mm=2.0),
    2: dict(sim_threshold=0.99, max_dist_mm=2.0),
    3: dict(sim_threshold=0.99, max_dist_mm=2.0),
    4: dict(sim_threshold=0.99, max_dist_mm=2.0),   # 660 fps → menos movimiento
}

# ---------- SAHI INFERENCE ----------
# Deben coincidir con los parámetros usados en entrenamiento (train_yolo26.py)
SAHI_SCALE_FACTOR  = 2      # upscale ×4 antes de tilear
SAHI_TILE_SIZE     = 640    # tamaño del tile en px
SAHI_OVERLAP_RATIO = 0.1    # solapamiento entre tiles
SAHI_IOU_THRESHOLD = 0.3    # NMS entre tiles solapados

# ---------- VISUALIZADOR EN TIEMPO REAL ----------
VIZ_TAIL_LENGTH  = 0    # 0 = trayectoria completa; N = últimos N frames
VIZ_UPDATE_EVERY = 1    # refrescar cada N frames (subir a 5 si va lento)

# ---------- CLEANUP ----------
DELETE_PREDICT_FOLDERS = False


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_tracking_params(cam: int) -> dict:
    """
    Retorna sim_threshold y max_dist_mm para una cámara.
    Fallback a cam1 si la cámara no está definida.
    """
    return CAM_TRACKING_PARAMS.get(cam, CAM_TRACKING_PARAMS[1])


def get_l_ref_px(cam: int, cam_profiles: dict) -> float:
    """
    Calcula el largo de referencia en píxeles para normalizar el vector de similitud.
    Usa px_per_mm de CAM_PROFILES y FIBER_LENGTH_MM.
    """
    px_per_mm = cam_profiles.get(cam, {}).get("px_per_mm", 7.8)
    return FIBER_LENGTH_MM * px_per_mm


def get_max_dist_px(cam: int, cam_profiles: dict) -> float:
    """
    Convierte max_dist_mm a píxeles usando px_per_mm de la cámara.
    """
    px_per_mm  = cam_profiles.get(cam, {}).get("px_per_mm", 7.8)
    max_dist_mm = get_tracking_params(cam)["max_dist_mm"]
    return max_dist_mm * px_per_mm