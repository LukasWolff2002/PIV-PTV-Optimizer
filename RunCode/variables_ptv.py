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
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 175,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 1.4737,
        'highpass_enabled': False,
        'highpass_size': 28,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1316,
        'max_intensity': 0.6447,
    },
    'cam2': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 155,
        'clahe_clip_limit': 0.0010,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 14,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1974,
        'max_intensity': 0.8421,
    },
    'cam3': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 159,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1184,
        'max_intensity': 0.7763,
    },
    'cam4': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 200,
        'clahe_clip_limit': 0.0635,
        'intensity_capping': True,
        'capping_n_std': 3.1053,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1053,
        'max_intensity': 0.8289,
    },
}

# ---------- MODELO YOLO TRACKING ----------
YOLO_TRACK_MODEL = PROJECT_ROOT / "PTV" / "Codes" / "Segmentation-Models" / "best.pt"
DEVICE_PTV = 0   # 0 = cuda:0, "cpu" = sin GPU

# ---------- PARÁMETROS PTV (FILTRO ABG) ----------
MAX_IMAGES      = 10
ALPHA           = 0.95
BETA            = 0.95
GAMMA           = 0.05
CONF_TRACK      = 0.1
MIN_FRAMES_KEEP = 20
ANNOTATE        = True

# ---------- GATE ESPACIAL (usado como salvaguarda en similarity search) ----------
# GATE_X y GATE_Y se mantienen por compatibilidad con annotate_frame,
# pero el tracker ya no los usa como filtro secuencial.
GATE_X     = 10
GATE_Y     = 10
GATE_ANGLE = 5

# ---------- SIMILARITY SEARCH SCHEME ----------
# Vector de características por fibra: [cos(2θ), sin(2θ), L/L_ref, cx/W, cy/H]
# Usar cos(2θ)/sin(2θ) en vez de cos(θ)/sin(θ) resuelve la simetría bidireccional:
# una fibra a 0° y la misma a 180° producen el mismo vector.

SIM_THRESHOLD = 0.85
# Similitud coseno mínima para aceptar un match.
# Rango [0, 1]. Subir para matches más estrictos, bajar si hay pocos tracks.

MAX_DIST_PX = 80.0
# Gate espacial duro (distancia Euclidiana, píxeles).
# Candidatos más lejos que este valor se rechazan independiente de la similitud.
# Ajustar según la velocidad máxima esperada de las fibras entre frames.

FEAT_WEIGHTS = (1.0, 1.0, 0.5, 1.5, 1.5)
# Pesos del vector de características: (w_cos2θ, w_sin2θ, w_largo, w_cx, w_cy).
# Pesos mayores = más influencia en la similitud.
# Subir w_cx / w_cy para priorizar posición; subir w_cos/w_sin para priorizar ángulo.

L_REF_PX = 101.4
# Largo de referencia para normalizar length_px.
# Default: 13 mm × 7.8 px/mm = 101.4 px (fibra completa a escala original).

# ---------- CLEANUP ----------
DELETE_PREDICT_FOLDERS = False

# ---------- VISUALIZADOR EN TIEMPO REAL ----------
VIZ_TAIL_LENGTH  = 0    # 0 = trayectoria completa, N = últimos N frames
VIZ_UPDATE_EVERY = 1    # refrescar visualizador cada N frames (1 = siempre)

# ---------- SAHI INFERENCE ----------
SAHI_SCALE_FACTOR  = 4
SAHI_TILE_SIZE     = 640
SAHI_OVERLAP_RATIO = 0.5
SAHI_IOU_THRESHOLD = 0.3

# ---------- VISUALIZADOR ----------
VIZ_TAIL_LENGTH = 0   # 0 = trayectoria completa; N = últimos N frames visibles
