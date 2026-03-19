from __future__ import annotations
from pathlib import Path

# ============================================================
# VARIABLES Y PARÁMETROS PTV
# ============================================================

# Obtener la raíz del proyecto (asumiendo que este archivo está en RunCode/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- DIRECTORIOS PTV ----------
PTV_BASE_DIR = PROJECT_ROOT / "PTV" / "Tomas"
RESULTS_PTV_ROOT = PROJECT_ROOT / "ResultadosPTV"
RUNS_SEGMENT_DIR = PROJECT_ROOT / "runs" / "segment"

# ---------- FILTRO PARA SELECCIONAR   CARPETAS ----------
PTV_METODO = "ptv"  # Filtra carpetas que terminen en "-ptv"

# ---------- PREPROCESAMIENTO PTV ----------
# Parámetros de preprocesamiento por cámara (NO MODIFICAR FRECUENTEMENTE)

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

# ---------- PARÁMETROS PTV ----------
MAX_IMAGES = 100
ALPHA = 0.95
BETA = 0.95
GAMMA = 0.05
GATE_X = 10
GATE_Y = 10
GATE_ANGLE = 5
CONF_TRACK = 0.1
MIN_FRAMES_KEEP = 20
ANNOTATE = True

# ---------- CLEANUP ----------
DELETE_PREDICT_FOLDERS = False