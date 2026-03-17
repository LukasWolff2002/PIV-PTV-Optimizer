from __future__ import annotations
from pathlib import Path
from PIV.Codes.PreProcessing.temporal_regions import TemporalRegion

# ============================================================
# VARIABLES Y PARÁMETROS PIV
# ============================================================

# Obtener la raíz del proyecto (asumiendo que este archivo está en RunCode/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- DIRECTORIOS PIV ----------
PRE_BASE_DIR = PROJECT_ROOT / "PIV" / "Tomas"
PROCESSED_ROOT = PROJECT_ROOT / "TomasProcesadas"
MASKS_ROOT = PROJECT_ROOT / "Masks"
RESULTS_PIV_ROOT = PROJECT_ROOT / "ResultadosPIV"
PIV_MODELS_DIR = PROJECT_ROOT / "PIV" / "Codes" / "Segmentation-Models"

# ---------- FILTRO PARA SELECCIONAR CARPETAS ----------
PIV_METODO = "piv"  # Filtra carpetas que terminen en "-piv"

# ---------- PREPROCESAMIENTO ----------
DELETE_EXISTING_PRE = True

# Parámetros de preprocesamiento por cámara (NO MODIFICAR FRECUENTEMENTE)
CAM_PREPROCESS_PARAMS = {
    'cam1': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 200,
        'clahe_clip_limit': 0.0010,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0395,
        'max_intensity': 1.0000,
    },
    'cam2': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 17,
        'clahe_clip_limit': 0.0492,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 14,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0000,
        'max_intensity': 1.0000,
    },
    'cam3': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 10,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0000,
        'max_intensity': 1.0000,
    },
    'cam4': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 10,
        'clahe_clip_limit': 0.0100,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0000,
        'max_intensity': 0.7237,
    },
}

# ---------- REGIONES TEMPORALES ----------
USE_TEMPORAL_REGIONS = True  # True = adaptativo, False = legacy

# Regiones temporales para CARBOPOL 02
TEMPORAL_REGIONS_CAR02 = {
    1: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=6.5,
            block_size=11,
            skip_inter=0,
            skip_final=9,
            fps=220.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=6.5,
            end_time=10.0,
            block_size=22,
            skip_inter=2,
            skip_final=18,
            fps=220.0
        ),
        TemporalRegion(
            name="baja_velocidad",
            start_time=10.0,
            end_time=20.0,
            block_size=22,
            skip_inter=8,
            skip_final=12,
            fps=220.0
        ),
    ],
    2: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=1.5,
            block_size=11,
            skip_inter=0,
            skip_final=9,
            fps=220.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=1.5,
            end_time=3.0,
            block_size=22,
            skip_inter=2,
            skip_final=18,
            fps=220.0
        ),
        TemporalRegion(
            name="baja_velocidad",
            start_time=3.0,
            end_time=6.0,
            block_size=22,
            skip_inter=4,
            skip_final=16,
            fps=220.0
        ),
        TemporalRegion(
            name="muy_baja_velocidad",
            start_time=6.0,
            end_time=20.0,
            block_size=22,
            skip_inter=8,
            skip_final=12,
            fps=220.0
        ),
    ],
    3: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=2.0,
            block_size=11,
            skip_inter=0,
            skip_final=9,
            fps=220.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=2.0,
            end_time=5.0,
            block_size=22,
            skip_inter=2,
            skip_final=18,
            fps=220.0
        ),
        TemporalRegion(
            name="baja_velocidad",
            start_time=5.0,
            end_time=7.0,
            block_size=22,
            skip_inter=4,
            skip_final=16,
            fps=220.0
        ),
        TemporalRegion(
            name="muy_baja_velocidad",
            start_time=7.0,
            end_time=20.0,
            block_size=22,
            skip_inter=8,
            skip_final=12,
            fps=220.0
        ),
    ],
    4: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=4.5,
            block_size=22,
            skip_inter=0,
            skip_final=20,
            fps=660.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=4.5,
            end_time=8.0,
            block_size=22,
            skip_inter=2,
            skip_final=18,
            fps=660.0
        ),
        TemporalRegion(
            name="baja_velocidad",
            start_time=8,
            end_time=30.0,
            block_size=22,
            skip_inter=8,
            skip_final=12,
            fps=660.0
        ),
    ],
}

# Regiones temporales para CARBOPOL 05
TEMPORAL_REGIONS_CAR05 = {
    1: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=0.2,
            block_size=11,
            skip_inter=0,
            skip_final=9,
            fps=220.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=0.2,
            end_time=5.0,
            block_size=22,
            skip_inter=1,
            skip_final=19,
            fps=220.0
        ),
        TemporalRegion(
            name="baja_velocidad",
            start_time=5.0,
            end_time=8.0,
            block_size=22,
            skip_inter=2,
            skip_final=18,
            fps=220.0
        ),
        TemporalRegion(
            name="muy_baja_velocidad",
            start_time=8.0,
            end_time=15.0,
            block_size=22,
            skip_inter=4,
            skip_final=16,
            fps=220.0
        ),
        TemporalRegion(
            name="extrema_baja_velocidad",
            start_time=15.0,
            end_time=30.0,
            block_size=44,
            skip_inter=8,
            skip_final=34,
            fps=220.0
        ),
        
    ],
    2: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=15,
            block_size=22,
            skip_inter=8,
            skip_final=12,
            fps=220.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=15.0,
            end_time=30,
            block_size=22,
            skip_inter=16,
            skip_final=4,
            fps=220.0
        ),
    ],
    3: [
        TemporalRegion(
            name="sin_datos",
            start_time=0.0,
            end_time=8.0,
            block_size=220,
            skip_inter=0,
            skip_final=218,
            fps=220.0
        ),
        TemporalRegion(
            name="alta_velocidad",
            start_time=8.0,
            end_time=30.0,
            block_size=22,
            skip_inter=16,
            skip_final=4,
            fps=220.0
        ),
    ],
    4: [
        TemporalRegion(
            name="alta_velocidad",
            start_time=0.0,
            end_time=3,
            block_size=11,
            skip_inter=0,
            skip_final=9,
            fps=660.0
        ),
        TemporalRegion(
            name="media_velocidad",
            start_time=3,
            end_time=6,
            block_size=22,
            skip_inter=2,
            skip_final=18,
            fps=660.0
        ),
    ],
}


# ---------- PARÁMETROS PIV POR CÁMARA Y CARBOPOL ----------

# Parámetros PIV para CARBOPOL 02
PIV_PARAMS_CAR02 = {
    1: {
        'window_sizes': [128, 64, 32],
        'overlaps': [64, 32, 16],
        'keep_percentile': 95.0,
    },
    2: {
        'window_sizes': [128, 64, 32],
        'overlaps': [64, 32, 16],
        'keep_percentile': 95.0,
    },
    3: {
        'window_sizes': [64, 32, 16],
        'overlaps': [32, 16, 8],
        'keep_percentile': 85.0,
    },
    4: {
        'window_sizes': [64, 32, 16],
        'overlaps': [32, 16, 8],
        'keep_percentile': 90.0,
    },
}

# Parámetros PIV para CARBOPOL 05
PIV_PARAMS_CAR05 = {
    1: {
        'window_sizes': [64, 32, 16],
        'overlaps': [32, 16, 8],
        'keep_percentile': 90.0,
    },
    2: {
        'window_sizes': [64, 32, 16],
        'overlaps': [32, 16, 8],
        'keep_percentile': 92.0,
    },
    3: {
        'window_sizes': [64, 32, 16],
        'overlaps': [32, 16, 8],
        'keep_percentile': 92.0,
    },
    4: {
        'window_sizes': [64, 32, 16],
        'overlaps': [32, 16, 8],
        'keep_percentile': 88.0,
    },
}


# Función helper para obtener parámetros PIV según cámara y carbopol
def get_piv_params(cam: int, carbopol: str) -> dict:
    """
    Retorna los parámetros PIV para una cámara y tipo de carbopol.
    
    Args:
        cam: Número de cámara (1-4)
        carbopol: "02", "2", "05", "5", etc.
    
    Returns:
        Dict con window_sizes, overlaps, keep_percentile
    """
    # Normalizar carbopol a formato "02" o "05"
    carbopol_normalized = carbopol.zfill(2)
    
    # Valores por defecto
    default_params = {
        'window_sizes': [128, 64, 32],
        'overlaps': [64, 32, 16],
        'keep_percentile': 95.0,
    }
    
    if carbopol_normalized == "02":
        return PIV_PARAMS_CAR02.get(cam, default_params)
    elif carbopol_normalized == "05":
        return PIV_PARAMS_CAR05.get(cam, default_params)
    
    return default_params


# Función helper para obtener regiones según carbopol
def get_temporal_regions(cam: int, carbopol: str) -> list[TemporalRegion] | None:
    """
    Retorna las regiones temporales para una cámara y tipo de carbopol.
    
    Args:
        cam: Número de cámara (1-4)
        carbopol: "02", "2", "05", "5", etc.
    
    Returns:
        Lista de TemporalRegion o None si no existe configuración
    """
    # Normalizar carbopol a formato "02" o "05"
    carbopol_normalized = carbopol.zfill(2)
    
    if carbopol_normalized == "02":
        return TEMPORAL_REGIONS_CAR02.get(cam)
    elif carbopol_normalized == "05":
        return TEMPORAL_REGIONS_CAR05.get(cam)
    return None


# ---------- PARÁMETROS LEGACY (solo si USE_TEMPORAL_REGIONS = False) ----------
BLOCKS = None  # None = todos los bloques posibles
BLOCK_SIZE = 22
SKIP_INTER = 0
SKIP_FINAL = 20

# ---------- PARÁMETROS PIV LEGACY (se sobrescriben con get_piv_params) ----------
WINDOW_SIZES = [128, 64, 32, 16]  # Valores por defecto, se sobrescriben dinámicamente
OVERLAPS = [64, 32, 16, 8]       # Valores por defecto, se sobrescriben dinámicamente
KEEP_PERCENTILE = 95.0        # Valor por defecto, se sobrescribe dinámicamente

# ---------- MÁSCARAS DINÁMICAS (YOLO) ----------
MASK_CONF = 0.25
MASK_DEVICE = "0"
INVERT_MASK = True
DELETE_EXISTING_MASKS = True

# ---------- PARÁMETROS PIV FIJOS ----------
SEARCH_AREA_FACTOR = 1
SIG2NOISE_METHOD = "peak2peak"
MASK_THRESHOLD = 0.0
LM_KERNEL = 1
LM_THRESH = 3.0
LM_EPS = 0.1
SHOW_VIEWERS = True
CLEAR_TXT = True

# ---------- CLEANUP ----------
DELETE_PROCESSED_SUBFOLDERS = True