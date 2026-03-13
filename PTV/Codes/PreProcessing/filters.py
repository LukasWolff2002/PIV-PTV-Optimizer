"""
PIV/Codes/PreProcessing/piv_preprocessing.py

Funciones de preprocesamiento para imágenes PIV
Basado en técnicas implementadas en PIVlab
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path


# ============================================================================
# FUNCIONES DE CARGA/GUARDADO
# ============================================================================

def load_image(filepath):
    """
    Cargar y normalizar imagen a rango 0-1
    
    Args:
        filepath: ruta al archivo de imagen (Path o str)
    
    Returns:
        numpy array (float64, rango 0-1)
    """
    filepath = Path(filepath)
    img = Image.open(filepath)
    img_array = np.array(img)
    
    # Convertir a grayscale si es RGB
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Normalizar según tipo de dato
    if img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float64) / 255.0
    elif img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float64) / 65535.0
    else:
        img_array = img_array.astype(np.float64)
        if img_array.max() > 1.0:
            img_array = img_array / img_array.max()
    
    return img_array


def save_image(image, filepath, bit_depth=16):
    """
    Guardar imagen procesada
    
    Args:
        image: numpy array (float64, rango 0-1)
        filepath: ruta donde guardar (Path o str)
        bit_depth: 8 o 16 bits
    """
    filepath = Path(filepath)
    
    if bit_depth == 8:
        img_save = (image * 255).astype(np.uint8)
    else:
        img_save = (image * 65535).astype(np.uint16)
    
    Image.fromarray(img_save).save(filepath)


# ============================================================================
# FUNCIONES DE FILTROS INDIVIDUALES
# ============================================================================

def apply_histogram_adjustment(img, min_intensity, max_intensity):
    """
    Ajuste de histograma (imadjust equivalente)
    
    Args:
        img: imagen (float64, 0-1)
        min_intensity: intensidad mínima (0-1)
        max_intensity: intensidad máxima (0-1)
    
    Returns:
        Imagen ajustada
    """
    if min_intensity < max_intensity and (min_intensity > 0 or max_intensity < 1):
        return np.clip((img - min_intensity) / (max_intensity - min_intensity), 0, 1)
    return img


def apply_intensity_capping(img, n_std):
    """
    Intensity Capping - Limitar puntos brillantes (Shavit et al.)
    
    Args:
        img: imagen (float64, 0-1)
        n_std: número de desviaciones estándar
    
    Returns:
        Imagen con capping aplicado
    """
    upper_limit = np.median(img) + n_std * np.std(img)
    result = np.clip(img, 0, upper_limit)
    
    # Renormalizar
    if result.max() > 0:
        result = result / result.max()
    
    return result


def apply_clahe(img, tile_size, clip_limit):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization
    
    Args:
        img: imagen (float64, 0-1)
        tile_size: tamaño de tiles en píxeles
        clip_limit: límite de clip
    
    Returns:
        Imagen con CLAHE aplicado
    """
    # Convertir a uint8
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Calcular grid de tiles
    h, w = img.shape
    grid_h = max(2, int(h / tile_size))
    grid_w = max(2, int(w / tile_size))
    
    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_w, grid_h))
    result = clahe.apply(img_uint8).astype(np.float64) / 255.0
    
    return result


def apply_highpass(img, kernel_size):
    """
    Filtro Highpass - Eliminar componentes de baja frecuencia
    
    Args:
        img: imagen (float64, 0-1)
        kernel_size: tamaño del kernel gaussiano
    
    Returns:
        Imagen con highpass aplicado
    """
    # Asegurar tamaño impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Filtro gaussiano para baja frecuencia
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), kernel_size/3)
    
    # Restar baja frecuencia
    result = img - blurred
    
    # Normalizar
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    
    return result


def apply_wiener(img, wiener_size, gaussian_size):
    """
    Filtro Wiener + Gaussian - Reducción de ruido
    
    Args:
        img: imagen (float64, 0-1)
        wiener_size: tamaño ventana wiener
        gaussian_size: tamaño kernel gaussian
    
    Returns:
        Imagen filtrada
    """
    # Asegurar tamaños impares
    if wiener_size % 2 == 0:
        wiener_size += 1
    if gaussian_size % 2 == 0:
        gaussian_size += 1
    
    # Convertir a uint8 para Wiener
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Aplicar filtro Non-local Means (aproximación a Wiener)
    result = cv2.fastNlMeansDenoising(img_uint8, None, h=10,
                                     templateWindowSize=wiener_size,
                                     searchWindowSize=wiener_size*2+1)
    result = result.astype(np.float64) / 255.0
    
    # Aplicar Gaussian
    result = cv2.GaussianBlur(result, (gaussian_size, gaussian_size), gaussian_size/2)
    
    return result


def apply_roi(img, x, y, width, height):
    """
    Extraer Region of Interest (ROI)
    
    Args:
        img: imagen (float64, 0-1)
        x, y: coordenadas esquina superior izquierda
        width, height: dimensiones del ROI
    
    Returns:
        ROI extraído
    """
    h, w = img.shape
    
    # Validar límites
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    return img[y:y+height, x:x+width]


# ============================================================================
# FUNCIÓN PRINCIPAL DE PREPROCESAMIENTO
# ============================================================================

def apply_preprocessing(img, params):
    """
    Aplicar todos los filtros de preprocesamiento según parámetros
    
    Args:
        img: imagen original (float64, 0-1)
        params: diccionario con parámetros de preprocesamiento
    
    Returns:
        Imagen procesada (float64, 0-1)
    
    Parámetros esperados en dict:
        - roi_enabled: bool
        - roi_x, roi_y, roi_width, roi_height: int
        - min_intensity, max_intensity: float (0-1)
        - intensity_capping: bool
        - capping_n_std: float
        - clahe_enabled: bool
        - clahe_tile_size: int
        - clahe_clip_limit: float
        - highpass_enabled: bool
        - highpass_size: int
        - wiener_enabled: bool
        - wiener_size, gaussian_size: int
    """
    if params is None:
        return img
    
    result = img.copy()
    
    # 0. ROI (si está habilitado)
    if params.get('roi_enabled', False):
        x = int(params.get('roi_x', 0))
        y = int(params.get('roi_y', 0))
        width = int(params.get('roi_width', 100))
        height = int(params.get('roi_height', 100))
        roi = apply_roi(result, x, y, width, height)
    else:
        roi = result
        x, y = 0, 0
    
    # 1. Ajuste de histograma
    min_i = params.get('min_intensity', 0.0)
    max_i = params.get('max_intensity', 1.0)
    if min_i > 0 or max_i < 1.0:
        roi = apply_histogram_adjustment(roi, min_i, max_i)
    
    # 2. Intensity Capping
    if params.get('intensity_capping', False):
        roi = apply_intensity_capping(roi, params.get('capping_n_std', 2.0))
    
    # 3. CLAHE
    if params.get('clahe_enabled', False):
        roi = apply_clahe(
            roi,
            int(params.get('clahe_tile_size', 50)),
            params.get('clahe_clip_limit', 0.01)
        )
    
    # 4. Highpass
    if params.get('highpass_enabled', False):
        roi = apply_highpass(roi, int(params.get('highpass_size', 15)))
    
    # 5. Wiener
    if params.get('wiener_enabled', False):
        roi = apply_wiener(
            roi,
            int(params.get('wiener_size', 3)),
            int(params.get('gaussian_size', 3))
        )
    
    # Reconstruir imagen completa si se usó ROI
    if params.get('roi_enabled', False):
        result[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        return result
    
    return roi


# ============================================================================
# FUNCIÓN DE CONVENIENCIA PARA PIPELINE
# ============================================================================

def preprocess_image_file(input_path, output_path, params, bit_depth=16):
    """
    Cargar, preprocesar y guardar imagen en un solo paso
    
    Args:
        input_path: ruta imagen original
        output_path: ruta imagen procesada
        params: diccionario con parámetros
        bit_depth: 8 o 16 bits para guardar
    
    Returns:
        True si se procesó, False si params es None
    """
    if params is None:
        return False
    
    # Cargar
    img = load_image(input_path)
    
    # Procesar
    img_processed = apply_preprocessing(img, params)
    
    # Guardar
    save_image(img_processed, output_path, bit_depth=bit_depth)
    
    return True