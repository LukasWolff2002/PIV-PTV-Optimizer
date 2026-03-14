"""
PIV/Codes/PreProcessing/blocks.py

Muestreo por bloques con preprocesamiento opcional de imágenes PIV
"""

from __future__ import annotations
from pathlib import Path
import os, re, shutil
from typing import List, Optional, Dict

# Importar funciones de preprocesamiento
try:
    import numpy as np
    from PIL import Image
    from .filters import load_image, apply_preprocessing, save_image
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
    print("[WARN] piv_preprocessing no disponible, preprocesamiento deshabilitado")


def natural_key(s: str):
    """Clave de ordenamiento natural (números ordenados correctamente)"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_sorted_files(folder: Path, natural_sort: bool) -> List[Path]:
    """
    Listar archivos de una carpeta ordenados
    
    Args:
        folder: carpeta a listar
        natural_sort: True para ordenamiento natural (1, 2, 10 en vez de 1, 10, 2)
    
    Returns:
        Lista de archivos ordenados
    """
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=(lambda p: natural_key(p.name)) if natural_sort else (lambda p: p.name))
    return files

def obtener_bit_depth_original(filepath: Path) -> int:
    """
    Determina la profundidad de bits original de la imagen.

    Retorna:
        8  -> si la imagen original es uint8
        16 -> si la imagen original es uint16
    """
    with Image.open(filepath) as img:
        img_array = np.array(img)

    if img_array.dtype == np.uint16:
        return 16

    return 8


def run_block_sampling(
    input_dir: Path,
    output_dir: Path,
    blocks: Optional[int],
    block_size: int,
    skip_inter: int,
    skip_final: int,
    delete_existing: bool,
    natural_sort: bool = True,
    overwrite: bool = True,
    preprocess_params: Optional[Dict] = None,
) -> None:
    """
    Muestrear imágenes por bloques con preprocesamiento opcional
    
    Args:
        input_dir: directorio con imágenes originales
        output_dir: directorio de salida
        blocks: número de bloques a procesar (None = todos los posibles)
        block_size: tamaño de cada bloque
        skip_inter: número de imágenes a saltar entre las 2 seleccionadas del bloque
        skip_final: número de imágenes a saltar al final de cada bloque
        delete_existing: eliminar directorio de salida si existe
        natural_sort: ordenamiento natural de archivos
        overwrite: sobrescribir archivos existentes
        preprocess_params: diccionario con parámetros de preprocesamiento PIV
                          Si es None, hace copia directa (sin preprocesamiento)
    
    Regla de bloque:
        2 (imágenes seleccionadas) + skip_inter + skip_final = block_size
    
    Ejemplo con block_size=22, skip_inter=1, skip_final=19:
        - Imagen 1: índice i
        - Imagen 2: índice i+2 (salta 1 imagen intermedia)
        - Salto final: 19 imágenes
        - Total: 2 + 1 + 19 = 22
    
    Preprocesamiento:
        Si preprocess_params no es None:
            - Carga imagen original
            - Aplica filtros (CLAHE, Highpass, Intensity Capping, etc.)
            - Guarda imagen procesada en output_dir
        Si preprocess_params es None:
            - Copia directa sin modificar (comportamiento original)
    """
    # Validaciones
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe input_dir: {input_dir}")

    # Eliminar carpeta de salida si se solicita
    if output_dir.exists() and delete_existing:
        shutil.rmtree(output_dir)

    # Crear carpeta de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Listar y ordenar archivos
    files = list_sorted_files(input_dir, natural_sort)
    total = len(files)
    
    if total == 0:
        raise RuntimeError(f"No hay archivos en: {input_dir}")

    # Validar regla de bloque
    if (2 + skip_inter + skip_final) != block_size:
        raise ValueError(
            f"Regla de bloque inválida: 2 + {skip_inter} + {skip_final} != {block_size}"
        )

    # Calcular bloques
    bloques_posibles = total // block_size
    blocks_max = bloques_posibles if blocks is None else min(int(blocks), bloques_posibles)

    # Determinar si hay preprocesamiento
    apply_preprocess = (preprocess_params is not None) and PREPROCESS_AVAILABLE
    
    if apply_preprocess:
        print(f"[PRE] Preprocesamiento HABILITADO para {input_dir.name}")
    else:
        if preprocess_params is not None and not PREPROCESS_AVAILABLE:
            print(f"[WARN] Preprocesamiento solicitado pero módulo no disponible")
        print(f"[PRE] Preprocesamiento DESHABILITADO - copia directa")

    # Procesar bloques
    i = 0
    hechos = 0
    
    while (i + block_size) <= total and hechos < blocks_max:
        # Índices de las 2 imágenes a seleccionar del bloque
        idx1 = i
        idx2 = i + 1 + skip_inter

        for idx in (idx1, idx2):
            src = files[idx]
            dst = output_dir / f"{src.stem}.png"
            
            # Saltar si existe y no se permite sobrescribir
            if dst.exists() and not overwrite:
                continue
            
            # PREPROCESAMIENTO O COPIA DIRECTA
            if apply_preprocess:
                try:
                    # 1) Detectar bit depth original
                    bit_depth_original = obtener_bit_depth_original(src)

                    # 2) Cargar imagen original normalizada a [0,1]
                    img = load_image(src)

                    # 3) Aplicar preprocesamiento
                    img_procesada = apply_preprocessing(img, preprocess_params)

                    # 4) Guardar como PNG manteniendo bit depth original
                    save_image(img_procesada, dst, bit_depth=bit_depth_original)

                except Exception as e:
                    print(f"[ERROR] Fallo preprocesamiento en {src.name}: {e}")

            else:
                try:
                    # Si no hay preprocesamiento, igual convertir a PNG
                    bit_depth_original = obtener_bit_depth_original(src)
                    img = load_image(src)
                    save_image(img, dst, bit_depth=bit_depth_original)

                except Exception as e:
                    print(f"[ERROR] Fallo conversión en {src.name}: {e}")

        # Avanzar al siguiente bloque
        i += block_size
        hechos += 1

    # Resumen
    modo = "preprocesadas" if apply_preprocess else "copiadas"
    print(f"[PRE] {input_dir.name}: {hechos}/{blocks_max} bloques, {hechos*2} imágenes {modo} → {output_dir}")


# ============================================================================
# FUNCIÓN DE AYUDA PARA DEBUG/TEST
# ============================================================================

def validate_preprocess_params(params: Optional[Dict]) -> bool:
    """
    Validar que el diccionario de parámetros tenga las claves esperadas
    
    Args:
        params: diccionario a validar
    
    Returns:
        True si es válido o None, False si falta alguna clave
    """
    if params is None:
        return True
    
    required_keys = [
        'roi_enabled', 'clahe_enabled', 'intensity_capping',
        'highpass_enabled', 'wiener_enabled',
        'min_intensity', 'max_intensity'
    ]
    
    missing = [k for k in required_keys if k not in params]
    
    if missing:
        print(f"[WARN] Parámetros faltantes: {missing}")
        return False
    
    return True


# ============================================================================
# EJEMPLO DE USO (para testing)
# ============================================================================

if __name__ == "__main__":
    # Ejemplo de parámetros de preprocesamiento
    example_params = {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 50,
        'clahe_clip_limit': 0.01,
        'intensity_capping': False,
        'capping_n_std': 2.0,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0,
        'max_intensity': 1.0,
    }
    
    # Validar parámetros
    if validate_preprocess_params(example_params):
        print("✓ Parámetros válidos")
    
    # Ejemplo de uso (descomentar para probar)
    """
    run_block_sampling(
        input_dir=Path("ruta/entrada"),
        output_dir=Path("ruta/salida"),
        blocks=50,
        block_size=22,
        skip_inter=1,
        skip_final=19,
        delete_existing=True,
        natural_sort=True,
        overwrite=True,
        preprocess_params=example_params,  # Con preprocesamiento
        # preprocess_params=None,          # Sin preprocesamiento (copia directa)
    )
    """