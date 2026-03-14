import os
import sys
import re
import random
from pathlib import Path

import numpy as np
from PIL import Image

# =========================================================
# AÑADIR LA RAÍZ DEL PROYECTO AL PATH
# =========================================================
# Esto permite importar desde PIV/... aunque el script esté en Others/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIV.Codes.PreProcessing.filters import (
    load_image,
    apply_preprocessing,
    save_image,
)

# =========================================================
# PARÁMETROS DE PREPROCESAMIENTO POR CÁMARA
# =========================================================
CAM_PREPROCESS_PARAMS = {
    "cam1": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 200,
        "clahe_clip_limit": 0.0010,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 15,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0395,
        "max_intensity": 1.0000,
    },
    "cam2": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 17,
        "clahe_clip_limit": 0.0492,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 14,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0000,
        "max_intensity": 1.0000,
    },
    "cam3": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 10,
        "clahe_clip_limit": 0.1000,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 15,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0000,
        "max_intensity": 1.0000,
    },
    "cam4": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 10,
        "clahe_clip_limit": 0.0100,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 15,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0000,
        "max_intensity": 0.7237,
    },
}


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def detectar_camara_desde_ruta(ruta):
    """
    Detecta el número de cámara dentro de una ruta.
    Acepta patrones como:
        cam-1
        cam_1
        cam1

    Retorna:
        'cam1', 'cam2', etc.
    """
    ruta_str = str(ruta).lower()
    match = re.search(r"cam[-_]?(\d+)", ruta_str)

    if not match:
        raise ValueError(
            f"No se pudo detectar el número de cámara en la ruta: {ruta}"
        )

    cam_key = f"cam{match.group(1)}"

    if cam_key not in CAM_PREPROCESS_PARAMS:
        raise ValueError(
            f"Se detectó '{cam_key}' en la ruta, pero no existe en CAM_PREPROCESS_PARAMS."
        )

    return cam_key


def obtener_bit_depth_original(filepath):
    """
    Determina la profundidad de bits original de la imagen.
    Se usa para guardar el PNG con la misma profundidad que el archivo fuente.

    Retorna:
        8  -> si la imagen original es uint8
        16 -> si la imagen original es uint16
    """
    filepath = Path(filepath)

    with Image.open(filepath) as img:
        img_array = np.array(img)

    if img_array.dtype == np.uint16:
        return 16

    # Todo lo demás se trata como 8 bits para el guardado de salida
    return 8


def copiar_preprocesar_y_convertir_fotos_random(carpeta_origen, carpeta_destino, cantidad):
    """
    Selecciona imágenes aleatorias desde carpeta_origen, detecta la cámara desde la ruta,
    aplica el preprocesamiento correspondiente y guarda cada imagen en PNG manteniendo
    la profundidad de bits original.

    Args:
        carpeta_origen: ruta de la carpeta con imágenes fuente
        carpeta_destino: ruta de la carpeta donde se guardarán los PNG
        cantidad: número de imágenes aleatorias a procesar
    """
    carpeta_origen = Path(carpeta_origen)
    carpeta_destino = Path(carpeta_destino)

    # -----------------------------------------------------
    # Validar existencia de carpeta origen
    # -----------------------------------------------------
    if not carpeta_origen.exists():
        print(f"Error: la carpeta de origen no existe: {carpeta_origen}")
        return

    if not carpeta_origen.is_dir():
        print(f"Error: la ruta de origen no es una carpeta: {carpeta_origen}")
        return

    # -----------------------------------------------------
    # Detectar cámara y parámetros asociados
    # -----------------------------------------------------
    try:
        cam_key = detectar_camara_desde_ruta(carpeta_origen)
        params = CAM_PREPROCESS_PARAMS[cam_key]
    except Exception as e:
        print(f"Error al detectar la cámara: {e}")
        return

    print(f"Cámara detectada: {cam_key}")
    print("Se aplicará el preprocesamiento correspondiente antes de guardar.")

    # -----------------------------------------------------
    # Crear carpeta destino si no existe
    # -----------------------------------------------------
    carpeta_destino.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta destino: {carpeta_destino}")

    # -----------------------------------------------------
    # Buscar imágenes válidas en el nivel superior
    # -----------------------------------------------------
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif')
    todas_las_fotos = [
        archivo for archivo in os.listdir(carpeta_origen)
        if archivo.lower().endswith(extensiones_validas)
    ]

    total_disponible = len(todas_las_fotos)
    print(f"Se encontraron {total_disponible} fotos en el origen.")

    # -----------------------------------------------------
    # Validar cantidad solicitada
    # -----------------------------------------------------
    if total_disponible == 0:
        print("No hay fotos válidas en la carpeta de origen.")
        return

    if cantidad <= 0:
        print("La cantidad de fotos debe ser mayor que cero.")
        return

    if cantidad > total_disponible:
        print(
            f"Ojo: pediste {cantidad} fotos, pero solo hay {total_disponible}. "
            f"Se procesarán todas."
        )
        cantidad = total_disponible

    # -----------------------------------------------------
    # Selección aleatoria sin repetición
    # -----------------------------------------------------
    fotos_seleccionadas = random.sample(todas_las_fotos, cantidad)

    exitosas = 0
    fallidas = 0

    # -----------------------------------------------------
    # Procesar una por una
    # -----------------------------------------------------
    for foto in fotos_seleccionadas:
        ruta_origen = carpeta_origen / foto
        nombre_sin_ext = ruta_origen.stem
        ruta_destino = carpeta_destino / f"{nombre_sin_ext}.png"

        try:
            # 1) Detectar bit depth original
            bit_depth_original = obtener_bit_depth_original(ruta_origen)

            # 2) Cargar imagen original normalizada a [0,1]
            img = load_image(ruta_origen)

            # 3) Aplicar preprocesamiento
            img_procesada = apply_preprocessing(img, params)

            # 4) Guardar como PNG manteniendo bit depth original
            save_image(img_procesada, ruta_destino, bit_depth=bit_depth_original)

            exitosas += 1
            print(
                f"Procesada y guardada -> {ruta_destino.name} "
                f"({bit_depth_original} bits)"
            )

        except Exception as e:
            fallidas += 1
            print(f"Error al procesar la imagen {foto}: {e}")

    # -----------------------------------------------------
    # Resumen final
    # -----------------------------------------------------
    print("\n¡Proceso terminado!")
    print(f"Cantidad solicitada: {cantidad}")
    print(f"Procesadas correctamente: {exitosas}")
    print(f"Con error: {fallidas}")
    print(f"Guardadas en: '{carpeta_destino}'")


# =========================================================
# CONFIGURACIÓN
# =========================================================
RUTA_ORIGEN = r"PIV\Tomas\m93-toma-2-cam-1-n-0000-car-05-piv"
RUTA_DESTINO = r"fotos_random"
NUMERO_DE_FOTOS = 500


# =========================================================
# EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    copiar_preprocesar_y_convertir_fotos_random(
        RUTA_ORIGEN,
        RUTA_DESTINO,
        NUMERO_DE_FOTOS,
    )