"""
PIV/Codes/OpenPIV/naming.py

Funciones para generar nombres de archivos PIV con metadata temporal
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import re


def extract_timestamp_from_filename(filename: str) -> float | None:
    """
    Extraer timestamp del nombre de archivo procesado
    
    Formato esperado: {nombre}_r{region}b{bloque}s{skip}{ext}
    
    Calcula el timestamp basado en:
    - Índice original del frame en la secuencia completa
    - FPS de la cámara
    
    Args:
        filename: Nombre del archivo (ej: "img_0220_r2b001s1.tiff")
    
    Returns:
        Timestamp en segundos, o None si no se puede extraer
    """
    # Intentar extraer índice original del nombre base
    # Asume formato: {prefix}_{índice}_r{region}b{bloque}s{skip}{ext}
    
    # Patrón: buscar el número antes del sufijo _r{region}
    pattern = r'_(\d+)_r\d+b\d+s\d+'
    match = re.search(pattern, filename)
    
    if match:
        frame_idx = int(match.group(1))
        return frame_idx  # Retorna índice, el timestamp se calcula con FPS
    
    return None


def generate_piv_result_filename(
    pair_metadata: Dict[str, Any],
    fps: float,
    extension: str = "txt"
) -> str:
    """
    Generar nombre de archivo PIV con timestamp de la toma
    
    Args:
        pair_metadata: Diccionario con metadata del par (de block_metadata.json)
        fps: FPS de la cámara
        extension: Extensión del archivo (default: "txt")
    
    Returns:
        Nombre de archivo con formato: pair_r{r}b{b}_t{time}s_dt{dt}ms.{ext}
    
    Ejemplo:
        pair_r1b001_t0.000s_dt4.545ms.txt
        pair_r2b015_t1.818s_dt9.091ms.txt
    """
    region_idx = pair_metadata["region_idx"]
    block_idx = pair_metadata["block_idx"]
    dt_ms = pair_metadata["dt_ms"]
    img1_idx = pair_metadata["img1_original_idx"]
    
    # Calcular timestamp en segundos
    timestamp_s = img1_idx / fps
    
    # Formato: pair_r{region}b{block}_t{time}s_dt{dt}ms.ext
    filename = (
        f"pair_r{region_idx+1}b{block_idx+1:03d}_"
        f"t{timestamp_s:.3f}s_"
        f"dt{dt_ms:.3f}ms.{extension}"
    )
    
    return filename


def generate_piv_result_filename_simple(
    region_idx: int,
    block_idx: int,
    timestamp_s: float,
    dt_ms: float,
    extension: str = "txt"
) -> str:
    """
    Generar nombre de archivo PIV (versión simplificada)
    
    Args:
        region_idx: Índice de región (0-based)
        block_idx: Índice de bloque (0-based)
        timestamp_s: Timestamp en segundos
        dt_ms: Delta tiempo en milisegundos
        extension: Extensión del archivo
    
    Returns:
        Nombre de archivo formateado
    """
    filename = (
        f"pair_r{region_idx+1}b{block_idx+1:03d}_"
        f"t{timestamp_s:.3f}s_"
        f"dt{dt_ms:.3f}ms.{extension}"
    )
    
    return filename


def parse_piv_result_filename(filename: str) -> Dict[str, Any] | None:
    """
    Parsear nombre de archivo PIV para extraer metadata
    
    Args:
        filename: Nombre del archivo PIV
    
    Returns:
        Diccionario con metadata extraída, o None si no coincide el patrón
    
    Ejemplo:
        Input: "pair_r1b001_t0.000s_dt4.545ms.txt"
        Output: {
            "region": 1,
            "block": 1,
            "timestamp_s": 0.000,
            "dt_ms": 4.545,
            "extension": "txt"
        }
    """
    # Patrón: pair_r{region}b{block}_t{time}s_dt{dt}ms.{ext}
    pattern = r'pair_r(\d+)b(\d+)_t([\d.]+)s_dt([\d.]+)ms\.(\w+)'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    return {
        "region": int(match.group(1)),
        "block": int(match.group(2)),
        "timestamp_s": float(match.group(3)),
        "dt_ms": float(match.group(4)),
        "extension": match.group(5),
    }


def get_piv_output_path_with_metadata(
    output_dir: Path,
    pair_metadata: Dict[str, Any],
    fps: float,
    extension: str = "txt"
) -> Path:
    """
    Obtener path completo para archivo de resultado PIV
    
    Args:
        output_dir: Directorio de salida
        pair_metadata: Metadata del par desde block_metadata.json
        fps: FPS de la cámara
        extension: Extensión del archivo
    
    Returns:
        Path completo al archivo de resultado
    """
    filename = generate_piv_result_filename(pair_metadata, fps, extension)
    return output_dir / filename


# ============================================================================
# FUNCIONES DE COMPATIBILIDAD (para uso en exporter.py)
# ============================================================================

def should_use_metadata_naming(config) -> bool:
    """
    Determinar si se debe usar naming con metadata o naming legacy
    
    Args:
        config: Objeto PIVConfig
    
    Returns:
        True si existe metadata JSON disponible
    """
    # Verificar si existe archivo de metadata
    if hasattr(config, 'images_dir'):
        metadata_path = config.images_dir / "block_metadata.json"
        return metadata_path.exists()
    return False


def load_pair_metadata_for_images(
    img1_filename: str,
    img2_filename: str,
    metadata_json_path: Path
) -> Dict[str, Any] | None:
    """
    Buscar metadata de un par de imágenes en el JSON
    
    Args:
        img1_filename: Nombre de imagen 1
        img2_filename: Nombre de imagen 2
        metadata_json_path: Path al archivo block_metadata.json
    
    Returns:
        Diccionario con metadata del par, o None si no se encuentra
    """
    import json
    
    if not metadata_json_path.exists():
        return None
    
    with open(metadata_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Buscar par que coincida con estos filenames
    for pair in data.get("pairs", []):
        if (pair["img1_filename"] == img1_filename and 
            pair["img2_filename"] == img2_filename):
            return pair
    
    return None