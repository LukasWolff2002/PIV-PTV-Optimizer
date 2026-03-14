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
    from .filters import preprocess_image_file
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
    print("[WARN] piv_preprocessing no disponible, preprocesamiento deshabilitado")

# Importar estructuras de regiones temporales
try:
    from .temporal_regions import (
        TemporalRegion,
        BlockMetadata,
        validate_regions,
        save_metadata_json,
        print_regions_summary,
    )
    TEMPORAL_REGIONS_AVAILABLE = True
except ImportError:
    TEMPORAL_REGIONS_AVAILABLE = False
    print("[WARN] temporal_regions no disponible, muestreo adaptativo deshabilitado")

@dataclass
class BlockMetadata:
    """Metadata de un par de imágenes procesado"""
    region_idx: int          # Índice de región (0, 1, 2, ...)
    region_name: str         # Nombre de región ("alta_velocidad", etc.)
    block_idx: int           # Índice del bloque dentro de la región
    skip_inter: int          # skip_inter usado
    dt_ms: float             # Delta tiempo calculado
    img1_original_idx: int   # Índice original en secuencia completa
    img2_original_idx: int   # Índice original en secuencia completa


def run_adaptive_block_sampling(
    input_dir: Path,
    output_dir: Path,
    regions: List[TemporalRegion],
    delete_existing: bool,
    natural_sort: bool = True,
    overwrite: bool = True,
    preprocess_params: Optional[Dict] = None,
    output_metadata: bool = True,  # Guardar JSON con metadata
) -> List[BlockMetadata]:
    """
    Muestreo adaptativo por regiones temporales
    
    Returns:
        Lista de metadata de cada par procesado
    """
    # Validaciones
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe input_dir: {input_dir}")
    
    # Eliminar y crear carpeta de salida
    if output_dir.exists() and delete_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Listar y ordenar archivos
    files = list_sorted_files(input_dir, natural_sort)
    total = len(files)
    
    if total == 0:
        raise RuntimeError(f"No hay archivos en: {input_dir}")
    
    # Validar que las regiones cubren frames disponibles
    max_frame_needed = max(r.end_frame for r in regions)
    if max_frame_needed > total:
        raise ValueError(
            f"Las regiones requieren {max_frame_needed} frames, "
            f"pero solo hay {total} disponibles"
        )
    
    # Validar que no hay solapamiento entre regiones
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            if not (r1.end_frame <= r2.start_frame or r2.end_frame <= r1.start_frame):
                raise ValueError(
                    f"Regiones '{r1.name}' y '{r2.name}' se solapan en frames"
                )
    
    # Procesar cada región
    metadata_list = []
    total_pairs = 0
    
    for region_idx, region in enumerate(regions):
        print(f"\n[REGIÓN {region_idx+1}/{len(regions)}] {region.name}")
        print(f"  Frames: {region.start_frame} → {region.end_frame} ({region.total_frames} frames)")
        print(f"  Block: size={region.block_size}, skip_inter={region.skip_inter}, skip_final={region.skip_final}")
        print(f"  Δt: {region.dt_ms:.3f} ms")
        
        # Procesar bloques en esta región
        i = region.start_frame
        block_idx = 0
        
        while (i + region.block_size) <= region.end_frame:
            # Índices de las 2 imágenes a seleccionar del bloque
            idx1 = i
            idx2 = i + 1 + region.skip_inter
            
            for local_idx, global_idx in enumerate([idx1, idx2]):
                src = files[global_idx]
                
                # Generar nombre con metadata
                stem = src.stem
                ext = src.suffix
                new_name = f"{stem}_r{region_idx+1}b{block_idx+1:03d}s{region.skip_inter}{ext}"
                dst = output_dir / new_name
                
                # Saltar si existe y no se permite sobrescribir
                if dst.exists() and not overwrite:
                    continue
                
                # Preprocesamiento o copia
                apply_preprocess = (preprocess_params is not None) and PREPROCESS_AVAILABLE
                
                if apply_preprocess:
                    try:
                        preprocess_image_file(
                            input_path=src,
                            output_path=dst,
                            params=preprocess_params,
                            bit_depth=16
                        )
                    except Exception as e:
                        print(f"[ERROR] Fallo preprocesamiento en {src.name}: {e}")
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            # Guardar metadata del par
            metadata_list.append(BlockMetadata(
                region_idx=region_idx,
                region_name=region.name,
                block_idx=block_idx,
                skip_inter=region.skip_inter,
                dt_ms=region.dt_ms,
                img1_original_idx=idx1,
                img2_original_idx=idx2,
            ))
            
            # Avanzar al siguiente bloque
            i += region.block_size
            block_idx += 1
            total_pairs += 1
        
        print(f"  Procesados: {block_idx} bloques ({block_idx * 2} imágenes)")
    
    # Guardar metadata como JSON
    if output_metadata:
        metadata_path = output_dir / "block_metadata.json"
        metadata_dict = {
            "total_pairs": total_pairs,
            "regions": [
                {
                    "idx": r_idx,
                    "name": r.name,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "fps": r.fps,
                    "block_size": r.block_size,
                    "skip_inter": r.skip_inter,
                    "dt_ms": r.dt_ms,
                }
                for r_idx, r in enumerate(regions)
            ],
            "pairs": [
                {
                    "region_idx": m.region_idx,
                    "region_name": m.region_name,
                    "block_idx": m.block_idx,
                    "skip_inter": m.skip_inter,
                    "dt_ms": m.dt_ms,
                    "img1_idx": m.img1_original_idx,
                    "img2_idx": m.img2_original_idx,
                }
                for m in metadata_list
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"\n[METADATA] Guardado en: {metadata_path}")
    
    # Resumen final
    print(f"\n[RESUMEN GLOBAL]")
    print(f"  Total pares procesados: {total_pairs}")
    print(f"  Total imágenes: {total_pairs * 2}")
    print(f"  Reducción total: {(1 - (total_pairs * 2) / total) * 100:.1f}%")
    
    return metadata_list


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
            dst = output_dir / src.name
            
            # Saltar si existe y no se permite sobrescribir
            if dst.exists() and not overwrite:
                continue
            
            # PREPROCESAMIENTO O COPIA DIRECTA
            if apply_preprocess:
                # Cargar → Procesar → Guardar
                try:
                    preprocess_image_file(
                        input_path=src,
                        output_path=dst,
                        params=preprocess_params,
                        bit_depth=16
                    )
                except Exception as e:
                    print(f"[ERROR] Fallo preprocesamiento en {src.name}: {e}")
                    print(f"[WARN] Copiando original sin procesar")
                    shutil.copy2(src, dst)
            else:
                # Copia directa sin modificar (comportamiento original)
                shutil.copy2(src, dst)

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