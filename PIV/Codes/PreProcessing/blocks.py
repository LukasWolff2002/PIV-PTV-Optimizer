"""
PIV/Codes/PreProcessing/blocks.py

Muestreo por bloques con preprocesamiento opcional de imágenes PIV
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass  # ← AGREGADO: faltaba este import
from typing import List, Optional, Dict
import os, re, shutil, json  # ← AGREGADO: json al final

# Importar funciones de preprocesamiento
try:
    import numpy as np
    from PIL import Image
    from .filters import load_image, apply_preprocessing, save_image
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


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

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


# ============================================================================
# MUESTREO ADAPTATIVO MULTI-REGIÓN
# ============================================================================

def run_adaptive_block_sampling(
    input_dir: Path,
    output_dir: Path,
    regions: List['TemporalRegion'],
    delete_existing: bool,
    natural_sort: bool = True,
    overwrite: bool = True,
    preprocess_params: Optional[Dict] = None,
    output_metadata: bool = True,
) -> List['BlockMetadata']:
    """
    Muestreo adaptativo por regiones temporales con diferentes parámetros de bloque
    
    Permite definir múltiples regiones temporales en la secuencia, cada una con sus
    propios parámetros de muestreo (block_size, skip_inter, skip_final).
    
    Args:
        input_dir: directorio con imágenes originales
        output_dir: directorio de salida
        regions: lista de TemporalRegion con parámetros por región
        delete_existing: eliminar directorio de salida si existe
        natural_sort: ordenamiento natural de archivos
        overwrite: sobrescribir archivos existentes
        preprocess_params: diccionario con parámetros de preprocesamiento PIV
        output_metadata: guardar archivo JSON con metadata completa
    
    Returns:
        Lista de BlockMetadata con info de cada par procesado
    """
    if not TEMPORAL_REGIONS_AVAILABLE:
        raise RuntimeError(
            "temporal_regions module not available. "
            "Cannot use adaptive block sampling."
        )
    
    # Validaciones
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe input_dir: {input_dir}")
    
    if not regions:
        raise ValueError("Lista de regiones vacía")
    
    # Eliminar y crear carpeta de salida
    if output_dir.exists() and delete_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Listar y ordenar archivos
    files = list_sorted_files(input_dir, natural_sort)
    total_files = len(files)
    
    if total_files == 0:
        raise RuntimeError(f"No hay archivos en: {input_dir}")
    
    # Validar regiones
    validate_regions(regions, total_files)
    
    # Imprimir configuración
    print_regions_summary(regions)
    
    # Determinar si hay preprocesamiento
    apply_preprocess = (preprocess_params is not None) and PREPROCESS_AVAILABLE
    
    if apply_preprocess:
        print(f"[PRE] Preprocesamiento HABILITADO para {input_dir.name}")
    else:
        if preprocess_params is not None and not PREPROCESS_AVAILABLE:
            print(f"[WARN] Preprocesamiento solicitado pero módulo no disponible")
        print(f"[PRE] Preprocesamiento DESHABILITADO - copia directa")
    
    # Procesar cada región
    metadata_list: List['BlockMetadata'] = []
    total_pairs = 0
    total_images_selected = 0
    
    for region_idx, region in enumerate(regions):
        print(f"\n{'='*70}")
        print(f"[REGIÓN {region_idx+1}/{len(regions)}] {region.name.upper()}")
        print(f"{'='*70}")
        print(f"  Tiempo:       {region.start_time:.2f}s → ", end="")
        if region.end_time is not None:
            print(f"{region.end_time:.2f}s")
        else:
            print("END (hasta final)")
        print(f"  Frames:       {region.start_frame} → {region.end_frame} ({region.total_frames} frames)")
        print(f"  Block:        size={region.block_size}, skip_inter={region.skip_inter}, skip_final={region.skip_final}")
        print(f"  Δt efectivo:  {region.dt_ms:.3f} ms")
        print(f"  Max bloques:  {region.max_blocks}")
        
        # Procesar bloques en esta región
        i = region.start_frame
        block_idx = 0
        region_pairs = 0
        region_images = 0
        
        while (i + region.block_size) <= region.end_frame:
            # Índices de las 2 imágenes a seleccionar del bloque
            idx1 = i
            idx2 = i + 1 + region.skip_inter
            
            # Nombres de archivos procesados para este par
            img1_filename = None
            img2_filename = None
            
            for img_order, global_idx in enumerate([idx1, idx2], start=1):
                src = files[global_idx]
                
                # Generar nombre con metadata de región
                # Formato: {original_stem}_r{region}b{block}s{skip}{ext}
                stem = src.stem
                ext = src.suffix
                new_name = f"{stem}_r{region_idx+1}b{block_idx+1:03d}s{region.skip_inter}{ext}"
                dst = output_dir / new_name
                
                # Guardar nombres para metadata
                if img_order == 1:
                    img1_filename = new_name
                else:
                    img2_filename = new_name
                
                # Saltar si existe y no se permite sobrescribir
                if dst.exists() and not overwrite:
                    continue
                
                # PREPROCESAMIENTO O COPIA DIRECTA
                if apply_preprocess:
                    try:
                        # Detectar bit depth original
                        bit_depth_original = obtener_bit_depth_original(src)
                        
                        # Cargar imagen original
                        img = load_image(src)
                        
                        # Aplicar preprocesamiento
                        img_procesada = apply_preprocessing(img, preprocess_params)
                        
                        # Guardar imagen procesada
                        save_image(img_procesada, dst, bit_depth=bit_depth_original)
                        
                    except Exception as e:
                        print(f"[ERROR] Fallo preprocesamiento en {src.name}: {e}")
                        print(f"[WARN] Copiando original sin procesar")
                        shutil.copy2(src, dst)
                else:
                    # Copia directa sin modificar
                    shutil.copy2(src, dst)
                
                region_images += 1
            
            # Guardar metadata del par (usando la clase importada de temporal_regions)
            metadata_list.append(BlockMetadata(
                region_idx=region_idx,
                region_name=region.name,
                block_idx=block_idx,
                skip_inter=region.skip_inter,
                dt_ms=region.dt_ms,
                img1_original_idx=idx1,
                img2_original_idx=idx2,
                img1_filename=img1_filename,
                img2_filename=img2_filename,
            ))
            
            # Avanzar al siguiente bloque
            i += region.block_size
            block_idx += 1
            region_pairs += 1
        
        total_pairs += region_pairs
        total_images_selected += region_images
        
        # Resumen de región
        reduction_region = (1 - region_images / region.total_frames) * 100
        print(f"\n  Resultados:")
        print(f"    Bloques procesados:  {block_idx}")
        print(f"    Pares generados:     {region_pairs}")
        print(f"    Imágenes copiadas:   {region_images}/{region.total_frames}")
        print(f"    Reducción:           {reduction_region:.1f}%")
    
    # Guardar metadata como JSON
    if output_metadata:
        metadata_path = output_dir / "block_metadata.json"
        save_metadata_json(
            metadata_list=metadata_list,
            regions=regions,
            output_path=metadata_path,
            total_images_input=total_files,
        )
    
    # Resumen global final
    reduction_total = (1 - total_images_selected / total_files) * 100
    
    print(f"\n{'='*70}")
    print(f"RESUMEN GLOBAL")
    print(f"{'='*70}")
    print(f"  Total imágenes input:      {total_files}")
    print(f"  Total pares procesados:    {total_pairs}")
    print(f"  Total imágenes copiadas:   {total_images_selected}")
    print(f"  Reducción total:           {reduction_total:.1f}%")
    print(f"  Directorio salida:         {output_dir}")
    if output_metadata:
        print(f"  Metadata JSON:             {output_dir / 'block_metadata.json'}")
    print(f"{'='*70}\n")
    
    return metadata_list


# ============================================================================
# MUESTREO LEGACY (BLOQUE ÚNICO)
# ============================================================================

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