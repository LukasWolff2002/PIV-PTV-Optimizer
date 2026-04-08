#!/usr/bin/env python3
"""
PIV SYSTEM - MAIN
==================

Pipeline automático completo de interpolación y visualización PIV.
Procesa automáticamente todas las tomas encontradas.

Uso:
    python main.py                    # Procesar todas las tomas
    python main.py --force            # Forzar re-generación
    python main.py --take "toma-1"    # Procesar solo una toma específica
"""

import sys
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

# Importar módulos
from .piv_interpolate_simple import generate_interpolated_files
from .piv_generate_cache import generate_cache_for_visualization
from .piv_visualize import render_animation
from .piv_config import INTERPOLATED_DIR, CACHE_DIR, ROOT_RESULTS, CAM_PROFILES, extract_subfolder_name
import Analisis.PIV.InterpolarPIV.piv_config as piv_config


ANIMATIONS_DIR = Path("animaciones")


def identify_takes(root_folder: Path) -> dict:
    """
    Identifica todas las tomas en la carpeta raíz.
    
    Returns:
        dict: {take_name: {cam_name: folder_path}}
    """
    print("\n🔍 Buscando tomas en la carpeta...")
    
    # Agrupar carpetas por nombre base (sin cam-X)
    takes = defaultdict(dict)
    
    for folder in root_folder.iterdir():
        if not folder.is_dir():
            continue
        
        # Extraer nombre base (sin cam-X)
        base_name = extract_subfolder_name(folder)
        
        # Detectar qué cámara es esta carpeta
        folder_name = folder.name
        cam_detected = None
        
        for cam_name in CAM_PROFILES.keys():
            # Buscar patrones como "cam-1", "cam_1", "Cam1", etc.
            cam_num = cam_name.split()[-1]  # "Cam 1" -> "1"
            patterns = [
                f"cam-{cam_num}",
                f"cam_{cam_num}",
                f"Cam-{cam_num}",
                f"Cam_{cam_num}",
                f"cam{cam_num}",
                f"Cam{cam_num}",
            ]
            
            for pattern in patterns:
                if pattern.lower() in folder_name.lower():
                    cam_detected = cam_name
                    break
            
            if cam_detected:
                break
        
        if cam_detected:
            takes[base_name][cam_detected] = folder
    
    # Filtrar tomas que tengan las 4 cámaras
    complete_takes = {}
    incomplete_takes = {}
    
    for take_name, cameras in takes.items():
        n_cameras = len(cameras)
        
        if n_cameras == len(CAM_PROFILES):
            complete_takes[take_name] = cameras
        else:
            incomplete_takes[take_name] = cameras
    
    # Mostrar resumen
    print(f"\n📊 Tomas encontradas:")
    print(f"   ✓ Completas (4 cámaras): {len(complete_takes)}")
    if incomplete_takes:
        print(f"   ⚠️  Incompletas: {len(incomplete_takes)}")
        for take_name, cameras in incomplete_takes.items():
            missing = set(CAM_PROFILES.keys()) - set(cameras.keys())
            print(f"      • {take_name}: {len(cameras)}/4 cámaras "
                  f"(faltan: {', '.join(missing)})")
    
    return complete_takes


def process_single_take(take_name: str, camera_folders: dict, force: bool = False):
    """
    Procesa una sola toma: interpolación → caché → video → limpieza
    
    Args:
        take_name: Nombre de la toma
        camera_folders: Dict {cam_name: folder_path}
        force: Forzar re-generación
    
    Returns:
        bool: True si se procesó exitosamente
    """
    print("\n" + "="*70)
    print(f"PROCESANDO TOMA: {take_name}")
    print("="*70)
    
    # Verificar si ya existe output
    output_txt_folder = INTERPOLATED_DIR / take_name
    output_video = ANIMATIONS_DIR / f"{take_name}.mp4"
    
    # ============================================================
    # NUEVO: Verificar si la toma ya está completamente analizada
    # ============================================================
    if output_video.exists() and not force:
        print(f"\n⏭️ La toma '{take_name}' ya fue analizada (existe {output_video.name}).")
        print("   Saltando al siguiente...")
        return True # Retornamos True para que el contador en main() no lo marque como error
    
    skip_interpolation = False
    if output_txt_folder.exists() and not force:
        n_existing = len(list(output_txt_folder.glob("*.txt")))
        if n_existing > 0:
            print(f"\n✓ Ya existen {n_existing} archivos .txt para esta toma")
            skip_interpolation = True
    
    # Crear carpeta temporal con enlaces simbólicos
    temp_root = Path(f"temp_take_{take_name}")
    temp_root.mkdir(exist_ok=True)
    
    try:
        # Crear enlaces simbólicos a las carpetas de cámaras
        for cam_name, folder in camera_folders.items():
            link_path = temp_root / folder.name
            if link_path.exists():
                if link_path.is_symlink():
                    link_path.unlink()
                else:
                    shutil.rmtree(link_path)
            link_path.symlink_to(folder.absolute())
        
        # Modificar temporalmente la configuración
        original_root = piv_config.ROOT_RESULTS
        
        piv_config.ROOT_RESULTS = temp_root
        
        # ============================================================
        # PASO 1: INTERPOLACIÓN
        # ============================================================
        
        if not skip_interpolation:
            print("\n" + "="*70)
            print("PASO 1/4: INTERPOLACIÓN")
            print("="*70)
            generate_interpolated_files()
        else:
            print("\n[PASO 1/4] ✓ Usando archivos .txt existentes")
        
        # NO RESTAURAR ROOT_RESULTS todavía - lo necesitamos para el caché
        
        # ============================================================
        # PASO 2: GENERAR CACHÉ TEMPORAL
        # ============================================================
        
        print("\n" + "="*70)
        print("PASO 2/4: GENERACIÓN DE CACHÉ TEMPORAL")
        print("="*70)
        generate_cache_for_visualization(subfolder_name=take_name)
        
        # ============================================================
        # PASO 3: RENDERIZAR VIDEO
        # ============================================================
        
        print("\n" + "="*70)
        print("PASO 3/4: RENDERIZADO DE VIDEO")
        print("="*70)
        
        ANIMATIONS_DIR.mkdir(exist_ok=True)
        
        print(f"Nombre del video: {take_name}.mp4")
        print(f"Ubicación: {ANIMATIONS_DIR.absolute()}/")
        
        render_animation(output_video, show_preview=False)
        
        # ============================================================
        # PASO 4: LIMPIAR CACHÉ
        # ============================================================
        
        print("\n" + "="*70)
        print("PASO 4/4: LIMPIEZA")
        print("="*70)
        cleanup_cache()
        
        # AHORA SÍ restaurar ROOT_RESULTS
        piv_config.ROOT_RESULTS = original_root
        
        print(f"\n✅ Toma completada: {take_name}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error procesando {take_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Limpiar carpeta temporal
        if temp_root.exists():
            for link in temp_root.iterdir():
                if link.is_symlink():
                    link.unlink()
                elif link.is_dir():
                    shutil.rmtree(link)
                else:
                    link.unlink()
            temp_root.rmdir()


def cleanup_cache():
    """Elimina el directorio de caché completo"""
    if CACHE_DIR.exists():
        print(f"\n🧹 Limpiando caché temporal...")
        shutil.rmtree(CACHE_DIR)
        print(f"   ✓ Caché eliminado: {CACHE_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline automático PIV: procesa todas las tomas automáticamente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
El script ejecuta automáticamente para cada toma:
  1. Genera archivos .txt interpolados en PIV_INTERPOLADO/
  2. Crea caché temporal en piv_cache/
  3. Renderiza video en animaciones/[nombre].mp4
  4. Elimina el caché temporal

Ejemplo:
  %(prog)s                      # Procesar todas las tomas
  %(prog)s --force              # Forzar re-generación
  %(prog)s --take "toma-1"      # Solo una toma específica
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Forzar re-generación de archivos .txt interpolados'
    )
    
    parser.add_argument(
        '--take',
        type=str,
        help='Procesar solo una toma específica (nombre base sin cam-X)'
    )
    
    parser.add_argument(
        '--root',
        type=Path,
        default=ROOT_RESULTS,
        help=f'Carpeta raíz con datos (default: {ROOT_RESULTS})'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("PIV SYSTEM - PROCESADOR AUTOMÁTICO DE MÚLTIPLES TOMAS")
    print("="*70)
    print(f"\nCarpeta de entrada: {args.root.absolute()}")
    print(f"Carpeta .txt: {INTERPOLATED_DIR.absolute()}")
    print(f"Carpeta videos: {ANIMATIONS_DIR.absolute()}")
    
    # Verificar que existe la carpeta raíz
    if not args.root.exists():
        print(f"\n❌ Error: No se encontró la carpeta {args.root}")
        sys.exit(1)
    
    # Identificar tomas
    takes = identify_takes(args.root)
    
    if not takes:
        print("\n❌ No se encontraron tomas completas (con 4 cámaras)")
        sys.exit(1)
    
    # Filtrar por toma específica si se especificó
    if args.take:
        if args.take in takes:
            takes = {args.take: takes[args.take]}
            print(f"\n🎯 Procesando solo: {args.take}")
        else:
            print(f"\n❌ Error: Toma '{args.take}' no encontrada")
            print(f"   Tomas disponibles: {', '.join(sorted(takes.keys()))}")
            sys.exit(1)
    
    # Mostrar resumen de tomas a procesar
    print(f"\n📋 Tomas a procesar: {len(takes)}")
    for idx, take_name in enumerate(sorted(takes.keys()), 1):
        print(f"   {idx}. {take_name}")
    
    # Confirmar procesamiento
    if not args.force and len(takes) > 1:
        response = input(f"\n¿Procesar {len(takes)} tomas? (S/n): ")
        if response.lower() == 'n':
            print("Cancelado.")
            sys.exit(0)
    
    # Procesar cada toma
    successful = 0
    failed = 0
    
    for idx, (take_name, cameras) in enumerate(sorted(takes.items()), 1):
        print(f"\n{'='*70}")
        print(f"PROGRESO: {idx}/{len(takes)}")
        print(f"{'='*70}")
        
        result = process_single_take(take_name, cameras, args.force)
        
        if result:
            successful += 1
        else:
            failed += 1
            
            if len(takes) > 1:
                response = input("\n¿Continuar con las siguientes tomas? (S/n): ")
                if response.lower() == 'n':
                    break
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    
    print("\n" + "="*70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("="*70)
    
    print(f"\n📊 Resumen:")
    print(f"   ✓ Tomas exitosas: {successful}/{len(takes)}")
    if failed > 0:
        print(f"   ❌ Tomas fallidas: {failed}/{len(takes)}")
    
    print(f"\n📁 Archivos interpolados (.txt):")
    print(f"   {INTERPOLATED_DIR.absolute()}/")
    if INTERPOLATED_DIR.exists():
        subfolders = [d for d in INTERPOLATED_DIR.iterdir() if d.is_dir()]
        total_txt = sum(len(list(d.glob("*.txt"))) for d in subfolders)
        print(f"   {len(subfolders)} toma(s), {total_txt} archivos")
    
    print(f"\n📹 Videos generados:")
    print(f"   {ANIMATIONS_DIR.absolute()}/")
    if ANIMATIONS_DIR.exists():
        videos = list(ANIMATIONS_DIR.glob("*.mp4"))
        total_size = sum(v.stat().st_size for v in videos) / 1024 / 1024
        print(f"   {len(videos)} video(s), {total_size:.1f} MB total")


if __name__ == "__main__":
    main()