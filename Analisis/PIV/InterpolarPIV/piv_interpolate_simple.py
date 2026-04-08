"""
INTERPOLACIÓN DE DATOS PIV
===========================
Genera archivos .txt interpolados por cámara.
NO genera caché .npz (eso es opcional para visualización).
"""

import bisect
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

from Analisis.PIV.InterpolarPIV.piv_config import (
    INTERPOLATED_DIR, CAM_PROFILES,
    MAX_TIME_GAP_FOR_INTERPOLATION,
    USE_MULTIPROCESSING, NUM_WORKERS,
    PIVFrame, build_camera_folder_map, build_time_index_for_camera,
    load_piv_frame, extract_subfolder_name, write_piv_txt,
    build_global_timeline
)
import Analisis.PIV.InterpolarPIV.piv_config as piv_config  # Para acceder a ROOT_RESULTS dinámicamente


# ============================================================
# INTERPOLACIÓN TEMPORAL
# ============================================================

def interpolate_piv_data(frame1: PIVFrame, frame2: PIVFrame, target_time: float) -> Optional[PIVFrame]:
    """
    Interpola linealmente entre dos frames PIV.
    Retorna None si los frames no son compatibles.
    """
    t1 = frame1.timestamp_s
    t2 = frame2.timestamp_s
    
    if t2 <= t1:
        return None
    
    # Factor de interpolación
    alpha = (target_time - t1) / (t2 - t1)
    if alpha < 0 or alpha > 1:
        return None
    
    # Verificar que tengan las mismas posiciones (x, y)
    if frame1.data.shape != frame2.data.shape:
        return None
    
    # Verificar que las posiciones x,y sean las mismas (tolerancia para floats)
    pos_match = (np.allclose(frame1.data[:, 0], frame2.data[:, 0], rtol=1e-5) and 
                 np.allclose(frame1.data[:, 1], frame2.data[:, 1], rtol=1e-5))
    
    if not pos_match:
        return None
    
    # Crear datos interpolados
    data_interp = frame1.data.copy()
    
    # Interpolar u, v (columnas 2 y 3)
    data_interp[:, 2] = (1 - alpha) * frame1.data[:, 2] + alpha * frame2.data[:, 2]  # u
    data_interp[:, 3] = (1 - alpha) * frame1.data[:, 3] + alpha * frame2.data[:, 3]  # v
    
    # Mantener validez solo si ambos frames son válidos
    valid1 = frame1.data[:, 5].astype(int) == 1
    valid2 = frame2.data[:, 5].astype(int) == 1
    data_interp[:, 5] = (valid1 & valid2).astype(int)
    
    return PIVFrame(
        path=frame1.path,
        timestamp_s=target_time,
        dt_ms=frame1.dt_ms,
        px_per_mm=frame1.px_per_mm,
        data=data_interp
    )


def load_camera_frame_with_interpolation(
    cam_name: str, 
    target_time_s: float, 
    time_index: List[Tuple[float, Path]]
) -> Optional[PIVFrame]:
    """
    Carga frame de UNA cámara para un tiempo específico.
    Intenta interpolación temporal si no hay datos exactos.
    """
    if not time_index:
        return None
    
    times = [t for t, _ in time_index]
    idx = bisect.bisect_left(times, target_time_s)
    
    # CASO 1: Tiempo exacto (o muy cercano < 1µs)
    if idx < len(times) and abs(times[idx] - target_time_s) < 1e-6:
        _, p_exact = time_index[idx]
        try:
            return load_piv_frame(p_exact)
        except Exception:
            return None
    
    # CASO 2: Antes del primer frame disponible
    if idx == 0:
        t_first, p_first = time_index[0]
        gap = abs(t_first - target_time_s)
        # Para flujos laminares, permitir extrapolación más allá
        if gap < 0.5:  # 500ms de tolerancia
            try:
                return load_piv_frame(p_first)
            except:
                return None
        return None
    
    # CASO 3: Después del último frame disponible
    if idx >= len(times):
        t_last, p_last = time_index[-1]
        gap = abs(t_last - target_time_s)
        # Para flujos laminares, permitir extrapolación más allá
        if gap < 0.5:  # 500ms de tolerancia
            try:
                return load_piv_frame(p_last)
            except:
                return None
        return None
    
    # CASO 4: Entre dos frames - interpolar siempre (flujo laminar)
    t_before, p_before = time_index[idx - 1]
    t_after, p_after = time_index[idx]
    
    gap = t_after - t_before
    
    # Para flujos laminares, interpolar siempre sin importar el gap
    if gap <= MAX_TIME_GAP_FOR_INTERPOLATION:
        try:
            frame_before = load_piv_frame(p_before)
            frame_after = load_piv_frame(p_after)
            frame_interp = interpolate_piv_data(frame_before, frame_after, target_time_s)
            
            if frame_interp is not None:
                return frame_interp
            else:
                # Interpolación falló, usar el más cercano
                if abs(t_after - target_time_s) < abs(t_before - target_time_s):
                    return frame_after
                else:
                    return frame_before
        except Exception:
            pass
    
    # CASO 5: Solo si MAX_TIME_GAP_FOR_INTERPOLATION es finito
    # (para flujos laminares con inf, nunca llega aquí)
    if abs(t_after - target_time_s) < abs(t_before - target_time_s):
        closest_t, closest_p = t_after, p_after
    else:
        closest_t, closest_p = t_before, p_before
    
    if abs(closest_t - target_time_s) < 0.5:  # 500ms de tolerancia
        try:
            return load_piv_frame(closest_p)
        except:
            return None
    
    return None


# ============================================================
# GUARDADO DE ARCHIVOS INTERPOLADOS
# ============================================================

def save_interpolated_camera_frames(
    target_time_s: float,
    camera_folder_map: Dict[str, Path]
) -> Dict[str, Path]:
    """
    Guarda UN archivo PIV fusionado por timestamp aplicando prioridad de cámaras.
    El target_time_s ya viene normalizado (desde 0) por build_global_timeline.
    """
    from Analisis.PIV.InterpolarPIV.piv_config import USE_CAMERA_PRIORITY, PRIORITY_MODE, CAMERA_PRIORITY_ORDER
    
    # Cargar datos de todas las cámaras para este timestamp
    camera_data = {}
    subfolder_name = None
    
    # Encontrar t_min global para normalizar los time_index
    t_min_global = None
    for cam_name in CAM_PROFILES.keys():
        folder = camera_folder_map.get(cam_name)
        if folder is None:
            continue
        
        time_index = build_time_index_for_camera(folder)
        if time_index and (t_min_global is None or time_index[0][0] < t_min_global):
            t_min_global = time_index[0][0]
    
    if t_min_global is None:
        return {}
    
    # Cargar frames de cada cámara
    for cam_name in CAM_PROFILES.keys():
        folder = camera_folder_map.get(cam_name)
        if folder is None:
            continue
        
        # Determinar nombre de subcarpeta
        if subfolder_name is None:
            subfolder_name = extract_subfolder_name(folder)
        
        # Obtener time_index sin normalizar
        time_index_raw = build_time_index_for_camera(folder)
        if not time_index_raw:
            continue
        
        # Normalizar time_index usando el mismo t_min que build_global_timeline
        time_index_normalized = [(t - t_min_global, path) for t, path in time_index_raw]
        
        # Cargar frame (con interpolación si es necesario)
        frame = load_camera_frame_with_interpolation(cam_name, target_time_s, time_index_normalized)
        if frame is None:
            continue
        
        camera_data[cam_name] = frame
    
    if not camera_data:
        return {}
    
    # Crear directorio de salida
    if subfolder_name is None:
        subfolder_name = "interpolated"
    
    output_dir = INTERPOLATED_DIR / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar nombre de archivo ÚNICO por timestamp
    timestamp_us = int(target_time_s * 1e6)
    output_file = output_dir / f"frame_{timestamp_us:09d}.txt"
    
    # Fusionar datos según modo de prioridad
    merged_data = merge_camera_data(camera_data, target_time_s)
    
    if merged_data is None:
        return {}
    
    # Headers del archivo fusionado
    cameras_used = ','.join(sorted(camera_data.keys()))
    extra_headers = {
        'cameras': cameras_used,
        'n_cameras': str(len(camera_data)),
        'priority_mode': PRIORITY_MODE if USE_CAMERA_PRIORITY else 'none',
    }
    
    # Escribir archivo fusionado
    write_piv_txt(
        output_file,
        merged_data,
        timestamp_s=target_time_s,
        dt_ms=None,
        px_per_mm=None,
        extra_headers=extra_headers
    )
    
    return {subfolder_name: output_file}


def merge_camera_data(camera_data: Dict[str, 'PIVFrame'], target_time_s: float) -> Optional[np.ndarray]:
    """
    Fusiona datos de múltiples cámaras en un solo array según prioridad.
    
    Returns:
        Array numpy fusionado (N x 6) o None si no hay datos
    """
    from Analisis.PIV.InterpolarPIV.piv_config import USE_CAMERA_PRIORITY, PRIORITY_MODE, CAMERA_PRIORITY_ORDER, build_transform
    
    # Transformar datos de cada cámara a coordenadas globales
    all_points = []
    
    # Ordenar cámaras por prioridad si está activo
    if USE_CAMERA_PRIORITY and PRIORITY_MODE == "override":
        # Procesar en orden de prioridad CRECIENTE (menor a mayor)
        sorted_cameras = sorted(
            camera_data.keys(),
            key=lambda cam: CAMERA_PRIORITY_ORDER.get(cam, 0)
        )
    else:
        sorted_cameras = list(camera_data.keys())
    
    # Diccionario para almacenar puntos por celda de grilla
    # key: (ix, iy), value: (x, y, u, v, mag, valid, priority)
    grid_points = {}
    
    for cam_name in sorted_cameras:
        frame = camera_data[cam_name]
        data = frame.data
        
        x = data[:, 0]
        y = data[:, 1]
        u = data[:, 2]
        v = data[:, 3]
        valid = data[:, 5].astype(int) == 1
        
        # Filtrar válidos
        mask = valid & np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
        if np.sum(mask) < 1:
            continue
        
        x = x[mask]
        y = y[mask]
        u = u[mask]
        v = v[mask]
        
        # Transformar a coordenadas globales
        tfm = build_transform(cam_name)
        xg, yg = tfm.local_mm_to_global_mm(x, y)
        ug, vg = tfm.local_vectors_to_global(u, v)
        
        # Calcular magnitud
        mag = np.sqrt(ug**2 + vg**2)
        
        # Prioridad de esta cámara
        priority = CAMERA_PRIORITY_ORDER.get(cam_name, 1) if USE_CAMERA_PRIORITY else 1
        
        # Agregar puntos
        for i in range(len(xg)):
            point_data = (xg[i], yg[i], ug[i], vg[i], mag[i], 1, priority)
            
            if USE_CAMERA_PRIORITY and PRIORITY_MODE == "override":
                # Modo sobrescritura: crear clave basada en posición aproximada
                # Redondear a grilla de 1mm para detectar solapamientos
                grid_key = (round(xg[i]), round(yg[i]))
                
                # Si ya existe un punto en esta celda, comparar prioridades
                if grid_key in grid_points:
                    existing_priority = grid_points[grid_key][6]
                    if priority > existing_priority:
                        # Esta cámara tiene mayor prioridad, sobrescribe
                        grid_points[grid_key] = point_data
                    # Si prioridad menor o igual, no sobrescribe
                else:
                    grid_points[grid_key] = point_data
            else:
                # Sin prioridad o modo weighted: mantener todos los puntos
                # Usar índice único para cada punto
                unique_key = (xg[i], yg[i], i, cam_name)
                grid_points[unique_key] = point_data
    
    if not grid_points:
        return None
    
    # Convertir a array numpy
    points_list = list(grid_points.values())
    merged_array = np.array([p[:6] for p in points_list])  # Solo primeras 6 columnas
    
    return merged_array


def process_single_timestamp(args):
    """Procesa un timestamp: genera archivos .txt para cada cámara"""
    idx, t, camera_folder_map = args
    
    saved_files = save_interpolated_camera_frames(t, camera_folder_map)
    
    stats = {
        'timestamp': t,
        'n_files': len(saved_files),
        'cameras': ','.join(saved_files.keys()) if saved_files else ''
    }
    
    return idx, stats


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def generate_interpolated_files():
    """
    Genera archivos PIV interpolados individuales por cámara en formato .txt.
    
    Returns:
        int: Número de timestamps procesados exitosamente
    """
    print("\n" + "="*70)
    print("GENERACIÓN DE ARCHIVOS PIV INTERPOLADOS")
    print("="*70)
    
    # Mostrar modo de fusión
    from Analisis.PIV.InterpolarPIV.piv_config import USE_CAMERA_PRIORITY, PRIORITY_MODE, CAMERA_PRIORITY_ORDER
    if USE_CAMERA_PRIORITY and PRIORITY_MODE == "override":
        print(f"🎯 Modo de fusión: SOBRESCRITURA por prioridad")
        priorities = [f"{cam}({CAMERA_PRIORITY_ORDER.get(cam, 0)})" 
                     for cam in sorted(CAMERA_PRIORITY_ORDER.keys(), 
                                      key=lambda c: CAMERA_PRIORITY_ORDER.get(c, 0), 
                                      reverse=True)]
        print(f"   Orden de prioridad: {' > '.join(priorities)}")
    elif USE_CAMERA_PRIORITY:
        print(f"⚖️  Modo de fusión: Promedio ponderado por prioridad")
    else:
        print(f"📊 Modo de fusión: Promedio igual de todas las cámaras")
    
    # Crear directorio de salida
    INTERPOLATED_DIR.mkdir(exist_ok=True)
    
    # DEBUG: Verificar qué ROOT_RESULTS se está usando
    print(f"\n🔍 DEBUG: ROOT_RESULTS = {piv_config.ROOT_RESULTS}")
    
    # Construir timeline
    timeline = build_global_timeline()
    print(f"Timestamps a procesar: {len(timeline)}")
    print(f"Rango temporal: {timeline[0]:.6f} - {timeline[-1]:.6f} s")
    print(f"Duración: {timeline[-1] - timeline[0]:.3f} s")
    
    # Mensaje sobre interpolación
    if MAX_TIME_GAP_FOR_INTERPOLATION == float('inf'):
        print(f"🌊 Interpolación: SIEMPRE (flujo laminar)")
    else:
        print(f"Max gap interpolación: {MAX_TIME_GAP_FOR_INTERPOLATION*1000:.1f} ms")
    
    # ANÁLISIS DE GAPS POR CÁMARA
    print("\n📊 Análisis de gaps temporales por cámara:")
    camera_folder_map = build_camera_folder_map(piv_config.ROOT_RESULTS)
    
    for cam_name in sorted(CAM_PROFILES.keys()):
        folder = camera_folder_map.get(cam_name)
        if folder is None:
            print(f"  {cam_name}: ❌ No encontrada")
            continue
        
        time_index = build_time_index_for_camera(folder)
        if not time_index:
            print(f"  {cam_name}: ❌ Sin datos")
            continue
        
        times = [t for t, _ in time_index]
        gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
        
        if gaps:
            max_gap = max(gaps)
            mean_gap = np.mean(gaps)
            
            if MAX_TIME_GAP_FOR_INTERPOLATION == float('inf'):
                # Modo flujo laminar: siempre interpola
                status = "✓"
                print(f"  {cam_name}: {len(time_index)} frames, "
                      f"gap medio={mean_gap*1000:.2f}ms, "
                      f"gap max={max_gap*1000:.2f}ms {status}")
                print(f"           └─ Interpolación lineal aplicada a todos los gaps")
            else:
                gaps_over_threshold = sum(1 for g in gaps if g > MAX_TIME_GAP_FOR_INTERPOLATION)
                status = "✓" if max_gap <= MAX_TIME_GAP_FOR_INTERPOLATION else "⚠️"
                print(f"  {cam_name}: {len(time_index)} frames, "
                      f"gap medio={mean_gap*1000:.2f}ms, "
                      f"gap max={max_gap*1000:.2f}ms {status}")
                
                if gaps_over_threshold > 0:
                    print(f"           └─ {gaps_over_threshold} gaps > {MAX_TIME_GAP_FOR_INTERPOLATION*1000:.0f}ms "
                          "(se usará frame más cercano)")
    
    # Mostrar estructura de carpetas que se crearán
    print("\n📁 Estructura de salida:")
    print(f"   {INTERPOLATED_DIR.absolute()}/")
    for cam_name in sorted(CAM_PROFILES.keys()):
        folder = camera_folder_map.get(cam_name)
        if folder:
            subfolder_name = extract_subfolder_name(folder)
            print(f"   ├── {subfolder_name}/  ← {folder.name}")
    
    print(f"\n🚀 Generando archivos ({NUM_WORKERS} workers)...")
    
    # Preparar argumentos
    args_list = [(idx, t, camera_folder_map) for idx, t in enumerate(timeline)]
    
    # Procesar en paralelo
    if USE_MULTIPROCESSING and NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_single_timestamp, args_list),
                total=len(args_list),
                desc="Interpolando"
            ))
    else:
        results = [process_single_timestamp(args) for args in tqdm(args_list, desc="Interpolando")]
    
    # Estadísticas
    successful = sum(1 for _, stats in results if stats is not None)
    total_files = sum(stats.get('n_files', 0) for _, stats in results if stats is not None)
    
    camera_usage = {cam: 0 for cam in CAM_PROFILES.keys()}
    for _, stats in results:
        if stats and stats['cameras']:
            for cam in stats['cameras'].split(','):
                if cam in camera_usage:
                    camera_usage[cam] += 1
    
    print(f"\n✓ Interpolación completada: {successful}/{len(timeline)} timestamps")
    print(f"✓ Total archivos generados: {total_files} archivos .txt")
    print(f"✓ Ubicación: {INTERPOLATED_DIR.absolute()}")
    
    # Mostrar archivos por subcarpeta
    if INTERPOLATED_DIR.exists():
        subfolders = sorted([d for d in INTERPOLATED_DIR.iterdir() if d.is_dir()])
        if subfolders:
            print(f"\n📁 Archivos por subcarpeta:")
            for subfolder in subfolders:
                n_files = len(list(subfolder.glob("*.txt")))
                print(f"   {subfolder.name}/ → {n_files} archivos")
    
    print(f"\n📊 Cobertura por cámara:")
    for cam_name in sorted(camera_usage.keys()):
        count = camera_usage[cam_name]
        percentage = 100 * count / successful if successful > 0 else 0
        print(f"  {cam_name}: {count}/{successful} timestamps ({percentage:.1f}%)")
    
    return successful


if __name__ == "__main__":
    generate_interpolated_files()