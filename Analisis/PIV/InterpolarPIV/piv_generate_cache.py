"""
GENERADOR DE CACHÉ PARA VISUALIZACIÓN
======================================
Lee archivos .txt de PIV_INTERPOLADO y genera caché .npz para rendering rápido.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from Analisis.PIV.InterpolarPIV.piv_config import (
    CACHE_DIR, INTERPOLATED_DIR, CAM_PROFILES, GRID_DX_MM, GRID_DY_MM,
    FINAL_SMOOTH_SIGMA, INCLUDE_ALL_CAMERA_FOOTPRINTS, USE_MULTIPROCESSING, NUM_WORKERS,
    USE_CAMERA_PRIORITY, PRIORITY_MODE, CAMERA_PRIORITY_ORDER,
    build_transform, get_camera_weight, compute_global_grid, 
    nan_gaussian, compute_vorticity, load_piv_frame, read_piv_header
)


# ============================================================
# LECTURA DE ARCHIVOS INTERPOLADOS
# ============================================================

def get_timeline_from_interpolated(subfolder_name: str = None) -> List[float]:
    """
    Extrae timeline de nombres de archivos en PIV_INTERPOLADO.
    
    Args:
        subfolder_name: Nombre de subcarpeta específica (ej: 'm70-toma-2-n-3000-car-02-piv')
                       Si es None, busca en todas las subcarpetas
    """
    timestamps = set()
    
    # Determinar qué subcarpetas procesar
    if subfolder_name:
        subfolders = [INTERPOLATED_DIR / subfolder_name]
    else:
        subfolders = [f for f in INTERPOLATED_DIR.iterdir() if f.is_dir()]
    
    for subfolder in subfolders:
        if not subfolder.exists() or not subfolder.is_dir():
            continue
        
        # Formato: frame_XXXXXXXXX.txt (sin sufijo de cámara)
        for txt_file in subfolder.glob("frame_*.txt"):
            try:
                # Extraer timestamp: frame_XXXXXXXXX.txt
                parts = txt_file.stem.split('_')
                if len(parts) == 2:  # frame_XXXXXXXXX
                    timestamp_us = int(parts[1])
                    timestamp_s = timestamp_us / 1e6
                    timestamps.add(timestamp_s)
            except (IndexError, ValueError):
                continue
    
    return sorted(list(timestamps))


def load_interpolated_frame(timestamp_s: float, cam_name: str) -> np.ndarray:
    """
    Carga un frame interpolado desde PIV_INTERPOLADO/.
    
    Returns:
        Array numpy con datos del frame o None si no existe
    """
    # Buscar archivo en subcarpetas
    timestamp_us = int(timestamp_s * 1e6)
    filename = f"frame_{timestamp_us:09d}.txt"
    
    for subfolder in INTERPOLATED_DIR.iterdir():
        if not subfolder.is_dir():
            continue
        
        filepath = subfolder / filename
        if filepath.exists():
            try:
                # Leer archivo y verificar que sea de la cámara correcta
                frame = load_piv_frame(filepath)
                # Verificar header de cámara si existe
                meta = read_piv_header(filepath)
                if 'camera' in meta and meta['camera'] == cam_name:
                    return frame.data
            except:
                continue
    
    return None


def load_and_transform_from_interpolated(timestamp_s: float, subfolder_name: str = None) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Carga datos desde PIV_INTERPOLADO.
    Los archivos ya están fusionados con prioridad aplicada, solo necesitamos cargarlos.
    
    Args:
        timestamp_s: Timestamp a cargar
        subfolder_name: Nombre de subcarpeta específica o None para buscar en todas
    """
    # Buscar archivo fusionado para este timestamp
    timestamp_us = int(timestamp_s * 1e6)
    filename = f"frame_{timestamp_us:09d}.txt"
    
    # Determinar qué subcarpetas buscar
    if subfolder_name:
        subfolders = [INTERPOLATED_DIR / subfolder_name]
    else:
        subfolders = [f for f in INTERPOLATED_DIR.iterdir() if f.is_dir()]
    
    for subfolder in subfolders:
        if not subfolder.exists() or not subfolder.is_dir():
            continue
        
        filepath = subfolder / filename
        if not filepath.exists():
            continue
        
        try:
            # Cargar frame fusionado
            frame = load_piv_frame(filepath)
            
            # Los datos ya están en coordenadas globales y fusionados
            data = frame.data
            xg = data[:, 0]
            yg = data[:, 1]
            ug = data[:, 2]
            vg = data[:, 3]
            valid = data[:, 5].astype(int) == 1
            
            # Filtrar válidos
            mask = valid & np.isfinite(xg) & np.isfinite(yg) & np.isfinite(ug) & np.isfinite(vg)
            if np.sum(mask) < 4:
                continue
            
            xg = xg[mask]
            yg = yg[mask]
            ug = ug[mask]
            vg = vg[mask]
            
            # Retornar como si fuera una "cámara virtual" fusionada
            return [("merged", xg, yg, ug, vg)]
        
        except Exception as e:
            print(f"⚠️  Error cargando {filepath}: {e}")
            continue
    
    return []


# ============================================================
# FUSIÓN EN GRILLA
# ============================================================

def accumulate_points_to_grid(xg, yg, u, v, x_edges, y_edges, cam_weight, sum_u, sum_v, sum_w):
    """Acumula puntos de velocidad en la grilla"""
    ix = np.searchsorted(x_edges, xg, side="right") - 1
    iy = np.searchsorted(y_edges, yg, side="right") - 1
    valid = ((ix >= 0) & (ix < sum_u.shape[1]) & (iy >= 0) & (iy < sum_u.shape[0]) & 
             np.isfinite(u) & np.isfinite(v))
    ix = ix[valid]
    iy = iy[valid]
    u = u[valid]
    v = v[valid]
    if len(ix) == 0:
        return
    nx = sum_u.shape[1]
    flat_idx = iy * nx + ix
    n_cells = sum_u.size
    add_u = np.bincount(flat_idx, weights=cam_weight * u, minlength=n_cells)
    add_v = np.bincount(flat_idx, weights=cam_weight * v, minlength=n_cells)
    add_w = np.bincount(flat_idx, weights=np.full_like(u, cam_weight, dtype=float), minlength=n_cells)
    sum_u += add_u.reshape(sum_u.shape)
    sum_v += add_v.reshape(sum_v.shape)
    sum_w += add_w.reshape(sum_w.shape)


def finalize_grid(sum_u, sum_v, sum_w):
    """Finaliza campos de velocidad promediando y suavizando"""
    U = np.full_like(sum_u, np.nan, dtype=float)
    V = np.full_like(sum_v, np.nan, dtype=float)
    support_mask = sum_w > 0
    U[support_mask] = sum_u[support_mask] / sum_w[support_mask]
    V[support_mask] = sum_v[support_mask] / sum_w[support_mask]
    U = nan_gaussian(U, sigma=FINAL_SMOOTH_SIGMA, allowed_mask=support_mask)
    V = nan_gaussian(V, sigma=FINAL_SMOOTH_SIGMA, allowed_mask=support_mask)
    U[~support_mask] = np.nan
    V[~support_mask] = np.nan
    return U, V


def merge_single_frame(transformed_data, X, Y, x_edges, y_edges):
    """
    Procesa datos pre-fusionados desde archivos .txt
    Los datos ya vienen fusionados con prioridad aplicada.
    """
    ny, nx = X.shape
    sum_u = np.zeros((ny, nx), dtype=float)
    sum_v = np.zeros((ny, nx), dtype=float)
    sum_w = np.zeros((ny, nx), dtype=float)
    
    # Los datos ya vienen fusionados, solo necesitamos mapearlos a la grilla
    for _, xg, yg, u, v in transformed_data:
        accumulate_points_to_grid(xg, yg, u, v, x_edges, y_edges, 1.0, sum_u, sum_v, sum_w)
    
    U, V = finalize_grid(sum_u, sum_v, sum_w)
    return U, V


# ============================================================
# PROCESAMIENTO
# ============================================================

def compute_and_save_cache_frame(args):
    """Computa y guarda un frame en caché .npz"""
    idx, t, X, Y, x_edges, y_edges, dx, dy, subfolder_name = args
    
    # Cargar datos interpolados
    data = load_and_transform_from_interpolated(t, subfolder_name)
    
    if not data:
        return idx, None
    
    # Fusionar
    U, V = merge_single_frame(data, X, Y, x_edges, y_edges)
    
    # Calcular campos derivados
    speed = np.sqrt(U**2 + V**2)
    vorticity = compute_vorticity(U, V, dx, dy)
    
    # Estadísticas
    valid_mask = np.isfinite(speed)
    if np.any(valid_mask):
        stats = {
            'v_mean': float(np.mean(speed[valid_mask])),
            'v_max': float(np.max(speed[valid_mask])),
            'v_std': float(np.std(speed[valid_mask])),
            'n_cams': len(data),
            'n_valid': int(np.sum(valid_mask))
        }
    else:
        stats = {'v_mean': 0.0, 'v_max': 0.0, 'v_std': 0.0, 'n_cams': len(data), 'n_valid': 0}
    
    # Guardar caché
    cache_file = CACHE_DIR / f"frame_{idx:05d}.npz"
    np.savez_compressed(
        cache_file,
        U=U.astype(np.float32),
        V=V.astype(np.float32),
        speed=speed.astype(np.float32),
        vorticity=vorticity.astype(np.float32),
        timestamp=t,
        **stats
    )
    
    return idx, stats


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def generate_cache_for_visualization(subfolder_name: str = None):
    """
    Genera caché .npz para visualización desde archivos .txt interpolados.
    
    Args:
        subfolder_name: Nombre de subcarpeta específica a procesar (ej: 'm70-toma-2-n-3000-car-02-piv')
                       Si es None, procesa todas las subcarpetas
    """
    print("\n" + "="*70)
    print("GENERACIÓN DE CACHÉ PARA VISUALIZACIÓN")
    print("="*70)
    
    # Verificar que existe PIV_INTERPOLADO
    if not INTERPOLATED_DIR.exists() or not any(INTERPOLATED_DIR.iterdir()):
        raise FileNotFoundError(
            f"No se encontraron archivos en {INTERPOLATED_DIR}. "
            "Ejecuta primero piv_main.py para generar archivos interpolados."
        )
    
    # Crear directorio de caché
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Extraer timeline de archivos
    timeline = get_timeline_from_interpolated(subfolder_name)
    print(f"Frames encontrados: {len(timeline)}")
    if timeline:
        print(f"Rango: {timeline[0]:.6f} - {timeline[-1]:.6f} s")
    
    # Calcular grilla global
    print("\n🗺️  Calculando dominio global...")
    all_points = []
    
    # Incluir footprints de cámaras
    if INCLUDE_ALL_CAMERA_FOOTPRINTS:
        for cam_name in CAM_PROFILES.keys():
            tfm = build_transform(cam_name)
            footprint = tfm.transformed_footprint_mm()
            all_points.append((footprint[:, 0], footprint[:, 1]))
    
    # Muestrear datos reales
    sample_indices = [0, len(timeline)//2, len(timeline)-1] if timeline else []
    for idx in sample_indices:
        sample_data = load_and_transform_from_interpolated(timeline[idx], subfolder_name)
        if sample_data:
            for cam_name, xg, yg, _, _ in sample_data:
                all_points.append((xg, yg))
    
    if not all_points:
        raise ValueError("No se pudieron cargar datos para calcular grilla")
    
    X, Y, x_edges, y_edges = compute_global_grid(all_points)
    dx, dy = GRID_DX_MM, GRID_DY_MM
    
    print(f"   Grilla: {Y.shape[0]} x {X.shape[1]} celdas")
    print(f"   Dominio: X=[{X.min():.1f}, {X.max():.1f}], Y=[{Y.min():.1f}, {Y.max():.1f}] mm")
    
    # Guardar metadatos
    metadata_file = CACHE_DIR / "metadata.npz"
    np.savez(metadata_file, X=X, Y=Y, x_edges=x_edges, y_edges=y_edges, 
             dx=dx, dy=dy, timeline=np.array(timeline))
    
    print(f"\n🚀 Procesando frames ({NUM_WORKERS} workers)...")
    print(f"   (Fusión con prioridad ya aplicada en archivos .txt)")
    
    # Preparar argumentos
    args_list = [(idx, t, X, Y, x_edges, y_edges, dx, dy, subfolder_name) 
                 for idx, t in enumerate(timeline)]
    
    # Procesar en paralelo
    if USE_MULTIPROCESSING and NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(compute_and_save_cache_frame, args_list),
                total=len(args_list),
                desc="Generando caché"
            ))
    else:
        results = [compute_and_save_cache_frame(args) for args in tqdm(args_list, desc="Generando caché")]
    
    # Contar exitosos
    successful = sum(1 for _, stats in results if stats is not None)
    
    print(f"\n✓ Caché generado: {successful}/{len(timeline)} frames")
    print(f"✓ Ubicación: {CACHE_DIR.absolute()}")
    
    return successful


if __name__ == "__main__":
    generate_cache_for_visualization()