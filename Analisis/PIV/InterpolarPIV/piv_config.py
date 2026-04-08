"""
CONFIGURACIÓN Y UTILIDADES COMPARTIDAS - PIV ANIMATION SYSTEM
==============================================================
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

# Directorios
ROOT_RESULTS = Path("PIV_Results")
ROOT_FOTOS = Path("FOTOS")
CACHE_DIR = Path("piv_cache")
INTERPOLATED_DIR = Path("PIV_INTERPOLADO")  # Nueva carpeta para archivos interpolados

# Optimización
USE_MULTIPROCESSING = True
NUM_WORKERS = 6  # Ajustar según CPU

# Sistema de referencia
REFERENCE_CAMERA = "Cam 2"

# Grilla
GRID_DX_MM = 2
GRID_DY_MM = 2

# Forzar inclusión de todas las cámaras configuradas
INCLUDE_ALL_CAMERA_FOOTPRINTS = True

# Suavizado
FINAL_SMOOTH_SIGMA = 1.5
VISUALIZATION_SMOOTH_SIGMA = 2.0

# Interpolación temporal
MAX_TIME_GAP_FOR_INTERPOLATION = float('inf')  # Interpolar siempre (flujo laminar)
# Nota: Para flujos laminares, la interpolación lineal es válida incluso con gaps grandes
# Si tu flujo fuera turbulento, usa un valor como 0.05 (50ms)

# Sistema de prioridad de cámaras
USE_CAMERA_PRIORITY = True  # Si True, usa prioridades definidas abajo

# Modo de prioridad:
# - "weighted": Promedio ponderado por prioridad (comportamiento original)
# - "override": Sobrescritura - cámara de mayor prioridad REEMPLAZA a las de menor prioridad
PRIORITY_MODE = "override"  # Cambiar a "weighted" para promedio ponderado

CAMERA_PRIORITY_ORDER = {
    # Prioridad más alta = se usa preferentemente en caso de solapamiento
    # Valores más altos tienen prioridad
    "Cam 4": 4,  # Máxima prioridad - SOBRESCRIBE a todas
    "Cam 2": 3,
    "Cam 3": 2,
    "Cam 1": 1,  # Mínima prioridad - es SOBRESCRITA por las demás
}

# Filtros temporales
START_TIME_S = None
END_TIME_S = None
TIME_STEP_S = None

# Perfiles de cámaras
CAM_PROFILES = {
    "Cam 1": dict(fps=220, dt_ms=1000 * (1 / 220), px_per_mm=8.0, width_px=1024, height_px=1024),
    "Cam 2": dict(fps=220, dt_ms=1000 * (1 / 220), px_per_mm=7.8, width_px=1024, height_px=1024),
    "Cam 3": dict(fps=220, dt_ms=1000 * (1 / 220), px_per_mm=7.8, width_px=1024, height_px=1024),
    "Cam 4": dict(fps=660, dt_ms=1000 * (1 / 660), px_per_mm=10.7, width_px=384, height_px=384),
}

EDITOR_CONFIG = {
    "Cam 1": {"ruta": "cam-1.tiff", "x": -432.0, "y": -930.0, "rot": -30.0, "exp": 0.21, "alpha": 0.66, "escala": 0.98},
    "Cam 2": {"ruta": "cam-2.tiff", "x": 450.0, "y": 0.0, "rot": 0.0, "exp": 0.19, "alpha": 0.43253968253968256, "escala": 1.0},
    "Cam 3": {"ruta": "cam-3.tiff", "x": 1350.0, "y": -65.0, "rot": 0, "exp": 1.5, "alpha": 0.29, "escala": 1.05},
    "Cam 4": {"ruta": "cam-4.tiff", "x": 320.0, "y": -165.0, "rot": -30.0, "exp": 0.0, "alpha": 0.5809817754262199, "escala": 0.7},
}

PRIORITY_WEIGHTS: Dict[str, float] = {}
DEFAULT_CAMERA_WEIGHT = 1.0

CAM_COLORS = {
    "Cam 1": "#1f77b4",
    "Cam 2": "#d62728",
    "Cam 3": "#2ca02c",
    "Cam 4": "#9467bd",
}


# ============================================================
# CLASES DE DATOS
# ============================================================

@dataclass
class CameraTransform:
    """Transformación de coordenadas de cámara local a global"""
    name: str
    px_per_mm: float
    width_px: float
    height_px: float
    scale: float
    rot_deg: float
    tx_canvas_px: float
    ty_canvas_px: float
    ref_px_per_mm: float

    @property
    def scaled_width_px(self) -> float:
        return self.width_px * self.scale

    @property
    def scaled_height_px(self) -> float:
        return self.height_px * self.scale

    @property
    def center_scaled_px(self) -> np.ndarray:
        return np.array([self.scaled_width_px / 2.0, self.scaled_height_px / 2.0], dtype=float)

    def rotation_matrix_image(self) -> np.ndarray:
        theta = -np.deg2rad(self.rot_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def rotated_bbox_shift(self) -> np.ndarray:
        corners = np.array([[0.0, 0.0], [self.scaled_width_px, 0.0],
                          [self.scaled_width_px, self.scaled_height_px], [0.0, self.scaled_height_px]], dtype=float)
        center = self.center_scaled_px
        R = self.rotation_matrix_image()
        rc = (corners - center) @ R.T + center
        return rc.min(axis=0)

    def local_mm_to_global_mm(self, x_mm: np.ndarray, y_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforma coordenadas locales (mm) a globales (mm)"""
        pts_mm_local = np.column_stack([x_mm, y_mm]).astype(float)
        pts_px_local = pts_mm_local * self.px_per_mm
        pts_px_scaled = pts_px_local * self.scale
        center = self.center_scaled_px
        R = self.rotation_matrix_image()
        pts_px_rot = (pts_px_scaled - center) @ R.T + center
        min_xy = self.rotated_bbox_shift()
        pts_px_expand = pts_px_rot - min_xy
        pts_canvas_px = pts_px_expand + np.array([self.tx_canvas_px, self.ty_canvas_px], dtype=float)
        pts_global_mm = pts_canvas_px / self.ref_px_per_mm
        return pts_global_mm[:, 0], pts_global_mm[:, 1]
    
    def local_vectors_to_global(self, u_local: np.ndarray, v_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforma vectores de velocidad locales a globales"""
        vecs = np.column_stack([u_local, v_local]).astype(float)
        R = self.rotation_matrix_image()
        vecs_rotated = vecs @ R.T
        return vecs_rotated[:, 0], vecs_rotated[:, 1]

    def transformed_footprint_mm(self) -> np.ndarray:
        """Retorna el footprint de la cámara en coordenadas globales"""
        xcorn = np.array([0.0, self.width_px, self.width_px, 0.0, 0.0]) / self.px_per_mm
        ycorn = np.array([0.0, 0.0, self.height_px, self.height_px, 0.0]) / self.px_per_mm
        gx, gy = self.local_mm_to_global_mm(xcorn, ycorn)
        return np.column_stack([gx, gy])


@dataclass
class PIVFrame:
    """Frame de datos PIV de una cámara"""
    path: Path
    timestamp_s: float
    dt_ms: Optional[float]
    px_per_mm: Optional[float]
    data: np.ndarray


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_reference_px_per_mm() -> float:
    """Obtiene la resolución de la cámara de referencia"""
    return float(CAM_PROFILES[REFERENCE_CAMERA]["px_per_mm"])


def get_camera_weight(camera_name: str) -> float:
    """Obtiene el peso/prioridad de una cámara"""
    return float(PRIORITY_WEIGHTS.get(camera_name, DEFAULT_CAMERA_WEIGHT))


def build_transform(camera_name: str) -> CameraTransform:
    """Construye la transformación de coordenadas para una cámara"""
    p = CAM_PROFILES[camera_name]
    e = EDITOR_CONFIG[camera_name]
    return CameraTransform(
        name=camera_name, px_per_mm=float(p["px_per_mm"]),
        width_px=float(p["width_px"]), height_px=float(p["height_px"]),
        scale=float(e["escala"]), rot_deg=float(e["rot"]),
        tx_canvas_px=float(e["x"]), ty_canvas_px=float(e["y"]),
        ref_px_per_mm=get_reference_px_per_mm(),
    )


def build_camera_folder_map(root_results: Path) -> Dict[str, Path]:
    """Mapea nombres de cámaras a sus carpetas de resultados"""
    folder_map: Dict[str, Path] = {}
    if not root_results.exists():
        raise FileNotFoundError(f"No existe la carpeta raíz: {root_results}")
    pattern = re.compile(r"(?:^|[-_ ])cam[-_ ]*(\d+)(?:$|[-_ ])", re.IGNORECASE)
    for sub in root_results.iterdir():
        if not sub.is_dir():
            continue
        match = pattern.search(sub.name)
        if not match:
            continue
        cam_num = int(match.group(1))
        cam_name = f"Cam {cam_num}"
        folder_map[cam_name] = sub
    return folder_map


def extract_subfolder_name(folder: Path) -> str:
    """
    Extrae el nombre de subcarpeta eliminando la parte 'cam-x'.
    Ejemplo: 'resultados-cam-1' -> 'resultados'
             'piv_cam_2_data' -> 'piv_data'
             'm93-toma-1-cam-1-n-0000' -> 'm93-toma-1-n-0000'
    """
    name = folder.name
    
    # Patrones más específicos para eliminar: -cam-1-, _cam_1_, -cam1-, etc.
    # Importante: eliminar también el guion/underscore anterior y posterior
    patterns = [
        r'[-_]cam[-_]?\d+[-_]',  # -cam-1-, _cam_1_, -cam1-, _cam-1-
        r'[-_]cam[-_]?\d+$',     # -cam-1, _cam_1, -cam1 al final
        r'^cam[-_]?\d+[-_]',     # cam-1-, cam_1_, cam1- al inicio
    ]
    
    clean_name = name
    for pattern in patterns:
        clean_name = re.sub(pattern, '-', clean_name, flags=re.IGNORECASE)
    
    # Limpiar guiones/underscores consecutivos o al inicio/final
    clean_name = re.sub(r'[-_]+', '-', clean_name)
    clean_name = clean_name.strip('-_')
    
    # Si quedó vacío, usar nombre original
    return clean_name if clean_name else name


def write_piv_txt(
    path: Path, 
    data: np.ndarray, 
    timestamp_s: float, 
    dt_ms: Optional[float] = None,
    px_per_mm: Optional[float] = None,
    extra_headers: Optional[Dict[str, str]] = None
) -> None:
    """
    Escribe un archivo PIV en formato texto con header.
    
    Args:
        path: Ruta del archivo a escribir
        data: Array numpy (N x 6) con [x, y, u, v, mag, valid]
        timestamp_s: Timestamp en segundos
        dt_ms: Delta tiempo en milisegundos (opcional)
        px_per_mm: Resolución en px/mm (opcional)
        extra_headers: Headers adicionales (opcional)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        # Escribir headers
        f.write(f"# timestamp_s: {timestamp_s:.6f}\n")
        if dt_ms is not None:
            f.write(f"# DT_ms: {dt_ms:.6f}\n")
        if px_per_mm is not None:
            f.write(f"# px_per_mm: {px_per_mm:.6f}\n")
        
        # Headers adicionales
        if extra_headers:
            for key, value in extra_headers.items():
                f.write(f"# {key}: {value}\n")
        
        # Columnas
        f.write("# Columns: x(mm) y(mm) u(mm/s) v(mm/s) magnitude(mm/s) valid(0/1)\n")
        
        # Datos
        np.savetxt(f, data, fmt='%.6f %.6f %.6f %.6f %.6f %d')


def read_piv_header(path: Path) -> Dict[str, str]:
    """Lee el header de un archivo PIV"""
    meta: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def load_piv_txt(path: Path) -> np.ndarray:
    """Carga los datos numéricos de un archivo PIV"""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"El archivo {path} no tiene 6 columnas.")
    return data


def load_piv_frame(path: Path) -> PIVFrame:
    """Carga un frame PIV completo (header + datos)"""
    meta = read_piv_header(path)
    if "timestamp_s" not in meta:
        raise ValueError(f"El archivo {path} no contiene 'timestamp_s' en el header.")
    timestamp_s = float(meta["timestamp_s"])
    dt_ms = float(meta["DT_ms"]) if "DT_ms" in meta else None
    px_per_mm = float(meta["px_per_mm"]) if "px_per_mm" in meta else None
    data = load_piv_txt(path)
    return PIVFrame(path=path, timestamp_s=timestamp_s, dt_ms=dt_ms, 
                   px_per_mm=px_per_mm, data=data)


def build_time_index_for_camera(folder: Path) -> List[Tuple[float, Path]]:
    """
    Construye índice temporal de archivos PIV para una cámara.
    Lee los timestamps tal como están en los archivos (sin normalizar).
    """
    items: List[Tuple[float, Path]] = []
    for path in sorted(folder.glob("*.txt")):
        try:
            meta = read_piv_header(path)
            if "timestamp_s" not in meta:
                continue
            t = float(meta["timestamp_s"])
            items.append((t, path))
        except Exception:
            pass
    
    items.sort(key=lambda x: x[0])
    return items


def nan_gaussian(data: np.ndarray, sigma: float, allowed_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Suavizado gaussiano con manejo de NaN"""
    if sigma <= 0:
        out = data.copy()
        if allowed_mask is not None:
            out[~allowed_mask] = np.nan
        return out
    if allowed_mask is None:
        allowed_mask = np.isfinite(data)
    valid = np.isfinite(data) & allowed_mask
    filled = np.zeros_like(data, dtype=float)
    filled[valid] = data[valid]
    num = gaussian_filter(filled, sigma=sigma)
    den = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.full_like(data, np.nan, dtype=float)
    mask = (den > 1e-12) & allowed_mask
    out[mask] = num[mask] / den[mask]
    out[~allowed_mask] = np.nan
    return out


def compute_global_grid(all_points: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la grilla global a partir de puntos de todas las cámaras"""
    all_x = np.concatenate([p[0] for p in all_points])
    all_y = np.concatenate([p[1] for p in all_points])
    min_x = float(np.nanmin(all_x))
    max_x = float(np.nanmax(all_x))
    min_y = float(np.nanmin(all_y))
    max_y = float(np.nanmax(all_y))
    dx = float(GRID_DX_MM)
    dy = float(GRID_DY_MM)
    min_x -= 1.5 * dx
    max_x += 1.5 * dx
    min_y -= 1.5 * dy
    max_y += 1.5 * dy
    x_edges = np.arange(min_x, max_x + dx, dx)
    y_edges = np.arange(min_y, max_y + dy, dy)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)
    return X, Y, x_edges, y_edges


def compute_vorticity(U: np.ndarray, V: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Calcula vorticidad a partir de campos de velocidad"""
    dv_dx = np.full_like(V, np.nan)
    du_dy = np.full_like(U, np.nan)
    dv_dx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * dx)
    du_dy[1:-1, :] = (U[2:, :] - U[:-2, :]) / (2 * dy)
    return dv_dx - du_dy


def build_global_timeline() -> List[float]:
    """
    Construye timeline global con todos los tiempos de todas las cámaras.
    Normaliza timestamps para empezar desde 0 para esta toma.
    """
    all_times = set()
    camera_folder_map = build_camera_folder_map(ROOT_RESULTS)
    
    # Recopilar todos los timestamps SIN normalizar (tal cual están en archivos)
    for cam_name in CAM_PROFILES.keys():
        folder = camera_folder_map.get(cam_name)
        if folder is None:
            continue
        time_index = build_time_index_for_camera(folder)
        for t, _ in time_index:
            all_times.add(t)
    
    if not all_times:
        return []
    
    timeline = sorted(list(all_times))
    
    # NORMALIZAR: restar el timestamp mínimo para que esta toma empiece en t=0
    t_min = timeline[0]
    timeline = [t - t_min for t in timeline]
    
    # Aplicar filtros temporales (después de normalizar)
    if START_TIME_S is not None:
        timeline = [t for t in timeline if t >= START_TIME_S]
    if END_TIME_S is not None:
        timeline = [t for t in timeline if t <= END_TIME_S]
    if TIME_STEP_S is not None and TIME_STEP_S > 0:
        if len(timeline) > 0:
            timeline_filtered = []
            for t in timeline:
                if len(timeline_filtered) == 0 or (t - timeline_filtered[-1]) >= TIME_STEP_S:
                    timeline_filtered.append(t)
            timeline = timeline_filtered
    
    return timeline