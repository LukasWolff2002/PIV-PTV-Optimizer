from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# VARIABLES Y PARÁMETROS PTV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- DIRECTORIOS PTV ----------
PTV_BASE_DIR        = PROJECT_ROOT / "PTV" / "Tomas"
RESULTS_PTV_ROOT    = PROJECT_ROOT / "ResultadosPTV"
RUNS_SEGMENT_DIR    = PROJECT_ROOT / "runs" / "segment"
PTV_PREPROCESSED    = PROJECT_ROOT / "PTVPreprocesadas"
PTV_MASKS_ROOT      = PROJECT_ROOT / "PTVMascaras"
PTV_MODELS_DIR      = PROJECT_ROOT / "PTV" / "Codes" / "Segmentation-Models"

# ---------- FILTRO PARA SELECCIONAR CARPETAS ----------
PTV_METODO = "ptv"

# ---------- PREPROCESAMIENTO PTV ----------
DELETE_EXISTING_PRE_PTV   = True
DELETE_EXISTING_MASKS_PTV = True

CAM_PREPROCESS_PARAMS_PTV = {
    'cam1': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 175,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 1.4737,
        'highpass_enabled': False, 'highpass_size': 28,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1316, 'max_intensity': 0.6447,
    },
    'cam2': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 155,
        'clahe_clip_limit': 0.0010,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False, 'highpass_size': 14,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1974, 'max_intensity': 0.8421,
    },
    'cam3': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 159,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False, 'highpass_size': 15,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1184, 'max_intensity': 0.7763,
    },
    'cam4': {
        'roi_enabled': False,
        'roi_x': 0, 'roi_y': 0, 'roi_width': 100, 'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 200,
        'clahe_clip_limit': 0.0635,
        'intensity_capping': True,
        'capping_n_std': 3.1053,
        'highpass_enabled': False, 'highpass_size': 15,
        'wiener_enabled': False, 'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.1053, 'max_intensity': 0.8289,
    },
}

# ---------- MÁSCARAS DINÁMICAS PTV ----------
MASK_CONF_PTV    = 0.25
MASK_DEVICE_PTV  = "0"
INVERT_MASK_PTV  = True

def ptv_mask_model_path(cam: int) -> Path:
    return PTV_MODELS_DIR / f"cam{cam}-ptv-yolo26.pt"

# ---------- MODELO YOLO TRACKING ----------
YOLO_TRACK_MODEL = PROJECT_ROOT / "PTV" / "Codes" / "Segmentation-Models" / "best.pt"
DEVICE_PTV = 0

# ---------- REGIONES TEMPORALES PTV ----------
# Estructura de un bloque: img_i → [skip frames] → img_{i+skip+1}
# dt efectivo = (skip_frames + 1) / fps
#
# max_dist_mm por región:
#   - Controla el gate espacial del tracker para esa región.
#   - None → usa el valor global de CAM_TRACKING_PARAMS[cam]["max_dist_mm"].
#   - Regla de dedo: max_dist_mm ≈ velocidad_max_esperada (mm/s) × dt_s × factor_seguridad
#     Ejemplo: flujo a 5 mm/s con dt=9/220 s ≈ 41 ms → despl. máx ≈ 0.2 mm → 0.5 mm con margen.

@dataclass
class PTVTemporalRegion:
    """
    Define una región temporal para el muestreo adaptativo en PTV.

    Un bloque PTV es: imagen_seleccionada + skip_frames → siguiente imagen seleccionada.
    dt efectivo = (skip_frames + 1) / fps

    La secuencia que ve el tracker dentro de una región es:
        img[start], img[start + (skip+1)], img[start + 2*(skip+1)], ...

    max_dist_mm:
        Gate espacial del tracker para esta región, en mm.
        None = usar valor global de CAM_TRACKING_PARAMS.
        Especificar explícitamente cuando el desplazamiento entre frames observados
        difiere mucho del de la región de referencia (alta_velocidad con skip=0).
    """
    name: str
    start_time: float           # segundos desde inicio de captura
    end_time: Optional[float]   # None = hasta el final
    skip_frames: int            # frames a saltar entre observaciones consecutivas
    fps: float
    max_dist_mm: Optional[float] = None   # gate espacial específico; None = global
    _total_frames_available: Optional[int] = field(default=None, repr=False)

    def __post_init__(self):
        if self.skip_frames < 0:
            raise ValueError(
                f"skip_frames no puede ser negativo en región '{self.name}'"
            )
        if self.end_time is not None and self.start_time >= self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) >= end_time ({self.end_time}) "
                f"en región '{self.name}'"
            )
        if self.start_time < 0:
            raise ValueError(f"start_time no puede ser negativo en región '{self.name}'")
        if self.max_dist_mm is not None and self.max_dist_mm <= 0:
            raise ValueError(
                f"max_dist_mm debe ser > 0 en región '{self.name}', "
                f"recibido: {self.max_dist_mm}"
            )

    @property
    def start_frame(self) -> int:
        return int(self.start_time * self.fps)

    @property
    def end_frame(self) -> int:
        if self.end_time is None:
            if self._total_frames_available is None:
                raise RuntimeError(
                    f"Región '{self.name}' tiene end_time=None pero "
                    f"_total_frames_available no está seteado."
                )
            return self._total_frames_available
        return int(self.end_time * self.fps)

    def set_total_frames_available(self, total: int) -> None:
        self._total_frames_available = total

    @property
    def total_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def dt_s(self) -> float:
        """Δt en segundos entre observaciones consecutivas del tracker."""
        return (self.skip_frames + 1) / self.fps

    @property
    def dt_ms(self) -> float:
        return self.dt_s * 1000.0

    @property
    def stride(self) -> int:
        """Paso en frames originales entre observaciones: skip + 1."""
        return self.skip_frames + 1

    @property
    def n_selected_frames(self) -> int:
        """Número de frames que el tracker efectivamente verá en esta región."""
        if self.total_frames <= 0:
            return 0
        return max(0, (self.total_frames - 1) // self.stride + 1)

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "start_time":   self.start_time,
            "end_time":     self.end_time,
            "start_frame":  self.start_frame,
            "end_frame":    self.end_frame,
            "total_frames": self.total_frames,
            "fps":          self.fps,
            "skip_frames":  self.skip_frames,
            "stride":       self.stride,
            "dt_ms":        self.dt_ms,
            "n_selected":   self.n_selected_frames,
            "max_dist_mm":  self.max_dist_mm,   # None si usa valor global
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PTVTemporalRegion":
        return cls(
            name=d["name"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            skip_frames=d["skip_frames"],
            fps=d["fps"],
            max_dist_mm=d.get("max_dist_mm"),   # None si no está en el JSON
        )

    def __repr__(self) -> str:
        end_str  = f"{self.end_time:.1f}s" if self.end_time is not None else "END"
        gate_str = f"{self.max_dist_mm}mm" if self.max_dist_mm is not None else "global"
        return (
            f"PTVTemporalRegion('{self.name}', "
            f"t={self.start_time:.1f}-{end_str}, "
            f"skip={self.skip_frames}, Δt={self.dt_ms:.2f}ms, gate={gate_str})"
        )


# ---------- REGIONES TEMPORALES CARBOPOL 02 ----------
# Criterio de max_dist_mm:
#   - alta_velocidad  (skip=0, dt≈4.5ms)  : flujo rápido, pero dt pequeño → 2.0 mm
#   - media_velocidad (skip=2, dt≈13.6ms) : flujo moderado → 3.0 mm
#   - baja_velocidad  (skip=8, dt≈40.9ms) : flujo lento, dt grande → 1.5 mm
#                     (en baja vel. la fibra se mueve poco aunque dt sea grande)
TEMPORAL_REGIONS_PTV_CAR02 = {
    1: [
        PTVTemporalRegion(name="alta_velocidad",     start_time=0.0,  end_time=1.5,  skip_frames=0, fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="media_velocidad",    start_time=1.5,  end_time=3.0,  skip_frames=2, fps=220.0, max_dist_mm=3.0),
        PTVTemporalRegion(name="baja_velocidad",     start_time=3.0,  end_time=6.0,  skip_frames=4, fps=220.0, max_dist_mm=3.0),
        PTVTemporalRegion(name="muy_baja_velocidad", start_time=6.0,  end_time=20.0, skip_frames=8, fps=220.0, max_dist_mm=4.0),
    ],
    2: [
        PTVTemporalRegion(name="alta_velocidad",     start_time=0.0,  end_time=1.5,  skip_frames=0, fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="media_velocidad",    start_time=1.5,  end_time=3.0,  skip_frames=2, fps=220.0, max_dist_mm=3.0),
        PTVTemporalRegion(name="baja_velocidad",     start_time=3.0,  end_time=6.0,  skip_frames=4, fps=220.0, max_dist_mm=3.0),
        PTVTemporalRegion(name="muy_baja_velocidad", start_time=6.0,  end_time=20.0, skip_frames=8, fps=220.0, max_dist_mm=4.0),
    ],
    3: [
        PTVTemporalRegion(name="alta_velocidad",     start_time=0.0,  end_time=2.0,  skip_frames=0, fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="media_velocidad",    start_time=2.0,  end_time=5.0,  skip_frames=2, fps=220.0, max_dist_mm=3.0),
        PTVTemporalRegion(name="baja_velocidad",     start_time=5.0,  end_time=7.0,  skip_frames=4, fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="muy_baja_velocidad", start_time=7.0,  end_time=20.0, skip_frames=8, fps=220.0, max_dist_mm=1.5),
    ],
    4: [
        PTVTemporalRegion(name="alta_velocidad",  start_time=0.0,  end_time=4.5,  skip_frames=0, fps=660.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="media_velocidad", start_time=4.5,  end_time=8.0,  skip_frames=2, fps=660.0, max_dist_mm=2.5),
        PTVTemporalRegion(name="baja_velocidad",  start_time=8.0,  end_time=20.0, skip_frames=8, fps=660.0, max_dist_mm=1.5),
    ],
}

# ---------- REGIONES TEMPORALES CARBOPOL 05 ----------
TEMPORAL_REGIONS_PTV_CAR05 = {
    1: [
        PTVTemporalRegion(name="alta_velocidad",         start_time=0.0,  end_time=0.2,  skip_frames=0,  fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="media_velocidad",        start_time=0.2,  end_time=5.0,  skip_frames=1,  fps=220.0, max_dist_mm=2.5),
        PTVTemporalRegion(name="baja_velocidad",         start_time=5.0,  end_time=8.0,  skip_frames=2,  fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="muy_baja_velocidad",     start_time=8.0,  end_time=15.0, skip_frames=4,  fps=220.0, max_dist_mm=1.5),
        PTVTemporalRegion(name="extrema_baja_velocidad", start_time=15.0, end_time=40.0, skip_frames=8,  fps=220.0, max_dist_mm=1.0),
    ],
    2: [
        PTVTemporalRegion(name="alta_velocidad",  start_time=0.0,  end_time=15.0, skip_frames=8,  fps=220.0, max_dist_mm=2.5),
        PTVTemporalRegion(name="media_velocidad", start_time=15.0, end_time=40.0, skip_frames=16, fps=220.0, max_dist_mm=1.5),
    ],
    3: [
        PTVTemporalRegion(name="sin_datos",      start_time=0.0,  end_time=8.0,  skip_frames=0,  fps=220.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="alta_velocidad", start_time=8.0,  end_time=40.0, skip_frames=16, fps=220.0, max_dist_mm=2.0),
    ],
    4: [
        PTVTemporalRegion(name="alta_velocidad",     start_time=0.0,  end_time=1.5,  skip_frames=0, fps=660.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="media_velocidad",    start_time=1.5,  end_time=7.5,  skip_frames=2, fps=660.0, max_dist_mm=2.5),
        PTVTemporalRegion(name="baja_velocidad",     start_time=7.5,  end_time=20.0, skip_frames=4, fps=660.0, max_dist_mm=2.0),
        PTVTemporalRegion(name="muy_baja_velocidad", start_time=20.0, end_time=40.0, skip_frames=8, fps=660.0, max_dist_mm=1.5),
    ],
}


def get_ptv_temporal_regions(cam: int, carbopol: str) -> list[PTVTemporalRegion] | None:
    """Retorna regiones temporales PTV para una cámara y carbopol."""
    carbopol_normalized = carbopol.zfill(2)
    if carbopol_normalized == "02":
        return TEMPORAL_REGIONS_PTV_CAR02.get(cam)
    elif carbopol_normalized == "05":
        return TEMPORAL_REGIONS_PTV_CAR05.get(cam)
    return None


# ---------- PARÁMETROS PTV ----------
MAX_IMAGES      = 2200
ALPHA           = 0.95
BETA            = 0.95
GAMMA           = 0.05
CONF_TRACK      = 0.1
MIN_FRAMES_KEEP = 5
ANNOTATE        = True
MAX_MISSES      = 2

SAVE_IMAGES = True

GATE_X     = 10
GATE_Y     = 10
GATE_ANGLE = 5

FIBER_LENGTH_MM = 13.0
FIBER_WIDTH_MM  = 0.2

FEAT_WEIGHTS = (1.0, 1.0, 0.3, 2.0, 2.0)

# max_dist_mm global: fallback cuando la región no especifica max_dist_mm.
CAM_TRACKING_PARAMS = {
    1: dict(sim_threshold=0.99, max_dist_mm=2.0),
    2: dict(sim_threshold=0.99, max_dist_mm=2.0),
    3: dict(sim_threshold=0.99, max_dist_mm=2.0),
    4: dict(sim_threshold=0.99, max_dist_mm=2.0),
}

SAHI_SCALE_FACTOR  = 2
SAHI_TILE_SIZE     = 640
SAHI_OVERLAP_RATIO = 0.1
SAHI_IOU_THRESHOLD = 0.3

VIZ_TAIL_LENGTH  = 0
VIZ_UPDATE_EVERY = 1

DELETE_PREDICT_FOLDERS = False

USE_TEMPORAL_REGIONS_PTV = True


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_tracking_params(cam: int) -> dict:
    return CAM_TRACKING_PARAMS.get(cam, CAM_TRACKING_PARAMS[1])


def get_l_ref_px(cam: int, cam_profiles: dict) -> float:
    px_per_mm = cam_profiles.get(cam, {}).get("px_per_mm", 7.8)
    return FIBER_LENGTH_MM * px_per_mm


def get_max_dist_px(cam: int, cam_profiles: dict) -> float:
    """Gate global (fallback). Usar get_max_dist_px_for_region cuando hay regiones."""
    px_per_mm   = cam_profiles.get(cam, {}).get("px_per_mm", 7.8)
    max_dist_mm = get_tracking_params(cam)["max_dist_mm"]
    return max_dist_mm * px_per_mm


def get_max_dist_px_for_region(
    region_max_dist_mm: Optional[float],
    cam: int,
    cam_profiles: dict,
) -> float:
    """
    Retorna max_dist_px efectivo para una región.
    Si la región define max_dist_mm, lo convierte a px.
    Si no (None), usa el valor global de CAM_TRACKING_PARAMS.
    """
    px_per_mm = cam_profiles.get(cam, {}).get("px_per_mm", 7.8)
    if region_max_dist_mm is not None:
        return region_max_dist_mm * px_per_mm
    return get_tracking_params(cam)["max_dist_mm"] * px_per_mm