"""
exporters.py
============
Exportación de detecciones y tracks a CSV y JSON.

Unidades de salida:
- Posición  : mm
- Velocidad : mm/s
- Aceleración: mm/s²
- Longitud/ancho: mm
- Ángulo    : grados
- dt_s      : segundos (timestep real de la observación)
- timestamp_s: segundos desde inicio de captura
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from .models import Detection, Track
from .image_utils import np_to_builtin


def export_detections_csv(detections: list[Detection], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "det_id", "frame_idx", "image_name",
            "cx_px", "cy_px", "angle_deg",
            "length_px", "width_px", "area_px", "score",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ])
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            w.writerow([
                d.det_id, d.frame_idx, d.image_name,
                d.cx, d.cy, d.angle_deg,
                d.length_px, d.width_px, d.area_px, d.score,
                x1, y1, x2, y2,
            ])


def export_tracks_csv(
    tracks: list[Track],
    fps: float,
    path: Path,
) -> None:
    """
    Exporta tracks con todas las unidades en mm.

    Nota: px_per_mm ya fue aplicado en TrackRecord durante el tracking.
    fps se incluye en el header para referencia pero no se usa para conversión.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            # Identificación
            "track_id", "frame_idx", "image_name",
            "region_name", "region_idx",
            # Tiempo
            "timestamp_s", "dt_s",
            # Posición
            "x_mm", "y_mm",
            # Velocidad
            "vx_mm_s", "vy_mm_s",
            # Aceleración
            "ax_mm_s2", "ay_mm_s2",
            # Orientación
            "angle_deg", "omega_deg_s", "alpha_ang_deg_s2",
            # Geometría
            "length_mm", "width_mm",
            # Detección origen
            "det_id",
        ])
        for tr in tracks:
            for rec in tr.history:
                w.writerow([
                    tr.track_id,
                    rec.frame_idx,
                    rec.image_name,
                    rec.region_name,
                    rec.region_idx,
                    f"{rec.timestamp_s:.6f}",
                    f"{rec.dt_s:.6f}",
                    f"{rec.x_mm:.6f}",
                    f"{rec.y_mm:.6f}",
                    f"{rec.vx_mm_s:.6f}",
                    f"{rec.vy_mm_s:.6f}",
                    f"{rec.ax_mm_s2:.6f}",
                    f"{rec.ay_mm_s2:.6f}",
                    f"{rec.angle_deg:.4f}",
                    f"{rec.omega_deg_s:.4f}",
                    f"{rec.alpha_ang_deg_s2:.4f}",
                    f"{rec.length_mm:.4f}",
                    f"{rec.width_mm:.4f}",
                    rec.det_id,
                ])


def export_tracks_json(
    tracks: list[Track],
    fps: float,
    temporal_regions: list | None,
    path: Path,
) -> None:
    """
    Exporta tracks a JSON con metadata completa de regiones temporales.

    Estructura:
    {
      "metadata": {
        "fps": ...,
        "units": {...},
        "temporal_regions": [...]
      },
      "tracks": [...]
    }
    """
    data = {
        "metadata": {
            "fps": fps,
            "units": {
                "position":     "mm",
                "velocity":     "mm/s",
                "acceleration": "mm/s2",
                "angle":        "degrees",
                "length":       "mm",
                "width":        "mm",
                "time":         "seconds",
            },
            "temporal_regions": temporal_regions or [],
        },
        "tracks": [tr.to_dict() for tr in tracks],
    }
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )