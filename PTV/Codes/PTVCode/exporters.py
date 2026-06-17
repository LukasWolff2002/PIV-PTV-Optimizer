"""
exporters.py
============
Exportación de detecciones, tracks y schedule a CSV y JSON.

Unidades de salida:
- Posición    : mm
- Velocidad   : mm/s
- Aceleración : mm/s²
- Longitud/ancho : mm
- Ángulo      : grados
- dt_s        : segundos (timestep real de la observación)
- timestamp_s : segundos desde inicio de captura
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
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
            # DPTV depth columns
            "defocus_score", "depth_blur_px", "depth_blur_mm",
            "depth_mm", "depth_confidence",
        ])
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            w.writerow([
                d.det_id, d.frame_idx, d.image_name,
                d.cx, d.cy, d.angle_deg,
                d.length_px, d.width_px, d.area_px, d.score,
                x1, y1, x2, y2,
                f"{d.defocus_score:.4f}",
                f"{d.depth_blur_px:.4f}",
                f"{d.depth_blur_mm:.4f}",
                f"{d.depth_mm:.4f}" if d.depth_mm is not None else "",
                f"{d.depth_confidence:.4f}",
            ])


def export_tracks_csv(
    tracks: list[Track],
    fps: float,
    path: Path,
) -> None:
    """Exporta tracks con todas las unidades en mm."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "frame_idx", "image_name",
            "region_name", "region_idx",
            "timestamp_s", "dt_s",
            "x_mm", "y_mm",
            "vx_mm_s", "vy_mm_s",
            "ax_mm_s2", "ay_mm_s2",
            "angle_deg", "omega_deg_s", "alpha_ang_deg_s2",
            "length_mm", "width_mm",
            "det_id",
            # DPTV depth columns
            "defocus_score", "depth_blur_px", "depth_blur_mm",
            "depth_mm", "depth_confidence",
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
                    f"{rec.defocus_score:.4f}",
                    f"{rec.depth_blur_px:.4f}",
                    f"{rec.depth_blur_mm:.4f}",
                    f"{rec.depth_mm:.4f}" if rec.depth_mm is not None else "",
                    f"{rec.depth_confidence:.4f}",
                ])


def export_tracks_json(
    tracks: list[Track],
    fps: float,
    temporal_regions: list | None,
    path: Path,
    dptv_config: dict | None = None,
) -> None:
    data = {
        "metadata": {
            "fps": fps,
            "units": {
                "position":        "mm",
                "velocity":        "mm/s",
                "acceleration":    "mm/s2",
                "angle":           "degrees",
                "length":          "mm",
                "width":           "mm",
                "time":            "seconds",
                "depth_blur":      "mm",
                "depth":           "mm",
                "defocus_score":   "dimensionless (1.0=in-focus)",
                "depth_confidence":"dimensionless [0,1]",
            },
            "temporal_regions": temporal_regions or [],
            "dptv": dptv_config or {"enabled": False},
        },
        "tracks": [tr.to_dict() for tr in tracks],
    }
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )


def export_schedule_csv(
    schedule: list[dict],
    tracks_filtered: list[Track],
    all_detections: list[Detection],
    path: Path,
) -> None:
    """
    Exporta una fila por frame analizado con contexto temporal completo.

    Columnas:
      sched_idx       : posición en el schedule (0-based)
      frame_idx       : índice del frame en la secuencia original de imágenes
      image_name      : nombre del archivo de imagen
      region_name     : nombre de la región temporal
      region_idx      : índice de la región (0-based)
      timestamp_s     : tiempo absoluto desde inicio de captura (s)
      dt_s            : timestep efectivo de esta observación (s)
      dt_ms           : timestep en milisegundos
      n_detections    : detecciones totales en este frame
      n_tracks_active : tracks activos (con registro en este frame)
      track_ids       : IDs de tracks activos separados por '|'

    Uso típico:
      - Construir ejes temporales correctos para animaciones (usar timestamp_s)
      - Calcular velocidades medias por frame con dt_s correcto
      - Identificar cambios de región y sus timesteps
      - Unir con tracks.csv por frame_idx para análisis combinado
    """
    # Indexar detecciones por frame
    dets_per_frame: dict[int, int] = defaultdict(int)
    for d in all_detections:
        dets_per_frame[d.frame_idx] += 1

    # Indexar tracks activos por frame (solo tracks filtrados)
    tracks_per_frame: dict[int, list[int]] = defaultdict(list)
    for tr in tracks_filtered:
        for rec in tr.history:
            tracks_per_frame[rec.frame_idx].append(tr.track_id)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sched_idx",
            "frame_idx",
            "image_name",
            "region_name",
            "region_idx",
            "timestamp_s",
            "dt_s",
            "dt_ms",
            "n_detections",
            "n_tracks_active",
            "track_ids",
        ])
        for i, entry in enumerate(schedule):
            fi          = entry["frame_idx_original"]
            track_ids   = sorted(tracks_per_frame.get(fi, []))
            track_ids_str = "|".join(str(t) for t in track_ids)
            dt_s        = entry["dt_s"]
            w.writerow([
                i,
                fi,
                entry["img_path"].name,
                entry["region_name"],
                entry["region_idx"],
                f"{entry['timestamp_s']:.6f}",
                f"{dt_s:.6f}",
                f"{dt_s * 1000:.4f}",
                dets_per_frame.get(fi, 0),
                len(track_ids),
                track_ids_str,
            ])