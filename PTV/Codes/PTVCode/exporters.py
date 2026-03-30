"""
exporters.py
============
Exportación de detecciones y tracks a CSV y JSON.
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
    px_per_mm: float,
    fps: float,
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "track_id", "frame_idx", "image_name",
            "x_px", "y_px", "x_mm", "y_mm",
            "vx_px_s", "vy_px_s", "vx_mm_s", "vy_mm_s",
            "ax_px_s2", "ay_px_s2", "ax_mm_s2", "ay_mm_s2",
            "angle_deg", "omega_deg_s", "alpha_ang_deg_s2",
            "length_px", "width_px", "det_id",
        ])
        for tr in tracks:
            for rec in tr.history:
                w.writerow([
                    tr.track_id, rec.frame_idx, rec.image_name,
                    rec.x, rec.y,
                    rec.x / px_per_mm, rec.y / px_per_mm,
                    rec.vx, rec.vy,
                    rec.vx / px_per_mm, rec.vy / px_per_mm,
                    rec.ax, rec.ay,
                    rec.ax / px_per_mm, rec.ay / px_per_mm,
                    rec.angle_deg, rec.omega, rec.alpha_ang,
                    rec.length_px, rec.width_px, rec.det_id,
                ])


def export_tracks_json(tracks: list[Track], path: Path) -> None:
    data = {"tracks": [tr.to_dict() for tr in tracks]}
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )
