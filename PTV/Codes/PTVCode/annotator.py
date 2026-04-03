"""
annotator.py
============
Dibuja detecciones y tracks sobre frames para revisión visual.
"""
from __future__ import annotations
import math
from pathlib import Path

import cv2
import numpy as np

from .models import Detection, Track


def annotate_frame(
    image_rgb: np.ndarray,
    detections: list[Detection],
    tracks: list[Track],
    frame_idx: int,
    image_name: str,
    out_path: Path,
    gate_x_px: float,
    gate_y_px: float,
) -> None:
    """
    Guarda frame anotado en BGR con:
    - Bounding boxes y centroides de detecciones (cian)
    - Eje de orientación de cada fibra detectada (amarillo)
    - Centroides de tracks activos con ID (verde)
    - Gate de búsqueda (magenta)
    """
    canvas_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    # Detecciones
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), (0, 220, 255), 1)
        cv2.circle(canvas_bgr, (int(round(d.cx)), int(round(d.cy))), 2, (0, 220, 255), -1)

        half = max(4, int(round(d.length_px / 2.0)))
        ang = math.radians(d.angle_deg)
        dx = int(round(math.cos(ang) * half))
        dy = int(round(math.sin(ang) * half))
        p1 = (int(round(d.cx - dx)), int(round(d.cy - dy)))
        p2 = (int(round(d.cx + dx)), int(round(d.cy + dy)))
        cv2.line(canvas_bgr, p1, p2, (255, 255, 0), 1)

    # Tracks
    for tr in tracks:
        recs = [r for r in tr.history
                if r.frame_idx == frame_idx and r.image_name == image_name]
        if not recs:
            continue
        rec = recs[-1]
        cx, cy = int(round(rec.x)), int(round(rec.y))
        cv2.circle(canvas_bgr, (cx, cy), 3, (0, 255, 0), -1)
        #cv2.putText(
        #    canvas_bgr, f"ID {tr.track_id}",
        #    (cx + 6, cy - 6),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
        #)
        gx, gy = int(round(gate_x_px)), int(round(gate_y_px))
        cv2.rectangle(canvas_bgr, (cx - gx, cy - gy), (cx + gx, cy + gy), (255, 0, 255), 1)

    cv2.imwrite(str(out_path), canvas_bgr)
