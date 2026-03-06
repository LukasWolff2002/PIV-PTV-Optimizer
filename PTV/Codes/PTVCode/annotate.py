# annotate.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

from models import Detection
from config import TrackingConfig


def annotate_and_save(
    processed_dir: Path,
    image_path: Path,
    detections: List[Detection],
    det_to_id: Dict[int, str],
    cfg: TrackingConfig,
) -> None:
    """
    Abre la imagen procesada por YOLO en processed_dir (mismo basename)
    y dibuja IDs + rectángulo de referencia.
    """
    out_path = processed_dir / image_path.name
    if not out_path.exists():
        # Ultralytics suele guardar en subcarpeta "predict*/" con otra estructura,
        # pero tu caso parecía guardar directo. Ajusta si tu carpeta difiere.
        return

    img = cv2.imread(str(out_path))
    if img is None:
        return

    # texto IDs
    for i, det in enumerate(detections):
        tid = det_to_id.get(i)
        if not tid:
            continue
        x1, y1, x2, y2 = det.box_xyxy
        cv2.putText(
            img,
            tid,
            (int(x1), max(0, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # rectángulo de referencia (top-right)
    h, w = img.shape[:2]
    rect_x1 = w - int(cfg.gate_x_px) - 10
    rect_y1 = 10
    rect_x2 = w - 10
    rect_y2 = 10 + int(cfg.gate_y_px)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)

    cv2.imwrite(str(out_path), img)