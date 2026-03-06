# yolo_detector.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import os
import numpy as np
from ultralytics import YOLO
from models import Detection
from config import TrackingConfig


def next_predict_folder(runs_segment_dir: Path) -> Path:
    runs_segment_dir.mkdir(parents=True, exist_ok=True)
    folders = [p for p in runs_segment_dir.iterdir() if p.is_dir() and p.name.startswith("predict")]

    nums = []
    for p in folders:
        if p.name == "predict":
            nums.append(0)
        else:
            try:
                nums.append(int(p.name.replace("predict", "")))
            except ValueError:
                pass

    if not nums:
        return runs_segment_dir / "predict"
    return runs_segment_dir / f"predict{max(nums) + 1}"


@dataclass
class YOLODetector:
    weights_path: Path
    cfg: TrackingConfig

    def __post_init__(self) -> None:
        self.model = YOLO(str(self.weights_path))

    def list_images(self, images_dir: Path) -> List[Path]:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        imgs = sorted(imgs)

        if self.cfg.max_images is not None:
            imgs = imgs[: self.cfg.max_images]
        if not imgs:
            raise RuntimeError(f"No hay imágenes en: {images_dir}")
        return imgs

    def predict(self, image_path: Path, save_dir: Path) -> List[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=float(self.cfg.conf),
            save=True,
            save_dir=str(save_dir),
            hide_labels=True,
            line_thickness=1
        )

        if not results or not results[0].boxes:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        dets: List[Detection] = []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            length_px = float(max(w, h))

            # OJO: esto NO es orientación real de fibra, solo bbox axis-aligned
            angle_deg = float(np.degrees(np.arctan2(h, w)))

            dets.append(Detection(
                cx=float(cx), cy=float(cy),
                angle_deg=angle_deg,
                length_px=length_px,
                score=float(sc),
                box_xyxy=np.array([x1, y1, x2, y2], dtype=float)
            ))

        return dets