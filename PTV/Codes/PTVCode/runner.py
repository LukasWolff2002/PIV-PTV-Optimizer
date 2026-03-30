"""
runner.py
=========
Loop principal del PTV: procesa frames, detecta, trackea y exporta.
Llamado por ptv_run.py (RunCode) después de leer el JSON de configuración.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrackingConfig
from .models import Detection, Track
from .detector import FiberYOLODetector
from .tracker import Tracker
from .image_utils import (
    ensure_dir, list_images, read_image_any,
    preprocess_frame_for_ptv, load_mask_as_bool, apply_static_mask_to_rgb,
    np_to_builtin,
)
from .exporters import export_detections_csv, export_tracks_csv, export_tracks_json
from .annotator import annotate_frame
from .visualizer import create_interactive_visualizer


def _save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )


def run_ptv(run_cfg: TrackingConfig, raw_cfg: dict) -> None:
    """
    Ejecuta el pipeline PTV completo:
      1. Preprocesamiento por frame
      2. Aplicación de máscara fija
      3. Detección YOLO
      4. Tracking ABG
      5. Exportación de resultados
      6. Visualizador HTML (si annotate=True)

    Args:
        run_cfg  : TrackingConfig ya validado
        raw_cfg  : dict original del JSON (para summary)
    """
    ensure_dir(run_cfg.out_dir)
    ann_dir = run_cfg.out_dir / "annotations"
    if run_cfg.annotate:
        ensure_dir(ann_dir)

    images = list_images(run_cfg.images_dir, run_cfg.max_images)
    if not images:
        raise RuntimeError(f"No hay imágenes en: {run_cfg.images_dir}")

    print(f"[PTV] images_dir   : {run_cfg.images_dir}", flush=True)
    print(f"[PTV] out_dir      : {run_cfg.out_dir}", flush=True)
    print(f"[PTV] weights_path : {run_cfg.weights_path}", flush=True)
    print(f"[PTV] frames       : {len(images)}", flush=True)

    detector = FiberYOLODetector(
        weights_path=run_cfg.weights_path,
        conf=run_cfg.conf,
        device=run_cfg.device,
    )
    tracker = Tracker(run_cfg)

    all_detections: list[Detection] = []
    static_mask_keep: np.ndarray | None = None
    next_det_id = 1

    for frame_idx, img_path in enumerate(images):
        print(f"[PTV] frame {frame_idx + 1}/{len(images)} -> {img_path.name}", flush=True)

        raw = read_image_any(img_path)

        # 1) Preprocesamiento
        rgb_u8 = preprocess_frame_for_ptv(raw, run_cfg.preprocess_params)

        h, w = rgb_u8.shape[:2]
        if (h, w) != (run_cfg.height_px, run_cfg.width_px):
            print(
                f"[WARN] Shape {img_path.name}: {(h, w)} "
                f"!= esperado {(run_cfg.height_px, run_cfg.width_px)}",
                flush=True,
            )

        # 2) Máscara fija
        if run_cfg.apply_static_mask:
            if static_mask_keep is None:
                static_mask_keep = load_mask_as_bool(
                    run_cfg.fixed_mask_path, expected_hw=(h, w)
                )
            rgb_u8 = apply_static_mask_to_rgb(rgb_u8, static_mask_keep)

        # 3) Detección
        detections, next_det_id = detector.detect(
            image_rgb_u8=rgb_u8,
            frame_idx=frame_idx,
            image_name=img_path.name,
            next_det_id=next_det_id,
        )
        all_detections.extend(detections)

        # 4) Tracking
        tracker.step(
            detections=detections,
            frame_idx=frame_idx,
            image_name=img_path.name,
        )

        # 5) Anotación
        if run_cfg.annotate:
            annotate_frame(
                image_rgb=rgb_u8,
                detections=detections,
                tracks=tracker.get_all_tracks(),
                frame_idx=frame_idx,
                image_name=img_path.name,
                out_path=ann_dir / f"{img_path.stem}.png",
                gate_x_px=run_cfg.gate_x_px,
                gate_y_px=run_cfg.gate_y_px,
            )

    tracker.close_all()
    tracks_all = tracker.get_all_tracks()
    tracks_filtered = [tr for tr in tracks_all if len(tr.history) >= run_cfg.min_frames_keep]

    # 6) Exportación
    export_detections_csv(all_detections, run_cfg.out_dir / "detections.csv")
    export_tracks_csv(
        tracks_filtered,
        px_per_mm=run_cfg.px_per_mm,
        fps=run_cfg.fps,
        path=run_cfg.out_dir / "tracks.csv",
    )
    export_tracks_json(tracks_filtered, run_cfg.out_dir / "tracks.json")

    summary = {
        "meta": raw_cfg.get("meta", {}),
        "camera": raw_cfg.get("camera", {}),
        "ptv": raw_cfg.get("ptv", {}),
        "results": {
            "n_frames": len(images),
            "n_detections": len(all_detections),
            "n_tracks_raw": len(tracks_all),
            "n_tracks_filtered": len(tracks_filtered),
            "min_frames_keep": run_cfg.min_frames_keep,
        },
    }
    _save_json(summary, run_cfg.out_dir / "summary.json")

    # 7) Visualizador HTML
    if run_cfg.annotate:
        create_interactive_visualizer(
            ann_dir=ann_dir,
            tracks=tracks_filtered,
            out_path=run_cfg.out_dir / "visualizer.html",
            width_px=run_cfg.width_px,
            height_px=run_cfg.height_px,
            fps=run_cfg.fps,
        )

    print("[PTV] Completado.", flush=True)
    print(f"[PTV] detections.csv  -> {run_cfg.out_dir / 'detections.csv'}", flush=True)
    print(f"[PTV] tracks.csv      -> {run_cfg.out_dir / 'tracks.csv'}", flush=True)
    print(f"[PTV] tracks.json     -> {run_cfg.out_dir / 'tracks.json'}", flush=True)
    print(f"[PTV] summary.json    -> {run_cfg.out_dir / 'summary.json'}", flush=True)
    if run_cfg.annotate:
        print(f"[PTV] visualizer.html -> {run_cfg.out_dir / 'visualizer.html'}", flush=True)
