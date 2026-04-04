"""
runner.py
=========
Loop principal del PTV con:
- Imágenes preprocesadas (leídas desde PTVPreprocesadas/)
- Procesamiento por regiones temporales con skip_frames variable
- dt_s variable pasado al tracker frame a frame
- Resultados en mm
- Prefetch asíncrono de imágenes (CPU overlapped con GPU)
"""
from __future__ import annotations
import json
import math
import queue
import re
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import TrackingConfig
from .models import Detection, Track
from .detector import FiberYOLODetector
from .tracker import Tracker
from .image_utils import (
    ensure_dir, read_image_any,
    normalize_to_uint8_for_yolo, np_to_builtin,
)
from .exporters import export_detections_csv, export_tracks_csv, export_tracks_json
from .visualizer import create_interactive_visualizer


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _list_images(folder: Path, max_images: int | None = None) -> list[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: _natural_key(p.name))
    if max_images is not None and max_images > 0:
        imgs = imgs[:max_images]
    return imgs


def _save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────
# REGIONES TEMPORALES
# ─────────────────────────────────────────────

def _build_frame_schedule(
    all_images: list[Path],
    fps: float,
    temporal_regions: list[dict],
) -> list[dict]:
    """
    Construye la lista ordenada de frames que el tracker debe procesar,
    con su dt_s, region_name y region_idx correspondientes.

    Estructura de un bloque PTV:
        img[i], img[i + stride], img[i + 2*stride], ...
    donde stride = skip_frames + 1.

    Entre regiones NO hay gap: la siguiente región comienza donde termina la anterior.
    Si el índice de inicio de una región no coincide con un frame seleccionado
    de la región anterior, se respeta el índice absoluto del frame (no hay duplicados).

    Returns:
        Lista de dicts con claves:
            img_path, frame_idx_original, timestamp_s, dt_s, region_name, region_idx
    """
    n_total = len(all_images)
    schedule: list[dict] = []
    seen_idx: set[int] = set()

    for r_idx, r in enumerate(temporal_regions):
        start_frame = int(r["start_time"] * fps)
        end_frame   = int(r["end_time"] * fps) if r["end_time"] is not None else n_total
        end_frame   = min(end_frame, n_total)
        skip        = int(r["skip_frames"])
        stride      = skip + 1
        dt_s        = stride / fps

        idx = start_frame
        while idx < end_frame:
            if idx < n_total and idx not in seen_idx:
                schedule.append({
                    "img_path":          all_images[idx],
                    "frame_idx_original": idx,
                    "timestamp_s":       idx / fps,
                    "dt_s":              dt_s,
                    "region_name":       r["name"],
                    "region_idx":        r_idx,
                })
                seen_idx.add(idx)
            idx += stride

    # Garantizar orden por frame_idx_original
    schedule.sort(key=lambda x: x["frame_idx_original"])
    return schedule


def _build_frame_schedule_no_regions(
    all_images: list[Path],
    fps: float,
) -> list[dict]:
    """Fallback: todos los frames consecutivos, dt = 1/fps."""
    dt_s = 1.0 / fps
    return [
        {
            "img_path":           p,
            "frame_idx_original": i,
            "timestamp_s":        i / fps,
            "dt_s":               dt_s,
            "region_name":        "default",
            "region_idx":         0,
        }
        for i, p in enumerate(all_images)
    ]


# ─────────────────────────────────────────────
# PREFETCH ASÍNCRONO
# ─────────────────────────────────────────────

def _load_one_preprocessed(img_path: Path) -> tuple[Path, np.ndarray]:
    """
    Carga imagen preprocesada (ya lista para YOLO).
    Solo convierte a uint8 RGB sin aplicar filtros adicionales.
    """
    raw = read_image_any(img_path)
    rgb_u8 = normalize_to_uint8_for_yolo(raw)
    return img_path, rgb_u8


def _prefetch_worker(
    schedule: list[dict],
    out_q: queue.Queue,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_load_one_preprocessed, entry["img_path"])
            for entry in schedule
        ]
        for fut in futures:
            out_q.put(fut.result())

    out_q.put(None)  # sentinel


# ─────────────────────────────────────────────
# GUARDAR FRAMES ANOTADOS
# ─────────────────────────────────────────────

def _save_annotated_frames(
    frames: list[np.ndarray],
    dets_per_frame: list[list],
    schedule: list[dict],
    tracks: list[Track],
    ann_dir: Path,
    tail_length: int = 0,
) -> None:
    import colorsys

    track_colors: dict[int, tuple] = {}

    def _color(tid: int) -> tuple:
        if tid not in track_colors:
            hue = (tid * 137.508) % 360
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.85, 0.95)
            track_colors[tid] = (int(b * 255), int(g * 255), int(r * 255))
        return track_colors[tid]

    # Mapear frame_idx_original → índice en schedule
    fidx_to_sched = {entry["frame_idx_original"]: i for i, entry in enumerate(schedule)}

    # Pre-indexar history por track: frame_idx_original → (x_mm, y_mm)
    # Para visualización en px: x_mm * px_per_mm  (recuperar px del mm)
    # Pero en frames_buffer guardamos el array ya en px, así que usamos x_mm directamente
    # en el visualizador (el HTML ya está en px porque dibuja sobre la imagen).
    # Nota: las posiciones en TrackRecord están en mm; para anotación recuperamos px
    # usando px_per_mm — pero no tenemos px_per_mm aquí. Guardamos en dict (x_mm, y_mm).
    track_history: dict[int, list[tuple]] = {}
    for tr in tracks:
        track_history[tr.track_id] = [
            (r.x_mm, r.y_mm, r.frame_idx)
            for r in tr.history
        ]

    for sched_i, (gray, dets, entry) in enumerate(
        zip(frames, dets_per_frame, schedule)
    ):
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fi_orig = entry["frame_idx_original"]

        for tr in tracks:
            hist_mm = [
                (x, y) for x, y, fi in track_history[tr.track_id]
                if fi <= fi_orig
            ]
            if tail_length > 0:
                hist_mm = hist_mm[-tail_length:]
            if len(hist_mm) < 1:
                continue

            color = _color(tr.track_id)
            n = len(hist_mm)

            # Nota: dibujamos en espacio de mm — el canvas es la imagen en px,
            # así que convertimos con el factor guardado en TrackRecord.
            # Como no tenemos px_per_mm en este scope, recuperamos las coordenadas
            # en px desde las detecciones originales cuando existen.
            # Para la trayectoria usamos las posiciones en mm directamente
            # con un factor dummy: se verán en unidades de mm sobre la imagen en px.
            # En producción real, pasar px_per_mm a esta función.
            if len(hist_mm) >= 2:
                for i in range(1, n):
                    # Las coordenadas están en mm; para overlay en imagen px
                    # esto es incorrecto sin px_per_mm, pero se mantiene la lógica
                    # para que el usuario lo corrija si necesita anotación en px.
                    pass

        out_png = ann_dir / f"{Path(entry['img_path'].name).stem}.png"
        cv2.imwrite(str(out_png), canvas)


def _save_annotated_frames_px(
    frames: list[np.ndarray],
    dets_per_frame: list[list],
    schedule: list[dict],
    tracks: list[Track],
    ann_dir: Path,
    px_per_mm: float,
    tail_length: int = 0,
) -> None:
    """
    Versión con px_per_mm para convertir posiciones mm → px al dibujar.
    """
    import colorsys

    track_colors: dict[int, tuple] = {}

    def _color(tid: int) -> tuple:
        if tid not in track_colors:
            hue = (tid * 137.508) % 360
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.85, 0.95)
            track_colors[tid] = (int(b * 255), int(g * 255), int(r * 255))
        return track_colors[tid]

    track_history: dict[int, list[tuple]] = {}
    for tr in tracks:
        track_history[tr.track_id] = [
            (r.x_mm * px_per_mm, r.y_mm * px_per_mm, r.frame_idx)
            for r in tr.history
        ]

    for gray, dets, entry in zip(frames, dets_per_frame, schedule):
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fi_orig = entry["frame_idx_original"]

        # Detecciones (en px, no necesitan conversión)
        for d in dets:
            cx_px = int(round(d.cx))
            cy_px = int(round(d.cy))
            half  = d.length_px / 2.0
            ang   = math.radians(d.angle_deg)
            dx_px = int(round(math.cos(ang) * half))
            dy_px = int(round(math.sin(ang) * half))
            cv2.line(canvas, (cx_px - dx_px, cy_px - dy_px),
                              (cx_px + dx_px, cy_px + dy_px), (0, 220, 255), 1)
            cv2.circle(canvas, (cx_px, cy_px), 2, (0, 220, 255), -1)

        # Trayectorias (convertidas a px)
        for tr in tracks:
            hist_px = [
                (int(round(x)), int(round(y)))
                for x, y, fi in track_history[tr.track_id]
                if fi <= fi_orig
            ]
            if tail_length > 0:
                hist_px = hist_px[-tail_length:]
            if len(hist_px) < 1:
                continue
            color = _color(tr.track_id)
            n = len(hist_px)
            if n >= 2:
                for i in range(1, n):
                    thickness = 1 if i < n * 0.6 else 2
                    cv2.line(canvas, hist_px[i - 1], hist_px[i], color, thickness)
            cv2.circle(canvas, hist_px[-1], 4, color, -1)

        out_png = ann_dir / f"{Path(entry['img_path'].name).stem}.png"
        cv2.imwrite(str(out_png), canvas)


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def run_ptv(run_cfg: TrackingConfig, raw_cfg: dict) -> None:
    ensure_dir(run_cfg.out_dir)

    # ── Listar imágenes preprocesadas ─────────────────────────────
    # El skip ya fue aplicado en preprocess_run_ptv.py: images_dir solo
    # contiene las imágenes a analizar (desde skip_first_images en adelante).
    all_images = _list_images(run_cfg.images_dir, max_images=run_cfg.max_images)
    if not all_images:
        raise RuntimeError(f"No hay imágenes preprocesadas en: {run_cfg.images_dir}")

    fps = run_cfg.fps

    # ── Construir schedule de frames ──────────────────────────────
    if run_cfg.use_temporal_regions and run_cfg.temporal_regions:
        schedule = _build_frame_schedule(all_images, fps, run_cfg.temporal_regions)
        mode_str = f"regiones temporales ({len(run_cfg.temporal_regions)} regiones)"
    else:
        schedule = _build_frame_schedule_no_regions(all_images, fps)
        mode_str = "sin regiones (frames consecutivos)"

    print(f"[PTV] images_dir (preprocesadas) : {run_cfg.images_dir}", flush=True)
    print(f"[PTV] out_dir                    : {run_cfg.out_dir}", flush=True)
    print(f"[PTV] total imágenes disponibles : {len(all_images)}", flush=True)
    print(f"[PTV] frames en schedule         : {len(schedule)}", flush=True)
    print(f"[PTV] modo temporal              : {mode_str}", flush=True)
    print(f"[PTV] px_per_mm                  : {run_cfg.px_per_mm}", flush=True)

    if run_cfg.use_temporal_regions and run_cfg.temporal_regions:
        for r in run_cfg.temporal_regions:
            skip   = r["skip_frames"]
            dt_ms  = (skip + 1) / fps * 1000
            end_t  = r["end_time"] if r["end_time"] is not None else "END"
            print(
                f"[PTV]   [{r['name']}] "
                f"t={r['start_time']:.1f}s→{end_t}  "
                f"skip={skip}  Δt={dt_ms:.2f}ms",
                flush=True,
            )

    # ── Detector y tracker ────────────────────────────────────────
    detector = FiberYOLODetector(
        weights_path  = run_cfg.weights_path,
        conf          = run_cfg.conf,
        device        = run_cfg.device,
        scale_factor  = run_cfg.sahi_scale_factor,
        tile_size     = run_cfg.sahi_tile_size,
        overlap_ratio = run_cfg.sahi_overlap_ratio,
        iou_threshold = run_cfg.sahi_iou_threshold,
    )
    tracker = Tracker(cfg=run_cfg)

    # ── Prefetch asíncrono ────────────────────────────────────────
    PREFETCH_SIZE = 8
    frame_q: queue.Queue = queue.Queue(maxsize=PREFETCH_SIZE)
    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(schedule, frame_q),
        daemon=True,
    )
    prefetch_thread.start()

    # ── Loop principal ────────────────────────────────────────────
    all_detections:  list[Detection] = []
    frames_buffer:   list[np.ndarray] = []
    dets_buffer:     list[list]       = []
    next_det_id = 1
    n_schedule  = len(schedule)

    for sched_i, entry in enumerate(schedule):
        item = frame_q.get()
        if item is None:
            break
        img_path, rgb_u8 = item

        dt_s         = entry["dt_s"]
        fi_orig      = entry["frame_idx_original"]
        timestamp_s  = entry["timestamp_s"]
        region_name  = entry["region_name"]
        region_idx   = entry["region_idx"]

        dt_ms_display = dt_s * 1000
        print(
            f"[PTV] frame {sched_i+1}/{n_schedule} "
            f"(orig={fi_orig}) {img_path.name} "
            f"[{region_name} Δt={dt_ms_display:.2f}ms]",
            flush=True,
        )

        h, w = rgb_u8.shape[:2]
        if (h, w) != (run_cfg.height_px, run_cfg.width_px):
            print(
                f"[WARN] Shape {img_path.name}: {(h,w)} "
                f"!= esperado {(run_cfg.height_px, run_cfg.width_px)}",
                flush=True,
            )

        detections, next_det_id = detector.detect(
            image_rgb_u8 = rgb_u8,
            frame_idx    = fi_orig,
            image_name   = img_path.name,
            next_det_id  = next_det_id,
        )
        all_detections.extend(detections)

        tracker.step(
            detections          = detections,
            frame_idx_original  = fi_orig,
            image_name          = img_path.name,
            dt_s                = dt_s,
            timestamp_s         = timestamp_s,
            region_name         = region_name,
            region_idx          = region_idx,
        )

        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY) if rgb_u8.ndim == 3 else rgb_u8
        frames_buffer.append(gray)
        dets_buffer.append(list(detections))

    tracker.close_all()
    tracks_all      = tracker.get_all_tracks()
    tracks_filtered = [tr for tr in tracks_all
                       if len(tr.history) >= run_cfg.min_frames_keep]

    print(f"[PTV] Tracks totales: {len(tracks_all)} | filtrados (≥{run_cfg.min_frames_keep} frames): {len(tracks_filtered)}", flush=True)

    # ── Exportación ───────────────────────────────────────────────
    export_detections_csv(all_detections, run_cfg.out_dir / "detections.csv")

    export_tracks_csv(
        tracks_filtered,
        fps=fps,
        path=run_cfg.out_dir / "tracks.csv",
    )

    export_tracks_json(
        tracks_filtered,
        fps=fps,
        temporal_regions=run_cfg.temporal_regions,
        path=run_cfg.out_dir / "tracks.json",
    )

    # ── Summary JSON ──────────────────────────────────────────────
    schedule_summary = []
    if run_cfg.temporal_regions:
        for r in run_cfg.temporal_regions:
            skip = r["skip_frames"]
            n_in_region = sum(
                1 for e in schedule if e["region_name"] == r["name"]
            )
            schedule_summary.append({
                "region":       r["name"],
                "start_time_s": r["start_time"],
                "end_time_s":   r["end_time"],
                "skip_frames":  skip,
                "dt_ms":        (skip + 1) / fps * 1000,
                "n_frames":     n_in_region,
            })

    summary = {
        "meta":   raw_cfg.get("meta", {}),
        "camera": raw_cfg.get("camera", {}),
        "ptv":    raw_cfg.get("ptv", {}),
        "schedule": schedule_summary,
        "results": {
            "n_frames_scheduled":  len(schedule),
            "n_frames_processed":  len(frames_buffer),
            "n_detections":        len(all_detections),
            "n_tracks_raw":        len(tracks_all),
            "n_tracks_filtered":   len(tracks_filtered),
            "min_frames_keep":     run_cfg.min_frames_keep,
            "units": {
                "position":     "mm",
                "velocity":     "mm/s",
                "acceleration": "mm/s2",
                "angle":        "degrees",
            },
        },
    }
    _save_json(summary, run_cfg.out_dir / "summary.json")

    print(f"[PTV] detections.csv → {run_cfg.out_dir / 'detections.csv'}", flush=True)
    print(f"[PTV] tracks.csv     → {run_cfg.out_dir / 'tracks.csv'}", flush=True)
    print(f"[PTV] tracks.json    → {run_cfg.out_dir / 'tracks.json'}", flush=True)
    print(f"[PTV] summary.json   → {run_cfg.out_dir / 'summary.json'}", flush=True)

    # ── Visualizador HTML ─────────────────────────────────────────
    if run_cfg.save_images and frames_buffer:
        ann_dir = run_cfg.out_dir / "annotations"
        ensure_dir(ann_dir)
        _save_annotated_frames_px(
            frames_buffer, dets_buffer, schedule,
            tracks_filtered, ann_dir,
            px_per_mm=run_cfg.px_per_mm,
            tail_length=run_cfg.viz_tail_length,
        )
        ann_images = list(ann_dir.glob("*.png"))
        if ann_images:
            create_interactive_visualizer(
                ann_dir   = ann_dir,
                tracks    = tracks_filtered,
                out_path  = run_cfg.out_dir / "visualizer.html",
                width_px  = run_cfg.width_px,
                height_px = run_cfg.height_px,
                fps       = fps,
            )
            print(f"[PTV] visualizer.html → {run_cfg.out_dir / 'visualizer.html'}", flush=True)

    print("[PTV] Completado.", flush=True)