"""
runner.py
=========
Loop principal del PTV con:
- Imágenes preprocesadas (leídas desde PTVPreprocesadas/)
- Procesamiento por regiones temporales con skip_frames variable
- dt_s y max_dist_px variables pasados al tracker frame a frame
- Resultados en mm
- Prefetch asíncrono de imágenes (CPU overlapped con GPU)
"""
from __future__ import annotations
import json
import math
import queue
import re
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import TrackingConfig
from .models import Detection, Track
from .detector import FiberYOLODetector
from .tracker import Tracker
from .dptv import DPTVEstimator, DPTVConfig
from .image_utils import (
    ensure_dir, read_image_any,
    normalize_to_uint8_for_yolo, np_to_builtin,
)
from .exporters import export_detections_csv, export_tracks_csv, export_tracks_json, export_schedule_csv
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


def _effective_frames(tr: "Track", fps: float) -> float:
    """
    Frames originales cubiertos por el track.

    Cada record del history tiene su dt_s real (variable entre regiones).
    Sumamos dt_s * fps por cada observacion: equivale a contar cuantos
    frames originales consecutivos abarca el track, sin importar el stride.

    Ejemplos:
      3 hits en muy_baja_velocidad (stride=9):  3 * 9  = 27 frames
      5 hits en alta_velocidad     (stride=1):  5 * 1  =  5 frames
      2 hits stride=1 + 2 hits stride=9:        2 + 18 = 20 frames
    """
    return sum(rec.dt_s * fps for rec in tr.history)


# ─────────────────────────────────────────────
# REGIONES TEMPORALES
# ─────────────────────────────────────────────

def _build_schedule_from_json(
    preprocessed_dir: Path,
    temporal_regions: list[dict],
    global_max_dist_px: float,
    px_per_mm: float,
) -> list[dict]:
    """
    Construye el schedule leyendo schedule.json generado por preprocess_run_ptv.py.

    preprocess_run_ptv.py guarda en PTVPreprocesadas/<sub>/schedule.json
    exactamente las imágenes que seleccionó, con su frame_idx_original,
    timestamp_s, dt_s y region_name ya calculados correctamente.

    El runner solo necesita:
      - resolver img_path = preprocessed_dir / entry["preprocessed_name"]
      - añadir max_dist_px desde la config de la región correspondiente

    De este modo no hay doble stride: el preprocesador ya aplicó los saltos
    al seleccionar las imágenes; el runner simplemente las lee en orden.

    Returns:
        Lista de dicts con claves:
            img_path, frame_idx_original, timestamp_s,
            dt_s, max_dist_px, region_name, region_idx
    """
    schedule_json = preprocessed_dir / "schedule.json"
    if not schedule_json.exists():
        raise FileNotFoundError(
            f"No se encontró schedule.json en {preprocessed_dir}.\n"
            f"Asegúrate de haber corrido preprocess_run_ptv.py antes del tracker."
        )

    entries = json.loads(schedule_json.read_text(encoding="utf-8"))

    # Mapa region_name → max_dist_px para lookup O(1)
    region_gate: dict[str, float] = {}
    for r in temporal_regions:
        mm = r.get("max_dist_mm")
        region_gate[r["name"]] = float(mm) * px_per_mm if mm is not None else global_max_dist_px

    schedule: list[dict] = []
    missing: list[str] = []

    for entry in entries:
        img_path = preprocessed_dir / entry["preprocessed_name"]
        if not img_path.exists():
            missing.append(entry["preprocessed_name"])
            continue

        region_name = entry["region_name"]
        max_dist_px = region_gate.get(region_name, global_max_dist_px)

        schedule.append({
            "img_path":           img_path,
            "frame_idx_original": int(entry["frame_idx_original"]),
            "timestamp_s":        float(entry["timestamp_s"]),
            "dt_s":               float(entry["dt_s"]),
            "max_dist_px":        max_dist_px,
            "region_name":        region_name,
            "region_idx":         int(entry["region_idx"]),
        })

    if missing:
        raise RuntimeError(
            f"{len(missing)} imágenes del schedule.json no existen en {preprocessed_dir}.\n"
            f"Ejemplo faltante: {missing[0]}\n"
            f"Vuelve a correr preprocess_run_ptv.py."
        )

    # schedule.json ya viene ordenado por frame_idx_original, pero lo
    # garantizamos explícitamente por si acaso.
    schedule.sort(key=lambda x: x["frame_idx_original"])
    return schedule


def _build_frame_schedule_no_regions(
    all_images: list[Path],
    fps: float,
    global_max_dist_px: float,
) -> list[dict]:
    """Fallback sin regiones temporales: todos los frames consecutivos, dt = 1/fps."""
    dt_s = 1.0 / fps
    return [
        {
            "img_path":           p,
            "frame_idx_original": i,
            "timestamp_s":        i / fps,
            "dt_s":               dt_s,
            "max_dist_px":        global_max_dist_px,
            "region_name":        "default",
            "region_idx":         0,
        }
        for i, p in enumerate(all_images)
    ]


# ─────────────────────────────────────────────
# PREFETCH ASÍNCRONO
# ─────────────────────────────────────────────

def _load_one_preprocessed(img_path: Path) -> tuple[Path, np.ndarray]:
    raw    = read_image_any(img_path)
    rgb_u8 = normalize_to_uint8_for_yolo(raw)
    return img_path, rgb_u8


def _prefetch_worker(schedule: list[dict], out_q: queue.Queue) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_load_one_preprocessed, entry["img_path"])
            for entry in schedule
        ]
        for fut in futures:
            out_q.put(fut.result())

    out_q.put(None)


# ─────────────────────────────────────────────
# GUARDAR FRAMES ANOTADOS
# ─────────────────────────────────────────────

def _save_annotated_frames_px(
    frames: list[np.ndarray],
    dets_per_frame: list[list],
    schedule: list[dict],
    tracks: list[Track],
    ann_dir: Path,
    px_per_mm: float,
    tail_length: int = 0,
) -> None:
    """Dibuja detecciones y trayectorias en coordenadas px sobre los frames."""
    import colorsys

    track_colors: dict[int, tuple] = {}

    def _color(tid: int) -> tuple:
        if tid not in track_colors:
            hue = (tid * 137.508) % 360
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.85, 0.95)
            track_colors[tid] = (int(b * 255), int(g * 255), int(r * 255))
        return track_colors[tid]

    # Pre-indexar history: track_id → lista (x_px, y_px, frame_idx)
    track_history: dict[int, list[tuple]] = {}
    for tr in tracks:
        track_history[tr.track_id] = [
            (r.x_mm * px_per_mm, r.y_mm * px_per_mm, r.frame_idx)
            for r in tr.history
        ]

    for gray, dets, entry in zip(frames, dets_per_frame, schedule):
        canvas  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fi_orig = entry["frame_idx_original"]

        # Detecciones (ya en px)
        for d in dets:
            cx_px = int(round(d.cx))
            cy_px = int(round(d.cy))
            half  = d.length_px / 2.0
            ang   = math.radians(d.angle_deg)
            dx_px = int(round(math.cos(ang) * half))
            dy_px = int(round(math.sin(ang) * half))
            cv2.line(canvas,
                     (cx_px - dx_px, cy_px - dy_px),
                     (cx_px + dx_px, cy_px + dy_px),
                     (0, 220, 255), 1)
            cv2.circle(canvas, (cx_px, cy_px), 2, (0, 220, 255), -1)

        # Trayectorias (mm → px)
        for tr in tracks:
            hist_px = [
                (int(round(x)), int(round(y)))
                for x, y, fi in track_history[tr.track_id]
                if fi <= fi_orig
            ]
            if tail_length > 0:
                hist_px = hist_px[-tail_length:]
            if not hist_px:
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

    fps            = run_cfg.fps
    px_per_mm      = run_cfg.px_per_mm
    # Gate global (fallback para regiones sin max_dist_mm explícito)
    global_max_dist_px = run_cfg.max_dist_px

    # ── Construir schedule de frames ──────────────────────────────
    if run_cfg.use_temporal_regions and run_cfg.temporal_regions:
        # Lee schedule.json generado por preprocess_run_ptv.py.
        # Ese archivo ya tiene los frame_idx_original, timestamp_s y dt_s
        # correctos — el preprocesador ya aplicó los strides al seleccionar
        # las imágenes; aquí solo las leemos en orden sin doble salto.
        schedule = _build_schedule_from_json(
            preprocessed_dir    = run_cfg.images_dir,
            temporal_regions    = run_cfg.temporal_regions,
            global_max_dist_px  = global_max_dist_px,
            px_per_mm           = px_per_mm,
        )
        mode_str = f"schedule.json ({len(run_cfg.temporal_regions)} regiones)"
    else:
        # Sin regiones: leer todas las imágenes preprocesadas en orden
        all_images = _list_images(run_cfg.images_dir, max_images=run_cfg.max_images)
        if not all_images:
            raise RuntimeError(f"No hay imágenes preprocesadas en: {run_cfg.images_dir}")
        schedule = _build_frame_schedule_no_regions(
            all_images, fps, global_max_dist_px,
        )
        mode_str = "sin regiones (frames consecutivos)"

    if not schedule:
        raise RuntimeError(
            f"El schedule está vacío. Comprueba que preprocess_run_ptv.py "
            f"generó schedule.json en {run_cfg.images_dir}."
        )

    print(f"[PTV] images_dir (preprocesadas) : {run_cfg.images_dir}", flush=True)
    print(f"[PTV] out_dir                    : {run_cfg.out_dir}", flush=True)
    print(f"[PTV] frames en schedule         : {len(schedule)}", flush=True)
    print(f"[PTV] modo temporal              : {mode_str}", flush=True)
    print(f"[PTV] px_per_mm                  : {px_per_mm}", flush=True)
    print(f"[PTV] gate global (fallback)     : {global_max_dist_px:.1f} px "
          f"({global_max_dist_px / px_per_mm:.2f} mm)", flush=True)

    if run_cfg.use_temporal_regions and run_cfg.temporal_regions:
        for r in run_cfg.temporal_regions:
            skip   = r["skip_frames"]
            dt_ms  = (skip + 1) / fps * 1000
            end_t  = r["end_time"] if r["end_time"] is not None else "END"
            gate_mm = r.get("max_dist_mm")
            gate_str = f"{gate_mm} mm" if gate_mm is not None else f"{global_max_dist_px / px_per_mm:.2f} mm (global)"
            print(
                f"[PTV]   [{r['name']}] "
                f"t={r['start_time']:.1f}s→{end_t}  "
                f"skip={skip}  Δt={dt_ms:.2f}ms  gate={gate_str}",
                flush=True,
            )

    # ── DPTV — Depth from Defocus estimator ──────────────────────
    dptv_estimator: DPTVEstimator | None = None
    dptv_config_dict: dict | None = None
    if run_cfg.dptv_enabled:
        dptv_cfg = DPTVConfig(
            fiber_width_mm=run_cfg.dptv_fiber_width_mm,
            fiber_length_mm=run_cfg.dptv_fiber_length_mm,
            noise_width_px=run_cfg.dptv_noise_width_px,
            k_blur_px_per_mm=run_cfg.dptv_k_blur_px_per_mm,
            w_ideal_px=run_cfg.dptv_w_ideal_px,
        )
        dptv_estimator = DPTVEstimator(dptv_cfg)
        dptv_config_dict = dptv_estimator.to_dict()
        calibrated = run_cfg.dptv_k_blur_px_per_mm is not None
        print(
            f"[PTV] DPTV habilitado — fibra {run_cfg.dptv_fiber_width_mm}mm × {run_cfg.dptv_fiber_length_mm}mm  "
            f"noise={run_cfg.dptv_noise_width_px}px  "
            f"k_blur={'%.3f' % run_cfg.dptv_k_blur_px_per_mm + ' px/mm' if calibrated else 'no calibrado (depth_mm=None)'}",
            flush=True,
        )
    else:
        print("[PTV] DPTV deshabilitado.", flush=True)

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

        dt_s          = entry["dt_s"]
        max_dist_px   = entry["max_dist_px"]   # gate específico de esta región
        fi_orig       = entry["frame_idx_original"]
        timestamp_s   = entry["timestamp_s"]
        region_name   = entry["region_name"]
        region_idx    = entry["region_idx"]

        dt_ms_display = dt_s * 1000
        print(
            f"[PTV] frame {sched_i+1}/{n_schedule} "
            f"(orig={fi_orig}) {img_path.name} "
            f"[{region_name} Δt={dt_ms_display:.2f}ms "
            f"gate={max_dist_px / px_per_mm:.2f}mm]",
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
            dptv         = dptv_estimator,
            px_per_mm    = px_per_mm,
        )
        all_detections.extend(detections)

        # max_dist_px se pasa frame a frame → tracker usa el gate correcto
        tracker.step(
            detections          = detections,
            frame_idx_original  = fi_orig,
            image_name          = img_path.name,
            dt_s                = dt_s,
            max_dist_px         = max_dist_px,
            timestamp_s         = timestamp_s,
            region_name         = region_name,
            region_idx          = region_idx,
        )

        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY) if rgb_u8.ndim == 3 else rgb_u8
        frames_buffer.append(gray)
        dets_buffer.append(list(detections))

    tracker.close_all()
    tracks_all      = tracker.get_all_tracks()
    tracks_filtered = [
        tr for tr in tracks_all
        if _effective_frames(tr, fps) >= run_cfg.min_frames_keep
    ]

    print(
        f"[PTV] Tracks totales: {len(tracks_all)} | "
        f"filtrados (≥{run_cfg.min_frames_keep} frames orig): {len(tracks_filtered)}",
        flush=True,
    )

    # ── Exportación ───────────────────────────────────────────────
    export_detections_csv(all_detections, run_cfg.out_dir / "detections.csv")
    export_tracks_csv(tracks_filtered, fps=fps, path=run_cfg.out_dir / "tracks.csv")
    export_tracks_json(
        tracks_filtered, fps=fps,
        temporal_regions=run_cfg.temporal_regions,
        path=run_cfg.out_dir / "tracks.json",
        dptv_config=dptv_config_dict,
    )


    export_schedule_csv(
        schedule        = schedule,
        tracks_filtered = tracks_filtered,
        all_detections  = all_detections,
        path            = run_cfg.out_dir / "schedule.csv",
    )

    # ── Summary JSON ──────────────────────────────────────────────
    schedule_summary = []
    if run_cfg.temporal_regions:
        for r in run_cfg.temporal_regions:
            skip       = r["skip_frames"]
            gate_mm    = r.get("max_dist_mm")
            n_in_region = sum(1 for e in schedule if e["region_name"] == r["name"])
            schedule_summary.append({
                "region":        r["name"],
                "start_time_s":  r["start_time"],
                "end_time_s":    r["end_time"],
                "skip_frames":   skip,
                "dt_ms":         (skip + 1) / fps * 1000,
                "max_dist_mm":   gate_mm,
                "n_frames":      n_in_region,
            })

    # DPTV aggregate statistics across all filtered tracks
    dptv_summary: dict = {"enabled": run_cfg.dptv_enabled}
    if run_cfg.dptv_enabled and tracks_filtered:
        from .dptv import _mean as _dptv_mean, _std as _dptv_std
        all_records = [r for tr in tracks_filtered for r in tr.history]
        dscores  = [r.defocus_score    for r in all_records]
        blurs_mm = [r.depth_blur_mm    for r in all_records]
        confs    = [r.depth_confidence for r in all_records]
        depths   = [r.depth_mm for r in all_records if r.depth_mm is not None]

        near_focus_count = sum(1 for s in dscores if s < 1.5)  # within 50% of ideal
        dptv_summary.update({
            "config":                dptv_config_dict,
            "n_records_total":       len(all_records),
            "n_depth_estimated":     len(depths),
            "mean_defocus_score":    _dptv_mean(dscores),
            "std_defocus_score":     _dptv_std(dscores),
            "mean_blur_mm":          _dptv_mean(blurs_mm),
            "mean_depth_confidence": _dptv_mean(confs),
            "pct_near_focus":        100.0 * near_focus_count / len(dscores) if dscores else 0.0,
            "mean_depth_mm":         _dptv_mean(depths) if depths else None,
            "std_depth_mm":          _dptv_std(depths)  if depths else None,
        })

    summary = {
        "meta":   raw_cfg.get("meta", {}),
        "camera": raw_cfg.get("camera", {}),
        "ptv":    raw_cfg.get("ptv", {}),
        "schedule": schedule_summary,
        "dptv":   dptv_summary,
        "results": {
            "n_frames_scheduled":  len(schedule),
            "n_frames_processed":  len(frames_buffer),
            "n_detections":        len(all_detections),
            "n_tracks_raw":        len(tracks_all),
            "n_tracks_filtered":   len(tracks_filtered),
            "min_frames_keep":     run_cfg.min_frames_keep,
            "units": {
                "position":        "mm",
                "velocity":        "mm/s",
                "acceleration":    "mm/s2",
                "angle":           "degrees",
                "depth":           "mm",
                "depth_blur":      "mm",
                "defocus_score":   "dimensionless (1.0=in-focus)",
                "depth_confidence":"dimensionless [0,1]",
            },
        },
    }
    _save_json(summary, run_cfg.out_dir / "summary.json")

    print(f"[PTV] detections.csv → {run_cfg.out_dir / 'detections.csv'}", flush=True)
    print(f"[PTV] tracks.csv     → {run_cfg.out_dir / 'tracks.csv'}", flush=True)
    print(f"[PTV] tracks.json    → {run_cfg.out_dir / 'tracks.json'}", flush=True)
    print(f"[PTV] summary.json   → {run_cfg.out_dir / 'summary.json'}", flush=True)

    # ── Visualizador HTML ─────────────────────────────────────────
    #if run_cfg.save_images and frames_buffer:
    #    ann_dir = run_cfg.out_dir / "annotations"
    #    ensure_dir(ann_dir)
    #    _save_annotated_frames_px(
     #       frames_buffer, dets_buffer, schedule,
    #        tracks_filtered, ann_dir,
    #        px_per_mm=px_per_mm,
    #        tail_length=run_cfg.viz_tail_length,
     #   )
    #    ann_images = list(ann_dir.glob("*.png"))
    #    if ann_images:
    #        create_interactive_visualizer(
    #            ann_dir   = ann_dir,
    #            tracks    = tracks_filtered,
    #            out_path  = run_cfg.out_dir / "visualizer.html",
    #            width_px  = run_cfg.width_px,
    #            height_px = run_cfg.height_px,
    #            fps       = fps,
    #            px_per_mm = px_per_mm,   # ← agregar esta línea
    #        )
    #        print(f"[PTV] visualizer.html → {run_cfg.out_dir / 'visualizer.html'}", flush=True)

    print("[PTV] Completado.", flush=True)