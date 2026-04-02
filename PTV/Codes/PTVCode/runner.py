"""
runner.py
=========
Loop principal del PTV con:
- Prefetch asíncrono de imágenes (CPU overlapped con GPU)
- Visualizador interactivo matplotlib con slider de frames
"""
from __future__ import annotations
import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import TrackingConfig
from .models import Detection, Track
from .detector import FiberYOLODetector
from .tracker import Tracker
from .image_utils import (
    ensure_dir, list_images, read_image_any,
    preprocess_frame_for_ptv, load_mask_as_bool,
    apply_static_mask_to_rgb, np_to_builtin,
)
from .exporters import export_detections_csv, export_tracks_csv, export_tracks_json
from .visualizer import create_interactive_visualizer


def _save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=np_to_builtin),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────
# PREFETCH ASÍNCRONO
# ─────────────────────────────────────────────

def _prefetch_worker(
    images: list[Path],
    preprocess_params: dict | None,
    static_mask_keep: np.ndarray | None,
    height_px: int,
    width_px: int,
    out_q: queue.Queue,
    n_ahead: int = 4,
) -> None:
    """
    Worker thread: carga y preprocesa imágenes por adelantado.
    Mientras la GPU procesa el frame N, este hilo prepara N+1..N+n_ahead.
    """
    for img_path in images:
        raw   = read_image_any(img_path)
        rgb_u8 = preprocess_frame_for_ptv(raw, preprocess_params)

        h, w = rgb_u8.shape[:2]
        if static_mask_keep is not None:
            rgb_u8 = apply_static_mask_to_rgb(rgb_u8, static_mask_keep)

        out_q.put((img_path, rgb_u8))  # bloquea si la cola está llena (backpressure)

    out_q.put(None)  # sentinel de fin


# ─────────────────────────────────────────────
# VISUALIZADOR INTERACTIVO
# ─────────────────────────────────────────────

class InteractiveVisualizer:
    """
    Ventana matplotlib interactiva que se actualiza en tiempo real.

    Muestra:
    - Panel izquierdo : frame actual con detecciones (líneas de fibra)
    - Panel derecho   : trayectorias de todos los tracks hasta el frame actual
    - Slider inferior : navegación manual por frames ya procesados
    - Texto info      : frame actual, n° tracks activos, velocidad media
    """

    def __init__(
        self,
        n_frames: int,
        width_px: int,
        height_px: int,
        fps: float,
        px_per_mm: float,
        ann_dir: Path | None = None,   # si se pasa, guarda PNGs anotados
        tail_length: int = 0,
        update_every: int = 1,
    ):
        import matplotlib
        matplotlib.use("TkAgg")           # backend con ventana interactiva
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.plt        = plt
        self.n_frames   = n_frames
        self.width_px   = width_px
        self.height_px  = height_px
        self.fps        = fps
        self.px_per_mm  = px_per_mm
        self.ann_dir    = ann_dir
        self.tail_length = tail_length
        self.update_every = update_every

        # Buffers de frames ya procesados
        self._frames:  list[np.ndarray | None] = [None] * n_frames
        self._dets:    list[list] = [[] for _ in range(n_frames)]
        self._tracks:  list[list] = [[] for _ in range(n_frames)]  # snapshot por frame
        self._current  = 0
        self._max_ready = -1   # último frame procesado

        # Paleta de colores por track_id (determinista)
        self._colors: dict[int, tuple] = {}

        # Construir figura
        self.fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.patch.set_facecolor("#1a1a1a")
        plt.subplots_adjust(bottom=0.15, hspace=0.05)

        self.ax_frame = axes[0]
        self.ax_track = axes[1]
        for ax in axes:
            ax.set_facecolor("#0a0a0a")
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#404040")

        self.ax_frame.set_title("Frame actual", color="white", fontsize=11)
        self.ax_track.set_title("Trayectorias (toma completa hasta frame)", color="white", fontsize=11)

        # Imagen dummy inicial
        blank = np.zeros((height_px, width_px), dtype=np.uint8)
        self.im_frame = self.ax_frame.imshow(blank, cmap="gray", vmin=0, vmax=255,
                                              interpolation="nearest")
        self.im_track = self.ax_track.imshow(blank, cmap="gray", vmin=0, vmax=255,
                                              interpolation="nearest")
        self.ax_frame.axis("off")
        self.ax_track.axis("off")

        # Info text
        self.txt_info = self.fig.text(
            0.5, 0.97, "Procesando...",
            ha="center", va="top", color="white", fontsize=10,
        )

        # Slider
        ax_slider = self.fig.add_axes([0.1, 0.04, 0.8, 0.03],
                                       facecolor="#2d2d2d")
        self.slider = Slider(
            ax_slider, "Frame", 0, max(n_frames - 1, 1),
            valinit=0, valstep=1, color="#00d4ff",
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        plt.ion()
        plt.show(block=False)

    def _track_color(self, track_id: int) -> tuple:
        if track_id not in self._colors:
            hue = (track_id * 137.508) % 360
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.75, 0.9)
            self._colors[track_id] = (r, g, b)
        return self._colors[track_id]

    def update(
        self,
        frame_idx: int,
        img_name: str,
        rgb_u8: np.ndarray,
        detections: list,
        tracker,
    ) -> None:
        """Llamado por runner después de cada frame detectado."""
        # Guardar frame en buffer
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY) if rgb_u8.ndim == 3 else rgb_u8
        self._frames[frame_idx] = gray
        self._dets[frame_idx]   = list(detections)

        # Snapshot de todos los tracks activos en este frame
        self._tracks[frame_idx] = [
            {
                "id":      tr.track_id,
                "history": [(r.x, r.y, r.frame_idx) for r in tr.history],
            }
            for tr in tracker.get_all_tracks()
            if len(tr.history) > 0
        ]
        self._max_ready = frame_idx

        # Auto-avanzar slider al último frame
        self._current = frame_idx
        self.slider.set_val(frame_idx)

        if frame_idx % self.update_every == 0:
            self._draw(frame_idx)
            self.plt.pause(0.001)

        # Guardar PNG anotado para visualizador HTML
        if self.ann_dir is not None:
            ann_img = self._render_annotated(frame_idx, gray)
            if ann_img is not None:
                import cv2 as _cv2
                out_png = self.ann_dir / f"{Path(img_name).stem}.png"
                _cv2.imwrite(str(out_png), _cv2.cvtColor(ann_img, _cv2.COLOR_RGB2BGR))

    def _render_annotated(self, frame_idx: int, gray: np.ndarray) -> np.ndarray | None:
        """Renderiza frame con detecciones y trayectorias como array RGB."""
        import math, colorsys
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        for d in self._dets[frame_idx]:
            cx, cy = int(round(d.cx)), int(round(d.cy))
            half = d.length_px / 2.0
            ang  = math.radians(d.angle_deg)
            dx   = int(round(math.cos(ang) * half))
            dy   = int(round(math.sin(ang) * half))
            cv2.line(canvas, (cx-dx, cy-dy), (cx+dx, cy+dy), (0, 220, 255), 1)
            cv2.circle(canvas, (cx, cy), 2, (0, 220, 255), -1)
        for tr in self._tracks[frame_idx]:
            tid  = tr["id"]
            hist = [(int(x), int(y)) for x, y, fi in tr["history"] if fi <= frame_idx]
            if self.tail_length > 0:
                hist = hist[-self.tail_length:]
            if len(hist) < 2:
                continue
            r, g, b = self._track_color(tid)
            color = (int(b*255), int(g*255), int(r*255))  # BGR para cv2
            for i in range(1, len(hist)):
                alpha = 0.2 + 0.8 * (i / len(hist))
                cv2.line(canvas, hist[i-1], hist[i], color, 1)
            cv2.circle(canvas, hist[-1], 3, color, -1)
            cv2.putText(canvas, str(tid), (hist[-1][0]+5, hist[-1][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return canvas

    def _on_slider(self, val: float) -> None:
        fi = int(val)
        if fi <= self._max_ready:
            self._current = fi
            self._draw(fi)
            self.plt.pause(0.001)

    def _draw(self, frame_idx: int) -> None:
        gray = self._frames[frame_idx]
        if gray is None:
            return

        # Panel izquierdo: imagen con detecciones de fibras (ejes orientados)
        self.im_frame.set_data(gray)
        for coll in self.ax_frame.collections + self.ax_frame.lines:
            coll.remove()
        for d in self._dets[frame_idx]:
            import math
            cx, cy = d.cx, d.cy
            half   = d.length_px / 2.0
            ang    = math.radians(d.angle_deg)
            dx, dy = math.cos(ang) * half, math.sin(ang) * half
            self.ax_frame.plot([cx-dx, cx+dx], [cy-dy, cy+dy],
                               "-", color="cyan", lw=1.0, alpha=0.7)

        # Panel derecho: SOLO trayectorias de centroides, sin fibras
        self.im_track.set_data(gray)
        for coll in self.ax_track.collections + self.ax_track.lines:
            coll.remove()

        tracks_snap = self._tracks[frame_idx]
        for tr in tracks_snap:
            tid  = tr["id"]
            hist = [(x, y) for x, y, fi in tr["history"] if fi <= frame_idx]
            if self.tail_length > 0:
                hist = hist[-self.tail_length:]
            if len(hist) < 1:
                continue
            color = self._track_color(tid)
            if len(hist) >= 2:
                xs = [p[0] for p in hist]
                ys = [p[1] for p in hist]
                n  = len(xs)
                for i in range(1, n):
                    alpha     = 0.25 + 0.75 * (i / n)
                    thickness = 1.5 if i >= n * 0.6 else 1.0
                    self.ax_track.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                                       "-", color=color, lw=thickness, alpha=alpha)
            # Centroide actual
            cx_now, cy_now = hist[-1]
            self.ax_track.plot(cx_now, cy_now, "o", color=color, ms=5)
            self.ax_track.text(cx_now + 6, cy_now - 6, str(tid),
                               color=color, fontsize=7, alpha=0.9)

        # Info
        t_s = frame_idx / max(self.fps, 1)
        n_active = len(tracks_snap)
        self.txt_info.set_text(
            f"Frame {frame_idx + 1}/{self.n_frames}  |  "
            f"t = {t_s:.3f} s  |  tracks: {n_active}"
        )
        self.fig.canvas.draw_idle()

    def close(self) -> None:
        """Mantiene la ventana abierta hasta que el usuario la cierre."""
        print("[VIZ] Ventana interactiva lista. Ciérrala para terminar.", flush=True)
        self.plt.ioff()
        self.plt.show(block=True)


# ─────────────────────────────────────────────
# GUARDAR FRAMES ANOTADOS PARA HTML
# ─────────────────────────────────────────────

def _save_annotated_frames(
    frames: list[np.ndarray],
    dets_per_frame: list[list],
    img_names: list[str],
    tracks: list,
    ann_dir: Path,
    tail_length: int = 0,
) -> None:
    """
    Guarda PNGs con solo las trayectorias de centroides (sin detecciones de fibras).
    Cada track tiene un color único. La opacidad aumenta hacia el frame más reciente.
    """
    import colorsys

    track_colors: dict[int, tuple] = {}
    def _color(tid: int) -> tuple:
        if tid not in track_colors:
            hue = (tid * 137.508) % 360
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.85, 0.95)
            track_colors[tid] = (int(b*255), int(g*255), int(r*255))  # BGR
        return track_colors[tid]

    # Pre-indexar history por track
    track_history: dict[int, list] = {
        tr.track_id: [(r.x, r.y, r.frame_idx) for r in tr.history]
        for tr in tracks
    }

    n_frames = len(frames)
    for frame_idx, (gray, img_name) in enumerate(zip(frames, img_names)):
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Solo trayectorias — sin líneas de fibra ni bounding boxes
        for tr in tracks:
            hist = [(int(round(x)), int(round(y)))
                    for x, y, fi in track_history[tr.track_id]
                    if fi <= frame_idx]
            if tail_length > 0:
                hist = hist[-tail_length:]
            if len(hist) < 2:
                # Punto único: solo el centroide actual
                if len(hist) == 1:
                    color = _color(tr.track_id)
                    cv2.circle(canvas, hist[0], 3, color, -1)
                continue

            color = _color(tr.track_id)
            n = len(hist)
            for i in range(1, n):
                # Grosor más grueso hacia el final de la trayectoria
                thickness = 1 if i < n * 0.6 else 2
                cv2.line(canvas, hist[i-1], hist[i], color, thickness)

            # Punto del centroide actual (más grande)
            cv2.circle(canvas, hist[-1], 4, color, -1)
            # ID del track
            cv2.putText(canvas, str(tr.track_id),
                        (hist[-1][0] + 6, hist[-1][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        out_png = ann_dir / f"{Path(img_name).stem}.png"
        cv2.imwrite(str(out_png), canvas)


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def run_ptv(run_cfg: TrackingConfig, raw_cfg: dict) -> None:
    ensure_dir(run_cfg.out_dir)

    # ── Listar imágenes: skip PRIMERO, luego max_images ─────────
    # El skip debe aplicarse antes de max_images para que MAX_IMAGES
    # controle cuántos frames analizar DESPUÉS del salto inicial.
    all_images = list_images(run_cfg.images_dir)   # sin límite
    if not all_images:
        raise RuntimeError(f"No hay imagenes en: {run_cfg.images_dir}")

    skip = max(0, run_cfg.skip_first_images)
    images_after_skip = all_images[skip:]
    if not images_after_skip:
        raise RuntimeError(
            f"No quedan imagenes despues de saltar {skip} frames "
            f"(total en carpeta: {len(all_images)})."
        )

    # Limitar con max_images DESPUÉS del skip
    if run_cfg.max_images is not None and run_cfg.max_images > 0:
        images = images_after_skip[:run_cfg.max_images]
    else:
        images = images_after_skip

    print(f"[PTV] images_dir        : {run_cfg.images_dir}", flush=True)
    print(f"[PTV] out_dir           : {run_cfg.out_dir}", flush=True)
    print(f"[PTV] weights_path      : {run_cfg.weights_path}", flush=True)
    print(f"[PTV] total en carpeta  : {len(all_images)}", flush=True)
    print(f"[PTV] skip_first_images : {skip}", flush=True)
    print(f"[PTV] despues del skip  : {len(images_after_skip)}", flush=True)
    print(f"[PTV] frames a analizar : {len(images)}", flush=True)

    # ── Prefetch de máscara fija ──────────────────────────────────
    static_mask_keep: np.ndarray | None = None
    if run_cfg.apply_static_mask and run_cfg.fixed_mask_path:
        sample = read_image_any(images[0])
        from .image_utils import preprocess_frame_for_ptv as _pp
        sample_rgb = _pp(sample, run_cfg.preprocess_params)
        h, w = sample_rgb.shape[:2]
        static_mask_keep = load_mask_as_bool(run_cfg.fixed_mask_path, (h, w))

    # ── Cola de prefetch ──────────────────────────────────────────
    PREFETCH_SIZE = 8
    frame_q: queue.Queue = queue.Queue(maxsize=PREFETCH_SIZE)
    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(images, run_cfg.preprocess_params, static_mask_keep,
              run_cfg.height_px, run_cfg.width_px, frame_q, PREFETCH_SIZE),
        daemon=True,
    )
    prefetch_thread.start()

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

    # ── Buffers para visualizador (se construye al final) ─────────
    frames_buffer:     list[np.ndarray] = []
    dets_buffer:       list[list]       = []
    img_names_buffer:  list[str]        = []

    # ── Loop principal ────────────────────────────────────────────
    all_detections: list[Detection] = []
    next_det_id = 1

    for frame_idx in range(len(images)):
        item = frame_q.get()
        if item is None:
            break
        img_path, rgb_u8 = item

        print(f"[PTV] frame {frame_idx+1}/{len(images)} -> {img_path.name}", flush=True)

        h, w = rgb_u8.shape[:2]
        if (h, w) != (run_cfg.height_px, run_cfg.width_px):
            print(f"[WARN] Shape {img_path.name}: {(h,w)} "
                  f"!= esperado {(run_cfg.height_px, run_cfg.width_px)}", flush=True)

        detections, next_det_id = detector.detect(
            image_rgb_u8 = rgb_u8,
            frame_idx    = frame_idx,
            image_name   = img_path.name,
            next_det_id  = next_det_id,
        )
        all_detections.extend(detections)

        tracker.step(
            detections = detections,
            frame_idx  = frame_idx,
            image_name = img_path.name,
        )

        # Guardar en buffers para visualizador posterior
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY) if rgb_u8.ndim == 3 else rgb_u8
        frames_buffer.append(gray)
        dets_buffer.append(list(detections))
        img_names_buffer.append(img_path.name)

    tracker.close_all()
    tracks_all      = tracker.get_all_tracks()
    tracks_filtered = [tr for tr in tracks_all
                       if len(tr.history) >= run_cfg.min_frames_keep]

    # ── Exportación ───────────────────────────────────────────────
    export_detections_csv(all_detections, run_cfg.out_dir / "detections.csv")
    export_tracks_csv(
        tracks_filtered,
        px_per_mm = run_cfg.px_per_mm,
        fps       = run_cfg.fps,
        path      = run_cfg.out_dir / "tracks.csv",
    )
    export_tracks_json(tracks_filtered, run_cfg.out_dir / "tracks.json")

    summary = {
        "meta":   raw_cfg.get("meta", {}),
        "camera": raw_cfg.get("camera", {}),
        "ptv":    raw_cfg.get("ptv", {}),
        "results": {
            "n_frames":          len(images),
            "n_detections":      len(all_detections),
            "n_tracks_raw":      len(tracks_all),
            "n_tracks_filtered": len(tracks_filtered),
            "min_frames_keep":   run_cfg.min_frames_keep,
            "skip_first_images": skip,
        },
    }
    _save_json(summary, run_cfg.out_dir / "summary.json")

    print("[PTV] Completado.", flush=True)
    print(f"[PTV] detections.csv -> {run_cfg.out_dir / 'detections.csv'}", flush=True)
    print(f"[PTV] tracks.csv     -> {run_cfg.out_dir / 'tracks.csv'}", flush=True)
    print(f"[PTV] summary.json   -> {run_cfg.out_dir / 'summary.json'}", flush=True)

    # ── Visualizador HTML offline ─────────────────────────────────
    ann_dir = run_cfg.out_dir / "annotations"
    ensure_dir(ann_dir)
    _save_annotated_frames(frames_buffer, dets_buffer, img_names_buffer,
                           tracks_filtered, ann_dir,
                           getattr(run_cfg, "viz_tail_length", 0))

    ann_images = list(ann_dir.glob("*.png"))
    if ann_images:
        create_interactive_visualizer(
            ann_dir   = ann_dir,
            tracks    = tracks_filtered,
            out_path  = run_cfg.out_dir / "visualizer.html",
            width_px  = run_cfg.width_px,
            height_px = run_cfg.height_px,
            fps       = run_cfg.fps,
        )
        print(f"[PTV] visualizer.html -> {run_cfg.out_dir / 'visualizer.html'}", flush=True)

    # ── Visualizador interactivo matplotlib (post-proceso) ────────
    print("[VIZ] Abriendo visualizador interactivo...", flush=True)
    viz = InteractiveVisualizer(
        n_frames     = len(images),
        width_px     = run_cfg.width_px,
        height_px    = run_cfg.height_px,
        fps          = run_cfg.fps,
        px_per_mm    = run_cfg.px_per_mm,
        tail_length  = getattr(run_cfg, "viz_tail_length", 0),
        update_every = 1,
    )
    # Cargar todos los frames en el visualizador de una vez
    for fi, (gray, dets, img_name) in enumerate(
            zip(frames_buffer, dets_buffer, img_names_buffer)):
        viz._frames[fi] = gray
        viz._dets[fi]   = dets
        viz._tracks[fi] = [
            {
                "id":      tr.track_id,
                "history": [(r.x, r.y, r.frame_idx) for r in tr.history],
            }
            for tr in tracks_filtered if len(tr.history) > 0
        ]
        viz._max_ready = fi

    viz._current = len(images) - 1
    viz.slider.set_val(len(images) - 1)
    viz._draw(len(images) - 1)
    viz.close()