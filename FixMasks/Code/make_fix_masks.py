#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import cv2
import tifffile


# ============================================================
# CONFIG USUARIO
# ============================================================
IMG_PATH = r"BasePhotos/PIV/cam6.tiff"   # acepta .tif, .tiff o .bmp
OUT_DIR  = r"FixMasks"                  # la máscara SIEMPRE se guarda como .tiff

WINDOW_NAME = "Polygon Mask Annotator"

# Extensiones soportadas
TIFF_EXTS = (".tif", ".tiff")
BMP_EXTS  = (".bmp",)
SUPPORTED_EXTS = TIFF_EXTS + BMP_EXTS
MASK_EXT = ".tiff"

# Polígonos
CLOSE_RADIUS_PX_ON_SCREEN = 14
MIN_POINTS_TO_CLOSE = 3

# Zoom
ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 30.0

# Panel inferior (UI)
BOTTOM_PANEL_H = 64

# Vista inicial: la imagen completa debe caber en pantalla
INIT_WIN_MAX_W = 1600   # tamaño máximo inicial de la ventana (px)
INIT_WIN_MAX_H = 850
FIT_MARGIN = 0.98       # <1 deja un pequeño borde alrededor de la imagen


@dataclass
class ViewState:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    dragging: bool = False
    last_mouse: tuple[int, int] | None = None


@dataclass
class PolyEditorState:
    polys_closed: list[list[tuple[float, float]]]
    current: list[tuple[float, float]]


# ---------------------------
# IO / imagen
# ---------------------------
def _load_tiff_raw(path: Path) -> np.ndarray:
    """Lee TIFF y devuelve (H,W) o (H,W,C) en orden RGB."""
    arr = tifffile.imread(str(path))

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # (C,H,W) -> (H,W,C) si parece canal-primero
        if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            return np.transpose(arr, (1, 2, 0))
        # (H,W,C)
        if arr.shape[2] in (1, 3, 4):
            return arr[..., 0] if arr.shape[2] == 1 else arr
        # Multipágina (N,H,W): usar la primera página
        print(f"[INFO] TIFF multipágina {arr.shape}, se usa la página 0")
        return arr[0]

    if arr.ndim == 4:
        # Multipágina a color (N,H,W,C): usar la primera página
        print(f"[INFO] TIFF multipágina {arr.shape}, se usa la página 0")
        return arr[0]

    raise ValueError(f"TIFF no soportado: shape={arr.shape}")


def _load_bmp_raw(path: Path) -> np.ndarray:
    """Lee BMP y devuelve (H,W) o (H,W,C) en orden RGB."""
    # np.fromfile + imdecode funciona también con rutas con tildes/ñ en Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"No se pudo leer el BMP: {path}")

    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    raise ValueError(f"BMP no soportado: shape={arr.shape}")


def read_image_as_float01(path: str) -> tuple[np.ndarray, dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el archivo: {p}")

    ext = p.suffix.lower()
    if ext in TIFF_EXTS:
        img = _load_tiff_raw(p)
    elif ext in BMP_EXTS:
        img = _load_bmp_raw(p)
    else:
        raise ValueError(f"Extensión no soportada '{ext}'. Usa: {', '.join(SUPPORTED_EXTS)}")

    meta = {"shape": tuple(img.shape), "dtype": str(img.dtype), "format": ext}

    img_f = img.astype(np.float32)

    mn = float(np.min(img_f))
    mx = float(np.max(img_f))
    if mx > mn:
        base01 = (img_f - mn) / (mx - mn)
    else:
        base01 = np.zeros_like(img_f, dtype=np.float32)

    # Si RGBA -> RGB
    if base01.ndim == 3 and base01.shape[2] == 4:
        base01 = base01[:, :, :3]

    return base01, meta


def robust_percentile_limits(base01_gray: np.ndarray, clip_percent: float) -> tuple[float, float]:
    """
    clip_percent en [0..10] típicamente. 1.0 => p1 y p99.
    """
    cp = float(np.clip(clip_percent, 0.0, 10.0))
    lo = float(np.percentile(base01_gray, cp))
    hi = float(np.percentile(base01_gray, 100.0 - cp))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def apply_display(base01: np.ndarray, ev_tenths: int, gamma_x100: int, clip_percent_x10: int,
                  cached_limits: tuple[float, float] | None = None) -> np.ndarray:
    """
    base01: float32 [0..1] (H,W) o (H,W,3)
    Retorna BGR uint8 listo para OpenCV.
    """
    ev = ev_tenths / 10.0
    factor = 2.0 ** ev

    gamma = max(gamma_x100 / 100.0, 0.05)
    clip_percent = max(clip_percent_x10 / 10.0, 0.0)

    img = (base01 * factor).astype(np.float32)

    if img.ndim == 2:
        gray = img
    else:
        gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    if cached_limits is None:
        lo, hi = robust_percentile_limits(gray, clip_percent)
    else:
        lo, hi = cached_limits

    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0.0, 1.0)

    img = np.power(img, 1.0 / gamma).astype(np.float32)

    u8 = (img * 255.0 + 0.5).astype(np.uint8)

    if u8.ndim == 2:
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(u8[..., :3], cv2.COLOR_RGB2BGR)

    return bgr


# ---------------------------
# Transformaciones view
# ---------------------------
def img_to_screen(pt_img: tuple[float, float], view: ViewState) -> tuple[int, int]:
    x, y = pt_img
    return int(round(x * view.scale + view.offset_x)), int(round(y * view.scale + view.offset_y))


def screen_to_img(pt_screen: tuple[int, int], view: ViewState) -> tuple[float, float]:
    sx, sy = pt_screen
    return (sx - view.offset_x) / view.scale, (sy - view.offset_y) / view.scale


def clamp_view(view: ViewState, img_hw: tuple[int, int], win_wh: tuple[int, int]) -> None:
    H, W = img_hw
    win_w, win_h = win_wh
    pad = 0.6
    min_off_x = -W * view.scale * pad + win_w * 0.1
    max_off_x = win_w * 0.9
    min_off_y = -H * view.scale * pad + win_h * 0.1
    max_off_y = win_h * 0.9
    view.offset_x = float(np.clip(view.offset_x, min_off_x, max_off_x))
    view.offset_y = float(np.clip(view.offset_y, min_off_y, max_off_y))


# ---------------------------
# UI / overlay
# ---------------------------
def draw_ui(canvas: np.ndarray, text_lines: list[str]) -> np.ndarray:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - BOTTOM_PANEL_H), (w, h), (0, 0, 0), -1)

    y = h - BOTTOM_PANEL_H + 22
    for line in text_lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return canvas


def draw_overlay(img_bgr_u8: np.ndarray, view: ViewState, st: PolyEditorState,
                 cursor_screen: tuple[int, int] | None,
                 ev_tenths: int, gamma_x100: int, clip_percent_x10: int) -> np.ndarray:
    H, W = img_bgr_u8.shape[:2]

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        if win_w <= 0 or win_h <= 0:
            win_w, win_h = W, H
    except Exception:
        win_w, win_h = W, H

    draw_h = max(100, win_h - BOTTOM_PANEL_H)

    M = np.array([[view.scale, 0, view.offset_x],
                  [0, view.scale, view.offset_y]], dtype=np.float32)

    img_area = cv2.warpAffine(
        img_bgr_u8, M, (win_w, draw_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    canvas = np.zeros((draw_h + BOTTOM_PANEL_H, win_w, 3), dtype=np.uint8)
    canvas[:draw_h, :, :] = img_area

    # Crosshair
    if cursor_screen is not None:
        cx, cy = cursor_screen
        if 0 <= cx < win_w and 0 <= cy < draw_h:
            cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1, cv2.LINE_AA)

    # Polígonos cerrados (amarillo)
    for poly in st.polys_closed:
        pts_s = [img_to_screen(p, view) for p in poly]
        if len(pts_s) >= 2:
            for a, b in zip(pts_s[:-1], pts_s[1:]):
                cv2.line(canvas, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        if len(pts_s) >= 3:
            cv2.line(canvas, pts_s[-1], pts_s[0], (0, 255, 255), 2, cv2.LINE_AA)

    # Polígono actual (cian)
    pts_s = [img_to_screen(p, view) for p in st.current]
    if len(pts_s) >= 2:
        for a, b in zip(pts_s[:-1], pts_s[1:]):
            cv2.line(canvas, a, b, (255, 255, 0), 2, cv2.LINE_AA)
    for i, p in enumerate(pts_s):
        color = (0, 0, 255) if i == 0 else (0, 255, 0)
        cv2.circle(canvas, p, 5, color, -1, cv2.LINE_AA)

    # Hint de cierre
    if cursor_screen is not None and len(pts_s) >= MIN_POINTS_TO_CLOSE:
        p0 = pts_s[0]
        d = math.hypot(cursor_screen[0] - p0[0], cursor_screen[1] - p0[1])
        if d <= CLOSE_RADIUS_PX_ON_SCREEN:
            cv2.circle(canvas, p0, int(CLOSE_RADIUS_PX_ON_SCREEN), (255, 255, 0), 2, cv2.LINE_AA)

    ev = ev_tenths / 10.0
    gamma = gamma_x100 / 100.0
    clip = clip_percent_x10 / 10.0

    if cursor_screen is not None and cursor_screen[1] < draw_h:
        ix, iy = screen_to_img(cursor_screen, view)
        coord = f"cursor img=(x={ix:.1f}, y={iy:.1f})"
    else:
        coord = "cursor img=(-, -)"

    lines = [
        f"polys_closed={len(st.polys_closed)} | current_pts={len(st.current)} | zoom={view.scale:.2f} | EV={ev:+.1f} | gamma={gamma:.2f} | clip%={clip:.1f} | {coord}",
        "Mouse: left=add/close | wheel=zoom | right-drag=pan   Keys: S=save  Z=undo  X=del poly  C=clear  R=fit view  +/-=zoom  ESC=exit (auto-save)",
    ]
    canvas = draw_ui(canvas, lines)
    return canvas


# ---------------------------
# Guardado máscara (siempre TIFF)
# ---------------------------
def mask_output_path(out_dir: str | Path, img_path: str) -> Path:
    """cam6.bmp -> FixMasks/cam6.tiff ; cam6.tif -> FixMasks/cam6.tiff"""
    return Path(out_dir) / (Path(img_path).stem + MASK_EXT)


def save_binary_mask_tiff(out_dir: str | Path, st: PolyEditorState, img_path: str,
                          img_shape_hw: tuple[int, int]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = img_shape_hw
    mask = np.full((H, W), 255, dtype=np.uint8)

    for poly in st.polys_closed:
        if len(poly) >= 3:
            pts = np.round(np.array(poly, dtype=np.float32)).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 0)

    out_path = mask_output_path(out_dir, img_path)
    tifffile.imwrite(str(out_path), mask)
    return out_path


# ============================================================
# MAIN
# ============================================================
def main():
    base01, meta = read_image_as_float01(IMG_PATH)
    print(f"[INFO] Imagen cargada: {IMG_PATH} | {meta}")
    print(f"[INFO] La máscara se guardará en: {mask_output_path(OUT_DIR, IMG_PATH)}")

    st = PolyEditorState(polys_closed=[], current=[])
    view = ViewState()
    cursor_screen = None

    H_img, W_img = base01.shape[:2]

    # Tamaño inicial de ventana: la imagen reducida (si hace falta) para caber en INIT_WIN_MAX_*
    init_s = min(1.0, INIT_WIN_MAX_W / W_img, INIT_WIN_MAX_H / H_img)
    init_w = max(400, int(round(W_img * init_s)))
    init_h = max(100, int(round(H_img * init_s))) + BOTTOM_PANEL_H

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, init_w, init_h)

    # Trackbars
    cv2.createTrackbar("EV x0.1", WINDOW_NAME, 50, 100, lambda v: None)       # 50 => 0.0
    cv2.createTrackbar("Gamma x0.01", WINDOW_NAME, 100, 300, lambda v: None)  # 1.00
    cv2.createTrackbar("Clip % x0.1", WINDOW_NAME, 10, 100, lambda v: None)   # 1.0%

    # Primer frame vacío para que la ventana exista y reporte su tamaño real
    cv2.imshow(WINDOW_NAME, np.zeros((init_h, init_w, 3), dtype=np.uint8))
    cv2.waitKey(1)

    def get_window_size() -> tuple[int, int]:
        """Devuelve (ancho, alto_zona_imagen) de la ventana."""
        try:
            _, _, ww, wh = cv2.getWindowImageRect(WINDOW_NAME)
            if ww > 0 and wh > 0:
                return ww, max(100, wh - BOTTOM_PANEL_H)
        except Exception:
            pass
        return init_w, init_h - BOTTOM_PANEL_H

    # Mientras sea True, la vista se reajusta sola si cambia el tamaño de la ventana.
    # Se desactiva cuando el usuario hace zoom o pan.
    auto_fit = True
    last_fit_size = None

    def fit_view():
        """Escala y centra la imagen para que se vea completa."""
        nonlocal last_fit_size
        win_w, draw_h = get_window_size()
        s = min(win_w / W_img, draw_h / H_img) * FIT_MARGIN
        view.scale = float(np.clip(s, ZOOM_MIN, ZOOM_MAX))
        view.offset_x = (win_w - W_img * view.scale) * 0.5
        view.offset_y = (draw_h - H_img * view.scale) * 0.5
        last_fit_size = (win_w, draw_h)

    def reset_view():
        nonlocal auto_fit
        auto_fit = True
        fit_view()

    reset_view()

    def zoom_at_point(x: int, y: int, factor: float):
        nonlocal auto_fit
        auto_fit = False
        old_scale = view.scale
        new_scale = float(np.clip(old_scale * factor, ZOOM_MIN, ZOOM_MAX))
        if abs(new_scale - old_scale) < 1e-12:
            return
        img_x, img_y = screen_to_img((x, y), view)
        view.scale = new_scale
        view.offset_x = x - img_x * view.scale
        view.offset_y = y - img_y * view.scale

    def try_close_polygon(click_xy_screen: tuple[int, int]) -> bool:
        if len(st.current) < MIN_POINTS_TO_CLOSE:
            return False
        p0_screen = img_to_screen(st.current[0], view)
        d = math.hypot(click_xy_screen[0] - p0_screen[0], click_xy_screen[1] - p0_screen[1])
        if d <= CLOSE_RADIUS_PX_ON_SCREEN:
            st.polys_closed.append(st.current.copy())
            st.current.clear()
            return True
        return False

    def on_mouse(event, x, y, flags, param):
        nonlocal cursor_screen, auto_fit
        cursor_screen = (x, y)

        # Pan con botón derecho
        if event == cv2.EVENT_RBUTTONDOWN:
            auto_fit = False
            view.dragging = True
            view.last_mouse = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and view.dragging:
            lx, ly = view.last_mouse if view.last_mouse else (x, y)
            view.offset_x += (x - lx)
            view.offset_y += (y - ly)
            view.last_mouse = (x, y)
            return

        if event == cv2.EVENT_RBUTTONUP:
            view.dragging = False
            view.last_mouse = None
            return

        # Click izquierdo: agregar punto / cerrar
        if event == cv2.EVENT_LBUTTONDOWN:
            _, draw_h = get_window_size()
            if y >= draw_h:
                return

            if try_close_polygon((x, y)):
                return

            px, py = screen_to_img((x, y), view)
            H, W = base01.shape[:2]
            px = float(np.clip(px, 0, W - 1))
            py = float(np.clip(py, 0, H - 1))
            st.current.append((px, py))
            return

        # Zoom con rueda
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            if delta > 0:
                factor = ZOOM_STEP
            elif delta < 0:
                factor = 1.0 / ZOOM_STEP
            else:
                return
            zoom_at_point(x, y, factor)
            return

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Cache de percentiles
    last_clip_x10 = None
    cached_limits = None

    def save(tag: str):
        out_path = save_binary_mask_tiff(OUT_DIR, st, IMG_PATH, (base01.shape[0], base01.shape[1]))
        print(f"[{tag}] Máscara guardada: {out_path}")

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        ev_tenths = int(cv2.getTrackbarPos("EV x0.1", WINDOW_NAME) - 50)
        gamma_x100 = int(max(5, cv2.getTrackbarPos("Gamma x0.01", WINDOW_NAME)))
        clip_percent_x10 = int(cv2.getTrackbarPos("Clip % x0.1", WINDOW_NAME))

        if last_clip_x10 != clip_percent_x10:
            if base01.ndim == 2:
                gray = base01
            else:
                gray = 0.2126 * base01[..., 0] + 0.7152 * base01[..., 1] + 0.0722 * base01[..., 2]
            cached_limits = robust_percentile_limits(gray, clip_percent_x10 / 10.0)
            last_clip_x10 = clip_percent_x10

        win_w, draw_h = get_window_size()

        # Si la ventana cambió de tamaño y el usuario no ha tocado la vista, reajustar
        if auto_fit and last_fit_size != (win_w, draw_h):
            fit_view()

        clamp_view(view, (base01.shape[0], base01.shape[1]), (win_w, draw_h))

        img_bgr_u8 = apply_display(
            base01,
            ev_tenths=ev_tenths,
            gamma_x100=gamma_x100,
            clip_percent_x10=clip_percent_x10,
            cached_limits=cached_limits,
        )

        frame = draw_overlay(img_bgr_u8, view, st, cursor_screen, ev_tenths, gamma_x100, clip_percent_x10)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(16) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord('z'), 8):  # undo punto
            if st.current:
                st.current.pop()
        elif key == ord('x'):  # borrar último polígono o limpiar actual
            if st.current:
                st.current.clear()
            elif st.polys_closed:
                st.polys_closed.pop()
        elif key == ord('c'):  # limpiar todo
            st.current.clear()
            st.polys_closed.clear()
        elif key == ord('r'):  # reset vista
            reset_view()
        elif key in (ord('+'), ord('=')):
            zoom_at_point(win_w // 2, draw_h // 2, ZOOM_STEP)
        elif key == ord('-'):
            zoom_at_point(win_w // 2, draw_h // 2, 1.0 / ZOOM_STEP)
        elif key == ord('s'):  # guardar manual
            save("OK")

    # Guardado automático al salir
    save("DONE")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()