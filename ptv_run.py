#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import cv2
import tifffile


# ============================================================
# CONFIG USUARIO
# ============================================================
TIFF_PATH = r"PIV\Tomas\m72-toma-1-cam-3-n-0-car-02-piv\Basler_acA1440-220uc__40671380__20260302_175123196_0000.tiff"

# Guardado de máscara (misma dimensión). Blanco=255, Polígonos=0.
MASK_TIFF_PATH = r"FixMasks\cam-1.tiff"


# ============================================================
# UI / Parámetros
# ============================================================
WINDOW_NAME = "TIFF Polygon Annotator"

CLOSE_RADIUS_PX_ON_SCREEN = 14
MIN_POINTS_TO_CLOSE = 3

ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 30.0

BOTTOM_PANEL_H = 64


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
# TIFF -> float [0..1] (para display)
# ---------------------------
def read_tiff_as_float01(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    arr = tifffile.imread(path)

    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        # (C,H,W) -> (H,W,C) si parece canal-primero
        if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            img = np.transpose(arr, (1, 2, 0))
        else:
            img = arr
    else:
        raise ValueError(f"TIFF no soportado: shape={arr.shape}")

    H, W = img.shape[:2]

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

    return base01, (H, W)


def robust_percentile_limits(base01_gray: np.ndarray, clip_percent: float) -> tuple[float, float]:
    cp = float(np.clip(clip_percent, 0.0, 10.0))
    lo_p = cp
    hi_p = 100.0 - cp
    lo = float(np.percentile(base01_gray, lo_p))
    hi = float(np.percentile(base01_gray, hi_p))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def apply_display(base01: np.ndarray, ev_tenths: int, gamma_x100: int, clip_percent_x10: int,
                  cached_limits: tuple[float, float] | None) -> np.ndarray:
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
# View transforms
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
# UI overlay
# ---------------------------
def draw_ui(canvas: np.ndarray, text_lines: list[str]) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - BOTTOM_PANEL_H), (w, h), (0, 0, 0), -1)
    y = h - BOTTOM_PANEL_H + 22
    for line in text_lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def draw_overlay(img_bgr_u8: np.ndarray, view: ViewState, st: PolyEditorState,
                 cursor_screen: tuple[int, int] | None,
                 ev_tenths: int, gamma_x100: int, clip_percent_x10: int) -> np.ndarray:
    H, W = img_bgr_u8.shape[:2]

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        if win_w <= 0 or win_h <= 0:
            win_w, win_h = W, H + BOTTOM_PANEL_H
    except Exception:
        win_w, win_h = W, H + BOTTOM_PANEL_H

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
        f"closed={len(st.polys_closed)} | current_pts={len(st.current)} | zoom={view.scale:.2f} | EV={ev:+.1f} | gamma={gamma:.2f} | clip%={clip:.1f} | {coord}",
        "Mouse: left=add/close | wheel=zoom | right-drag=pan   Keys: S=save-mask  Z=undo  X=del poly  C=clear  R=reset view  ESC=exit (auto-save)",
    ]
    draw_ui(canvas, lines)
    return canvas


# ---------------------------
# Guardado máscara TIFF
# ---------------------------
def save_mask_tiff(mask_path: str | Path,
                   polys_closed: list[list[tuple[float, float]]],
                   img_hw: tuple[int, int]) -> None:
    """
    Guarda máscara TIFF uint8:
      - fondo 255 (blanco)
      - polígonos 0 (negro)
    """
    H, W = img_hw
    mask = np.full((H, W), 255, dtype=np.uint8)

    # fillPoly requiere int32 con shape (N,1,2) en (x,y)
    for poly in polys_closed:
        if len(poly) < 3:
            continue
        pts = np.array([[int(round(x)), int(round(y))] for (x, y) in poly], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 0)

    mask_path = Path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(mask_path), mask, photometric="minisblack")


# ============================================================
# MAIN
# ============================================================
def main():
    base01, (H, W) = read_tiff_as_float01(TIFF_PATH)

    st = PolyEditorState(polys_closed=[], current=[])
    view = ViewState()
    cursor_screen = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Sliders (display)
    cv2.createTrackbar("EV x0.1", WINDOW_NAME, 50, 100, lambda v: None)          # 50 => 0.0
    cv2.createTrackbar("Gamma x0.01", WINDOW_NAME, 100, 300, lambda v: None)     # 1.00
    cv2.createTrackbar("Clip % x0.1", WINDOW_NAME, 10, 100, lambda v: None)      # 1.0%

    def reset_view():
        view.scale = 1.0
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            draw_h = max(100, win_h - BOTTOM_PANEL_H)
            view.offset_x = (win_w - W) * 0.5
            view.offset_y = (draw_h - H) * 0.5
        except Exception:
            view.offset_x = 0.0
            view.offset_y = 0.0

    reset_view()

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
        nonlocal cursor_screen
        cursor_screen = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
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

        if event == cv2.EVENT_LBUTTONDOWN:
            # evita clicks en panel inferior
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
                draw_h = max(100, win_h - BOTTOM_PANEL_H)
            except Exception:
                draw_h = H

            if y >= draw_h:
                return

            if try_close_polygon((x, y)):
                return

            px, py = screen_to_img((x, y), view)
            px = float(np.clip(px, 0, W - 1))
            py = float(np.clip(py, 0, H - 1))
            st.current.append((px, py))
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = ZOOM_STEP if flags > 0 else (1.0 / ZOOM_STEP)
            old_scale = view.scale
            new_scale = float(np.clip(old_scale * factor, ZOOM_MIN, ZOOM_MAX))
            if abs(new_scale - old_scale) < 1e-12:
                return
            img_x, img_y = screen_to_img((x, y), view)
            view.scale = new_scale
            view.offset_x = x - img_x * view.scale
            view.offset_y = y - img_y * view.scale
            return

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Cache percentiles
    last_clip_x10 = None
    cached_limits = None

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

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            draw_h = max(100, win_h - BOTTOM_PANEL_H)
        except Exception:
            win_w, draw_h = W, H

        clamp_view(view, (H, W), (win_w, draw_h))

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
        elif key in (ord('z'), 8):
            if st.current:
                st.current.pop()
        elif key == ord('x'):
            if st.current:
                st.current.clear()
            elif st.polys_closed:
                st.polys_closed.pop()
        elif key == ord('c'):
            st.current.clear()
            st.polys_closed.clear()
        elif key == ord('r'):
            reset_view()
        elif key == ord('s'):
            save_mask_tiff(MASK_TIFF_PATH, st.polys_closed, (H, W))
            print(f"[OK] Máscara guardada: {MASK_TIFF_PATH}")

    # Guardado automático al salir (solo con polígonos cerrados)
    save_mask_tiff(MASK_TIFF_PATH, st.polys_closed, (H, W))
    print(f"[DONE] Máscara final guardada: {MASK_TIFF_PATH}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()