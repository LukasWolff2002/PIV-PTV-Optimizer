"""
image_utils.py
==============
Utilidades de carga, preprocesamiento, máscaras y geometría de contornos.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile


# ─────────────────────────────────────────────
# HELPERS GENERALES
# ─────────────────────────────────────────────

def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wrap_angle_deg(angle_deg: float) -> float:
    """Normaliza ángulo a [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Diferencia angular mínima en grados."""
    return abs(wrap_angle_deg(a_deg - b_deg))


def np_to_builtin(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Tipo no serializable: {type(obj)}")


# ─────────────────────────────────────────────
# LISTADO DE IMÁGENES
# ─────────────────────────────────────────────

def list_images(images_dir: Path, max_images: int | None = None) -> list[Path]:
    valid_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    imgs = [p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext]
    imgs.sort(key=lambda p: natural_key(p.name))
    if max_images is not None:
        imgs = imgs[:max_images]
    return imgs


# ─────────────────────────────────────────────
# LECTURA DE IMAGEN
# ─────────────────────────────────────────────

def read_image_any(path: Path) -> np.ndarray:
    """
    Carga imagen como:
    - grayscale: (H, W)
    - color:     (H, W, 3) en RGB
    """
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"No se pudo leer imagen: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def normalize_to_uint8_for_yolo(img: np.ndarray) -> np.ndarray:
    """Convierte a uint8 RGB listo para YOLO."""
    if img.ndim == 2:
        base = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        base = img[..., :3]
    else:
        raise ValueError(f"Formato no soportado: shape={img.shape}")

    if base.dtype == np.uint8:
        out = base.copy()
    elif base.dtype == np.uint16:
        out = np.clip(base / 257.0, 0, 255).astype(np.uint8)
    else:
        base_f = base.astype(np.float32)
        mn, mx = float(np.min(base_f)), float(np.max(base_f))
        if mx <= mn:
            out = np.zeros_like(base_f, dtype=np.uint8)
        else:
            out = ((base_f - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out


def image_to_float01_grayscale(img: np.ndarray) -> np.ndarray:
    """Convierte imagen a float64 [0, 1] en escala de grises."""
    arr = img.copy()
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY) if arr.shape[2] >= 3 else arr[..., 0]
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    elif arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    else:
        arr = arr.astype(np.float64)
        mx = arr.max()
        return arr / mx if mx > 1.0 else arr


def preprocess_frame_for_ptv(raw_img: np.ndarray, preprocess_params: dict | None) -> np.ndarray:
    """
    Aplica preprocesamiento a imagen PTV.
    Salida: RGB uint8 lista para YOLO.
    """
    if not preprocess_params:
        return normalize_to_uint8_for_yolo(raw_img)

    from PTV.Codes.PreProcessing.filters import apply_preprocessing
    img01 = image_to_float01_grayscale(raw_img)
    img01_proc = apply_preprocessing(img01, preprocess_params)
    img_u8 = np.clip(img01_proc * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────────
# MÁSCARAS
# ─────────────────────────────────────────────

def load_mask_as_bool(
    mask_path: Path, expected_hw: tuple[int, int] | None = None
) -> np.ndarray:
    """
    Carga máscara binaria.
    Convención: negro (0) = mantener, blanco = eliminar.
    """
    mask = read_image_any(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask == 0
    if expected_hw is not None:
        h, w = expected_hw
        if mask_bool.shape != (h, w):
            raise ValueError(
                f"Máscara shape {mask_bool.shape}, esperado {(h, w)}"
            )
    return mask_bool


def apply_static_mask_to_rgb(
    rgb: np.ndarray, static_mask_keep: np.ndarray
) -> np.ndarray:
    """Aplica máscara booleana sobre imagen RGB: False → negro."""
    out = rgb.copy()
    out[~static_mask_keep] = 0
    return out


# ─────────────────────────────────────────────
# GEOMETRÍA
# ─────────────────────────────────────────────

def polygon_to_mask(poly_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def contour_geometry_from_mask(mask_u8: np.ndarray) -> dict | None:
    """
    Extrae geometría de fibra desde máscara binaria.
    Retorna: cx, cy, angle_deg, length_px, width_px, area_px, bbox_xyxy
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return None

    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-12:
        return None

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(cnt)
    bbox_xyxy = [float(x), float(y), float(x + w), float(y + h)]

    rect = cv2.minAreaRect(cnt)
    (_, _), (rw, rh), angle = rect

    if rw >= rh:
        length_px, width_px = float(rw), float(rh)
        angle_deg = float(angle)
    else:
        length_px, width_px = float(rh), float(rw)
        angle_deg = float(angle + 90.0)

    angle_deg = wrap_angle_deg(angle_deg)

    return {
        "cx": cx,
        "cy": cy,
        "angle_deg": angle_deg,
        "length_px": length_px,
        "width_px": width_px,
        "area_px": area,
        "bbox_xyxy": bbox_xyxy,
    }
