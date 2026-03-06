#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera máscaras binarias a partir de detecciones YOLO sobre imágenes 1024x1024.

Entrada esperada:
- IMAGES_DIR contiene subcarpetas (una por toma), y dentro las imágenes.
Salida:
- OUTPUT_DIR replica la estructura:
    OUTPUT_DIR/<subcarpeta>/<imagen>_mask.tiff
- Opción: si existe OUTPUT_DIR/<subcarpeta>, se puede borrar antes de reprocesar (DELETE_EXISTING)
"""

from pathlib import Path
import re
import shutil
from typing import List

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from tqdm import tqdm
import torch
import cv2  # postproceso morfológico

# =========================
# CONFIGURACIÓN DEL USUARIO
# =========================
Camara = 3
MODEL_PATH   = "PIV-Codes/Segmentation-Models/DynamicMask.pt"
IMAGES_DIR   = f"TomasProcesadas/Cam{Camara}"
OUTPUT_DIR   = f"Masks/Cam{Camara}"
DELETE_EXISTING   = True     # si True y existe OUTPUT_DIR/subcarpeta => borrar y reprocesar
OVERLAY_ORIGINAL  = False

CONF_THRESH  = 0.25
DEVICE       = "0"   # "cpu" o "0" para GPU 0, etc.

# --- Parámetros de suavizado (post-proceso) ---
APPLY_SMOOTHING   = True
KERNEL_SIZE       = 5
MORPH_ITER        = 1
REMOVE_MIN_AREA   = 50
GAUSSIAN_BLUR_K   = 0
FINAL_THRESHOLD   = 128

# --- Overlay ---

OVERLAY_ALPHA     = 128

# --- Salida ---
INVERT_MASK       = True

# --- Procesar estructura por subcarpetas ---
PROCESS_RECURSIVE = True      # True = procesa subcarpetas; False = carpeta plana

# --- NUEVO: manejo de subcarpetas ya existentes en OUTPUT_DIR ---
SKIP_IF_EXISTS    = True      # si ya existe OUTPUT_DIR/subcarpeta => skip (si DELETE_EXISTING=False)


# =========================

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def ensure_gray(img: Image.Image) -> Image.Image:
    return img.convert("L") if img.mode != "L" else img

def prepare_for_model(gray_img: Image.Image) -> np.ndarray:
    arr = np.array(gray_img)               # (H, W)
    arr3 = np.stack([arr, arr, arr], -1)   # (H, W, 3)
    return arr3

def draw_segmentation_masks(mask_canvas: Image.Image, result) -> bool:
    if getattr(result, "masks", None) is None:
        return False
    masks = result.masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        return False

    h, w = masks.orig_shape
    data = masks.data
    if not isinstance(data, torch.Tensor):
        return False

    data_np = data.cpu().numpy()
    canvas_np = np.array(mask_canvas, dtype=np.uint8)
    drew_any = False

    for m in data_np:
        m_img = Image.fromarray((m * 255).astype(np.uint8), mode="L").resize((w, h), resample=Image.NEAREST)
        m_arr = np.array(m_img)
        canvas_np = np.maximum(canvas_np, (m_arr > 0).astype(np.uint8) * 255)
        drew_any = True

    mask_canvas.paste(Image.fromarray(canvas_np, mode="L"))
    return drew_any

def draw_boxes(mask_canvas: Image.Image, result) -> bool:
    # Fallback: si el modelo no entrega masks (solo boxes)
    if getattr(result, "boxes", None) is None or result.boxes is None or len(result.boxes) == 0:
        return False
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask_canvas)
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    drew_any = False
    for x1, y1, x2, y2 in boxes_xyxy:
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], fill=255)
        drew_any = True
    return drew_any

def postprocess_mask(mask_img: Image.Image) -> Image.Image:
    if not APPLY_SMOOTHING:
        return mask_img

    mask = np.array(mask_img, dtype=np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    k = max(3, KERNEL_SIZE | 1)  # asegura impar >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER)

    if REMOVE_MIN_AREA > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(num_labels, dtype=bool)
        for i in range(1, num_labels):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= REMOVE_MIN_AREA
        cleaned = np.zeros_like(mask)
        cleaned[np.isin(labels, np.where(keep)[0])] = 255
        mask = cleaned

    if GAUSSIAN_BLUR_K and GAUSSIAN_BLUR_K >= 3 and GAUSSIAN_BLUR_K % 2 == 1:
        mask = cv2.GaussianBlur(mask, (GAUSSIAN_BLUR_K, GAUSSIAN_BLUR_K), 0)
        _, mask = cv2.threshold(mask, FINAL_THRESHOLD, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask, mode="L")

def create_overlay(original_gray: Image.Image, mask: Image.Image, alpha: int) -> Image.Image:
    base = original_gray.convert("RGBA")
    mask_arr = np.array(mask, dtype=np.uint8)

    overlay_arr = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
    overlay_arr[mask_arr > 0] = [255, 255, 255, alpha]
    overlay = Image.fromarray(overlay_arr, mode="RGBA")

    return Image.alpha_composite(base, overlay)

def process_image(model: YOLO, img_path: Path, out_dir: Path) -> None:
    with Image.open(img_path) as im:
        im.load()
        gray = ensure_gray(im)

    arr3 = prepare_for_model(gray)
    results = model.predict(source=arr3, imgsz=1024, conf=CONF_THRESH, device=DEVICE, verbose=False)

    mask = Image.new("L", gray.size, color=0)
    if results:
        r = results[0]
        drew_masks = draw_segmentation_masks(mask, r)
        if not drew_masks:
            draw_boxes(mask, r)
        mask = postprocess_mask(mask)

    positive_mask = mask
    final_mask = ImageOps.invert(positive_mask) if INVERT_MASK else positive_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_mask.tiff"
    final_mask.convert("L").save(out_path, format="TIFF", compression="raw")

    if OVERLAY_ORIGINAL:
        overlay_img = create_overlay(gray, positive_mask, OVERLAY_ALPHA)
        out_overlay = out_dir / f"{img_path.stem}_overlay.tiff"
        overlay_img.save(out_overlay, format="TIFF", compression="raw")

def list_images(folder: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF")
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=lambda p: natural_key(p.name))

def prepare_output_subfolder(out_sf: Path) -> bool:
    """
    Maneja la existencia previa de la carpeta destino.
    - Si DELETE_EXISTING=True y existe => borra y reprocesa
    - Si DELETE_EXISTING=False y SKIP_IF_EXISTS=True y existe => salta
    - En otros casos => continúa (sobrescribe archivo a archivo si OVERWRITE=True)
    """
    if out_sf.exists():
        if DELETE_EXISTING:
            print(f"[DEL] '{out_sf.name}' existe en salida. Eliminando para reprocesar...")
            shutil.rmtree(out_sf)
        elif SKIP_IF_EXISTS:
            print(f"[SKIP] '{out_sf.name}' ya procesada anteriormente.")
            return False
    return True

def main():
    images_dir = Path(IMAGES_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe IMAGES_DIR: {images_dir}")

    model = YOLO(MODEL_PATH)

    if PROCESS_RECURSIVE:
        subfolders = [p for p in images_dir.iterdir() if p.is_dir()]
        subfolders = sorted(subfolders, key=lambda p: natural_key(p.name))

        if not subfolders:
            print("No se encontraron subcarpetas en IMAGES_DIR.")
            return

        for sf in subfolders:
            out_sf = output_dir / sf.name

            if not prepare_output_subfolder(out_sf):
                continue

            files = list_images(sf)
            if not files:
                print(f"[WARN] '{sf.name}': sin imágenes.")
                continue

            for f in tqdm(files, desc=f"Procesando {sf.name}", leave=False):
                try:
                    process_image(model, f, out_sf)
                except Exception as e:
                    print(f"[ADVERTENCIA] Error en {sf.name}/{f.name}: {e}")

        print(f"Listo. Máscaras guardadas en: {output_dir}")

    else:
        files = list_images(images_dir)
        if not files:
            print("No se encontraron imágenes.")
            return

        for f in tqdm(files, desc="Procesando"):
            try:
                process_image(model, f, output_dir)
            except Exception as e:
                print(f"[ADVERTENCIA] Error procesando {f.name}: {e}")

        print(f"Listo. Máscaras guardadas en: {output_dir}")

if __name__ == "__main__":
    main()