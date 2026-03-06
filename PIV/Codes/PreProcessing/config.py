# PIV/Codes/PreProcessing/masks.py
from __future__ import annotations
from pathlib import Path
import re
import shutil
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from tqdm import tqdm
import torch
import cv2


# ---------- helpers ----------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def ensure_gray(img: Image.Image) -> Image.Image:
    return img.convert("L") if img.mode != "L" else img

def prepare_for_model(gray_img: Image.Image) -> np.ndarray:
    arr = np.array(gray_img)               # (H, W)
    arr3 = np.stack([arr, arr, arr], -1)   # (H, W, 3)
    return arr3

def list_images(folder: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.png",
            "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF", "*.PNG")
    files: List[Path] = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=lambda p: natural_key(p.name))

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

def postprocess_mask(
    mask_img: Image.Image,
    apply_smoothing: bool = True,
    kernel_size: int = 5,
    morph_iter: int = 1,
    remove_min_area: int = 50,
    gaussian_blur_k: int = 0,
    final_threshold: int = 128,
) -> Image.Image:
    if not apply_smoothing:
        return mask_img

    mask = np.array(mask_img, dtype=np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    k = max(3, kernel_size | 1)  # impar >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

    if remove_min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(num_labels, dtype=bool)
        for i in range(1, num_labels):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= remove_min_area
        cleaned = np.zeros_like(mask)
        cleaned[np.isin(labels, np.where(keep)[0])] = 255
        mask = cleaned

    if gaussian_blur_k and gaussian_blur_k >= 3 and gaussian_blur_k % 2 == 1:
        mask = cv2.GaussianBlur(mask, (gaussian_blur_k, gaussian_blur_k), 0)
        _, mask = cv2.threshold(mask, final_threshold, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask, mode="L")

# ===== NUEVO: fixed mask loader (positiva: 255=enmascarado) =====
def load_fixed_mask(
    fixed_mask_path: Path,
    target_size: Tuple[int, int],
    threshold: int = 127,
    resize_if_needed: bool = True,
) -> np.ndarray:
    if not fixed_mask_path.exists():
        raise FileNotFoundError(f"No existe máscara fija: {fixed_mask_path}")

    with Image.open(fixed_mask_path) as im:
        im.load()
        im = im.convert("L")

    if im.size != target_size:
        if not resize_if_needed:
            raise RuntimeError(
                f"Máscara fija tamaño {im.size} != target {target_size} y resize_if_needed=False"
            )
        im = im.resize(target_size, resample=Image.NEAREST)

    arr = np.array(im, dtype=np.uint8)
    arr = np.where(arr >= threshold, 255, 0).astype(np.uint8)  # binario
    return arr

def intersect_positive_masks(dynamic_pos: Image.Image, fixed_pos_arr: np.ndarray) -> Image.Image:
    dyn = np.array(dynamic_pos, dtype=np.uint8)
    dyn_bin = np.where(dyn > 0, 255, 0).astype(np.uint8)

    # Intersección de zonas enmascaradas (255=enmascarado)
    inter = np.where((dyn_bin == 255) & (fixed_pos_arr == 255), 255, 0).astype(np.uint8)
    return Image.fromarray(inter, mode="L")


def process_one_image(
    model: YOLO,
    img_path: Path,
    out_dir: Path,
    conf_thresh: float,
    device: str,
    invert_mask: bool,
    # NUEVO:
    apply_static_mask: bool,
    fixed_mask_path: Optional[Path],
    fixed_mask_threshold: int,
    resize_fixed_mask_if_needed: bool,
) -> None:
    with Image.open(img_path) as im:
        im.load()
        gray = ensure_gray(im)

    arr3 = prepare_for_model(gray)
    results = model.predict(
        source=arr3,
        imgsz=1024,
        conf=conf_thresh,
        device=device,
        verbose=False,
    )

    # dinámica positiva: 255 = enmascarado
    mask = Image.new("L", gray.size, color=0)
    if results:
        r = results[0]
        drew = draw_segmentation_masks(mask, r)
        if not drew:
            draw_boxes(mask, r)
        mask = postprocess_mask(mask)

    positive_mask = mask  # 255=enmascarado

    # ===== NUEVO: intersección con máscara fija =====
    if apply_static_mask:
        if fixed_mask_path is None:
            raise ValueError("apply_static_mask=True pero fixed_mask_path=None")
        fixed_arr = load_fixed_mask(
            fixed_mask_path,
            target_size=gray.size,  # (W,H)
            threshold=fixed_mask_threshold,
            resize_if_needed=resize_fixed_mask_if_needed,
        )
        positive_mask = intersect_positive_masks(positive_mask, fixed_arr)

    # Guardado final (posible invert)
    final_mask = ImageOps.invert(positive_mask) if invert_mask else positive_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_mask.tiff"
    final_mask.convert("L").save(out_path, format="TIFF", compression="raw")


# ---------- API que llama preprocess_run.py ----------
def run_masks_yolo(
    model_path: Path,
    images_dir: Path,
    output_dir: Path,
    conf_thresh: float,
    device: str,
    invert_mask: bool,
    delete_existing: bool,
    # ===== NUEVO =====
    apply_static_mask: bool = False,
    fixed_mask_path: Optional[Path] = None,
    fixed_mask_threshold: int = 127,
    resize_fixed_mask_if_needed: bool = True,
) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe images_dir: {images_dir}")

    if delete_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if apply_static_mask:
        if fixed_mask_path is None:
            raise ValueError("apply_static_mask=True pero fixed_mask_path=None")
        if not fixed_mask_path.exists():
            raise FileNotFoundError(f"fixed_mask_path no existe: {fixed_mask_path}")

    model = YOLO(str(model_path))

    files = list_images(images_dir)
    if not files:
        raise RuntimeError(f"No hay imágenes en: {images_dir}")

    for f in tqdm(files, desc=f"Masks {images_dir.name}", unit="img"):
        process_one_image(
            model=model,
            img_path=f,
            out_dir=output_dir,
            conf_thresh=conf_thresh,
            device=device,
            invert_mask=invert_mask,
            apply_static_mask=apply_static_mask,
            fixed_mask_path=fixed_mask_path,
            fixed_mask_threshold=fixed_mask_threshold,
            resize_fixed_mask_if_needed=resize_fixed_mask_if_needed,
        )

    msg = f"[MASKS] listo: {output_dir} ({len(files)} imgs)"
    if apply_static_mask:
        msg += f" | static={fixed_mask_path}"
    print(msg)