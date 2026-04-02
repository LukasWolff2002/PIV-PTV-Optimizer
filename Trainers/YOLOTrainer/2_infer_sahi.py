"""
PASO 2 (CORREGIDO): Inferencia SAHI + YOLOv8-seg para fibras
=============================================================
Fixes aplicados:
  - Coordenadas de máscara correctamente rescaladas ÷ scale
  - Ángulo extraído con PCA sobre los píxeles de la máscara (más robusto que fitEllipse)
  - Sistema de coordenadas consistente entre extracción y visualización
  - Ruta temporal compatible con Windows

Uso:
    python 2_infer_sahi.py --img cam-1.tiff --model runs/fiber_seg/train/weights/best.pt
"""

import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image as PILImage
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────
PX_PER_MM      = 7.8
SCALE_FACTOR   = 3
TILE_SIZE      = 512
OVERLAP_RATIO  = 0.5
CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.3


def load_image_as_bgr(path):
    """Carga imagen (TIFF 16-bit o 8-bit, grayscale) → BGR uint8."""
    img = np.array(PILImage.open(path))
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def mask_to_fiber_params(mask_arr, scale):
    """
    Dado un array binario de máscara (en espacio upscaleado),
    extrae centroide y ángulo usando PCA sobre los píxeles activos.

    PCA es más robusto que fitEllipse para máscaras delgadas e irregulares:
    - El 1er componente principal = dirección del eje largo (orientación)
    - La proyección sobre ese eje = largo
    - La proyección sobre el 2do eje = ancho

    Retorna dict con valores en espacio ORIGINAL (÷ scale).
    """
    ys, xs = np.where(mask_arr > 0)
    if len(xs) < 5:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    # Centroide en espacio upscaleado
    cx_up = pts[:, 0].mean()
    cy_up = pts[:, 1].mean()

    # PCA
    pts_centered = pts - np.array([cx_up, cy_up])
    cov = np.cov(pts_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Eje principal: eigenvector del mayor eigenvalue
    main_axis = eigenvectors[:, -1]   # (dx, dy) en coords imagen

    # Ángulo respecto al eje horizontal [0°, 180°)
    angle_deg = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
    if angle_deg < 0:
        angle_deg += 180.0

    # Largo y ancho por proyección
    proj_main  = pts_centered @ main_axis
    minor_axis = eigenvectors[:, 0]
    proj_minor = pts_centered @ minor_axis

    length_up = proj_main.max() - proj_main.min()
    width_up  = proj_minor.max() - proj_minor.min()

    # Rescalear al espacio original
    return {
        "cx":        cx_up / scale,
        "cy":        cy_up / scale,
        "angle_deg": angle_deg,
        "main_axis": main_axis,
        "length_px": length_up / scale,
        "width_px":  width_up  / scale,
        "length_mm": (length_up / scale) / PX_PER_MM,
    }


def infer_sahi(img_path, model_path, scale=SCALE_FACTOR):
    print(f"Cargando imagen: {img_path}")
    img_bgr = load_image_as_bgr(img_path)
    src_h, src_w = img_bgr.shape[:2]
    print(f"  Tamaño original: {src_w}x{src_h}")

    img_up = cv2.resize(img_bgr, (src_w * scale, src_h * scale),
                        interpolation=cv2.INTER_CUBIC)
    print(f"  Upscaleado a: {src_w*scale}x{src_h*scale}")

    tmp_path = Path("fiber_upscaled_tmp.jpg")
    cv2.imwrite(str(tmp_path), img_up)

    print(f"Cargando modelo: {model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type           = "ultralytics",
        model_path           = str(model_path),
        confidence_threshold = CONF_THRESHOLD,
        device               = "cuda:0",
    )

    print(f"Corriendo SAHI (tiles {TILE_SIZE}x{TILE_SIZE}, overlap {OVERLAP_RATIO})...")
    result = get_sliced_prediction(
        str(tmp_path),
        detection_model,
        slice_height                = TILE_SIZE,
        slice_width                 = TILE_SIZE,
        overlap_height_ratio        = OVERLAP_RATIO,
        overlap_width_ratio         = OVERLAP_RATIO,
        postprocess_type            = "NMS",
        postprocess_match_threshold = IOU_THRESHOLD,
    )

    tmp_path.unlink(missing_ok=True)
    print(f"  Detecciones post-NMS: {len(result.object_prediction_list)}")
    return result, img_bgr, src_w, src_h, scale


def extract_fibers(result, scale):
    fibras, skipped = [], 0
    for pred in result.object_prediction_list:
        if pred.score.value < CONF_THRESHOLD or pred.mask is None:
            skipped += 1
            continue
        mask_arr = pred.mask.bool_mask.astype(np.uint8)
        params = mask_to_fiber_params(mask_arr, scale)
        if params is None:
            skipped += 1
            continue
        params["conf"] = pred.score.value
        fibras.append(params)
    print(f"  Fibras validas: {len(fibras)}  (descartadas: {skipped})")
    return fibras


def visualize(img_bgr, fibras, out_path):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    n = len(fibras)

    cmap = cm.get_cmap("gist_rainbow", max(n, 1))
    colors = [cmap(i) for i in range(n)]
    np.random.seed(42)
    np.random.shuffle(colors)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Panel izquierdo: imagen original sin anotaciones
    axes[0].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Imagen original", fontsize=13)
    axes[0].axis("off")

    # Panel central: imagen completa con detecciones
    axes[1].imshow(img_gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Fibras detectadas: {n}", fontsize=13)
    axes[1].axis("off")

    for i, f in enumerate(fibras):
        cx, cy = f["cx"], f["cy"]
        half   = f["length_px"] / 2.0
        v      = f["main_axis"]
        dx, dy = v[0] * half, v[1] * half
        c = colors[i]
        axes[1].plot([cx-dx, cx+dx], [cy-dy, cy+dy], "-", color=c, lw=1.8, alpha=0.88)
        axes[1].plot(cx, cy, "o", color=c, ms=2.5, alpha=0.7)

    # Panel derecho: zoom automático en zona más densa
    if fibras:
        cxs = np.array([f["cx"] for f in fibras])
        cys = np.array([f["cy"] for f in fibras])
        zoom_cx, zoom_cy = np.median(cxs), np.median(cys)
        hz = 200
        c1, c2 = max(0, int(zoom_cx-hz)), min(w, int(zoom_cx+hz))
        r1, r2 = max(0, int(zoom_cy-hz)), min(h, int(zoom_cy+hz))
    else:
        r1, r2, c1, c2 = h//4, 3*h//4, w//4, 3*w//4

    axes[2].imshow(img_gray[r1:r2, c1:c2], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Zoom zona densa ({c2-c1}x{r2-r1} px)", fontsize=13)
    axes[2].axis("off")

    for i, f in enumerate(fibras):
        cx_l, cy_l = f["cx"] - c1, f["cy"] - r1
        if not (0 < cx_l < c2-c1 and 0 < cy_l < r2-r1):
            continue
        half   = f["length_px"] / 2.0
        v      = f["main_axis"]
        dx, dy = v[0] * half, v[1] * half
        c = colors[i]
        axes[2].plot([cx_l-dx, cx_l+dx], [cy_l-dy, cy_l+dy], "-", color=c, lw=2.2, alpha=0.92)
        axes[2].plot(cx_l, cy_l, "o", color=c, ms=4, alpha=0.8)
        axes[2].text(cx_l+3, cy_l-4, f"{f['angle_deg']:.0f}°",
                     color=c, fontsize=6, alpha=0.9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Visualizacion guardada: {out_path}")


def export_csv(fibras, out_path):
    import csv
    fields = ["id","cx_px","cy_px","angle_deg","length_px","length_mm","width_px","conf"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, fib in enumerate(fibras):
            w.writerow({
                "id":        i,
                "cx_px":     f"{fib['cx']:.2f}",
                "cy_px":     f"{fib['cy']:.2f}",
                "angle_deg": f"{fib['angle_deg']:.2f}",
                "length_px": f"{fib['length_px']:.1f}",
                "length_mm": f"{fib['length_mm']:.2f}",
                "width_px":  f"{fib['width_px']:.2f}",
                "conf":      f"{fib['conf']:.3f}",
            })
    print(f"CSV guardado: {out_path}  ({len(fibras)} fibras)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",   required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out",   default="output")
    parser.add_argument("--conf",  type=float, default=CONF_THRESHOLD)
    parser.add_argument("--scale", type=int,   default=SCALE_FACTOR)
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    result, img_bgr, src_w, src_h, scale = infer_sahi(args.img, args.model, args.scale)
    fibras = extract_fibers(result, scale)

    stem = Path(args.img).stem
    visualize(img_bgr, fibras, out_dir / f"{stem}_detections.png")
    export_csv(fibras,         out_dir / f"{stem}_fibers.csv")

    if fibras:
        lengths = [f["length_mm"] for f in fibras]
        angles  = [f["angle_deg"] for f in fibras]
        print(f"\nResumen:")
        print(f"  Fibras:          {len(fibras)}")
        print(f"  Largo medio:     {np.mean(lengths):.1f} mm proyectados")
        print(f"  Rango angulos:   {np.min(angles):.0f} - {np.max(angles):.0f} grados")
        print(f"  Confianza media: {np.mean([f['conf'] for f in fibras]):.3f}")