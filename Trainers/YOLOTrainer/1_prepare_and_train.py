"""
PASO 1: Preparación del dataset + Entrenamiento YOLOv8-seg
==========================================================
Requisitos:
    pip install ultralytics sahi opencv-python-headless tqdm

Dataset esperado (exportado desde Roboflow como "YOLOv8 Segmentation"):
    dataset/
        train/images/   *.jpg / *.png / *.tiff
        train/labels/   *.txt  (formato YOLO seg: class x1 y1 x2 y2 ...)
        valid/images/
        valid/labels/
        data.yaml

Parámetros físicos de la fibra:
    PX_PER_MM  = 7.8
    largo      = 13 mm  → 101 px a escala original
    ancho      = 0.2 mm → 1.56 px a escala original
"""

import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURACIÓN  — ajusta estas rutas
# ─────────────────────────────────────────────
DATASET_DIR   = Path("dataset")          # raíz del dataset Roboflow
OUTPUT_DIR    = Path("dataset_tiled")    # dataset con tiles generados
RUNS_DIR      = Path("runs/fiber_seg")   # carpeta de entrenamiento

# Parámetros de tiling
SCALE_FACTOR  = 3          # upscale antes de tilear (fibra pasa de 1.5px a ~4.7px)
TILE_SIZE     = 512        # tamaño del tile en espacio upscaleado
OVERLAP       = 0.5        # solapamiento entre tiles (50%)
TILE_STRIDE   = int(TILE_SIZE * (1 - OVERLAP))  # 256 px

# Parámetros de entrenamiento
MODEL_SIZE    = "n"        # "n"=nano (rápido), "s"=small (mejor), "m"=medium
EPOCHS        = 30
IMGSZ         = 512        # tamaño de entrada al modelo
BATCH         = 16         # reducir a 8 si hay OOM
DEVICE        = 0          # GPU 0; usar "cpu" si no hay GPU

# ─────────────────────────────────────────────
# FUNCIONES DE TILING
# ─────────────────────────────────────────────

def scale_yolo_polygon(poly_norm, src_w, src_h, tile_x, tile_y,
                        tile_size, scale):
    """
    Convierte un polígono YOLO normalizado (espacio original) a
    coordenadas normalizadas en el espacio del tile upscaleado.
    Retorna None si el polígono no intersecta el tile.
    """
    # Desnormalizar al espacio original
    pts = np.array(poly_norm, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= src_w
    pts[:, 1] *= src_h

    # Escalar al espacio upscaleado
    pts *= scale

    # Recortar al tile
    pts[:, 0] -= tile_x
    pts[:, 1] -= tile_y

    # Clip al bounding del tile
    pts[:, 0] = np.clip(pts[:, 0], 0, tile_size)
    pts[:, 1] = np.clip(pts[:, 1], 0, tile_size)

    # Verificar que hay área útil
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    if x_range < 2 or y_range < 2:
        return None

    # Normalizar al tile
    pts[:, 0] /= tile_size
    pts[:, 1] /= tile_size
    pts = np.clip(pts, 0, 1)

    return pts.flatten().tolist()


def tile_image_and_labels(img_path, label_path, out_img_dir,
                           out_lbl_dir, scale=SCALE_FACTOR,
                           tile_size=TILE_SIZE, stride=TILE_STRIDE):
    """
    Upscalea una imagen, la divide en tiles solapados y ajusta las
    anotaciones YOLO-seg a cada tile.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Intentar con PIL para TIFF
        from PIL import Image as PILImage
        img = np.array(PILImage.open(img_path))

    src_h, src_w = img.shape[:2]

    # Upscale
    new_w, new_h = src_w * scale, src_h * scale
    img_up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Convertir a 3 canales (YOLOv8 espera RGB)
    img_up = cv2.cvtColor(img_up, cv2.COLOR_GRAY2BGR)

    # Leer etiquetas
    annotations = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cls = int(parts[0])
                poly = [float(x) for x in parts[1:]]
                annotations.append((cls, poly))

    stem = img_path.stem
    tile_count = 0

    for y0 in range(0, new_h - tile_size + 1, stride):
        for x0 in range(0, new_w - tile_size + 1, stride):
            x1, y1 = x0 + tile_size, y0 + tile_size
            tile_img = img_up[y0:y1, x0:x1]

            tile_labels = []
            for cls, poly in annotations:
                new_poly = scale_yolo_polygon(
                    poly, src_w, src_h,
                    x0, y0, tile_size, scale
                )
                if new_poly is not None:
                    tile_labels.append((cls, new_poly))

            # Guardar solo tiles con al menos una anotación
            if not tile_labels:
                continue

            tile_name = f"{stem}_t{tile_count:04d}"
            cv2.imwrite(str(out_img_dir / f"{tile_name}.jpg"), tile_img)

            with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
                for cls, poly in tile_labels:
                    coords = " ".join(f"{v:.6f}" for v in poly)
                    f.write(f"{cls} {coords}\n")

            tile_count += 1

    return tile_count


def build_tiled_dataset(dataset_dir, output_dir):
    """Genera el dataset completo con tiles para train y valid."""
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)

    total_tiles = 0
    for split in ["train", "valid"]:
        img_in  = dataset_dir / split / "images"
        lbl_in  = dataset_dir / split / "labels"
        img_out = output_dir  / split / "images"
        lbl_out = output_dir  / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        img_files = list(img_in.glob("*"))
        img_files = [f for f in img_files
                     if f.suffix.lower() in {".jpg",".jpeg",".png",".tiff",".tif"}]

        print(f"\n[{split}] Procesando {len(img_files)} imágenes...")
        for img_path in tqdm(img_files):
            lbl_path = lbl_in / (img_path.stem + ".txt")
            n = tile_image_and_labels(img_path, lbl_path, img_out, lbl_out)
            total_tiles += n

        print(f"  → {total_tiles} tiles generados acumulado")

    # Copiar y actualizar data.yaml
    src_yaml = dataset_dir / "data.yaml"
    dst_yaml = output_dir  / "data.yaml"
    if src_yaml.exists():
        import yaml
        with open(src_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["path"]  = str(output_dir.resolve())
        cfg["train"] = "train/images"
        cfg["val"]   = "valid/images"
        with open(dst_yaml, "w") as f:
            yaml.dump(cfg, f)
        print(f"\ndata.yaml actualizado → {dst_yaml}")
    else:
        # Crear data.yaml mínimo (una sola clase: fiber)
        with open(dst_yaml, "w") as f:
            f.write(f"path: {output_dir.resolve()}\n")
            f.write("train: train/images\n")
            f.write("val:   valid/images\n")
            f.write("nc: 1\n")
            f.write("names: ['fiber']\n")
        print(f"\ndata.yaml creado → {dst_yaml}")

    print(f"\nTotal tiles generados: {total_tiles}")
    return dst_yaml


# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

def train(data_yaml, model_size=MODEL_SIZE, epochs=EPOCHS,
          imgsz=IMGSZ, batch=BATCH, device=DEVICE):
    """Fine-tune YOLOv8-seg desde pesos COCO."""
    model = YOLO(f"yolov8{model_size}-seg.pt")

    results = model.train(
        data      = str(data_yaml),
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        device    = device,
        project   = str(RUNS_DIR),
        name      = "train",
        # Augmentaciones útiles para fibras
        flipud    = 0.5,
        fliplr    = 0.5,
        degrees   = 180.0,   # fibras pueden estar en cualquier orientación
        scale     = 0.3,
        hsv_v     = 0.3,     # variación de brillo (grayscale convertido a BGR)
        hsv_s     = 0.0,     # sin saturación (grayscale)
        mosaic    = 0.5,
        # Parámetros de optimización
        optimizer = "AdamW",
        lr0       = 1e-3,
        lrf       = 0.01,
        warmup_epochs = 5,
        patience  = 30,      # early stopping
        save      = True,
        plots     = True,
    )
    print(f"\nEntrenamiento completo. Mejor modelo: {RUNS_DIR}/train/weights/best.pt")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE: Fibras → YOLOv8-seg + SAHI")
    print("=" * 60)

    # Paso 1: Generar dataset con tiles
    print("\n[1/2] Generando dataset con tiles solapados...")
    if OUTPUT_DIR.exists():
        print(f"  Ya existe {OUTPUT_DIR}, saltando tiling.")
        data_yaml = OUTPUT_DIR / "data.yaml"
    else:
        data_yaml = build_tiled_dataset(DATASET_DIR, OUTPUT_DIR)

    # Paso 2: Entrenar
    print("\n[2/2] Entrenando YOLOv8-seg...")
    train(data_yaml)
