"""
dino_seg_train.py
=================
Entrena DINOv2 + cabeza de segmentación para detectar fibras.

Arquitectura
------------
  DINOv2 ViT-S/14  →  features (73×73, 384 dims)
  Upsample ×14     →  (1024×1024, 384 dims)
  Conv 1×1         →  (1024×1024, 1)   ← solo esta parte se entrena (o + decoder)
  Sigmoid          →  máscara P(fibra) ∈ [0,1]

Ventaja sobre YOLO/SAM para fibras densas
------------------------------------------
  - Opera a nivel de patch (14px), no de instancia
  - No necesita separar fibras individuales
  - Robusto cuando las fibras se solapan
  - La máscara semántica es continua: cada píxel = probabilidad de fibra

Formato de datos esperado (exportar desde Roboflow)
----------------------------------------------------
  Exportar como: COCO JSON (Segmentation)
  Estructura esperada:
    dataset/
      train/
        _annotations.coco.json
        imagen1.bmp (o .jpg/.png/.tiff)
        imagen2.bmp
        ...
      valid/
        _annotations.coco.json
        ...

Instalación
-----------
  pip install torch torchvision
  pip install pycocotools albumentations tqdm opencv-python pillow

Uso
---
  1. Exportar dataset de Roboflow como COCO JSON
  2. Editar CONFIGURACIÓN al inicio del archivo
  3. Ejecutar:
       python dino_seg_train.py

  El script guarda checkpoints cada epoch y el mejor modelo en:
    dino_seg_best.pt   ← usar este para inferencia
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import json
import time
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
except ImportError:
    raise ImportError("Instalar: pip install pycocotools")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[!] albumentations no disponible — sin augmentation. "
          "Instalar: pip install albumentations")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  —  editar estas variables
# ═══════════════════════════════════════════════════════════════════════════════

# Rutas del dataset (exportado de Roboflow como COCO JSON)
TRAIN_DIR  = "dataset/train"        # carpeta con _annotations.coco.json + imágenes
VALID_DIR  = "dataset/valid"        # carpeta con _annotations.coco.json + imágenes

# Modelo DINOv2
DINO_MODEL = "dinov2_vits14"        # vits14=rápido(384d), vitb14=mejor(768d)

# Estrategia de entrenamiento
# "linear"   : solo entrenar Conv 1×1 + decoder  (rápido, ~5 min)
# "finetune" : + descongelar últimas 4 capas DINO (mejor, ~30 min)
TRAIN_MODE = "finetune"

# Hiperparámetros
IMAGE_SIZE  = 1036          # múltiplo de 14 más cercano a 1024 que cabe en RAM
                             # 518 = 37×14 | 1022 = 73×14 (requiere más RAM)
BATCH_SIZE  = 4             # reducir a 2 si da OOM en Mac
EPOCHS      = 50
LR          = 1e-4          # learning rate inicial
LR_DINO     = 1e-5          # lr para las capas de DINO (más bajo = más conservador)
WEIGHT_DECAY= 1e-4

# Umbral de binarización para la máscara de salida (inferencia)
THRESHOLD   = 0.5

# Loss weights
BCE_WEIGHT  = 0.5           # peso del Binary Cross-Entropy
DICE_WEIGHT = 0.5           # peso del Dice Loss (mejor para objetos finos)

# Checkpoint
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL     = "dino_seg_best.pt"
SAVE_EVERY     = 5          # guardar checkpoint cada N epochs

# Reproducibilidad
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class FiberDataset(Dataset):
    """
    Dataset de fibras desde anotaciones COCO JSON.
    Convierte los polígonos de segmentación a máscaras binarias.
    """

    # Normalización ImageNet (igual que DINOv2)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, data_dir: str, image_size: int, augment: bool = False):
        self.data_dir   = Path(data_dir)
        self.image_size = image_size
        self.augment    = augment

        # Cargar anotaciones COCO
        ann_path = self.data_dir / "_annotations.coco.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"No se encontró {ann_path}\n"
                "Exportar desde Roboflow como 'COCO JSON'"
            )
        self.coco    = COCO(str(ann_path))
        self.img_ids = list(self.coco.imgs.keys())

        # Augmentations con albumentations
        if augment and HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.GaussNoise(variance=(5, 30), p=0.4), # Corregido a 'variance'
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.ElasticTransform(
                    alpha=30, sigma=5, p=0.3),
                # PADDING en lugar de Resize: rellena con negro (0) hasta 1036
                A.PadIfNeeded(min_height=1036, min_width=1036, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                # PADDING también en validación para que el tamaño coincida
                A.PadIfNeeded(min_height=1036, min_width=1036, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]) if HAS_ALBUMENTATIONS else None

    def __len__(self):
        return len(self.img_ids)

    def _load_mask(self, img_id: int, h: int, w: int) -> np.ndarray:
        """Convierte anotaciones COCO a máscara binaria uint8 (0/255)."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)
        mask    = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if "segmentation" not in ann:
                continue
            seg = ann["segmentation"]
            if isinstance(seg, list):
                # Polígonos → máscara rasterizada
                for poly in seg:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            elif isinstance(seg, dict):
                # RLE → máscara
                rle  = coco_mask.frPyObjects(seg, h, w)
                m    = coco_mask.decode(rle)
                mask = np.maximum(mask, m.astype(np.uint8))
        return mask   # valores 0 o 1

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]

        # Cargar imagen
        img_path = self.data_dir / img_info["file_name"]
        pil      = Image.open(img_path)
        if pil.mode != "L":
            pil = pil.convert("L")
        # Gris → RGB (DINOv2 espera 3 canales)
        img_np = np.stack([np.array(pil)] * 3, axis=-1)   # (H,W,3)

        h, w = img_np.shape[:2]
        mask = self._load_mask(img_id, h, w)               # (H,W) uint8

        if self.transform is not None:
            out   = self.transform(image=img_np, mask=mask)
            image = out["image"].float()                    # (3,H,W)
            mask  = out["mask"].float().unsqueeze(0)        # (1,H,W)
        else:
            # Fallback sin albumentations
            img_r = cv2.resize(img_np, (self.image_size, self.image_size))
            mask_r= cv2.resize(mask,   (self.image_size, self.image_size),
                               interpolation=cv2.INTER_NEAREST)
            mean  = np.array(self.MEAN, dtype=np.float32)
            std   = np.array(self.STD,  dtype=np.float32)
            img_r = (img_r.astype(np.float32)/255.0 - mean) / std
            image = torch.from_numpy(img_r.transpose(2,0,1)).float()
            mask  = torch.from_numpy(mask_r).float().unsqueeze(0)

        return image, mask


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELO
# ═══════════════════════════════════════════════════════════════════════════════

class DINOSegmentation(nn.Module):
    """
    DINOv2 + cabeza de segmentación semántica.

    Arquitectura:
      1. DINOv2 extrae features patch-level: (B, n_patches, D)
         donde D=384 (vits14) o D=768 (vitb14)
      2. Reshape a grilla espacial: (B, D, n_h, n_w)
      3. Decoder ligero: upsampling progresivo + convoluciones
      4. Interpolación final a tamaño original
      5. Sigmoid → P(fibra) por píxel

    El decoder usa skip connections desde las últimas capas de DINO
    para preservar detalle fino (importante para fibras de 1-2px).
    """

    def __init__(self, dino_model_name: str, image_size: int,
                 train_mode: str = "finetune"):
        super().__init__()

        # ── Cargar DINOv2 ────────────────────────────────────────────────────
        print(f"[→] Cargando {dino_model_name}…")
        self.dino = torch.hub.load(
            "facebookresearch/dinov2", dino_model_name,
            pretrained=True, verbose=False, trust_repo=True,
        )
        self.patch_size = self.dino.patch_size   # 14
        self.feat_dim   = self.dino.embed_dim    # 384 (vits14) o 768 (vitb14)
        self.image_size = image_size
        self.n_patches  = image_size // self.patch_size  # lado de la grilla

        # ── Estrategia de congelamiento ──────────────────────────────────────
        # Congelar todo primero
        for p in self.dino.parameters():
            p.requires_grad = False

        if train_mode == "finetune":
            # Descongelar las últimas 4 capas transformer + norm final
            n_blocks = len(self.dino.blocks)
            for blk in self.dino.blocks[n_blocks-4:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.dino.norm.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.dino.parameters()
                           if p.requires_grad)
            print(f"    DINO fine-tune: {trainable/1e6:.1f}M parámetros "
                  f"entrenables (últimas 4 capas)")
        else:
            print("    DINO frozen: solo se entrena el decoder")

        # ── Decoder de segmentación ──────────────────────────────────────────
        # Upsampling progresivo: patch_grid → imagen completa
        # n_patches = image_size/14 (ej: 37 para 518px)
        # Necesitamos ×14 de upsampling total
        # Lo hacemos en 3 pasos: ×2, ×2, ×3.5 aprox → se interpola al final
        D = self.feat_dim
        self.decoder = nn.Sequential(
            # Paso 1: reducir dimensionalidad y añadir no-linealidad
            nn.Conv2d(D, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Paso 2: upsample ×2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Paso 3: upsample ×2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Paso 4: upsample ×2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Clasificación final
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Inicialización del decoder
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        n_dec = sum(p.numel() for p in self.decoder.parameters())
        print(f"    Decoder: {n_dec/1e6:.2f}M parámetros")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)  normalizado ImageNet
        returns: (B, 1, H, W)  logits (sin sigmoid — aplica en loss/inferencia)
        """
        B, C, H, W = x.shape

        # ── Features DINOv2 ──────────────────────────────────────────────────
        # get_intermediate_layers retorna lista de tensores (B, n_tokens, D)
        # n_tokens = n_patches + 1 (cls token)
        feats = self.dino.get_intermediate_layers(x, n=1)[0]  # (B, n_tokens, D)
        #feats = feats[:, 1:, :]   # quitar cls token → (B, n_patches, D)

        # Reshape a grilla espacial (B, D, n_h, n_w)
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        feats = feats.reshape(B, n_h, n_w, self.feat_dim)
        feats = feats.permute(0, 3, 1, 2)   # (B, D, n_h, n_w)

        # ── Decoder ──────────────────────────────────────────────────────────
        out = self.decoder(feats)   # (B, 1, ~H/2, ~W/2) aprox

        # Interpolación final exacta al tamaño original
        out = F.interpolate(out, size=(H, W), mode="bilinear",
                            align_corners=False)
        return out   # logits (B, 1, H, W)

    def predict_mask(self, x: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
        """Inferencia: retorna máscara binaria (B, 1, H, W) uint8."""
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            return (probs >= threshold).to(torch.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Dice Loss para segmentación de objetos finos.
    Mejor que BCE puro cuando las fibras son pequeñas respecto al fondo.
    pred   : logits (B, 1, H, W)
    target : máscara float (B, 1, H, W)  valores 0 o 1
    """
    pred   = torch.sigmoid(pred)
    pred   = pred.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  bce_w: float = 0.5, dice_w: float = 0.5) -> torch.Tensor:
    """BCE + Dice — estándar para segmentación de estructuras finas."""
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_w * bce + dice_w * dice


# ═══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_logits: torch.Tensor,
                    target: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """Calcula IoU, Dice, Precision y Recall sobre el batch."""
    pred_bin = (torch.sigmoid(pred_logits) >= threshold).float()
    tgt      = target.float()

    tp = (pred_bin * tgt).sum().item()
    fp = (pred_bin * (1 - tgt)).sum().item()
    fn = ((1 - pred_bin) * tgt).sum().item()

    iou       = tp / (tp + fp + fn + 1e-6)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)

    return {"iou": iou, "dice": dice,
            "precision": precision, "recall": recall}


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[GPU] Apple MPS (Mac Silicon)")
    else:
        dev = torch.device("cpu")
        print("[CPU] Sin GPU — puede ser lento")
    return dev


def train_one_epoch(model, loader, optimizer, device, bce_w, dice_w):
    model.train()
    total_loss = 0.0
    metrics    = {"iou": 0, "dice": 0, "precision": 0, "recall": 0}

    for imgs, masks in tqdm(loader, desc="  train", ncols=65, leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        pred  = model(imgs)
        loss  = combined_loss(pred, masks, bce_w, dice_w)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        m = compute_metrics(pred.detach(), masks)
        for k in metrics:
            metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate(model, loader, device, bce_w, dice_w):
    model.eval()
    total_loss = 0.0
    metrics    = {"iou": 0, "dice": 0, "precision": 0, "recall": 0}

    for imgs, masks in tqdm(loader, desc="  valid", ncols=65, leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)
        pred  = model(imgs)
        loss  = combined_loss(pred, masks, bce_w, dice_w)
        total_loss += loss.item()
        m = compute_metrics(pred, masks)
        for k in metrics:
            metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    print(f"[→] Cargando datasets…")
    train_ds = FiberDataset(TRAIN_DIR, IMAGE_SIZE, augment=True)
    valid_ds = FiberDataset(VALID_DIR, IMAGE_SIZE, augment=False)
    print(f"    Train: {len(train_ds)} imágenes")
    print(f"    Valid: {len(valid_ds)} imágenes")

    # workers=0 en Mac para evitar problemas con MPS + multiprocessing
    n_workers = 0 if str(device) == "mps" else 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=False)

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = DINOSegmentation(DINO_MODEL, IMAGE_SIZE, TRAIN_MODE).to(device)

    # ── Optimizer con learning rates distintos ───────────────────────────────
    # DINO layers: lr más bajo para no destruir los features preentrenados
    # Decoder: lr normal
    dino_params    = [p for p in model.dino.parameters()    if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": dino_params,    "lr": LR_DINO},
        {"params": decoder_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    # Scheduler coseno: reduce lr gradualmente hasta 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[→] Parámetros entrenables: {total_params/1e6:.2f}M")
    print(f"[→] Iniciando entrenamiento: {EPOCHS} epochs, "
          f"batch={BATCH_SIZE}, img={IMAGE_SIZE}px")
    print(f"    Checkpoints en: {CHECKPOINT_DIR}/")
    print()

    # ── Loop de entrenamiento ────────────────────────────────────────────────
    best_iou     = 0.0
    history      = []
    t_start      = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, device, BCE_WEIGHT, DICE_WEIGHT)
        valid_loss, valid_m = validate(
            model, valid_loader, device, BCE_WEIGHT, DICE_WEIGHT)
        scheduler.step()

        elapsed = time.time() - t_ep
        lr_now  = optimizer.param_groups[1]["lr"]

        print(
            f"Ep {epoch:3d}/{EPOCHS}  "
            f"loss {train_loss:.4f}/{valid_loss:.4f}  "
            f"IoU {train_m['iou']:.3f}/{valid_m['iou']:.3f}  "
            f"Dice {valid_m['dice']:.3f}  "
            f"lr {lr_now:.1e}  "
            f"{elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "valid_loss": round(valid_loss, 5),
            "train_iou":  round(train_m["iou"], 4),
            "valid_iou":  round(valid_m["iou"], 4),
            "valid_dice": round(valid_m["dice"], 4),
            "valid_precision": round(valid_m["precision"], 4),
            "valid_recall":    round(valid_m["recall"], 4),
        })

        # Guardar mejor modelo
        if valid_m["iou"] > best_iou:
            best_iou = valid_m["iou"]
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "valid_iou":  best_iou,
                "config": {
                    "dino_model":  DINO_MODEL,
                    "image_size":  IMAGE_SIZE,
                    "train_mode":  TRAIN_MODE,
                    "threshold":   THRESHOLD,
                }
            }, BEST_MODEL)
            print(f"    ↑ Mejor modelo guardado (IoU={best_iou:.4f})")

        # Checkpoint periódico
        if epoch % SAVE_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt_path)

    # ── Resumen final ────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print()
    print(f"[✓] Entrenamiento completado en {total_time/60:.1f} min")
    print(f"    Mejor IoU (validación): {best_iou:.4f}")
    print(f"    Modelo guardado en:     {BEST_MODEL}")

    # Guardar historial
    import json
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"    Historial guardado en:  training_history.json")
    print()
    print("    Siguiente paso:")
    print("      python dino_seg_infer.py")


if __name__ == "__main__":
    main()