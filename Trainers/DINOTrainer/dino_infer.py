"""
dino_seg_infer.py
=================
Inferencia del modelo DINOv2 entrenado sobre la secuencia completa de frames.

Genera para cada frame:
  - Máscara binaria PNG (0=fondo, 255=fibra)
  - JSON con detecciones de fibras (centroide, longitud, ángulo)
  - Video MP4 con dos paneles: original | segmentación superpuesta

Uso
---
  python dino_seg_infer.py

  Requiere dino_seg_best.pt generado por dino_seg_train.py
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import glob
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

FRAMES_DIR    = "FOTOS"
FILE_PATTERN  = "*.bmp"
FRAME_START   = None
FRAME_END     = 200

MODEL_PATH    = "dino_seg_best.pt"      # generado por dino_seg_train.py

# Salidas
OUT_DIR_MASKS = "output_masks"          # carpeta para PNGs de máscaras
OUT_VIDEO     = "dino_seg_result.mp4"   # video de visualización
OUT_JSON      = "dino_seg_detections.json"  # detecciones de fibras

VIDEO_FPS     = 30
VIDEO_SIZE    = (1400, 700)             # (ancho, alto)

# Umbral de segmentación (puede ajustarse después del entrenamiento)
THRESHOLD     = 0.5

# Post-procesamiento de la máscara → detección de fibras
MIN_FIBER_LEN_PX   = 25       # longitud mínima de segmento (px)
MIN_STRAIGHTNESS   = 0.65     # rectitud mínima [0-1]
MERGE_ANGLE_TOL    = 8        # tolerancia angular para fusión de segmentos (°)
MERGE_PERP_TOL     = 12       # distancia perpendicular máxima para fusión (px)
MERGE_GAP_TOL      = 25       # brecha axial máxima para fusión (px)

PX_PER_MM     = 7.8


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORTS DEL MODELO (desde dino_seg_train.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Importar la arquitectura del archivo de entrenamiento
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dino_seg_train import DINOSegmentation
except ImportError:
    raise ImportError(
        "No se pudo importar DINOSegmentation.\n"
        "Asegúrate de que dino_seg_train.py esté en la misma carpeta."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, device: torch.device) -> tuple:
    """Carga el modelo entrenado desde checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}\n"
            "Ejecutar primero: python dino_seg_train.py"
        )
    ckpt   = torch.load(model_path, map_location=device)
    config = ckpt["config"]

    model = DINOSegmentation(
        dino_model_name = config["dino_model"],
        image_size      = config["image_size"],
        train_mode      = "linear",   # no importa para inferencia
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[✓] Modelo cargado: IoU={ckpt.get('valid_iou',0):.4f}  "
          f"epoch={ckpt.get('epoch',0)}  "
          f"img_size={config['image_size']}")
    return model, config


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(gray: np.ndarray, image_size: int,
               device: torch.device) -> torch.Tensor:
    """
    Prepara un frame gris para inferencia:
      - Resize a image_size × image_size
      - Gris → RGB (replicar canal)
      - Normalizar ImageNet
      - (1,3,H,W) tensor
    """
    resized = cv2.resize(gray, (image_size, image_size))
    rgb     = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    rgb     = (rgb - MEAN) / STD
    t       = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return t.to(device)


@torch.no_grad()
def predict(model, gray: np.ndarray, image_size: int,
            threshold: float, device: torch.device) -> np.ndarray:
    """
    Retorna máscara binaria uint8 (0/255) en la resolución original del frame.
    """
    h_orig, w_orig = gray.shape
    t = preprocess(gray, image_size, device)

    logits = model(t)                             # (1,1,H,W) logits
    probs  = torch.sigmoid(logits)[0, 0]          # (H,W) float
    probs_np = probs.float().cpu().numpy()

    # Escalar de vuelta a resolución original
    mask_resized = cv2.resize(probs_np, (w_orig, h_orig),
                              interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask_resized >= threshold).astype(np.uint8) * 255
    return mask_bin, mask_resized


# ═══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESAMIENTO: MÁSCARA → FIBRAS
# ═══════════════════════════════════════════════════════════════════════════════

def mask_to_fibers(mask_bin: np.ndarray,
                   min_len: float, min_straight: float,
                   merge_angle: float, merge_perp: float,
                   merge_gap: float) -> list:
    """
    Convierte la máscara binaria a lista de segmentos de fibra.
    Pipeline: closing → skeletonize → corte en bifurcaciones
              → fitLine por componente → fusión de colineales
    """
    # Limpiar ruido pequeño
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kc)
    binary = cv2.morphologyEx(binary,   cv2.MORPH_OPEN,  ko)

    # Esqueletizar
    skel = skeletonize(binary > 127).astype(np.uint8) * 255

    # Cortar en bifurcaciones
    k3   = np.ones((3, 3), dtype=np.uint8)
    nc   = cv2.filter2D((skel > 0).astype(np.uint8), -1, k3)
    junc = ((skel > 0) & (nc >= 4)).astype(np.uint8) * 255
    kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skel_cut = skel.copy()
    skel_cut[cv2.dilate(junc, kdil) > 0] = 0

    # fitLine por componente
    nl, labels, stats, cents = cv2.connectedComponentsWithStats(
        skel_cut, connectivity=8)

    segments = []
    for i in range(1, nl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_len:
            continue
        pts = np.argwhere(labels == i)[:, ::-1].astype(np.float32)
        if len(pts) < 5:
            continue

        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx=float(vx[0]); vy=float(vy[0]); cx=float(cx[0]); cy=float(cy[0])
        tv = (pts[:, 0]-cx)*vx + (pts[:, 1]-cy)*vy
        x1=cx+tv.min()*vx; y1=cy+tv.min()*vy
        x2=cx+tv.max()*vx; y2=cy+tv.max()*vy
        L  = math.hypot(x2-x1, y2-y1)
        if L < min_len:
            continue

        angle = math.degrees(math.atan2(vy, vx)) % 180.0
        perp  = np.abs((pts[:,0]-cx)*vy - (pts[:,1]-cy)*vx)
        straight = 1.0 - min(1.0, float(perp.mean()) / max(1.0, L/2.0))
        if straight < min_straight:
            continue

        segments.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1+x2)/2, "cy": (y1+y2)/2,
            "length": L, "angle": angle, "straightness": straight,
            "skeleton_pixels": area,
        })

    # Fusionar colineales
    return _merge_collinear(segments, merge_angle, merge_perp, merge_gap)


def _merge_collinear(segs, angle_tol, perp_tol, gap_tol):
    if not segs:
        return []
    used = [False] * len(segs)
    merged = []
    for i, s1 in enumerate(segs):
        if used[i]:
            continue
        group = [s1]; used[i] = True
        a1  = s1["angle"]
        vx1 = math.cos(math.radians(a1))
        vy1 = math.sin(math.radians(a1))
        cx1, cy1, L1 = s1["cx"], s1["cy"], s1["length"]

        for j, s2 in enumerate(segs):
            if used[j]: continue
            da = abs(a1 - s2["angle"]); da = min(da, 180-da)
            if da > angle_tol: continue
            perp = abs((s2["cy"]-cy1)*vx1 - (s2["cx"]-cx1)*vy1)
            if perp > perp_tol: continue
            t2 = [(s2["x1"]-cx1)*vx1+(s2["y1"]-cy1)*vy1,
                  (s2["x2"]-cx1)*vx1+(s2["y2"]-cy1)*vy1]
            gap = max(0.0, float(max(min(t2),-L1/2) - min(max(t2),L1/2)))
            if gap > gap_tol: continue
            group.append(s2); used[j] = True

        if len(group) == 1:
            s = s1.copy(); s["n_merged"] = 1; merged.append(s); continue

        all_pts = np.array(
            [[s["x1"],s["y1"]] for s in group] +
            [[s["x2"],s["y2"]] for s in group], dtype=np.float32)
        vx,vy,cx,cy = cv2.fitLine(all_pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx=float(vx[0]);vy=float(vy[0]);cx=float(cx[0]);cy=float(cy[0])
        tv   = (all_pts[:,0]-cx)*vx + (all_pts[:,1]-cy)*vy
        x1f  = cx+tv.min()*vx; y1f = cy+tv.min()*vy
        x2f  = cx+tv.max()*vx; y2f = cy+tv.max()*vy
        Lf   = math.hypot(x2f-x1f, y2f-y1f)
        angf = math.degrees(math.atan2(vy,vx)) % 180.0
        merged.append({
            "x1": x1f, "y1": y1f, "x2": x2f, "y2": y2f,
            "cx": (x1f+x2f)/2, "cy": (y1f+y2f)/2,
            "length": Lf, "angle": angf,
            "straightness": float(np.mean([s["straightness"] for s in group])),
            "skeleton_pixels": sum(s["skeleton_pixels"] for s in group),
            "n_merged": len(group),
        })
    merged.sort(key=lambda f: -f["length"])
    for rank, fb in enumerate(merged, 1):
        fb["id"] = rank
        fb["length_mm"] = round(fb["length"] / PX_PER_MM, 2)
        fb["x1"] = round(fb["x1"],1); fb["y1"] = round(fb["y1"],1)
        fb["x2"] = round(fb["x2"],1); fb["y2"] = round(fb["y2"],1)
        fb["centroid_x"] = round(fb["cx"],1)
        fb["centroid_y"] = round(fb["cy"],1)
        fb["length_px"]  = round(fb["length"],1)
        fb["angle_deg"]  = round(fb["angle"],1)
        fb["straightness"] = round(fb["straightness"],3)
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDERIZADO
# ═══════════════════════════════════════════════════════════════════════════════

def render_frame(gray: np.ndarray, prob_map: np.ndarray,
                 fibers: list, frame_name: str,
                 frame_idx: int, out_size: tuple) -> np.ndarray:
    """Dos paneles: original | segmentación + detecciones."""
    W_out, H_out = out_size
    pw = W_out // 2

    orig_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    left     = cv2.resize(orig_bgr, (pw, H_out))

    # Panel derecho: imagen original + máscara de probabilidad + fibras
    right_bg = (left.astype(np.float32) * 0.45).astype(np.uint8)

    # Colorizar mapa de probabilidad
    import matplotlib.pyplot as plt
    cmap     = plt.get_cmap("inferno")
    prob_rgb = (cmap(prob_map)[:,:,:3]*255).astype(np.uint8)
    prob_bgr = cv2.cvtColor(cv2.resize(prob_rgb,(pw,H_out)), cv2.COLOR_RGB2BGR)
    alpha    = cv2.resize((prob_map*255).astype(np.uint8),(pw,H_out)).astype(np.float32)/255
    alpha3   = np.stack([alpha]*3, axis=-1)
    right    = (prob_bgr*alpha3*0.8 + right_bg*(1-alpha3*0.8)).astype(np.uint8)

    # Dibujar fibras detectadas
    scale_x = pw / gray.shape[1]
    scale_y = H_out / gray.shape[0]
    for fb in fibers:
        ang = fb["angle_deg"]
        hue = int(ang/180*179)
        col = cv2.cvtColor(np.uint8([[[hue,220,210]]]),cv2.COLOR_HSV2RGB)[0][0]
        col = (int(col[0]),int(col[1]),int(col[2]))
        x1 = int(fb["x1"]*scale_x); y1 = int(fb["y1"]*scale_y)
        x2 = int(fb["x2"]*scale_x); y2 = int(fb["y2"]*scale_y)
        lw = max(1, int(1.5 + fb["length_px"]/80))
        cv2.line(right, (x1,y1), (x2,y2), col, lw, cv2.LINE_AA)

    out = np.hstack([left, right])

    font = cv2.FONT_HERSHEY_SIMPLEX
    def put_text(img, text, x, y, s=0.58, t=1):
        cv2.putText(img,text,(x+1,y+1),font,s,(0,0,0),t+1,cv2.LINE_AA)
        cv2.putText(img,text,(x,y),font,s,(255,255,255),t,cv2.LINE_AA)

    put_text(out, "ORIGINAL",                          12,    28, 0.68)
    put_text(out, f"DINOv2 SEG  |  {len(fibers)} fibras", pw+12, 28, 0.62)
    put_text(out, f"Frame: {frame_name}  [{frame_idx+1}]", 12, H_out-12, 0.50)
    cv2.line(out, (pw,0), (pw,H_out), (60,60,60), 2)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[GPU] Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[CPU]")
    return dev


def main():
    device = get_device()

    # ── Modelo ───────────────────────────────────────────────────────────────
    model, config = load_model(MODEL_PATH, device)
    image_size    = config["image_size"]
    threshold     = config.get("threshold", THRESHOLD)

    # ── Frames ───────────────────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(FRAMES_DIR, FILE_PATTERN)))
    paths = paths[FRAME_START or 0 : FRAME_END]
    print(f"[→] {len(paths)} frames a procesar")

    os.makedirs(OUT_DIR_MASKS, exist_ok=True)
    out_w, out_h = VIDEO_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, VIDEO_FPS, (out_w, out_h))

    all_results = []
    times       = []

    for idx, path in enumerate(tqdm(paths, desc="Inferencia", ncols=70)):
        t0 = time.time()

        gray       = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        fname      = Path(path).stem.split("__")[-1]

        # Segmentación
        mask_bin, prob_map = predict(model, gray, image_size, threshold, device)

        # Guardar máscara PNG
        mask_path = os.path.join(OUT_DIR_MASKS, f"{Path(path).stem}_mask.png")
        cv2.imwrite(mask_path, mask_bin)

        # Detección de fibras desde la máscara
        fibers = mask_to_fibers(
            mask_bin, MIN_FIBER_LEN_PX, MIN_STRAIGHTNESS,
            MERGE_ANGLE_TOL, MERGE_PERP_TOL, MERGE_GAP_TOL,
        )

        all_results.append({
            "frame_index": idx,
            "frame_name":  fname,
            "frame_path":  path,
            "mask_path":   mask_path,
            "n_fibers":    len(fibers),
            "fibers":      fibers,
        })

        # Renderizar y escribir video
        frame_out = render_frame(gray, prob_map, fibers, fname, idx, VIDEO_SIZE)
        writer.write(frame_out)

        times.append(time.time() - t0)
        if (idx+1) % 200 == 0:
            avg = np.mean(times[-200:])
            eta = avg * (len(paths)-idx-1)
            tqdm.write(f"  [{idx+1}/{len(paths)}] {avg:.2f}s/frame  ETA {eta/60:.1f}min")

    writer.release()

    # Guardar JSON
    with open(OUT_JSON, "w") as f:
        json.dump({"n_frames": len(all_results), "frames": all_results},
                  f, indent=2)

    elapsed = sum(times)
    print()
    print(f"[✓] Completado en {elapsed/60:.1f} min  ({np.mean(times):.2f}s/frame)")
    print(f"    Máscaras → {OUT_DIR_MASKS}/")
    print(f"    Video   → {OUT_VIDEO}")
    print(f"    JSON    → {OUT_JSON}")
    print(f"\n    open {OUT_VIDEO}")


if __name__ == "__main__":
    main()