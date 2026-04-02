"""
dino_raft_fiber.py  —  v2
==========================
Pipeline combinado DINOv2 + RAFT para PTV de fibras.
Versión corregida: resuelve el problema de no-superposición entre
la máscara DINO y el flujo RAFT.

Problemas de la versión anterior y sus soluciones
--------------------------------------------------
  Problema 1: RAFT operaba sobre residuo×3
    → el ruido del fondo se amplificaba → flujo falso en zonas sin fibra
    Solución: RAFT opera sobre el FRAME ORIGINAL siempre.
              El residuo temporal se usa solo como MÁSCARA, no como input.

  Problema 2: máscara DINO y flujo RAFT no se superponían
    → DINO ve frame_t, RAFT ve el movimiento entre frame_t y frame_t+1.
      Si la fibra se movió 3-5px entre frames, la máscara de t no cubre
      la posición en t+1.
    Solución: dilatar la máscara DINO antes del AND. La dilatación
              (kernel 15px) absorbe desplazamientos de hasta ~7px entre frames.

  Problema 3: AND estricto descartaba demasiado
    → DINO×RAFT necesitan estar perfectamente alineados pixel a pixel.
    Solución: fusión suave con pesos configurables.
              El resultado es un mapa de confianza, no un AND binario.

Estrategia correcta (3 señales independientes)
-----------------------------------------------
  Señal A — DINO:   máscara P(fibra) en frame_t, dilatada 15px
  Señal B — RAFT:   magnitud del flujo entre frame_t y frame_t+1
                    calculada sobre el FRAME ORIGINAL (sin residuo boost)
  Señal C — Residuo temporal: |frame_t − mediana_bg| > umbral
                    indica si hay señal de fibra en este frame concreto

  Fusión: confianza = A × B_normalizada × C_suave
          → alto donde hay fibra (DINO) Y movimiento real (RAFT)
            Y señal de objeto en este frame (residuo)

Salidas
-------
  dino_raft_result.mp4       — 4 paneles: original | DINO | RAFT | fusión
  dino_raft_detections.json  — id, centroide (px+mm), velocidad (mm/s),
                               ángulo fibra, ángulo movimiento por frame

Uso
---
  python dino_raft_fiber.py

  Prueba rápida (100 frames):
    FRAME_END = 100
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os, sys, glob, json, math, time
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

# ── DINO ─────────────────────────────────────────────────────────────────────
DINO_MODEL_PT  = "dino_seg_best.pt"
DINO_THRESHOLD = 0.4           # umbral de binarización de la máscara DINO
                                # bajar si DINO detecta pocas fibras

# Dilatación de la máscara DINO antes del AND con RAFT.
# Compensa el desplazamiento de la fibra entre frame_t y frame_t+1.
# Regla: DINO_DILATION_PX ≈ velocidad_max_px_frame × 2
DINO_DILATION_PX = 15          # en px; aumentar si las fibras son muy rápidas

# ── RAFT ─────────────────────────────────────────────────────────────────────
RAFT_MODEL    = "small"        # "small" (rápido) | "large" (mejor calidad)
RAFT_ITERS    = 12             # iteraciones de refinamiento RAFT

# RAFT siempre opera sobre el frame original (no el residuo amplificado).
# El residuo temporal se usa solo como máscara post-procesamiento.
FLOW_MIN_PX   = 0.4            # magnitud mínima para considerar movimiento real
FLOW_MAX_PX   = 40.0           # magnitud máxima esperada (por encima = error)

# ── Residuo temporal (fondo) ──────────────────────────────────────────────────
N_BG_FRAMES   = 50
BG_FRAME_STEP = 20
# Umbral de residuo: píxeles con |frame - bg| > este valor tienen señal de fibra
RESIDUO_THRESH = 12            # en niveles de gris [0-255]

# ── Fusión de las 3 señales ───────────────────────────────────────────────────
# Pesos relativos de cada señal en el mapa de confianza final.
# Si una señal falla, se puede reducir su peso para que las otras dominen.
W_DINO        = 0.40           # peso de la máscara DINO
W_RAFT        = 0.40           # peso de la magnitud RAFT
W_RESIDUO     = 0.20           # peso del residuo temporal

# ── Extracción de fibras ──────────────────────────────────────────────────────
MIN_FIBER_PX  = 25
MIN_STRAIGHT  = 0.55
MERGE_ANGLE   = 8
MERGE_PERP    = 12
MERGE_GAP     = 25

# ── Física ────────────────────────────────────────────────────────────────────
PX_PER_MM     = 7.8
FPS_REAL      = 10.0

# ── Salidas ───────────────────────────────────────────────────────────────────
OUTPUT_VIDEO  = "dino_raft_result.mp4"
OUTPUT_JSON   = "dino_raft_detections.json"
VIDEO_FPS     = 10
VIDEO_SIZE    = (2048, 512)    # 4 paneles de 512×512


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[GPU] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[CPU]")
    return d


def load_gray(path):
    pil = Image.open(path)
    if pil.mode != "L":
        pil = pil.convert("L")
    return np.array(pil, dtype=np.uint8)


def compute_background(paths, n_frames, step):
    indices = list(range(0, len(paths), step))[:n_frames]
    if len(indices) < 2:
        indices = list(range(min(n_frames, len(paths))))
    print(f"[→] Calculando fondo con {len(indices)} frames…")
    stack = np.stack(
        [load_gray(paths[i]).astype(np.float32)
         for i in tqdm(indices, desc="   bg", ncols=55)],
        axis=0,
    )
    bg = np.median(stack, axis=0).astype(np.float32)
    print(f"    OK: {bg.shape}  [{bg.min():.0f}, {bg.max():.0f}]")
    return bg


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO DINO
# ═══════════════════════════════════════════════════════════════════════════════

DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_dino(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró '{model_path}'\n"
            "  Ejecutar primero: python dino_seg_train.py"
        )
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dino_seg_train import DINOSegmentation

    ckpt   = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    model  = DINOSegmentation(
        config["dino_model"], config["image_size"], train_mode="linear"
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"    DINOv2 OK — IoU={ckpt.get('valid_iou',0):.4f}  "
          f"img_size={config['image_size']}")
    return model, config


@torch.no_grad()
def dino_predict(model, gray, image_size, threshold, device):
    """
    Segmentación de fibras con DINOv2.
    Input:  frame gris (H, W) uint8
    Output: prob_map float32 (H, W) en [0,1]
            mask_bin uint8  (H, W) 0/255
    """
    h, w    = gray.shape
    resized = cv2.resize(gray, (image_size, image_size))
    rgb     = np.stack([resized]*3, axis=-1).astype(np.float32)/255.0
    rgb     = (rgb - DINO_MEAN) / DINO_STD
    t       = torch.from_numpy(
        rgb.transpose(2,0,1)).unsqueeze(0).float().to(device)

    logits    = model(t)
    probs     = torch.sigmoid(logits)[0,0].float().cpu().numpy()
    prob_full = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_bin  = (prob_full >= threshold).astype(np.uint8) * 255
    return prob_full, mask_bin


# ═══════════════════════════════════════════════════════════════════════════════
#  MÓDULO RAFT
# ═══════════════════════════════════════════════════════════════════════════════

def load_raft(model_size, device):
    from torchvision.models.optical_flow import (
        raft_small, raft_large,
        Raft_Small_Weights, Raft_Large_Weights,
    )
    print(f"    Cargando RAFT-{model_size}…")
    if model_size == "small":
        model = raft_small(weights=Raft_Small_Weights.C_T_V2)
    else:
        model = raft_large(weights=Raft_Large_Weights.C_T_V2)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())/1e6
    print(f"    RAFT OK — {n:.1f}M parámetros")
    return model


def preprocess_raft(gray, device):
    """
    Prepara un frame ORIGINAL (uint8) para RAFT.
    NO usa residuo ni amplificación — RAFT ve la imagen tal cual.
    """
    rgb = np.stack([gray]*3, axis=0)
    t   = torch.from_numpy(rgb).float() / 255.0
    t   = (t - 0.5) / 0.5          # normalizar a [-1, 1]
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def raft_predict(model, frame1, frame2, device, n_iters):
    """
    Flujo óptico entre frame1 y frame2.
    Ambos son frames ORIGINALES — no se aplica ninguna transformación previa.
    Output: float32 (H, W, 2) — (u=dx, v=dy) en píxeles/frame
    """
    t1 = preprocess_raft(frame1, device)
    t2 = preprocess_raft(frame2, device)
    flows = model(t1, t2, num_flow_updates=n_iters)
    return flows[-1][0].permute(1,2,0).float().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  FUSIÓN CORREGIDA: DINO + RAFT + RESIDUO
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_signals(dino_prob, dino_mask, raft_flow, frame_gray, bg,
                  dilation_px, flow_min, flow_max,
                  residuo_thresh, w_dino, w_raft, w_res):
    """
    Combina las 3 señales de forma robusta.

    Señal A — DINO (dilatado):
      La máscara se dilata DINO_DILATION_PX píxeles para cubrir el
      desplazamiento de la fibra entre frame_t y frame_t+1.
      Sin dilatación, una fibra que se mueve 3px queda fuera de la máscara.

    Señal B — RAFT magnitude:
      Calculado sobre el frame ORIGINAL → sin ruido amplificado.
      Se normaliza al rango [0,1] usando FLOW_MAX_PX como referencia.
      Píxeles con mag < flow_min o > flow_max se descartan.

    Señal C — Residuo temporal:
      |frame_t - mediana_bg| > residuo_thresh
      Indica si hay señal de fibra en este frame concreto.
      Suavizado gaussiano para tolerar bordes imprecisos del fondo.

    Fusión: mapa_confianza = w_A×A + w_B×B + w_C×C
    """
    H, W = frame_gray.shape

    # ── Señal A: DINO dilatado ───────────────────────────────────────────────
    if dilation_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px*2+1, dilation_px*2+1))
        dino_dilated = cv2.dilate(dino_mask, k)
    else:
        dino_dilated = dino_mask.copy()
    sig_a = (dino_dilated / 255.0).astype(np.float32)

    # ── Señal B: RAFT magnitude normalizada ──────────────────────────────────
    mag = np.sqrt(raft_flow[...,0]**2 + raft_flow[...,1]**2)
    # Máscara de movimiento válido
    valid_flow = (mag >= flow_min) & (mag <= flow_max)
    sig_b = np.clip(mag / flow_max, 0, 1) * valid_flow.astype(np.float32)

    # ── Señal C: residuo temporal suavizado ──────────────────────────────────
    residuo    = np.abs(frame_gray.astype(np.float32) - bg)
    residuo_s  = gaussian_filter(residuo, sigma=3.0)
    sig_c      = np.clip(residuo_s / (residuo_thresh * 3), 0, 1).astype(np.float32)

    # ── Mapa de confianza combinado ──────────────────────────────────────────
    conf = w_dino * sig_a + w_raft * sig_b + w_res * sig_c
    # Normalizar
    c_max = np.percentile(conf[conf > 0], 95) if (conf > 0).any() else 1.0
    if c_max > 0:
        conf = np.clip(conf / c_max, 0, 1)

    # ── Máscara binaria final ────────────────────────────────────────────────
    # Umbral adaptativo: p80 del mapa de confianza
    thr = max(0.15, float(np.percentile(conf, 80)))
    fiber_mask = (conf >= thr).astype(np.uint8) * 255

    # Limpiar ruido puntual
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fiber_mask = cv2.morphologyEx(fiber_mask, cv2.MORPH_OPEN,  ko)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fiber_mask = cv2.morphologyEx(fiber_mask, cv2.MORPH_CLOSE, kc)

    # Flujo enmascarado: solo dentro de fibras detectadas
    flow_masked = raft_flow * (fiber_mask[..., np.newaxis] / 255.0)

    return conf, fiber_mask, flow_masked, sig_a, sig_b, sig_c


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE FIBRAS CON VELOCIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def extract_fibers(fiber_mask, flow_masked, min_len, min_straight,
                    merge_angle, merge_perp, merge_gap, px_per_mm, fps):
    """
    Desde la máscara fusionada → segmentos de fibra con velocidad asignada.
    """
    skel = skeletonize(fiber_mask > 127).astype(np.uint8) * 255
    k3   = np.ones((3,3), dtype=np.uint8)
    nc   = cv2.filter2D((skel>0).astype(np.uint8), -1, k3)
    junc = ((skel>0)&(nc>=4)).astype(np.uint8)*255
    kdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    skel_cut = skel.copy()
    skel_cut[cv2.dilate(junc,kdil)>0] = 0

    nl, labels, stats, cents = cv2.connectedComponentsWithStats(
        skel_cut, connectivity=8)

    candidates = []
    for i in range(1, nl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_len:
            continue
        pts_yx = np.argwhere(labels==i)
        pts_xy = pts_yx[:,::-1].astype(np.float32)
        if len(pts_xy) < 5:
            continue

        vx,vy,cx,cy = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
        vx=float(vx[0]); vy=float(vy[0]); cx=float(cx[0]); cy=float(cy[0])
        tv = (pts_xy[:,0]-cx)*vx + (pts_xy[:,1]-cy)*vy
        x1=cx+tv.min()*vx; y1=cy+tv.min()*vy
        x2=cx+tv.max()*vx; y2=cy+tv.max()*vy
        length = math.hypot(x2-x1, y2-y1)
        if length < min_len:
            continue

        angle_fiber = math.degrees(math.atan2(vy,vx)) % 180.0
        perp = np.abs((pts_xy[:,0]-cx)*vy - (pts_xy[:,1]-cy)*vx)
        straight = 1.0 - min(1.0, float(perp.mean())/max(1.0,length/2.0))
        if straight < min_straight:
            continue

        # Velocidad media del flujo RAFT en los píxeles de esta componente
        mask_i  = (labels==i)
        u_vals  = flow_masked[mask_i, 0]
        v_vals  = flow_masked[mask_i, 1]
        u_mean  = float(u_vals.mean()) if len(u_vals) > 0 else 0.0
        v_mean  = float(v_vals.mean()) if len(v_vals) > 0 else 0.0
        speed   = math.hypot(u_mean, v_mean)

        # Omitir fibras estáticas (sin flujo)
        if speed < FLOW_MIN_PX:
            continue

        candidates.append({
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
            "cx":(x1+x2)/2,"cy":(y1+y2)/2,
            "centroid_x":float(cents[i][0]),"centroid_y":float(cents[i][1]),
            "length":length,"angle_fiber":angle_fiber,"straightness":straight,
            "u_px":u_mean,"v_px":v_mean,"speed_px":speed,
            "speed_mm_s":speed*fps/px_per_mm,
            "angle_motion":math.degrees(math.atan2(v_mean,u_mean))%360,
            "skeleton_pixels":area,
        })

    fibers = _merge(candidates, merge_angle, merge_perp, merge_gap)
    fibers.sort(key=lambda f: -f["length"])
    for rank, fb in enumerate(fibers, 1):
        fb["id"]              = rank
        fb["x1"]              = round(fb["x1"],1)
        fb["y1"]              = round(fb["y1"],1)
        fb["x2"]              = round(fb["x2"],1)
        fb["y2"]              = round(fb["y2"],1)
        fb["centroid_x"]      = round(fb["centroid_x"],1)
        fb["centroid_y"]      = round(fb["centroid_y"],1)
        fb["centroid_x_mm"]   = round(fb["centroid_x"]/px_per_mm, 3)
        fb["centroid_y_mm"]   = round(fb["centroid_y"]/px_per_mm, 3)
        fb["length_px"]       = round(fb["length"],1)
        fb["length_mm"]       = round(fb["length"]/px_per_mm, 2)
        fb["angle_fiber_deg"] = round(fb["angle_fiber"],1)
        fb["angle_motion_deg"]= round(fb["angle_motion"],1)
        fb["speed_px_frame"]  = round(fb["speed_px"],3)
        fb["speed_mm_s"]      = round(fb["speed_mm_s"],2)
        fb["u_px_frame"]      = round(fb["u_px"],3)
        fb["v_px_frame"]      = round(fb["v_px"],3)
        fb["straightness"]    = round(fb["straightness"],3)
        for k in ["cx","cy","length","angle_fiber","u_px","v_px","speed_px","angle_motion"]:
            fb.pop(k, None)
    return fibers


def _merge(segs, angle_tol, perp_tol, gap_tol):
    if not segs: return []
    used=[False]*len(segs); merged=[]
    for i,s1 in enumerate(segs):
        if used[i]: continue
        group=[s1]; used[i]=True
        a1=s1["angle_fiber"]; vx1=math.cos(math.radians(a1))
        vy1=math.sin(math.radians(a1)); cx1=s1["cx"]; cy1=s1["cy"]; L1=s1["length"]
        for j,s2 in enumerate(segs):
            if used[j]: continue
            da=abs(a1-s2["angle_fiber"]); da=min(da,180-da)
            if da>angle_tol: continue
            if abs((s2["cy"]-cy1)*vx1-(s2["cx"]-cx1)*vy1)>perp_tol: continue
            t2=[(s2["x1"]-cx1)*vx1+(s2["y1"]-cy1)*vy1,
                (s2["x2"]-cx1)*vx1+(s2["y2"]-cy1)*vy1]
            if max(0.0,float(max(min(t2),-L1/2)-min(max(t2),L1/2)))>gap_tol: continue
            group.append(s2); used[j]=True
        if len(group)==1:
            s=s1.copy(); s["n_merged"]=1; merged.append(s); continue
        pts=np.array([[s["x1"],s["y1"]] for s in group]+
                     [[s["x2"],s["y2"]] for s in group],dtype=np.float32)
        vx,vy,cx,cy=cv2.fitLine(pts,cv2.DIST_L2,0,0.01,0.01)
        vx=float(vx[0]);vy=float(vy[0]);cx=float(cx[0]);cy=float(cy[0])
        tv=(pts[:,0]-cx)*vx+(pts[:,1]-cy)*vy
        x1f=cx+tv.min()*vx; y1f=cy+tv.min()*vy
        x2f=cx+tv.max()*vx; y2f=cy+tv.max()*vy
        u_m=float(np.mean([s["u_px"] for s in group]))
        v_m=float(np.mean([s["v_px"] for s in group]))
        merged.append({
            "x1":x1f,"y1":y1f,"x2":x2f,"y2":y2f,
            "cx":(x1f+x2f)/2,"cy":(y1f+y2f)/2,
            "centroid_x":float(np.mean([s["centroid_x"] for s in group])),
            "centroid_y":float(np.mean([s["centroid_y"] for s in group])),
            "length":math.hypot(x2f-x1f,y2f-y1f),
            "angle_fiber":math.degrees(math.atan2(vy,vx))%180,
            "straightness":float(np.mean([s["straightness"] for s in group])),
            "u_px":u_m,"v_px":v_m,
            "speed_px":math.hypot(u_m,v_m),
            "speed_mm_s":math.hypot(u_m,v_m)*FPS_REAL/PX_PER_MM,
            "angle_motion":math.degrees(math.atan2(v_m,u_m))%360,
            "skeleton_pixels":sum(s["skeleton_pixels"] for s in group),
            "n_merged":len(group),
        })
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def flow_to_hsv(flow, max_disp, gamma=0.4, adaptive_pct=95):
    """
    Visualización HSV del campo de flujo con contraste adaptativo.

    Normalización por percentil adaptativo (no por máximo fijo):
      - Calcula el percentil `adaptive_pct` de la magnitud en este frame
      - Normaliza por ese valor → el rango útil ocupa toda la escala
      - Aplica corrección gamma (< 1) para realzar velocidades bajas
        que de otro modo quedarían casi negras

    Sin gamma y con max fijo (versión anterior), zonas de 3 px/frame
    sobre una escala de 40 px/frame = 7% de brillo → casi invisible.
    Con p95 + gamma=0.4, el mismo flujo llega al 80% de brillo.

    Hue   = dirección del movimiento (rojo=derecha, cian=izquierda, etc.)
    Value = velocidad (brillo adaptativo con gamma)
    """
    u, v = flow[..., 0], flow[..., 1]
    mag  = np.sqrt(u**2 + v**2)
    ang  = np.arctan2(v, u)

    # Percentil adaptativo del flujo en este frame
    mag_active = mag[mag > 0.1]
    if len(mag_active) > 100:
        scale = float(np.percentile(mag_active, adaptive_pct))
    else:
        scale = max_disp
    scale = max(scale, 0.5)   # evitar división por cero

    # Normalizar y aplicar gamma
    mag_norm = np.clip(mag / scale, 0, 1)
    mag_gamma = np.power(mag_norm, gamma)

    h   = ((np.degrees(ang) + 180) / 360 * 179).astype(np.uint8)
    s   = np.full_like(h, 255)
    val = (mag_gamma * 255).astype(np.uint8)

    return cv2.cvtColor(np.stack([h, s, val], axis=-1), cv2.COLOR_HSV2BGR)


def render_frame(gray, dino_prob, raft_flow, conf_map, fibers,
                  fname, idx, stats, out_size):
    """
    4 paneles:
      1. Frame original
      2. Máscara DINO (probabilidad)
      3. Flujo RAFT (HSV encoding)
      4. Mapa de confianza fusionado + fibras detectadas con vectores
    """
    import matplotlib.pyplot as plt
    W_total, H_out = out_size
    pw = W_total // 4
    sx = pw / gray.shape[1]
    sy = H_out / gray.shape[0]

    def bg_overlay(img, alpha_map, cmap_name):
        bg  = cv2.resize(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),(pw,H_out))
        col = (plt.get_cmap(cmap_name)(alpha_map)[:,:,:3]*255).astype(np.uint8)
        col = cv2.cvtColor(cv2.resize(col,(pw,H_out)), cv2.COLOR_RGB2BGR)
        a   = cv2.resize((alpha_map*255).astype(np.uint8),(pw,H_out)).astype(np.float32)/255
        a3  = np.stack([a]*3,axis=-1)
        return (col*a3*0.85 + (bg*0.3).astype(np.uint8)*(1-a3*0.85)).astype(np.uint8)

    p1 = cv2.resize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),(pw,H_out))
    p2 = bg_overlay(gray, dino_prob, "plasma")
    p3 = cv2.resize(flow_to_hsv(raft_flow, FLOW_MAX_PX), (pw, H_out))
    p4 = bg_overlay(gray, conf_map, "inferno")

    # Dibujar fibras y vectores en panel 4
    for fb in fibers:
        ang = fb.get("angle_fiber_deg",0)
        hue = int(ang/180*179)
        col = cv2.cvtColor(np.uint8([[[hue,220,210]]]),cv2.COLOR_HSV2RGB)[0][0]
        col = (int(col[0]),int(col[1]),int(col[2]))
        x1i=int(fb["x1"]*sx); y1i=int(fb["y1"]*sy)
        x2i=int(fb["x2"]*sx); y2i=int(fb["y2"]*sy)
        cv2.line(p4,(x1i,y1i),(x2i,y2i),col,2,cv2.LINE_AA)
        # Vector velocidad (amarillo)
        cxi=int(fb["centroid_x"]*sx); cyi=int(fb["centroid_y"]*sy)
        ex =int(np.clip(cxi+fb.get("u_px_frame",0)*sx*5,0,pw-1))
        ey =int(np.clip(cyi+fb.get("v_px_frame",0)*sy*5,0,H_out-1))
        cv2.arrowedLine(p4,(cxi,cyi),(ex,ey),(0,220,220),1,cv2.LINE_AA,tipLength=0.35)

    out = np.hstack([p1,p2,p3,p4])

    font = cv2.FONT_HERSHEY_SIMPLEX
    def pt(img,txt,x,y,s=0.48,t=1):
        cv2.putText(img,txt,(x+1,y+1),font,s,(0,0,0),t+1,cv2.LINE_AA)
        cv2.putText(img,txt,(x,y),font,s,(255,255,255),t,cv2.LINE_AA)

    pt(out,"ORIGINAL",       12,    22,0.55)
    pt(out,"DINO mask",      pw+12, 22,0.52)
    pt(out,"RAFT flow",      pw*2+12,22,0.52)
    pt(out,f"FUSION  {len(fibers)} fibras",pw*3+12,22,0.52)
    pt(out,f"Frame {fname} [{idx+1}]",12,H_out-10,0.42)
    spd=stats.get("speed_mean_mm_s",0)
    pt(out,f"vel.media {spd:.1f} mm/s",pw*3+12,H_out-10,0.42)
    for i in range(1,4):
        cv2.line(out,(pw*i,0),(pw*i,H_out),(50,50,50),2)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    device  = get_device()

    paths = sorted(glob.glob(os.path.join(FRAMES_DIR, FILE_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"No se encontraron '{FILE_PATTERN}' en '{FRAMES_DIR}'")
    paths = paths[FRAME_START or 0 : FRAME_END]
    print(f"[→] {len(paths)} frames  ({len(paths)-1} pares)")

    bg_cache = OUTPUT_VIDEO.replace(".mp4","_bg.npy")
    if os.path.exists(bg_cache):
        print(f"[→] Fondo cacheado: {bg_cache}")
        bg = np.load(bg_cache).astype(np.float32)
    else:
        bg = compute_background(paths, N_BG_FRAMES, BG_FRAME_STEP)
        np.save(bg_cache, bg)

    print("[→] Cargando modelos…")
    dino_model, dino_config = load_dino(DINO_MODEL_PT, device)
    raft_model              = load_raft(RAFT_MODEL, device)
    dino_img_size           = dino_config["image_size"]

    W_out, H_out = VIDEO_SIZE
    writer = cv2.VideoWriter(OUTPUT_VIDEO,
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              VIDEO_FPS, (W_out, H_out))
    n_pairs = len(paths)-1
    print(f"[→] Procesando {n_pairs} pares…")

    all_results = []
    times       = []

    for idx in tqdm(range(n_pairs), desc="DINO+RAFT", ncols=70):
        t0 = time.time()

        frame1 = load_gray(paths[idx])
        frame2 = load_gray(paths[idx+1])
        fname  = Path(paths[idx]).stem.split("__")[-1]

        # 1. DINO: segmentación sobre frame original
        dino_prob, dino_mask = dino_predict(
            dino_model, frame1, dino_img_size, DINO_THRESHOLD, device)

        # 2. RAFT: flujo sobre FRAMES ORIGINALES (corrección clave)
        flow = raft_predict(raft_model, frame1, frame2, device, RAFT_ITERS)

        # 3. Fusión con dilatación DINO y residuo temporal
        conf, fiber_mask, flow_masked, _, _, _ = fuse_signals(
            dino_prob, dino_mask, flow, frame1, bg,
            DINO_DILATION_PX, FLOW_MIN_PX, FLOW_MAX_PX,
            RESIDUO_THRESH, W_DINO, W_RAFT, W_RESIDUO,
        )

        # 4. Extracción de fibras con velocidad
        fibers = extract_fibers(
            fiber_mask, flow_masked,
            MIN_FIBER_PX, MIN_STRAIGHT,
            MERGE_ANGLE, MERGE_PERP, MERGE_GAP,
            PX_PER_MM, FPS_REAL,
        )

        speeds = [fb["speed_mm_s"] for fb in fibers]
        frame_stats = {
            "frame_index"     : idx,
            "frame_name"      : fname,
            "n_fibers"        : len(fibers),
            "speed_mean_mm_s" : round(float(np.mean(speeds)),2) if speeds else 0,
            "speed_max_mm_s"  : round(float(max(speeds)),2)     if speeds else 0,
        }
        all_results.append({**frame_stats, "fibers": fibers})

        frame_out = render_frame(
            frame1, dino_prob, flow, conf, fibers,
            fname, idx, frame_stats, VIDEO_SIZE,
        )
        writer.write(frame_out)

        times.append(time.time()-t0)
        if (idx+1)%100==0:
            avg=np.mean(times[-100:])
            tqdm.write(f"  [{idx+1}/{n_pairs}]  {avg:.2f}s/f  "
                       f"ETA {avg*(n_pairs-idx-1)/60:.1f}min  "
                       f"fibras={len(fibers)}  "
                       f"vel={frame_stats['speed_mean_mm_s']:.1f}mm/s")

    writer.release()

    all_spd=[s["speed_mean_mm_s"] for s in all_results if s["n_fibers"]>0]
    summary={
        "pipeline":"DINOv2 + RAFT (v2 — fusión corregida)",
        "n_frames":len(all_results),
        "px_per_mm":PX_PER_MM,"fps_real":FPS_REAL,
        "config":{
            "dino_threshold":DINO_THRESHOLD,
            "dino_dilation_px":DINO_DILATION_PX,
            "raft_model":RAFT_MODEL,
            "flow_min_px":FLOW_MIN_PX,
            "flow_max_px":FLOW_MAX_PX,
            "weights":{"dino":W_DINO,"raft":W_RAFT,"residuo":W_RESIDUO},
        },
        "global_stats":{
            "mean_fibers_per_frame":round(np.mean([s["n_fibers"] for s in all_results]),1),
            "speed_global_mean_mm_s":round(float(np.mean(all_spd)),2) if all_spd else 0,
            "speed_global_max_mm_s":round(float(max(all_spd)),2) if all_spd else 0,
        },
        "frames":all_results,
    }
    with open(OUTPUT_JSON,"w") as f:
        json.dump(summary, f, indent=2)

    elapsed=time.time()-t_start
    g=summary["global_stats"]
    print()
    print(f"[✓] Completado en {elapsed/60:.1f} min  ({np.mean(times):.2f}s/frame)")
    print(f"    Video  → {OUTPUT_VIDEO}")
    print(f"    JSON   → {OUTPUT_JSON}")
    print(f"    Fibras/frame (media) : {g['mean_fibers_per_frame']}")
    print(f"    Velocidad media      : {g['speed_global_mean_mm_s']} mm/s")
    print(f"    Velocidad máxima     : {g['speed_global_max_mm_s']} mm/s")
    print()
    print("─── Si aún hay desalineación, ajustar ────────────────────────────")
    print(f"    DINO_DILATION_PX = {DINO_DILATION_PX}  "
          f"→ sube si las fibras se mueven rápido")
    print(f"    DINO_THRESHOLD   = {DINO_THRESHOLD}  "
          f"→ baja si DINO detecta pocas fibras")
    print(f"    W_DINO={W_DINO} W_RAFT={W_RAFT} W_RESIDUO={W_RESIDUO}  "
          f"→ sube W_RAFT si DINO falla")
    print(f"    open {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()