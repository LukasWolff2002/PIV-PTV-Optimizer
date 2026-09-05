#!/usr/bin/env python3
"""
pintar_fibras_video.py
======================
Genera un video .mp4 por CADA toma presente en PTV/Tomas/, pintando las fibras
detectadas (YOLO-seg + SAHI, la MISMA logica del pipeline PTV) de un color fijo.
NO trackea, NO estima profundidad: solo detecta por frame y pinta la silueta.

Que hace, punto por punto:
  1. Recorre todas las carpetas de PTV/Tomas/ con nombre PTV valido
     (m{mezcla}-toma-{toma}-cam-{cam}-n-{n}-car-{car}-ptv).
  2. Para cada toma, lee del Google Sheet cuantas fotos saltar al inicio
     ('Fotos Saltar', con el ajuste x'Razon FPS' para la cam 4) — igual que
     pipeline_global.py.
  3. Toma el resto de parametros de variables_ptv.py:
       - fps de captura, resolucion y mascara fija por camara (CAM_PROFILES_PTV)
       - filtros de preprocesamiento por camara (CAM_PREPROCESS_PARAMS_PTV)
       - confianza y parametros SAHI de deteccion (CONF_TRACK, SAHI_*)
       - modelo de segmentacion de fibras (YOLO_TRACK_MODEL = best.pt)
       - regiones temporales -> la DURACION del video es el end_time de la
         ultima region de la toma (car02 -> 20 s, car05 -> 40 s, etc.).
  4. Muestrea a VELOCIDAD REAL (paso constante = fps_captura / FPS_SALIDA), NO
     usa el muestreo adaptativo del tracker: 1 s grabado dura 1 s en el video.
  5. Detecta las fibras en cada frame muestreado, descarta las que caen en la
     zona enmascarada (mascara fija, convencion blanco=ignorar, filtro por
     centroide como el tracker) y pinta cada fibra con COLOR_FIBRA_HEX, ya sea
     como LINEA estimada por PCA (MODO_PINTURA="linea") o como la silueta YOLO.
  6. Escribe el .mp4 en la carpeta de salida.

IMPORTANTE — entorno:
  Este script corre la deteccion, asi que necesita el entorno con ultralytics +
  torch (el 'yolox4' del pipeline) MAS imageio/imageio-ffmpeg para escribir video:
      conda run -n yolox4 python Videos/pintar_fibras_video.py
  Si falta el escritor de video:  pip install imageio imageio-ffmpeg
"""

from __future__ import annotations

import csv
import math
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# CONFIGURACION  (edita solo esta seccion)
# ----------------------------------------------------------------------------

COLOR_FIBRA_HEX = "#5e6ca2"   # color con el que se pintan las fibras
ALPHA_PINTURA   = 1.0         # 1.0 = opaco; <1.0 = semitransparente (ambos modos)

# Que se pinta por fibra:
#   "linea"   -> la fibra estimada como una LINEA: centroide + angulo + largo por
#                PCA sobre la sombra, calculada con la ruta COMPLETA del PTV
#                (SAHI + NMS + PCA via det.detect). Es la geometria que usa el PTV.
#   "silueta" -> toda la sombra segmentada por YOLO (relleno del contorno crudo).
MODO_PINTURA   = "linea"
GROSOR_LINEA   = 2            # grosor de la linea en px (modo "linea")
DIBUJAR_CENTRO = False        # dibujar un punto en el centroide de cada fibra

FPS_SALIDA = 30.0             # fps del video final (velocidad real preservada)

# Fondo del video:
#   "raw"          -> foto original normalizada (video limpio, estilo tiff_a_video)
#   "preprocesado" -> lo que ve el detector (CLAHE/capping); util para verificar
FONDO = "raw"

# Normalizacion del fondo a 8 bits (solo aplica si FONDO = "raw"):
#   "global"  -> min/max sobre los frames seleccionados (recomendado)
#   "frame"   -> min/max por frame (puede parpadear)
#   "rango"   -> divide por el maximo del tipo de dato (65535, 4095...)
#   "ninguna" -> asume que ya viene en 8 bits
NORMALIZACION = "global"
BITS_SENSOR   = 16
PERCENTILES   = (0.5, 99.5)

# Skip inicial:
#   None -> se lee del Google Sheet por toma (recomendado)
#   int  -> fuerza ese skip para TODAS las tomas (ignora el Sheet)
SALTAR_INICIO = None
SKIP_FALLBACK = 0             # si el Sheet no responde o no encuentra la fila

# Duracion cuando la toma no tiene regiones temporales definidas:
#   None -> hasta el final de la secuencia disponible
DURACION_FALLBACK_SEG = None

# Procesar todas las tomas presentes (True) o respetar la columna 'Usar' del
# Sheet y saltar las marcadas con Usar=no (False).
PROCESAR_TODAS = True

# Dispositivo de inferencia:  None = auto (cuda si hay, si no cpu) | "cpu" | 0
DISPOSITIVO = None

# Salida
SALIDA_DIR = Path(__file__).resolve().parent / "salida"
CODEC       = "libx264"
CALIDAD_CRF = 18              # 0 sin perdida, 18 muy buena, 23 normal, 28 baja

# URL del Google Sheet publicado como CSV (misma que pipeline_global.py).
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQj9_M-1BdxtMDhG9u7j0qpO5WKN5tGZMe6lXdm-DZi-CIhwKY907aNLCLAXHppkda2AI5g2qX_p24S/pub?output=csv"

# ----------------------------------------------------------------------------
# IMPORTS DEL PROYECTO  (variables PTV + detector real)
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNCODE_DIR  = PROJECT_ROOT / "RunCode"
for _p in (str(PROJECT_ROOT), str(RUNCODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import variables_ptv as V                                   # noqa: E402
from PTV.Codes.PTVCode.detector import FiberYOLODetector    # noqa: E402
from PTV.Codes.PTVCode.image_utils import (                 # noqa: E402
    read_image_any,
    preprocess_frame_for_ptv,
    natural_key,
)

import tifffile                                             # noqa: E402
import imageio.v2 as imageio                                # noqa: E402

EXTENSIONES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# ----------------------------------------------------------------------------
# COLOR
# ----------------------------------------------------------------------------

def hex_a_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


COLOR_FIBRA_RGB = hex_a_rgb(COLOR_FIBRA_HEX)   # (94, 108, 162) para #5e6ca2


# ----------------------------------------------------------------------------
# GOOGLE SHEET  (espejo minimo de pipeline_global.py, sin pandas/requests)
# ----------------------------------------------------------------------------

_SHEET_ROWS: list[dict] | None = None
_SHEET_INTENTADO = False


def cargar_sheet() -> list[dict] | None:
    """Descarga el CSV una vez y lo cachea como lista de dicts (o None si falla)."""
    global _SHEET_ROWS, _SHEET_INTENTADO
    if _SHEET_INTENTADO:
        return _SHEET_ROWS
    _SHEET_INTENTADO = True
    try:
        print("[INFO] Descargando Google Sheet...", flush=True)
        with urllib.request.urlopen(GOOGLE_SHEET_CSV_URL, timeout=15) as resp:
            texto = resp.read().decode("utf-8", errors="replace")
        _SHEET_ROWS = list(csv.DictReader(StringIO(texto)))
        print(f"[INFO] Google Sheet cargado: {len(_SHEET_ROWS)} filas", flush=True)
    except Exception as e:
        print(f"[WARN] No se pudo cargar Google Sheet: {e} -> skip={SKIP_FALLBACK}", flush=True)
        _SHEET_ROWS = None
    return _SHEET_ROWS


def _a_float(x, default=0.0) -> float:
    if x is None:
        return default
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return default


def _parse_fps_ratio(x) -> float:
    """'660:220' -> 3.0 ; '3' -> 3.0 ; vacio -> 1.0  (como pipeline_global)."""
    if x is None or str(x).strip() == "":
        return 1.0
    s = str(x).strip()
    if ":" in s:
        a, _, b = s.partition(":")
        na, nb = _a_float(a, 0.0), _a_float(b, 0.0)
        if na != 0:
            return nb / na
    return _a_float(s, 1.0)


def fila_para_toma(mezcla: str, toma: int, carbopol: str, metodo: str = "ptv") -> dict | None:
    """Busca la fila del Sheet para (mezcla, toma, tipo, metodo)."""
    rows = cargar_sheet()
    if not rows:
        return None
    tipo_obj = float(carbopol) / 10.0      # "02" -> 0.2 ; "05" -> 0.5
    mez_obj  = mezcla.upper()
    for r in rows:
        try:
            if str(r.get("Mezcla", "")).strip().upper() != mez_obj:
                continue
            if int(_a_float(r.get("Toma"), -1)) != toma:
                continue
            if abs(_a_float(r.get("Tipo"), -1.0) - tipo_obj) > 1e-6:
                continue
            if "Metodo" in r and str(r.get("Metodo", "")).strip().lower() != metodo.lower():
                continue
            return r
        except Exception:
            continue
    return None


def skip_para_toma(mezcla: str, toma: int, carbopol: str, cam: int, metodo: str = "ptv") -> int:
    """Fotos a saltar al inicio segun el Sheet (x Razon FPS si cam==4)."""
    if SALTAR_INICIO is not None:
        return int(SALTAR_INICIO)
    row = fila_para_toma(mezcla, toma, carbopol, metodo)
    if row is None:
        print(f"[WARN] Sin fila en Sheet para {mezcla}-Toma{toma}-Car{carbopol}-{metodo.upper()} "
              f"-> skip={SKIP_FALLBACK}", flush=True)
        return SKIP_FALLBACK
    skip_base = int(_a_float(row.get("Fotos Saltar"), 0))
    if cam == 4:
        ratio = _parse_fps_ratio(row.get("Razon FPS", 1.0))
        skip_adj = int(skip_base * ratio)
        print(f"[INFO] {mezcla}-Toma{toma}-Car{carbopol} Cam4: skip={skip_base} x ratio={ratio:.1f} = {skip_adj}", flush=True)
        return skip_adj
    print(f"[INFO] {mezcla}-Toma{toma}-Car{carbopol} Cam{cam}: skip={skip_base}", flush=True)
    return skip_base


def usar_toma(mezcla: str, toma: int, carbopol: str, metodo: str = "ptv") -> bool:
    """Columna 'Usar' del Sheet (si no hay Sheet/fila -> True)."""
    if PROCESAR_TODAS:
        return True
    row = fila_para_toma(mezcla, toma, carbopol, metodo)
    if row is None:
        return True
    val = str(row.get("Usar", "si")).strip().lower()
    return val in ("si", "sí", "s", "yes", "y", "")


# ----------------------------------------------------------------------------
# NOMBRE DE TOMA
# ----------------------------------------------------------------------------

import re                                                   # noqa: E402

NAME_RE = re.compile(
    r"^m(?P<mezcla>\d+)-toma-(?P<toma>\d+)-cam-(?P<cam>\d+)-n-(?P<n>\d+)-car-(?P<car>\d+)-(?P<metodo>[A-Za-z0-9_]+)$"
)


def parse_nombre(name: str) -> dict | None:
    m = NAME_RE.match(name)
    return m.groupdict() if m else None


def listar_tomas_ptv(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        info = parse_nombre(p.name)
        if info and info["metodo"].lower() == V.PTV_METODO.lower():
            out.append(p)
    out.sort(key=lambda p: natural_key(p.name))
    return out


def listar_imagenes(carpeta: Path) -> list[Path]:
    imgs = [p for p in carpeta.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES]
    imgs.sort(key=lambda p: natural_key(p.name))
    return imgs


# ----------------------------------------------------------------------------
# MASCARA FIJA  (convencion del tracker: blanco/alto = IGNORAR)
# ----------------------------------------------------------------------------

def _auto_threshold(m: np.ndarray) -> float:
    vmax = float(m.max())
    if vmax <= 1.0:
        return 0.5
    if vmax <= 255.0:
        return 127.0
    return 32767.0


def cargar_keep_mask(cam: int, H: int, W: int) -> np.ndarray | None:
    """
    Devuelve mascara booleana (H, W): True = zona VALIDA (analizar / pintar).
    Replica PTVCode/tracker._load_static_mask (blanco=ignorar, sin invertir).
    """
    path = PROJECT_ROOT / "FixMasks" / f"cam-{cam}.tiff"
    if not path.exists():
        print(f"[WARN] No existe mascara fija {path} -> se pinta todo.", flush=True)
        return None
    try:
        raw = tifffile.imread(str(path)).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        ignore = raw > _auto_threshold(raw)     # True = ignorar
        keep = ~ignore
        if keep.shape != (H, W):
            keep = cv2.resize(keep.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        pct = 100.0 * int((~keep).sum()) / keep.size
        print(f"[INFO] Mascara fija cam{cam}: {pct:.1f}% ignorado", flush=True)
        return keep
    except Exception as e:
        print(f"[WARN] No se pudo leer mascara fija {path}: {e} -> se pinta todo.", flush=True)
        return None


# ----------------------------------------------------------------------------
# SELECCION DE FRAMES A VELOCIDAD REAL  (misma idea que tiff_a_video.py)
# ----------------------------------------------------------------------------

def seleccionar_indices(n_disponibles: int, fps_captura: float, fps_salida: float,
                        saltar_inicio: int, duracion_seg: float | None):
    if saltar_inicio >= n_disponibles:
        raise ValueError(
            f"skip={saltar_inicio} pero solo hay {n_disponibles} imagenes.")

    n_restantes = n_disponibles - saltar_inicio
    dur_maxima  = n_restantes / fps_captura
    if duracion_seg is None:
        duracion = dur_maxima
    else:
        duracion = min(duracion_seg, dur_maxima)
        if duracion_seg > dur_maxima + 1e-9:
            print(f"  [aviso] Se pidieron {duracion_seg:g}s pero solo hay "
                  f"{dur_maxima:.2f}s -> se recorta.", flush=True)

    paso = fps_captura / fps_salida
    n_frames = int(round(duracion * fps_salida))
    indices = []
    for i in range(n_frames):
        idx = saltar_inicio + int(round(i * paso))
        if idx >= n_disponibles:
            break
        indices.append(idx)
    return indices, duracion, paso


def duracion_maxima_toma(cam: int, carbopol: str) -> float | None:
    """end_time de la ULTIMA region temporal de la toma (None = hasta el final)."""
    regiones = V.get_ptv_temporal_regions(cam, carbopol)
    if not regiones:
        return DURACION_FALLBACK_SEG
    return regiones[-1].end_time   # puede ser None -> secuencia completa


# ----------------------------------------------------------------------------
# NORMALIZACION DEL FONDO
# ----------------------------------------------------------------------------

def _a_gris(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        return cv2.cvtColor(raw[..., :3], cv2.COLOR_RGB2GRAY)
    return raw


def rango_global(archivos: list[Path], indices: list[int], max_muestras=40):
    muestra = indices[:: max(1, len(indices) // max_muestras)]
    los, his = [], []
    for i in muestra:
        g = _a_gris(np.asarray(read_image_any(archivos[i])))
        lo, hi = np.percentile(g, PERCENTILES)
        los.append(lo); his.append(hi)
    return float(np.min(los)), float(np.max(his))


def fondo_rgb_desde_raw(raw: np.ndarray, lo=None, hi=None) -> np.ndarray:
    g = _a_gris(raw)
    if NORMALIZACION == "ninguna":
        g8 = g.astype(np.uint8, copy=False)
    elif NORMALIZACION == "rango":
        maxv = float(2 ** BITS_SENSOR - 1)
        g8 = np.clip(g.astype(np.float32) * (255.0 / maxv), 0, 255).astype(np.uint8)
    else:  # "global" o "frame"
        if NORMALIZACION == "frame" or lo is None or hi is None:
            lo, hi = np.percentile(g, PERCENTILES)
        if hi <= lo:
            hi = lo + 1.0
        g8 = np.clip((g.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return np.stack([g8, g8, g8], axis=-1)


def a_dims_pares(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        frame = frame[: h - h % 2, : w - w % 2]
    return frame


# ----------------------------------------------------------------------------
# DETECCION DE FIBRAS
# ----------------------------------------------------------------------------
# Los dos modos usan el MISMO detector del pipeline: mismo upsampling (x scale)
# y los mismos parches solapados SAHI, con los parametros de variables_ptv.
#   - "linea"  : det.detect(...) -> Detection por fibra con centroide, angulo y
#                largo estimados por PCA (con NMS entre parches). Es exactamente
#                la geometria que calcula el PTV a partir de la sombra.
#   - "silueta": se rasteriza el contorno crudo de YOLO (la sombra completa).

def detectar_fibras_pca(det: FiberYOLODetector, rgb_u8: np.ndarray) -> list:
    """Ruta COMPLETA del pipeline (SAHI + NMS + PCA). Devuelve list[Detection]."""
    detections, _ = det.detect(
        image_rgb_u8 = rgb_u8,
        frame_idx    = 0,
        image_name   = "frame",
        next_det_id  = 1,
        dptv         = None,     # sin profundidad
        px_per_mm    = None,
    )
    return detections


def detectar_poligonos_fibra(det: FiberYOLODetector, rgb_u8: np.ndarray) -> list[np.ndarray]:
    """
    Poligonos crudos de YOLO (la sombra) en coordenadas de la imagen ORIGINAL.
    Mismo upscale + tiles SAHI que el pipeline. Usado por el modo "silueta".
    """
    img_bgr_up = det._upscale_to_bgr(rgb_u8)
    tiles, offsets = det._slice_views(img_bgr_up)
    if not tiles:
        return []

    resultados = det.model.predict(
        source=tiles, conf=det.conf, verbose=False,
        device=det._device_str, half=det._use_fp16,
    )

    s = float(det.scale_factor)
    polys: list[np.ndarray] = []
    for res, (ox, oy) in zip(resultados, offsets):
        if res.masks is None:
            continue
        boxes = res.boxes
        for i, seg in enumerate(res.masks.xy):
            score = float(boxes.conf[i].item()) if boxes is not None else 1.0
            if score < det.conf:
                continue
            if seg is None or len(seg) < 3:
                continue
            poly = np.asarray(seg, dtype=np.float32).copy()
            poly[:, 0] = (poly[:, 0] + ox) / s   # tile -> upscaled -> original
            poly[:, 1] = (poly[:, 1] + oy) / s
            polys.append(poly)
    return polys


# ----------------------------------------------------------------------------
# PINTURA
# ----------------------------------------------------------------------------

def _centroide_valido(cx: float, cy: float, keep_mask, H: int, W: int) -> bool:
    """True si el centroide cae en zona valida (misma logica que tracker._det_in_mask)."""
    if keep_mask is None:
        return True
    xi = min(max(int(round(cx)), 0), W - 1)
    yi = min(max(int(round(cy)), 0), H - 1)
    return bool(keep_mask[yi, xi])


def _capa_lineas(detections: list, keep_mask, H: int, W: int) -> tuple[np.ndarray, int]:
    """Dibuja cada fibra como una linea: centroide + angulo + largo (PCA)."""
    capa = np.zeros((H, W), dtype=np.uint8)
    n = 0
    for d in detections:
        if not _centroide_valido(d.cx, d.cy, keep_mask, H, W):
            continue
        half = d.length_px / 2.0
        ang  = math.radians(d.angle_deg)
        dx, dy = math.cos(ang) * half, math.sin(ang) * half
        p1 = (int(round(d.cx - dx)), int(round(d.cy - dy)))
        p2 = (int(round(d.cx + dx)), int(round(d.cy + dy)))
        cv2.line(capa, p1, p2, 255, GROSOR_LINEA, lineType=cv2.LINE_AA)
        if DIBUJAR_CENTRO:
            cv2.circle(capa, (int(round(d.cx)), int(round(d.cy))),
                       max(2, GROSOR_LINEA), 255, -1, lineType=cv2.LINE_AA)
        n += 1
    return capa, n


def _capa_siluetas(polys: list[np.ndarray], keep_mask, H: int, W: int) -> tuple[np.ndarray, int]:
    """Rellena la sombra segmentada de cada fibra (contorno crudo de YOLO)."""
    capa = np.zeros((H, W), dtype=np.uint8)
    n = 0
    for poly in polys:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        M = cv2.moments(pts)
        if abs(M["m00"]) > 1e-9:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        if not _centroide_valido(cx, cy, keep_mask, H, W):
            continue
        cv2.fillPoly(capa, [pts], 255)
        n += 1
    return capa, n


def _aplicar_capa(fondo_rgb: np.ndarray, capa_u8: np.ndarray) -> None:
    """Mezcla COLOR_FIBRA sobre fondo_rgb segun cobertura (capa) y ALPHA_PINTURA."""
    m = capa_u8 > 0
    if not m.any():
        return
    cov = (capa_u8[m].astype(np.float32) / 255.0) * float(ALPHA_PINTURA)  # (K,)
    cov = cov[:, None]
    color = np.array(COLOR_FIBRA_RGB, dtype=np.float32)
    fondo_rgb[m] = (fondo_rgb[m].astype(np.float32) * (1.0 - cov)
                    + color * cov).astype(np.uint8)


def pintar_fibras(det: FiberYOLODetector, rgb_det: np.ndarray,
                  fondo_rgb: np.ndarray, keep_mask) -> int:
    """Detecta segun MODO_PINTURA y pinta sobre fondo_rgb. Devuelve cuantas pinto."""
    H, W = fondo_rgb.shape[:2]
    if MODO_PINTURA == "linea":
        capa, n = _capa_lineas(detectar_fibras_pca(det, rgb_det), keep_mask, H, W)
    else:
        capa, n = _capa_siluetas(detectar_poligonos_fibra(det, rgb_det), keep_mask, H, W)
    _aplicar_capa(fondo_rgb, capa)
    return n


# ----------------------------------------------------------------------------
# PROCESAR UNA TOMA
# ----------------------------------------------------------------------------

def procesar_toma(det: FiberYOLODetector, carpeta: Path, salida_mp4: Path) -> bool:
    info = parse_nombre(carpeta.name)
    if info is None:
        print(f"  [salto] Nombre PTV invalido: {carpeta.name}", flush=True)
        return False

    cam      = int(info["cam"])
    carbopol = info["car"]
    mezcla   = f"M{info['mezcla']}"
    toma     = int(info["toma"])
    metodo   = info["metodo"]

    if cam not in V.CAM_PROFILES_PTV:
        print(f"  [salto] Sin perfil de camara para cam={cam}", flush=True)
        return False

    if not usar_toma(mezcla, toma, carbopol, metodo):
        print(f"  [skip] {carpeta.name}: Usar=no en el Sheet", flush=True)
        return False

    prof = V.CAM_PROFILES_PTV[cam]
    fps_captura = float(prof["fps"])
    preprocess_params = V.CAM_PREPROCESS_PARAMS_PTV.get(f"cam{cam}", {})

    archivos = listar_imagenes(carpeta)
    if not archivos:
        print(f"  [salto] Sin imagenes en {carpeta}", flush=True)
        return False

    skip = skip_para_toma(mezcla, toma, carbopol, cam, metodo)
    dur  = duracion_maxima_toma(cam, carbopol)

    try:
        indices, duracion, paso = seleccionar_indices(
            len(archivos), fps_captura, FPS_SALIDA, skip, dur)
    except ValueError as e:
        print(f"  [salto] {e}", flush=True)
        return False
    if not indices:
        print("  [salto] Seleccion vacia.", flush=True)
        return False

    print(f"  cam={cam} car={carbopol} fps_captura={fps_captura:g}", flush=True)
    print(f"  imagenes={len(archivos)}  skip={skip}  "
          f"duracion={'END' if dur is None else f'{dur:g}s'} -> {duracion:.2f}s", flush=True)
    print(f"  paso={paso:.3f} img/frame  frames_video={len(indices)}", flush=True)

    # Rango global para el fondo (si aplica)
    lo = hi = None
    if FONDO == "raw" and NORMALIZACION == "global":
        lo, hi = rango_global(archivos, indices)
        print(f"  rango intensidad fondo: {lo:.1f} - {hi:.1f}", flush=True)

    # Dimensiones y mascara fija
    raw0 = np.asarray(read_image_any(archivos[indices[0]]))
    H, W = raw0.shape[:2]
    keep_mask = None
    if bool(prof.get("apply_static_mask", False)):
        keep_mask = cargar_keep_mask(cam, H, W)

    salida_mp4.parent.mkdir(parents=True, exist_ok=True)
    escritor = imageio.get_writer(
        str(salida_mp4), fps=FPS_SALIDA, codec=CODEC, quality=None,
        pixelformat="yuv420p", macro_block_size=None,
        ffmpeg_params=["-crf", str(CALIDAD_CRF)],
    )

    total_fibras = 0
    try:
        for n, idx in enumerate(indices, 1):
            raw = np.asarray(read_image_any(archivos[idx]))

            # Imagen que ve el detector (la misma que usa el PTV: preprocesada)
            rgb_det = preprocess_frame_for_ptv(raw, preprocess_params)  # RGB uint8

            # Fondo del video: el preprocesado (reutiliza rgb_det) o el raw limpio
            if FONDO == "preprocesado":
                fondo = rgb_det
            else:
                fondo = fondo_rgb_desde_raw(raw, lo, hi)

            # Detecta (SAHI + NMS + PCA) y pinta segun MODO_PINTURA
            total_fibras += pintar_fibras(det, rgb_det, fondo, keep_mask)

            escritor.append_data(a_dims_pares(fondo))
            if n % 25 == 0 or n == len(indices):
                print(f"\r  procesando... {n}/{len(indices)}  "
                      f"(fibras acumuladas: {total_fibras})", end="", flush=True)
    finally:
        escritor.close()

    print(f"\n  -> {salida_mp4}\n", flush=True)
    return True


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main() -> None:
    if not V.PTV_BASE_DIR.exists():
        sys.exit(f"No existe la carpeta de tomas PTV: {V.PTV_BASE_DIR}")

    tomas = listar_tomas_ptv(V.PTV_BASE_DIR)
    if not tomas:
        sys.exit(f"No hay tomas PTV validas en {V.PTV_BASE_DIR}")

    print(f"[PINTAR] Color fibras : {COLOR_FIBRA_HEX} {COLOR_FIBRA_RGB}", flush=True)
    print(f"[PINTAR] Tomas PTV    : {len(tomas)}", flush=True)
    print(f"[PINTAR] Modelo       : {V.YOLO_TRACK_MODEL}", flush=True)
    print(f"[PINTAR] conf={V.CONF_TRACK}  SAHI scale={V.SAHI_SCALE_FACTOR} "
          f"tile={V.SAHI_TILE_SIZE} overlap={V.SAHI_OVERLAP_RATIO}", flush=True)
    print(f"[PINTAR] Salida       : {SALIDA_DIR}\n", flush=True)

    # Un solo detector para todas las tomas (mismo modelo best.pt)
    det = FiberYOLODetector(
        weights_path  = V.YOLO_TRACK_MODEL,
        conf          = V.CONF_TRACK,
        device        = DISPOSITIVO,
        scale_factor  = V.SAHI_SCALE_FACTOR,
        tile_size     = V.SAHI_TILE_SIZE,
        overlap_ratio = V.SAHI_OVERLAP_RATIO,
        iou_threshold = V.SAHI_IOU_THRESHOLD,
    )

    ok = 0
    for carpeta in tomas:
        print(f"[{carpeta.name}]", flush=True)
        salida = SALIDA_DIR / f"{carpeta.name}_{int(FPS_SALIDA)}fps_fibras.mp4"
        try:
            if procesar_toma(det, carpeta, salida):
                ok += 1
        except Exception as e:
            print(f"  [error] {e}\n", flush=True)

    print(f"[PINTAR] Completado. {ok}/{len(tomas)} videos generados en {SALIDA_DIR}", flush=True)


if __name__ == "__main__":
    main()
