from __future__ import annotations
from pathlib import Path
import json
import subprocess
import re
import sys
import pandas as pd
import requests
from io import StringIO

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import variables_piv as piv_vars
import variables_ptv as ptv_vars

# ============================================================
# CONFIGURACIÓN PRINCIPAL
# ============================================================

RUN_MODE = "ptv"  # "piv" | "ptv" | "both"
ALLOW_BOTH_WITHOUT_PTV = True

CONDA_BAT_OPTIONS = [
    Path(r"C:\Users\SarumanPM\anaconda3\condabin\conda.bat"),
    Path(r"C:\Users\MBX\anaconda3\condabin\conda.bat"),
    Path(r"C:\Users\lkwol\anaconda3\condabin\conda.bat"),
]
CONDA_BAT = next((str(p) for p in CONDA_BAT_OPTIONS if p.exists()), None)

ENV_YOLO           = "yolov11"
ENV_PIV            = "piv"
ENV_PREPROCESS_PTV = "yolov11"
ENV_PTV            = "yolox4"

RUNCODE_DIR            = PROJECT_ROOT / "RunCode"
PREPROCESS_SCRIPT      = RUNCODE_DIR / "preprocess_run.py"
PREPROCESS_PTV_SCRIPT  = RUNCODE_DIR / "preprocess_run_ptv.py"
PIV_SCRIPT             = RUNCODE_DIR / "piv_run.py"
PTV_SCRIPT             = RUNCODE_DIR / "ptv_run.py"
CLEANUP_SCRIPT         = RUNCODE_DIR / "cleanup_run.py"
CFG_PATH               = RUNCODE_DIR / "pipeline_config.json"

FIX_MASKS_DIR = PROJECT_ROOT / "FixMasks"

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQj9_M-1BdxtMDhG9u7j0qpO5WKN5tGZMe6lXdm-DZi-CIhwKY907aNLCLAXHppkda2AI5g2qX_p24S/pub?output=csv"

# ============================================================
# PERFILES DE CÁMARA — importados desde variables_piv/ptv
# ============================================================
# Para cambiar máscaras o parámetros de cámara, editar directamente:
#   variables_piv.py → CAM_PROFILES_PIV
#   variables_ptv.py → CAM_PROFILES_PTV

CAM_PROFILES_PIV = piv_vars.CAM_PROFILES_PIV
CAM_PROFILES_PTV = ptv_vars.CAM_PROFILES_PTV


# ============================================================
# GOOGLE SHEETS INTEGRATION
# ============================================================

def should_process_folder(mezcla: str, toma: int, carbopol: str, metodo: str = "piv") -> bool:
    df = load_experiment_config()
    if df is None:
        print(f"[WARN] No se pudo cargar Google Sheet, procesando por defecto", flush=True)
        return True
    carbopol_float    = float(carbopol) / 10.0
    mezcla_normalized = mezcla.upper()
    mask = (
        (df['Mezcla'].astype(str).str.upper() == mezcla_normalized) &
        (df['Toma'] == toma) &
        (df['Tipo'] == carbopol_float)
    )
    if 'Metodo' in df.columns:
        mask = mask & (df['Metodo'].astype(str).str.lower() == metodo.lower())

    matches = df[mask]
    if len(matches) == 0:
        print(f"[WARN] No se encontró config para {mezcla}-Toma{toma}-Car{carbopol}-{metodo.upper()}, procesando por defecto", flush=True)
        return True
    row         = matches.iloc[0]
    usar_value  = row.get('Usar', 'si')
    if pd.isna(usar_value):
        usar_value = 'si'
    usar_normalized = str(usar_value).strip().lower()
    should_use = usar_normalized in ['si', 'sí', 's', 'yes', 'y']
    if not should_use:
        print(f"[SKIP] {mezcla}-Toma{toma}-Car{carbopol}-{metodo.upper()}: Usar={usar_value} → SALTANDO", flush=True)
    return should_use


def load_experiment_config() -> pd.DataFrame | None:
    try:
        print(f"[INFO] Descargando Google Sheet...", flush=True)
        response = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if 'Tipo' in df.columns:
            df['Tipo'] = df['Tipo'].astype(str).str.replace(',', '.').astype(float)
        if 'Fotos Saltar' in df.columns:
            df['Fotos Saltar'] = df['Fotos Saltar'].astype(str).str.replace(',', '.', regex=False)
            df['Fotos Saltar'] = pd.to_numeric(df['Fotos Saltar'], errors='coerce').fillna(0)
        print(f"[INFO] Google Sheet cargado: {len(df)} filas", flush=True)
        return df
    except Exception as e:
        print(f"[WARN] No se pudo cargar Google Sheet: {e}", flush=True)
        return None


def parse_fps_ratio(ratio_str: str | float) -> float:
    if pd.isna(ratio_str):
        return 1.0
    if isinstance(ratio_str, (int, float)):
        return float(ratio_str)
    ratio_str = str(ratio_str).strip()
    if ':' in ratio_str:
        parts = ratio_str.split(':')
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if num != 0:
                    return den / num
            except ValueError:
                pass
    try:
        return float(ratio_str)
    except ValueError:
        return 1.0


def get_skip_images_for_folder(mezcla: str, toma: int, carbopol: str, cam: int, metodo: str = "piv") -> int:
    df = load_experiment_config()
    if df is None:
        return 0
    carbopol_float    = float(carbopol) / 10.0
    mezcla_normalized = mezcla.upper()
    mask = (
        (df['Mezcla'].astype(str).str.upper() == mezcla_normalized) &
        (df['Toma'] == toma) &
        (df['Tipo'] == carbopol_float)
    )
    if 'Metodo' in df.columns:
        mask = mask & (df['Metodo'].astype(str).str.lower() == metodo.lower())

    matches = df[mask]
    if len(matches) == 0:
        print(f"[WARN] No se encontró config para {mezcla}-Toma{toma}-Car{carbopol}-{metodo.upper()}, usando skip=0", flush=True)
        return 0
    row        = matches.iloc[0]
    skip_value = row.get('Fotos Saltar', 0)
    if pd.isna(skip_value):
        skip_value = 0
    skip_base  = int(skip_value)
    fps_ratio  = parse_fps_ratio(row.get('Razon FPS', 1.0))
    if cam == 4:
        skip_adjusted = int(skip_base * fps_ratio)
        print(f"[INFO] {mezcla}-Toma{toma}-Car{carbopol}-{metodo.upper()} Cam{cam}: skip={skip_base} × ratio={fps_ratio:.1f} = {skip_adjusted}", flush=True)
        return skip_adjusted
    else:
        print(f"[INFO] {mezcla}-Toma{toma}-Car{carbopol}-{metodo.upper()} Cam{cam}: skip={skip_base}", flush=True)
        return skip_base


# ============================================================
# HELPERS
# ============================================================

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


NAME_RE = re.compile(
    r"^m(?P<mezcla>\d+)-toma-(?P<toma>\d+)-cam-(?P<cam>\d+)-n-(?P<n>\d+)-car-(?P<car>\d+)-(?P<metodo>[A-Za-z0-9_]+)$"
)


def parse_subfolder_name(name: str) -> dict | None:
    m = NAME_RE.match(name)
    return m.groupdict() if m else None


def list_matching_subfolders(root: Path, metodo: str | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        info = parse_subfolder_name(p.name)
        if info is None:
            continue
        if metodo is not None and info["metodo"].lower() != metodo.lower():
            continue
        out.append(p)
    out.sort(key=lambda p: natural_key(p.name))
    return out


def fixed_mask_path_for_cam(cam: int) -> Path:
    return FIX_MASKS_DIR / f"cam-{cam}.tiff"


def piv_model_path_for_cam(cam: int) -> Path:
    return piv_vars.PIV_MODELS_DIR / f"cam{cam}-piv-yolo26.pt"


def ptv_mask_model_path_for_cam(cam: int) -> Path:
    return ptv_vars.PTV_MODELS_DIR / f"cam{cam}-ptv-yolo26.pt"


def cam_profile_for_folder(folder: Path, profiles: dict) -> tuple[int, str, dict, int]:
    info = parse_subfolder_name(folder.name)
    if info is None:
        raise RuntimeError(f"Nombre inválido: {folder.name}")
    cam      = int(info["cam"])
    carbopol = info["car"]
    mezcla   = f"M{info['mezcla']}"
    toma     = int(info["toma"])
    metodo   = info["metodo"]
    skip_images = get_skip_images_for_folder(mezcla, toma, carbopol, cam, metodo=metodo)
    if cam not in profiles:
        raise RuntimeError(f"No hay perfil definido para cam={cam}")
    prof = dict(profiles[cam])
    prof["fixed_mask_path"] = str(fixed_mask_path_for_cam(cam))
    return cam, carbopol, prof, skip_images


def run_env(env: str, script: Path) -> None:
    subprocess.run(
        [CONDA_BAT, "run", "-n", env, "python", str(script), str(CFG_PATH)],
        check=True, cwd=str(PROJECT_ROOT),
    )


def run_any(script: Path) -> None:
    subprocess.run(
        ["python", str(script), str(CFG_PATH)],
        check=True, cwd=str(PROJECT_ROOT),
    )


# ============================================================
# BUILD CONFIG JSON
# ============================================================

def write_cfg(
    pre_sub: Path | None,
    ptv_sub: Path | None,
    cam: int,
    carbopol: str,
    prof_piv: dict,
    prof_ptv: dict,
    skip_first_images: int,
) -> None:
    pre_info = parse_subfolder_name(pre_sub.name) if pre_sub else None
    ptv_info = parse_subfolder_name(ptv_sub.name) if ptv_sub else None
    pre_name = pre_sub.name if pre_sub else ""
    ptv_name = ptv_sub.name if ptv_sub else ""

    preprocess_params     = piv_vars.CAM_PREPROCESS_PARAMS.get(f"cam{cam}", {})
    ptv_preprocess_params = ptv_vars.CAM_PREPROCESS_PARAMS_PTV.get(f"cam{cam}", {})
    fixed_mask_path       = Path(prof_piv["fixed_mask_path"]) if prof_piv.get("fixed_mask_path") else None
    piv_model_path        = piv_model_path_for_cam(cam)
    piv_params            = piv_vars.get_piv_params(cam, carbopol)

    piv_apply_dynamic = bool(prof_piv["apply_dynamic_mask"])
    piv_apply_static  = bool(prof_piv["apply_static_mask"])
    ptv_apply_dynamic = bool(prof_ptv["apply_dynamic_mask"])
    ptv_apply_static  = bool(prof_ptv["apply_static_mask"])

    # ── Regiones temporales PIV ──────────────────────────────────
    temporal_regions_config = None
    if piv_vars.USE_TEMPORAL_REGIONS:
        regions = piv_vars.get_temporal_regions(cam, carbopol)
        if regions:
            temporal_regions_config = [
                {
                    "name":       r.name,
                    "start_time": r.start_time,
                    "end_time":   r.end_time,
                    "block_size": r.block_size,
                    "skip_inter": r.skip_inter,
                    "skip_final": r.skip_final,
                    "fps":        r.fps,
                }
                for r in regions
            ]
        else:
            print(f"[WARN] No hay regiones PIV para cam={cam}, carbopol={carbopol}", flush=True)

    # ── Regiones temporales PTV ──────────────────────────────────
    ptv_temporal_regions_config = None
    if ptv_vars.USE_TEMPORAL_REGIONS_PTV:
        ptv_regions = ptv_vars.get_ptv_temporal_regions(cam, carbopol)
        if ptv_regions:
            ptv_temporal_regions_config = [r.to_dict() for r in ptv_regions]
        else:
            print(f"[WARN] No hay regiones PTV para cam={cam}, carbopol={carbopol}", flush=True)

    cfg = {
        "meta": {
            "cam":               cam,
            "carbopol":          carbopol,
            "skip_first_images": skip_first_images,
            "cam_profile_piv":   prof_piv,
            "cam_profile_ptv":   prof_ptv,
            "pre_subfolder":     pre_sub.name if pre_sub else None,
            "ptv_subfolder":     ptv_sub.name if ptv_sub else None,
            "pre_info":          pre_info,
            "ptv_info":          ptv_info,
        },

        "camera": {
            "cam":       cam,
            "fps":       prof_piv["fps"],
            "dt_ms":     prof_piv["dt_ms"],
            "px_per_mm": prof_piv["px_per_mm"],
            "width_px":  prof_piv["width_px"],
            "height_px": prof_piv["height_px"],
        },

        # ── PIV: preprocesamiento ─────────────────────────────────
        "pre": {
            "input_subdir":         str(pre_sub) if pre_sub else None,
            "dest_out_dir":         str(piv_vars.PROCESSED_ROOT / pre_name) if pre_sub else None,
            "masks_out_dir":        str(piv_vars.MASKS_ROOT / pre_name) if pre_sub else None,
            "blocks":               piv_vars.BLOCKS,
            "block_size":           piv_vars.BLOCK_SIZE,
            "skip_inter":           piv_vars.SKIP_INTER,
            "skip_final":           piv_vars.SKIP_FINAL,
            "skip_first_images":    skip_first_images,
            "delete_existing":      piv_vars.DELETE_EXISTING_PRE,
            "preprocess_params":    preprocess_params,
            "use_temporal_regions": piv_vars.USE_TEMPORAL_REGIONS and temporal_regions_config is not None,
            "temporal_regions":     temporal_regions_config,
        },

        # ── PIV: máscaras dinámicas ───────────────────────────────
        "masks": {
            "model_path":         str(piv_model_path),
            "images_dir":         str(piv_vars.PROCESSED_ROOT / pre_name) if pre_sub else None,
            "output_dir":         str(piv_vars.MASKS_ROOT / pre_name) if pre_sub else None,
            "conf_thresh":        piv_vars.MASK_CONF,
            "device":             piv_vars.MASK_DEVICE,
            "invert_mask":        piv_vars.INVERT_MASK,
            "delete_existing":    piv_vars.DELETE_EXISTING_MASKS,
            "apply_dynamic_mask": piv_apply_dynamic,
            "apply_static_mask":  piv_apply_static,
            "fixed_mask_path":    str(fixed_mask_path) if fixed_mask_path else None,
            "imgsz":              int(prof_piv["width_px"]),
        },

        # ── PIV: cómputo ─────────────────────────────────────────
        "piv": {
            "images_dir":              str(piv_vars.PROCESSED_ROOT / pre_name) if pre_sub else None,
            "masks_dir":               str(piv_vars.MASKS_ROOT / pre_name) if pre_sub else None,
            "out_dir":                 str(piv_vars.RESULTS_PIV_ROOT / pre_name) if pre_sub else None,
            "dt_ms":                   prof_piv["dt_ms"],
            "px_per_mm":               prof_piv["px_per_mm"],
            "width_px":                prof_piv["width_px"],
            "height_px":               prof_piv["height_px"],
            "apply_dynamic_mask":      piv_apply_dynamic,
            "apply_static_mask":       piv_apply_static,
            "fixed_mask_path":         str(fixed_mask_path) if fixed_mask_path else None,
            "window_sizes":            piv_params['window_sizes'],
            "overlaps":                piv_params['overlaps'],
            "search_area_factor":      piv_vars.SEARCH_AREA_FACTOR,
            "sig2noise_method":        piv_vars.SIG2NOISE_METHOD,
            "mask_threshold":          piv_vars.MASK_THRESHOLD,
            "keep_percentile":         piv_params['keep_percentile'],
            "lm_kernel":               piv_vars.LM_KERNEL,
            "lm_thresh":               piv_vars.LM_THRESH,
            "lm_eps":                  piv_vars.LM_EPS,
            "show_viewers":            piv_vars.SHOW_VIEWERS,
            "clear_txt_before_export": piv_vars.CLEAR_TXT,
        },

        # ── PTV: preprocesamiento de imágenes ─────────────────────
        "pre_ptv": {
            "input_dir":         str(ptv_sub) if ptv_sub else None,
            "output_dir":        str(ptv_vars.PTV_PREPROCESSED / ptv_name) if ptv_sub else None,
            "skip_first_images": skip_first_images,
            "delete_existing":   ptv_vars.DELETE_EXISTING_PRE_PTV,
            "preprocess_params": ptv_preprocess_params,
        },

        # ── PTV: máscaras dinámicas ───────────────────────────────
        "masks_ptv": {
            "model_path":           str(ptv_mask_model_path_for_cam(cam)) if ptv_sub else None,
            "images_dir":           str(ptv_vars.PTV_PREPROCESSED / ptv_name) if ptv_sub else None,
            "output_dir":           str(ptv_vars.PTV_MASKS_ROOT / ptv_name) if ptv_sub else None,
            "conf_thresh":          ptv_vars.MASK_CONF_PTV,
            "device":               str(ptv_vars.MASK_DEVICE_PTV),
            "invert_mask":          ptv_vars.INVERT_MASK_PTV,
            "delete_existing":      ptv_vars.DELETE_EXISTING_MASKS_PTV,
            "apply_dynamic_mask":   ptv_apply_dynamic,
            "apply_static_mask":    ptv_apply_static,
            "fixed_mask_path":      str(fixed_mask_path) if fixed_mask_path else None,
            "fixed_mask_threshold": 127,
            "imgsz":                int(prof_ptv["width_px"]),
        },

        # ── PTV: tracking ─────────────────────────────────────────
        "ptv": {
            "images_dir":       str(ptv_sub) if ptv_sub else None,
            "preprocessed_dir": str(ptv_vars.PTV_PREPROCESSED / ptv_name) if ptv_sub else None,
            "masks_dir":        str(ptv_vars.PTV_MASKS_ROOT / ptv_name) if ptv_sub else None,
            "out_dir":          str(ptv_vars.RESULTS_PTV_ROOT / ptv_name) if ptv_sub else None,
            "weights_path":     str(ptv_vars.YOLO_TRACK_MODEL),
            "runs_segment_dir": str(ptv_vars.RUNS_SEGMENT_DIR),
            "device":           ptv_vars.DEVICE_PTV,
            "fps":              prof_ptv["fps"],
            "width_px":         prof_ptv["width_px"],
            "height_px":        prof_ptv["height_px"],
            "px_per_mm":        prof_ptv["px_per_mm"],
            "apply_dynamic_mask": ptv_apply_dynamic,
            "apply_static_mask":  ptv_apply_static,
            "fixed_mask_path":    str(fixed_mask_path) if fixed_mask_path else None,
            "mask_threshold":     127.0,
            "max_images":         ptv_vars.MAX_IMAGES,
            "preprocess_params":  ptv_preprocess_params,
            "use_temporal_regions": ptv_vars.USE_TEMPORAL_REGIONS_PTV and ptv_temporal_regions_config is not None,
            "temporal_regions":     ptv_temporal_regions_config,
            "alpha": ptv_vars.ALPHA,
            "beta":  ptv_vars.BETA,
            "gamma": ptv_vars.GAMMA,
            "alpha_ang":       ptv_vars.ALPHA_ANG,
            "beta_ang":        ptv_vars.BETA_ANG,
            "gamma_ang":       ptv_vars.GAMMA_ANG,
            "omega_decay":     ptv_vars.OMEGA_DECAY,
            "alpha_ang_decay": ptv_vars.ALPHA_ANG_DECAY,
            "conf":            ptv_vars.CONF_TRACK,
            "min_frames_keep": ptv_vars.MIN_FRAMES_KEEP,
            "annotate":        ptv_vars.ANNOTATE,
            "max_misses":      ptv_vars.MAX_MISSES,
            "gate_x_px":      ptv_vars.GATE_X,
            "gate_y_px":      ptv_vars.GATE_Y,
            "gate_angle_deg": ptv_vars.GATE_ANGLE,
            "sim_threshold": ptv_vars.get_tracking_params(cam)["sim_threshold"],
            "max_dist_px":   ptv_vars.get_max_dist_px(cam, CAM_PROFILES_PTV),
            "feat_weights":  list(ptv_vars.FEAT_WEIGHTS),
            "l_ref_px":      ptv_vars.get_l_ref_px(cam, CAM_PROFILES_PTV),
            "sahi_scale_factor":  ptv_vars.SAHI_SCALE_FACTOR,
            "sahi_tile_size":     ptv_vars.SAHI_TILE_SIZE,
            "sahi_overlap_ratio": ptv_vars.SAHI_OVERLAP_RATIO,
            "sahi_iou_threshold": ptv_vars.SAHI_IOU_THRESHOLD,
            "viz_tail_length":  ptv_vars.VIZ_TAIL_LENGTH,
            "viz_update_every": ptv_vars.VIZ_UPDATE_EVERY,
            "save_images":      ptv_vars.SAVE_IMAGES,
            # DPTV — Depth from Defocus
            "dptv_enabled":          ptv_vars.DPTV_ENABLED,
            "dptv_fiber_width_mm":   ptv_vars.FIBER_WIDTH_MM,
            "dptv_fiber_length_mm":  ptv_vars.FIBER_LENGTH_MM,
            "dptv_noise_width_px":   ptv_vars.DPTV_NOISE_WIDTH_PX,
            "dptv_k_blur_px_per_mm": ptv_vars.DPTV_K_BLUR_PX_PER_MM,
        },

        # ── Cleanup ───────────────────────────────────────────────
        "cleanup": {
            "processed_dir_to_delete":     str(piv_vars.PROCESSED_ROOT / pre_name) if pre_sub else None,
            "masks_dir_to_delete":         str(piv_vars.MASKS_ROOT / pre_name) if pre_sub else None,
            "delete_processed_subfolders": piv_vars.DELETE_PROCESSED_SUBFOLDERS,
            "ptv_preprocessed_to_delete":  str(ptv_vars.PTV_PREPROCESSED / ptv_name) if ptv_sub else None,
            "ptv_masks_to_delete":         str(ptv_vars.PTV_MASKS_ROOT / ptv_name) if ptv_sub else None,
            "delete_predict_folders":      ptv_vars.DELETE_PREDICT_FOLDERS,
            "runs_segment_dir":            str(ptv_vars.RUNS_SEGMENT_DIR),
        },
    }

    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# VALIDACIONES PREVIAS
# ============================================================

def _validate_mask_setup(cam: int, prof: dict, metodo: str) -> None:
    apply_dynamic = bool(prof["apply_dynamic_mask"])
    apply_static  = bool(prof["apply_static_mask"])

    if apply_static:
        fmp = Path(prof["fixed_mask_path"])
        if not fmp.exists():
            raise FileNotFoundError(
                f"apply_static_mask=True pero no existe máscara fija para cam={cam}: {fmp}"
            )

    if apply_dynamic:
        model = piv_model_path_for_cam(cam) if metodo == "piv" else ptv_mask_model_path_for_cam(cam)
        label = metodo.upper()
        if not model.exists():
            raise FileNotFoundError(
                f"apply_dynamic_mask=True pero no existe modelo {label} para cam={cam}: {model}"
            )

    if not apply_dynamic and not apply_static:
        print(f"[INFO] cam={cam} {metodo.upper()}: sin máscaras", flush=True)


# ============================================================
# PIPELINES
# ============================================================

def run_one_piv_folder(pre_sub: Path) -> None:
    cam, carbopol, prof_piv, skip_images = cam_profile_for_folder(pre_sub, CAM_PROFILES_PIV)
    info = parse_subfolder_name(pre_sub.name)
    if info:
        if not should_process_folder(f"M{info['mezcla']}", int(info['toma']), carbopol, metodo=info['metodo']):
            print(f"[SKIP] {pre_sub.name}", flush=True)
            return

    _validate_mask_setup(cam, prof_piv, metodo="piv")

    piv_model = piv_model_path_for_cam(cam)
    if prof_piv["apply_dynamic_mask"] and not piv_model.exists():
        raise FileNotFoundError(f"No existe modelo PIV para cam={cam}: {piv_model}")

    write_cfg(pre_sub=pre_sub, ptv_sub=None, cam=cam, carbopol=carbopol,
              prof_piv=prof_piv, prof_ptv=prof_piv, skip_first_images=skip_images)

    print(f"\n[PIPE] === PIV: {pre_sub.name} (cam={cam}, car={carbopol}, skip={skip_images}) ===", flush=True)
    print(f"[PIPE] Máscaras PIV: dynamic={prof_piv['apply_dynamic_mask']} static={prof_piv['apply_static_mask']}", flush=True)

    run_env(ENV_YOLO, PREPROCESS_SCRIPT)
    run_env(ENV_PIV, PIV_SCRIPT)
    run_any(CLEANUP_SCRIPT)
    print(f"[OK] PIV completado: {pre_sub.name}\n", flush=True)


def run_one_ptv_folder(ptv_sub: Path) -> None:
    cam, carbopol, prof_ptv, skip_images = cam_profile_for_folder(ptv_sub, CAM_PROFILES_PTV)
    info = parse_subfolder_name(ptv_sub.name)
    if info:
        if not should_process_folder(f"M{info['mezcla']}", int(info['toma']), carbopol, metodo=info['metodo']):
            print(f"[SKIP] {ptv_sub.name}", flush=True)
            return

    _validate_mask_setup(cam, prof_ptv, metodo="ptv")

    write_cfg(pre_sub=None, ptv_sub=ptv_sub, cam=cam, carbopol=carbopol,
              prof_piv=prof_ptv, prof_ptv=prof_ptv, skip_first_images=skip_images)

    print(f"\n[PIPE] === PTV: {ptv_sub.name} (cam={cam}, car={carbopol}, skip={skip_images}) ===", flush=True)
    print(f"[PIPE] Máscaras PTV: dynamic={prof_ptv['apply_dynamic_mask']} static={prof_ptv['apply_static_mask']}", flush=True)

    run_env(ENV_PREPROCESS_PTV, PREPROCESS_PTV_SCRIPT)
    run_env(ENV_PTV, PTV_SCRIPT)
    run_any(CLEANUP_SCRIPT)
    print(f"[OK] PTV completado: {ptv_sub.name}\n", flush=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if RUN_MODE in ("piv", "both"):
        if not piv_vars.PRE_BASE_DIR.exists():
            raise FileNotFoundError(f"PRE_BASE_DIR no existe: {piv_vars.PRE_BASE_DIR}")
        if not piv_vars.PIV_MODELS_DIR.exists():
            raise FileNotFoundError(f"PIV_MODELS_DIR no existe: {piv_vars.PIV_MODELS_DIR}")

    if RUN_MODE in ("ptv", "both"):
        if not ptv_vars.PTV_BASE_DIR.exists():
            raise FileNotFoundError(f"PTV_BASE_DIR no existe: {ptv_vars.PTV_BASE_DIR}")

    needs_static = (
        any(CAM_PROFILES_PIV[c]["apply_static_mask"] for c in CAM_PROFILES_PIV) or
        any(CAM_PROFILES_PTV[c]["apply_static_mask"] for c in CAM_PROFILES_PTV)
    )
    if needs_static and not FIX_MASKS_DIR.exists():
        raise FileNotFoundError(f"FixMasks dir no existe: {FIX_MASKS_DIR}")

    piv_folders: list[Path] = []
    ptv_folders: list[Path] = []

    if RUN_MODE in ("piv", "both"):
        piv_folders = list_matching_subfolders(piv_vars.PRE_BASE_DIR, metodo=piv_vars.PIV_METODO)
        if not piv_folders:
            raise RuntimeError(f"No hay carpetas PIV con metodo={piv_vars.PIV_METODO}")

    if RUN_MODE in ("ptv", "both"):
        ptv_folders = list_matching_subfolders(ptv_vars.PTV_BASE_DIR, metodo=ptv_vars.PTV_METODO)
        if RUN_MODE == "ptv" and not ptv_folders:
            raise RuntimeError(f"No hay carpetas PTV con metodo={ptv_vars.PTV_METODO}")
        if RUN_MODE == "both" and not ptv_folders:
            msg = f"[WARN] No hay carpetas PTV con metodo={ptv_vars.PTV_METODO}"
            if ALLOW_BOTH_WITHOUT_PTV:
                print(msg + " → omitiendo PTV.", flush=True)
            else:
                raise RuntimeError(msg)

    if RUN_MODE == "piv":
        print(f"[PIPE] PIV ONLY | carpetas={len(piv_folders)}", flush=True)
        for f in piv_folders:
            run_one_piv_folder(f)
        print("[OK] Pipeline PIV completo.", flush=True)
        return

    if RUN_MODE == "ptv":
        print(f"[PIPE] PTV ONLY | carpetas={len(ptv_folders)}", flush=True)
        for f in ptv_folders:
            run_one_ptv_folder(f)
        print("[OK] Pipeline PTV completo.", flush=True)
        return

    print(f"[PIPE] BOTH | PIV={len(piv_folders)} | PTV={len(ptv_folders)}", flush=True)
    for f in piv_folders:
        run_one_piv_folder(f)
    for f in ptv_folders:
        run_one_ptv_folder(f)
    print("[OK] Pipeline BOTH completo.", flush=True)


if __name__ == "__main__":
    main()