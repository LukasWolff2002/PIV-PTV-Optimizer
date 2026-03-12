from __future__ import annotations
from pathlib import Path
import json
import subprocess
import re


# ============================================================
# 1) USER INPUTS
# ============================================================

RUN_MODE = "piv"  # "piv" | "ptv" | "both"
ALLOW_BOTH_WITHOUT_PTV = True

CONDA_BAT = r"C:\Users\MBX\anaconda3\condabin\conda.bat"
ENV_YOLO = "yolov11"
ENV_PIV  = "piv"

# ------------------------------------------------------------
# ROOTS
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNCODE_DIR = PROJECT_ROOT / "RunCode"

# Scripts
PREPROCESS_SCRIPT = RUNCODE_DIR / "preprocess_run.py"
PIV_SCRIPT        = RUNCODE_DIR / "piv_run.py"
PTV_SCRIPT        = RUNCODE_DIR / "ptv_run.py"
CLEANUP_SCRIPT    = RUNCODE_DIR / "cleanup_run.py"

# Config JSON
CFG_PATH = RUNCODE_DIR / "pipeline_config.json"

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------
PRE_BASE_DIR = PROJECT_ROOT / "PIV" / "Tomas"
PTV_BASE_DIR = PROJECT_ROOT / "PTV" / "Tomas"

PROCESSED_ROOT = PROJECT_ROOT / "TomasProcesadas"
MASKS_ROOT     = PROJECT_ROOT / "Masks"

RESULTS_PIV_ROOT = PROJECT_ROOT / "ResultadosPIV"
RESULTS_PTV_ROOT = PROJECT_ROOT / "ResultadosPTV"

# Si quieres filtrar por método en el nombre
PIV_METODO = "piv"
PTV_METODO = "ptv"   # si no aplica, pon None

# ---------- FIX MASKS ----------
# Estructura esperada:
#   FixMasks/
#     cam-1.tiff
#     cam-2.tiff
#     cam-3.tiff
#     cam-4.tiff
FIX_MASKS_DIR = PROJECT_ROOT / "FixMasks"

# ---------- Perfiles por cámara ----------
CAM_PROFILES = {
    1: dict(
        fps=200,
        dt_ms=1.0,
        px_per_mm=20.0,
        width_px=1024,
        height_px=1024,
        apply_dynamic_mask=True,
        apply_static_mask=True,
    ),
    2: dict(
        fps=200,
        dt_ms=1.0,
        px_per_mm=20.0,
        width_px=1024,
        height_px=1024,
        apply_dynamic_mask=True,
        apply_static_mask=False,
    ),
    3: dict(
        fps=200,
        dt_ms=1.0,
        px_per_mm=20.0,
        width_px=1024,
        height_px=1024,
        apply_dynamic_mask=True,
        apply_static_mask=False,
    ),
    4: dict(
        fps=600,
        dt_ms=1.0,
        px_per_mm=20.0,
        width_px=1024,
        height_px=1024,
        apply_dynamic_mask=True,
        apply_static_mask=False,
    ),
}

CAM_PREPROCESS_PARAMS = {
    "cam1": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 200,
        "clahe_clip_limit": 0.0010,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 15,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0395,
        "max_intensity": 1.0000,
    },
    "cam2": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 17,
        "clahe_clip_limit": 0.0492,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 14,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0000,
        "max_intensity": 1.0000,
    },
    "cam3": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 10,
        "clahe_clip_limit": 0.1000,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 15,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0000,
        "max_intensity": 1.0000,
    },
    "cam4": {
        "roi_enabled": False,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 100,
        "roi_height": 100,
        "clahe_enabled": True,
        "clahe_tile_size": 10,
        "clahe_clip_limit": 0.0100,
        "intensity_capping": True,
        "capping_n_std": 5.0000,
        "highpass_enabled": False,
        "highpass_size": 15,
        "wiener_enabled": False,
        "wiener_size": 3,
        "gaussian_size": 3,
        "min_intensity": 0.0000,
        "max_intensity": 0.7237,
    },
}

# --- Preprocess: muestreo por bloques ---
BLOCKS      = 50
BLOCK_SIZE  = 22
SKIP_INTER  = 0
SKIP_FINAL  = 20
DELETE_EXISTING_PRE = True

# --- Modelo YOLO máscaras (para PIV) ---
MASK_MODEL = PROJECT_ROOT / "PIV" / "Codes" / "Segmentation-Models" / "DynamicMask.pt"
MASK_CONF = 0.25
MASK_DEVICE = "0"
INVERT_MASK = True
DELETE_EXISTING_MASKS = True

# --- Parámetros PIV (comunes) ---
WINDOW_SIZES = [64, 32, 16]
OVERLAPS     = [32, 16, 8]
SEARCH_AREA_FACTOR = 1
SIG2NOISE_METHOD   = "peak2peak"
MASK_THRESHOLD = 0.0
KEEP_PERCENTILE = 80.0
LM_KERNEL = 1
LM_THRESH = 3.0
LM_EPS    = 0.1
SHOW_VIEWERS = True
CLEAR_TXT = True

# --- Modelo YOLO tracking (PTV) ---
YOLO_TRACK_MODEL = PROJECT_ROOT / "PTV-Codes" / "Segmentation-Models" / "yolo11ssef.pt"
RUNS_SEGMENT_DIR = PROJECT_ROOT / "runs" / "segment"

# --- Parámetros PTV (comunes) ---
MAX_IMAGES = None
ALPHA = 0.95
BETA  = 0.95
GAMMA = 0.05
GATE_X = 10
GATE_Y = 10
GATE_ANGLE = 5
CONF_TRACK = 0.25
MIN_FRAMES_KEEP = 20
ANNOTATE = True

# --- Cleanup ---
DELETE_PROCESSED_SUBFOLDERS = True
DELETE_PREDICT_FOLDERS = False


# ============================================================
# 2) HELPERS
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

def cam_profile_for_folder(folder: Path) -> tuple[int, dict]:
    info = parse_subfolder_name(folder.name)
    if info is None:
        raise RuntimeError(f"Nombre inválido: {folder.name}")

    cam = int(info["cam"])
    if cam not in CAM_PROFILES:
        raise RuntimeError(f"No hay perfil definido para cam={cam}. Define CAM_PROFILES[{cam}]")

    prof = dict(CAM_PROFILES[cam])
    prof["fixed_mask_path"] = str(fixed_mask_path_for_cam(cam))
    return cam, prof

def run_env(env: str, script: Path) -> None:
    subprocess.run(
        [CONDA_BAT, "run", "-n", env, "python", str(script), str(CFG_PATH)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

def run_any(script: Path) -> None:
    subprocess.run(
        ["python", str(script), str(CFG_PATH)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


# ============================================================
# 3) BUILD CONFIG JSON (por carpeta)
# ============================================================

def write_cfg(pre_sub: Path | None, ptv_sub: Path | None, cam: int, prof: dict) -> None:
    pre_info = parse_subfolder_name(pre_sub.name) if pre_sub else None
    ptv_info = parse_subfolder_name(ptv_sub.name) if ptv_sub else None

    pre_name = pre_sub.name if pre_sub else ""
    ptv_name = ptv_sub.name if ptv_sub else ""

    preprocess_params = CAM_PREPROCESS_PARAMS.get(f"cam{cam}", {})

    fixed_mask_path = Path(prof["fixed_mask_path"]) if prof.get("fixed_mask_path") else None

    cfg = {
        "meta": {
            "cam": cam,
            "cam_profile": prof,
            "pre_subfolder": pre_sub.name if pre_sub else None,
            "ptv_subfolder": ptv_sub.name if ptv_sub else None,
            "pre_info": pre_info,
            "ptv_info": ptv_info,
        },

        "camera": {
            "cam": cam,
            "fps": prof["fps"],
            "dt_ms": prof["dt_ms"],
            "px_per_mm": prof["px_per_mm"],
            "width_px": prof["width_px"],
            "height_px": prof["height_px"],
            "apply_dynamic_mask": bool(prof["apply_dynamic_mask"]),
            "apply_static_mask": bool(prof["apply_static_mask"]),
            "fixed_mask_path": str(fixed_mask_path) if fixed_mask_path else None,
        },

        "pre": {
            "input_subdir": str(pre_sub) if pre_sub else None,
            "dest_out_dir": str(PROCESSED_ROOT / pre_name) if pre_sub else None,
            "masks_out_dir": str(MASKS_ROOT / pre_name) if pre_sub else None,
            "blocks": BLOCKS,
            "block_size": BLOCK_SIZE,
            "skip_inter": SKIP_INTER,
            "skip_final": SKIP_FINAL,
            "delete_existing": DELETE_EXISTING_PRE,
            "preprocess_params": preprocess_params,
        },

        "masks": {
            "model_path": str(MASK_MODEL),
            "images_dir": str(PROCESSED_ROOT / pre_name) if pre_sub else None,
            "output_dir": str(MASKS_ROOT / pre_name) if pre_sub else None,
            "conf_thresh": MASK_CONF,
            "device": MASK_DEVICE,
            "invert_mask": INVERT_MASK,
            "delete_existing": DELETE_EXISTING_MASKS,
            "apply_dynamic_mask": bool(prof["apply_dynamic_mask"]),
            "apply_static_mask": bool(prof["apply_static_mask"]),
            "fixed_mask_path": str(fixed_mask_path) if fixed_mask_path else None,
        },

        "piv": {
            "images_dir": str(PROCESSED_ROOT / pre_name) if pre_sub else None,
            "masks_dir": str(MASKS_ROOT / pre_name) if pre_sub else None,
            "out_dir": str(RESULTS_PIV_ROOT / pre_name) if pre_sub else None,
            "dt_ms": prof["dt_ms"],
            "px_per_mm": prof["px_per_mm"],
            "width_px": prof["width_px"],
            "height_px": prof["height_px"],
            "apply_dynamic_mask": bool(prof["apply_dynamic_mask"]),
            "apply_static_mask": bool(prof["apply_static_mask"]),
            "fixed_mask_path": str(fixed_mask_path) if fixed_mask_path else None,
            "window_sizes": WINDOW_SIZES,
            "overlaps": OVERLAPS,
            "search_area_factor": SEARCH_AREA_FACTOR,
            "sig2noise_method": SIG2NOISE_METHOD,
            "mask_threshold": MASK_THRESHOLD,
            "keep_percentile": KEEP_PERCENTILE,
            "lm_kernel": LM_KERNEL,
            "lm_thresh": LM_THRESH,
            "lm_eps": LM_EPS,
            "show_viewers": SHOW_VIEWERS,
            "clear_txt_before_export": CLEAR_TXT,
        },

        "ptv": {
            "images_dir": str(ptv_sub) if ptv_sub else None,
            "out_dir": str(RESULTS_PTV_ROOT / ptv_name) if ptv_sub else None,
            "weights_path": str(YOLO_TRACK_MODEL),
            "runs_segment_dir": str(RUNS_SEGMENT_DIR),
            "fps": prof["fps"],
            "width_px": prof["width_px"],
            "height_px": prof["height_px"],
            "apply_dynamic_mask": bool(prof["apply_dynamic_mask"]),
            "apply_static_mask": bool(prof["apply_static_mask"]),
            "fixed_mask_path": str(fixed_mask_path) if fixed_mask_path else None,
            "max_images": MAX_IMAGES,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "gate_x_px": GATE_X,
            "gate_y_px": GATE_Y,
            "gate_angle_deg": GATE_ANGLE,
            "conf": CONF_TRACK,
            "min_frames_keep": MIN_FRAMES_KEEP,
            "annotate": ANNOTATE,
        },

        "cleanup": {
            "processed_dir_to_delete": str(PROCESSED_ROOT / pre_name) if pre_sub else None,
            "masks_dir_to_delete": str(MASKS_ROOT / pre_name) if pre_sub else None,
            "delete_processed_subfolders": DELETE_PROCESSED_SUBFOLDERS,
            "delete_predict_folders": DELETE_PREDICT_FOLDERS,
            "runs_segment_dir": str(RUNS_SEGMENT_DIR),
        },
    }

    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 4) PIPELINES (por carpeta)
# ============================================================

def run_one_piv_folder(pre_sub: Path) -> None:
    cam, prof = cam_profile_for_folder(pre_sub)

    if prof["apply_static_mask"]:
        fmp = Path(prof["fixed_mask_path"])
        if not fmp.exists():
            raise FileNotFoundError(
                f"apply_static_mask=True pero no existe máscara fija para cam={cam}: {fmp}"
            )

    write_cfg(pre_sub=pre_sub, ptv_sub=None, cam=cam, prof=prof)

    print(f"\n[PIPE] === PIV folder: {pre_sub.name} (cam={cam}) ===", flush=True)
    print(f"[PIPE] cam profile: {prof}", flush=True)

    print("[PIPE] 1) PRE + MASKS", flush=True)
    run_env(ENV_YOLO, PREPROCESS_SCRIPT)

    print(f"[PIPE] 2) PIV -> {RESULTS_PIV_ROOT / pre_sub.name}", flush=True)
    run_env(ENV_PIV, PIV_SCRIPT)

    print("[PIPE] 3) CLEANUP pre outputs", flush=True)
    run_any(CLEANUP_SCRIPT)

    print(f"[OK] PIV listo: {pre_sub.name}", flush=True)

def run_one_ptv_folder(ptv_sub: Path) -> None:
    cam, prof = cam_profile_for_folder(ptv_sub)

    if prof["apply_static_mask"]:
        fmp = Path(prof["fixed_mask_path"])
        if not fmp.exists():
            raise FileNotFoundError(
                f"apply_static_mask=True pero no existe máscara fija para cam={cam}: {fmp}"
            )

    write_cfg(pre_sub=None, ptv_sub=ptv_sub, cam=cam, prof=prof)

    print(f"\n[PIPE] === PTV folder: {ptv_sub.name} (cam={cam}) ===", flush=True)
    print(f"[PIPE] cam profile: {prof}", flush=True)

    print(f"[PIPE] PTV -> {RESULTS_PTV_ROOT / ptv_sub.name}", flush=True)
    run_env(ENV_YOLO, PTV_SCRIPT)

    print(f"[OK] PTV listo: {ptv_sub.name}", flush=True)


# ============================================================
# 5) MAIN
# ============================================================

def main() -> None:
    if RUN_MODE in ("piv", "both") and not PRE_BASE_DIR.exists():
        raise FileNotFoundError(f"PRE_BASE_DIR no existe: {PRE_BASE_DIR}")

    if RUN_MODE in ("piv", "both") and not MASK_MODEL.exists():
        raise FileNotFoundError(f"MASK_MODEL no existe: {MASK_MODEL}")

    if any(CAM_PROFILES[c]["apply_static_mask"] for c in CAM_PROFILES):
        if not FIX_MASKS_DIR.exists():
            raise FileNotFoundError(f"FixMasks dir no existe: {FIX_MASKS_DIR}")

    piv_folders: list[Path] = []
    ptv_folders: list[Path] = []

    if RUN_MODE in ("piv", "both"):
        piv_folders = list_matching_subfolders(PRE_BASE_DIR, metodo=PIV_METODO)
        if not piv_folders:
            raise RuntimeError(f"No encontré carpetas PIV metodo={PIV_METODO} en {PRE_BASE_DIR}")

    if RUN_MODE in ("ptv", "both"):
        ptv_folders = list_matching_subfolders(PTV_BASE_DIR, metodo=PTV_METODO)
        if RUN_MODE == "ptv" and not ptv_folders:
            raise RuntimeError(f"No encontré carpetas PTV metodo={PTV_METODO} en {PTV_BASE_DIR}")
        if RUN_MODE == "both" and not ptv_folders:
            msg = f"[WARN] No encontré carpetas PTV metodo={PTV_METODO} en {PTV_BASE_DIR}"
            if ALLOW_BOTH_WITHOUT_PTV:
                print(msg + " -> omitiendo PTV.", flush=True)
            else:
                raise RuntimeError(msg)

    if RUN_MODE == "piv":
        print(f"[PIPE] PIV ONLY | folders={len(piv_folders)}", flush=True)
        for f in piv_folders:
            run_one_piv_folder(f)
        print("\n[OK] Pipeline PIV completo.", flush=True)
        return

    if RUN_MODE == "ptv":
        print(f"[PIPE] PTV ONLY | folders={len(ptv_folders)}", flush=True)
        for f in ptv_folders:
            run_one_ptv_folder(f)
        print("\n[OK] Pipeline PTV completo.", flush=True)
        return

    print(f"[PIPE] BOTH | PIV folders={len(piv_folders)} | PTV folders={len(ptv_folders)}", flush=True)

    for f in piv_folders:
        run_one_piv_folder(f)

    for f in ptv_folders:
        run_one_ptv_folder(f)

    print("\n[OK] Pipeline BOTH completo.", flush=True)


if __name__ == "__main__":
    main()