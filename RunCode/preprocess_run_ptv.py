"""
RunCode/preprocess_run_ptv.py
==============================
Preprocesamiento PTV en subproceso separado (entorno yolov11).

Responsabilidades:
1. Aplicar filtros de imagen (CLAHE, capping, etc.) a las imágenes originales
   y guardarlas en PTVPreprocesadas/<subfolder>/
2. Generar máscaras dinámicas con camX-ptv-yolo26.pt y guardarlas en
   PTVMascaras/<subfolder>/

Este script es invocado por pipeline_global.py antes de ptv_run.py,
con el mismo JSON de configuración como argumento.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path) -> list[Path]:
    exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg",
            "*.TIF", "*.TIFF", "*.PNG", "*.JPG", "*.JPEG")
    files: list[Path] = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=lambda p: natural_key(p.name))


# ─────────────────────────────────────────────
# PREPROCESAMIENTO DE IMÁGENES
# ─────────────────────────────────────────────

def run_preprocess_images(
    input_dir: Path,
    output_dir: Path,
    preprocess_params: dict | None,
    skip_first_images: int,
    delete_existing: bool,
) -> list[Path]:
    """
    Aplica filtros de preprocesamiento a las imágenes originales
    y las guarda en output_dir como TIFF 16-bit.

    Returns:
        Lista de rutas de imágenes preprocesadas (en orden natural).
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe images_dir: {input_dir}")

    if delete_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_images(input_dir)
    if not all_images:
        raise RuntimeError(f"No hay imágenes en: {input_dir}")

    images = all_images[skip_first_images:]
    if not images:
        raise RuntimeError(
            f"No quedan imágenes después de saltar {skip_first_images} frames "
            f"(total: {len(all_images)})"
        )

    print(f"[PRE-PTV] Preprocesando {len(images)} imágenes → {output_dir}", flush=True)

    if preprocess_params is None:
        # Sin preproceso: copiar directamente
        for img_path in images:
            dst = output_dir / img_path.name
            shutil.copy2(img_path, dst)
        print(f"[PRE-PTV] Sin preproceso: {len(images)} imágenes copiadas.", flush=True)
        return list_images(output_dir)

    # Con preproceso: aplicar filtros
    from PTV.Codes.PreProcessing.filters import apply_preprocessing, load_image, save_image

    for img_path in images:
        img = load_image(img_path)
        img_proc = apply_preprocessing(img, preprocess_params)
        dst = output_dir / img_path.name
        save_image(img_proc, dst, bit_depth=16)

    processed = list_images(output_dir)
    print(f"[PRE-PTV] Preproceso completado: {len(processed)} imágenes.", flush=True)
    return processed


# ─────────────────────────────────────────────
# MÁSCARAS DINÁMICAS (camX-ptv-yolo26.pt)
# ─────────────────────────────────────────────

def run_masks_ptv(
    model_path: Path,
    images_dir: Path,
    output_dir: Path,
    conf_thresh: float,
    device: str,
    invert_mask: bool,
    delete_existing: bool,
    apply_static_mask: bool,
    fixed_mask_path: Path | None,
    fixed_mask_threshold: int,
    imgsz: int,
) -> None:
    """
    Genera máscaras dinámicas para las imágenes preprocesadas PTV.
    Reutiliza run_masks_yolo del módulo PIV (mismo formato de salida).
    """
    from PIV.Codes.PreProcessing.masks import run_masks_yolo

    run_masks_yolo(
        model_path=model_path,
        images_dir=images_dir,
        output_dir=output_dir,
        conf_thresh=conf_thresh,
        device=device,
        invert_mask=invert_mask,
        delete_existing=delete_existing,
        apply_dynamic_mask=True,
        apply_static_mask=apply_static_mask,
        fixed_mask_path=fixed_mask_path,
        fixed_mask_threshold=fixed_mask_threshold,
        resize_fixed_mask_if_needed=True,
        imgsz=imgsz,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python RunCode/preprocess_run_ptv.py RunCode/pipeline_config.json")

    cfg_path = Path(sys.argv[1]).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    pre_ptv   = cfg["pre_ptv"]
    masks_ptv = cfg["masks_ptv"]

    # ── 1) Preprocesar imágenes ──────────────────────────────────
    run_preprocess_images(
        input_dir=Path(pre_ptv["input_dir"]),
        output_dir=Path(pre_ptv["output_dir"]),
        preprocess_params=pre_ptv.get("preprocess_params"),
        skip_first_images=int(pre_ptv.get("skip_first_images", 0)),
        delete_existing=bool(pre_ptv.get("delete_existing", True)),
    )

    # ── 2) Máscaras dinámicas ────────────────────────────────────
    apply_dynamic = bool(masks_ptv.get("apply_dynamic_mask", True))
    apply_static  = bool(masks_ptv.get("apply_static_mask", False))
    fixed_path    = masks_ptv.get("fixed_mask_path")
    fixed_path    = Path(fixed_path) if fixed_path else None

    if apply_dynamic:
        run_masks_ptv(
            model_path=Path(masks_ptv["model_path"]),
            images_dir=Path(masks_ptv["images_dir"]),
            output_dir=Path(masks_ptv["output_dir"]),
            conf_thresh=float(masks_ptv["conf_thresh"]),
            device=str(masks_ptv["device"]),
            invert_mask=bool(masks_ptv["invert_mask"]),
            delete_existing=bool(masks_ptv.get("delete_existing", True)),
            apply_static_mask=apply_static,
            fixed_mask_path=fixed_path,
            fixed_mask_threshold=int(masks_ptv.get("fixed_mask_threshold", 127)),
            imgsz=int(masks_ptv.get("imgsz", 1024)),
        )
    else:
        print("[PRE-PTV] apply_dynamic_mask=False → máscaras dinámicas omitidas.", flush=True)

    print("[PRE-PTV] Preprocesamiento PTV completado.", flush=True)


if __name__ == "__main__":
    main()