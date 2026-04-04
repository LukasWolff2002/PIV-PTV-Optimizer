"""
RunCode/preprocess_run_ptv.py
==============================
Preprocesamiento PTV en subproceso separado (entorno yolov11).

Responsabilidades:
1. Calcular qué imágenes va a necesitar el tracker según las regiones temporales
2. Preprocesar SOLO esas imágenes y guardarlas en PTVPreprocesadas/<subfolder>/
3. Generar máscaras dinámicas SOLO para esas imágenes con camX-ptv-yolo26.pt

El número de imágenes procesadas debe coincidir exactamente con las que
el tracker va a analizar — no se procesa la toma completa.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_all_images(folder: Path) -> list[Path]:
    """Lista TODAS las imágenes de una carpeta en orden natural."""
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg",
            ".TIF", ".TIFF", ".PNG", ".JPG", ".JPEG"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix in exts]
    return sorted(files, key=lambda p: natural_key(p.name))


# ─────────────────────────────────────────────
# SCHEDULE DE FRAMES (réplica de runner.py)
# ─────────────────────────────────────────────

def compute_selected_indices(
    all_images: list[Path],
    fps: float,
    temporal_regions: list[dict],
    max_images: int | None,
) -> list[int]:
    """
    Calcula los índices (en all_images) de las imágenes que el tracker
    va a procesar según las regiones temporales.

    Replica exactamente la lógica de _build_frame_schedule() en runner.py
    para garantizar que preproceso y tracker usan el mismo conjunto de imágenes.

    Args:
        all_images       : lista de imágenes ya con skip_first_images aplicado
        fps              : frames por segundo de la cámara
        temporal_regions : lista de dicts con name, start_time, end_time, skip_frames
        max_images       : límite máximo (igual que en runner.py)

    Returns:
        Lista de índices ordenados, sin duplicados.
    """
    images = all_images[:max_images] if max_images and max_images > 0 else all_images
    n_total = len(images)

    selected: set[int] = set()

    for r in temporal_regions:
        start_frame = int(r["start_time"] * fps)
        end_frame   = int(r["end_time"] * fps) if r["end_time"] is not None else n_total
        end_frame   = min(end_frame, n_total)
        stride      = int(r["skip_frames"]) + 1

        idx = start_frame
        while idx < end_frame:
            if idx < n_total:
                selected.add(idx)
            idx += stride

    return sorted(selected)


def compute_selected_indices_no_regions(
    all_images: list[Path],
    max_images: int | None,
) -> list[int]:
    """Fallback sin regiones: todos los frames consecutivos."""
    images = all_images[:max_images] if max_images and max_images > 0 else all_images
    return list(range(len(images)))


# ─────────────────────────────────────────────
# PREPROCESAMIENTO DE IMÁGENES SELECCIONADAS
# ─────────────────────────────────────────────

def run_preprocess_selected(
    selected_paths: list[Path],
    output_dir: Path,
    preprocess_params: dict | None,
    delete_existing: bool,
) -> None:
    """
    Aplica filtros de preprocesamiento SOLO a las imágenes seleccionadas
    y las guarda en output_dir manteniendo el nombre original.
    """
    if not selected_paths:
        raise RuntimeError("No hay imágenes seleccionadas para preprocesar.")

    if delete_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[PRE-PTV] Preprocesando {len(selected_paths)} imágenes → {output_dir}",
        flush=True,
    )

    if preprocess_params is None:
        for src in selected_paths:
            shutil.copy2(src, output_dir / src.name)
        print(f"[PRE-PTV] Sin filtros: {len(selected_paths)} imágenes copiadas.", flush=True)
        return

    from PIV.Codes.PreProcessing.filters import apply_preprocessing, load_image, save_image

    for src in selected_paths:
        img      = load_image(src)
        img_proc = apply_preprocessing(img, preprocess_params)
        save_image(img_proc, output_dir / src.name, bit_depth=16)

    print(f"[PRE-PTV] Filtros aplicados a {len(selected_paths)} imágenes.", flush=True)


# ─────────────────────────────────────────────
# MÁSCARAS DINÁMICAS PARA IMÁGENES SELECCIONADAS
# ─────────────────────────────────────────────

def run_masks_for_selected(
    model_path: Path,
    selected_paths: list[Path],
    output_dir: Path,
    conf_thresh: float,
    device: str,
    invert_mask: bool,
    delete_existing: bool,
    apply_static_mask: bool,
    fixed_mask_path: Optional[Path],
    fixed_mask_threshold: int,
    imgsz: int,
) -> None:
    """
    Genera máscaras dinámicas solo para las imágenes seleccionadas.
    Itera explícitamente sobre selected_paths en lugar de escanear el directorio.
    """
    from ultralytics import YOLO
    from tqdm import tqdm
    from PIV.Codes.PreProcessing.masks import process_one_image

    if not selected_paths:
        print("[PRE-PTV] Sin imágenes seleccionadas para máscaras.", flush=True)
        return

    if delete_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if apply_static_mask:
        if fixed_mask_path is None:
            raise ValueError("apply_static_mask=True pero fixed_mask_path=None")
        if not fixed_mask_path.exists():
            raise FileNotFoundError(f"fixed_mask_path no existe: {fixed_mask_path}")

    model = YOLO(str(model_path))

    print(
        f"[PRE-PTV] Generando máscaras para {len(selected_paths)} imágenes → {output_dir}",
        flush=True,
    )

    for img_path in tqdm(
        selected_paths,
        desc="Masks PTV",
        unit="img",
        dynamic_ncols=True,
        ascii=True,
    ):
        process_one_image(
            model=model,
            img_path=img_path,
            out_dir=output_dir,
            conf_thresh=conf_thresh,
            device=device,
            invert_mask=invert_mask,
            apply_dynamic_mask=True,
            apply_static_mask=apply_static_mask,
            fixed_mask_path=fixed_mask_path,
            fixed_mask_threshold=fixed_mask_threshold,
            resize_fixed_mask_if_needed=True,
            imgsz=imgsz,
        )

    print(f"[PRE-PTV] Máscaras completadas: {len(selected_paths)} generadas.", flush=True)


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
    ptv       = cfg["ptv"]

    input_dir         = Path(pre_ptv["input_dir"])
    output_dir        = Path(pre_ptv["output_dir"])
    skip_first_images = int(pre_ptv.get("skip_first_images", 0))
    delete_existing   = bool(pre_ptv.get("delete_existing", True))
    preprocess_params = pre_ptv.get("preprocess_params")

    fps        = float(ptv["fps"])
    max_images = ptv.get("max_images")
    use_tr     = bool(ptv.get("use_temporal_regions", False))
    tr_list    = ptv.get("temporal_regions") or []

    # ── 1) Listar todas las imágenes y aplicar skip inicial ──────
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe input_dir: {input_dir}")

    all_images_full = list_all_images(input_dir)
    if not all_images_full:
        raise RuntimeError(f"No hay imágenes en: {input_dir}")

    all_images = all_images_full[skip_first_images:]
    if not all_images:
        raise RuntimeError(
            f"No quedan imágenes tras skip_first_images={skip_first_images} "
            f"(total en carpeta: {len(all_images_full)})"
        )

    print(f"[PRE-PTV] Total imágenes en carpeta   : {len(all_images_full)}", flush=True)
    print(f"[PRE-PTV] Tras skip={skip_first_images}: {len(all_images)}", flush=True)

    # ── 2) Calcular qué imágenes necesita el tracker ─────────────
    # Cuando se usan regiones temporales, son ellas las que definen el rango
    # a procesar (start_time/end_time). max_images NO se aplica al schedule
    # para no truncar regiones finales como muy_baja_velocidad.
    # max_images solo tiene efecto en modo sin regiones (frames consecutivos).
    if use_tr and tr_list:
        selected_indices = compute_selected_indices(
            all_images=all_images,
            fps=fps,
            temporal_regions=tr_list,
            max_images=None,   # regiones temporales controlan el rango
        )
        mode_str = f"regiones temporales ({len(tr_list)} regiones)"
    else:
        selected_indices = compute_selected_indices_no_regions(
            all_images=all_images,
            max_images=max_images,
        )
        mode_str = "frames consecutivos (sin regiones)"

    selected_paths = [all_images[i] for i in selected_indices]

    print(f"[PRE-PTV] Modo                        : {mode_str}", flush=True)
    print(f"[PRE-PTV] Imágenes a preprocesar      : {len(selected_paths)}", flush=True)

    # Desglose por región
    if use_tr and tr_list:
        for r in tr_list:
            skip   = int(r["skip_frames"])
            stride = skip + 1
            start  = int(r["start_time"] * fps)
            end_t  = r["end_time"]
            end    = int(end_t * fps) if end_t is not None else len(all_images)
            end    = min(end, len(all_images))
            n_region = sum(1 for idx in selected_indices if start <= idx < end)
            dt_ms  = stride / fps * 1000
            label  = f"{end_t:.1f}s" if end_t is not None else "END"
            print(
                f"[PRE-PTV]   [{r['name']}] "
                f"t={r['start_time']:.1f}→{label}  "
                f"skip={skip}  Δt={dt_ms:.2f}ms  "
                f"frames={n_region}",
                flush=True,
            )

    if not selected_paths:
        raise RuntimeError(
            "El schedule no seleccionó ninguna imagen. "
            "Revisa start_time/end_time de las regiones temporales."
        )

    # ── 3) Guardar schedule.json — fuente de verdad para runner.py ──
    # Crear output_dir antes de escribir el JSON (run_preprocess_selected
    # también lo crea, pero el JSON debe guardarse primero).
    if delete_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Cada entrada relaciona el nombre del archivo preprocesado con su
    # metadata temporal (dt_s, timestamp_s, region_name, region_idx,
    # frame_idx_original). El runner lo leerá para construir el loop
    # sin recalcular el schedule desde cero.
    schedule_entries = []

    if use_tr and tr_list:
        # Reconstruir a qué región pertenece cada índice seleccionado
        for idx in selected_indices:
            # La región de un frame es la que contiene su índice
            region_hit = None
            for r_i, r in enumerate(tr_list):
                start = int(r["start_time"] * fps)
                end_t = r["end_time"]
                end   = int(end_t * fps) if end_t is not None else len(all_images)
                end   = min(end, len(all_images))
                if start <= idx < end:
                    region_hit = (r_i, r)
                    break
            if region_hit is None:
                # Fuera de todas las regiones — usar la última
                region_hit = (len(tr_list) - 1, tr_list[-1])

            r_i, r = region_hit
            skip   = int(r["skip_frames"])
            stride = skip + 1
            dt_s   = stride / fps
            schedule_entries.append({
                "preprocessed_name":  all_images[idx].name,
                "frame_idx_original": idx + skip_first_images,  # índice global
                "timestamp_s":        (idx + skip_first_images) / fps,
                "dt_s":               dt_s,
                "region_name":        r["name"],
                "region_idx":         r_i,
            })
    else:
        # Sin regiones: todos consecutivos
        dt_s = 1.0 / fps
        for local_i, idx in enumerate(selected_indices):
            schedule_entries.append({
                "preprocessed_name":  all_images[idx].name,
                "frame_idx_original": idx + skip_first_images,
                "timestamp_s":        (idx + skip_first_images) / fps,
                "dt_s":               dt_s,
                "region_name":        "default",
                "region_idx":         0,
            })

    schedule_json_path = output_dir / "schedule.json"
    schedule_json_path.write_text(
        json.dumps(schedule_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[PRE-PTV] schedule.json guardado: {schedule_json_path}", flush=True)

    # ── 4) Preprocesar SOLO las imágenes seleccionadas ──────────
    # output_dir ya existe (creado en el paso 3 junto con schedule.json).
    # Pasamos delete_existing=False para no borrar el schedule recién guardado.
    run_preprocess_selected(
        selected_paths=selected_paths,
        output_dir=output_dir,
        preprocess_params=preprocess_params,
        delete_existing=False,   # el directorio ya existe con schedule.json
    )

    # ── 5) Máscaras para las imágenes seleccionadas ───────────────
    apply_dynamic = bool(masks_ptv.get("apply_dynamic_mask", True))
    apply_static  = bool(masks_ptv.get("apply_static_mask", False))
    fixed_path    = masks_ptv.get("fixed_mask_path")
    fixed_path    = Path(fixed_path) if fixed_path else None

    if apply_dynamic:
        # Apuntar a las imágenes ya preprocesadas en output_dir
        preprocessed_paths = [output_dir / p.name for p in selected_paths]
        missing = [p for p in preprocessed_paths if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Faltan {len(missing)} imágenes en {output_dir}. "
                f"Ejemplo faltante: {missing[0].name}"
            )

        run_masks_for_selected(
            model_path=Path(masks_ptv["model_path"]),
            selected_paths=preprocessed_paths,
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

    elif apply_static:
        # Solo máscara fija: el tracker la lee directamente, no se genera nada aquí
        print(
            "[PRE-PTV] apply_dynamic_mask=False, apply_static_mask=True → "
            "máscara fija aplicada por el tracker directamente.",
            flush=True,
        )
    else:
        print("[PRE-PTV] Sin máscaras (dynamic=False, static=False).", flush=True)

    print(
        f"[PRE-PTV] Completado. {len(selected_paths)} imágenes listas en {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()