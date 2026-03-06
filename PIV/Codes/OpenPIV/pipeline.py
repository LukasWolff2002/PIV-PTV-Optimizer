# pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .config import PIVConfig
from .models import PairJob, PIVResult, PIVResultFinal
from .utils import ensure_folder, pair_indices
from .workers import compute_pair_worker, validate_pair_worker


def _list_images(images_dir: Path) -> List[Path]:
    imgs = sorted(images_dir.glob("*.tif*"))
    return imgs


def _build_mask_map(masks_dir: Path) -> Dict[str, Path]:
    """
    Crea mapa por stem: <img_stem> -> <mask_path>
    Espera archivos tipo: frame_0001_mask.tiff
    """
    masks = list(masks_dir.glob("*.tif*"))
    out: Dict[str, Path] = {}

    for m in masks:
        stem = m.stem
        if stem.endswith("_mask"):
            img_stem = stem[:-5]
        else:
            continue

        if img_stem not in out:
            out[img_stem] = m
        else:
            try:
                if m.stat().st_mtime > out[img_stem].stat().st_mtime:
                    out[img_stem] = m
            except Exception:
                out[img_stem] = m

    return out


def _mask_for_image(mask_map: Dict[str, Path], img_path: Path) -> Path:
    key = img_path.stem
    if key not in mask_map:
        raise FileNotFoundError(
            f"No encontré máscara para imagen '{img_path.name}'. "
            f"Se esperaba '{key}_mask.tif' o '{key}_mask.tiff' en masks_dir."
        )
    return mask_map[key]


class PIVPipeline:
    def __init__(self, cfg: PIVConfig) -> None:
        self.cfg = cfg

    def build_jobs(self) -> List[PairJob]:
        cfg = self.cfg
        ensure_folder(cfg.images_dir, "IMAGES_DIR")

        images = _list_images(cfg.images_dir)

        if len(images) < 2:
            raise RuntimeError(f"Se necesitan al menos 2 imágenes en {cfg.images_dir}")
        if len(cfg.window_sizes) != len(cfg.overlaps):
            raise RuntimeError("WINDOW_SIZES y OVERLAPS deben tener el mismo largo.")
        if cfg.dt_s() <= 0:
            raise RuntimeError(f"DT_MS debe ser > 0. Valor actual: {cfg.dt_ms}")

        use_masks = bool(getattr(cfg, "apply_dynamic_mask", True))

        mask_map: Dict[str, Path] = {}
        if use_masks:
            ensure_folder(cfg.masks_dir, "MASKS_DIR")
            mask_map = _build_mask_map(cfg.masks_dir)
            if not mask_map:
                raise RuntimeError(f"No encontré máscaras '*_mask.tif*' en {cfg.masks_dir}")

        jobs: List[PairJob] = []
        for pair_id, ia, ib in pair_indices(len(images)):
            img_a = images[ia]
            img_b = images[ib]

            if use_masks:
                m_a = _mask_for_image(mask_map, img_a)
                m_b = _mask_for_image(mask_map, img_b)
            else:
                m_a = Path("")
                m_b = Path("")

            name = f"{img_a.name} - {img_b.name}"
            jobs.append(PairJob(pair_id, img_a, img_b, m_a, m_b, name))

        if not jobs:
            raise RuntimeError("No hay pares A-B (necesitas al menos 2 imágenes).")

        return jobs

    def compute_all(self, jobs: List[PairJob]) -> Tuple[List[PIVResult], List[str]]:
        cfg = self.cfg
        max_workers = max(1, (os.cpu_count() or 2) - 1)

        results: List[PIVResult] = []
        names: List[str] = [""] * len(jobs)

        use_masks = bool(getattr(cfg, "apply_dynamic_mask", True))

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for job in jobs:
                names[job.pair_id] = job.name
                futures.append(
                    ex.submit(
                        compute_pair_worker,
                        job.pair_id,
                        str(job.img_a),
                        str(job.img_b),
                        str(job.mask_a) if use_masks else "",
                        str(job.mask_b) if use_masks else "",
                        cfg.dt_s(),
                        cfg.window_sizes,
                        cfg.overlaps,
                        cfg.search_area_factor,
                        cfg.sig2noise_method,
                        cfg.mm_per_px(),
                        cfg.mask_threshold,
                        use_masks,
                        False,   # apply_static_mask (no usado en este modo)
                        "",      # fixed_mask_path
                        cfg.replace_outliers_kernel,   # MEJORA #5
                        cfg.replace_outliers_max_iter, # MEJORA #5
                    )
                )

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Procesando PIV",
                unit="par",
                ascii=True,
                dynamic_ncols=True,
            ):
                pair_id, x_mm, y_mm, u_mms, v_mms, in_mask, bg_disp, a, b = fut.result()
                results.append(
                    PIVResult(
                        pair_id=pair_id,
                        x_mm=x_mm,
                        y_mm=y_mm,
                        u_mms=u_mms,
                        v_mms=v_mms,
                        in_mask=in_mask,
                        bg_display=bg_disp,
                        img_a=Path(a),
                        img_b=Path(b),
                    )
                )

        results.sort(key=lambda r: r.pair_id)
        return results, names

    def validate_all(self, results: List[PIVResult]) -> List[PIVResultFinal]:
        cfg = self.cfg
        max_workers = max(1, (os.cpu_count() or 2) - 1)

        futures = []
        finals: List[PIVResultFinal] = []

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for r in results:
                futures.append(
                    ex.submit(
                        validate_pair_worker,
                        r.pair_id,
                        r.x_mm,
                        r.y_mm,
                        r.u_mms,
                        r.v_mms,
                        r.in_mask,
                        r.bg_display,
                        str(r.img_a),
                        str(r.img_b),
                        cfg.keep_percentile,
                        cfg.lm_kernel,
                        cfg.lm_thresh,
                        cfg.lm_eps,
                        cfg.replace_outliers_kernel,   # MEJORA #5
                        cfg.replace_outliers_max_iter, # MEJORA #5
                    )
                )

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validando PIV",
                unit="par",
                ascii=True,
                dynamic_ncols=True,
            ):
                pair_id, x_mm, y_mm, u3, v3, in_mask, bg, a, b = fut.result()
                finals.append(
                    PIVResultFinal(
                        pair_id=pair_id,
                        x_mm=x_mm,
                        y_mm=y_mm,
                        u_mms=u3,
                        v_mms=v3,
                        in_mask=in_mask,
                        bg_display=bg,
                        img_a=Path(a),
                        img_b=Path(b),
                    )
                )

        finals.sort(key=lambda r: r.pair_id)
        return finals