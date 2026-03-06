# piv/run.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import PIVConfig
from .pipeline import PIVPipeline
from .viewer import PIVViewer
from .exporter import TxtExporter
from .utils import clear_txt_in_out_dir


@dataclass(frozen=True)
class PIVRunOptions:
    show_viewers: bool = True
    clear_txt_before_export: bool = True


def _mask_mode_str(cfg: PIVConfig) -> str:
    # En este modo, PIV solo usa máscaras por frame desde masks_dir.
    dyn = bool(getattr(cfg, "apply_dynamic_mask", True))
    return "DINÁMICA (desde masks_dir)" if dyn else "SIN MÁSCARA"


def _validate_mask_inputs(cfg: PIVConfig) -> None:
    """
    Validaciones mínimas:
    - Si apply_dynamic_mask=True: masks_dir debe existir.
    - No validamos máscara fija, aunque el cfg la traiga por compatibilidad.
    """
    dyn = bool(getattr(cfg, "apply_dynamic_mask", True))
    if dyn:
        if not getattr(cfg, "masks_dir", None):
            raise RuntimeError("[PIV] apply_dynamic_mask=True pero cfg.masks_dir es None/vacío.")
        if not cfg.masks_dir.exists():
            raise FileNotFoundError(f"[PIV] No existe masks_dir: {cfg.masks_dir}")
        if not cfg.masks_dir.is_dir():
            raise NotADirectoryError(f"[PIV] masks_dir no es carpeta: {cfg.masks_dir}")


def run_piv(cfg: PIVConfig, opt: Optional[PIVRunOptions] = None) -> None:
    opt = opt or PIVRunOptions()

    # Validación de inputs (solo masks_dir)
    _validate_mask_inputs(cfg)

    pipe = PIVPipeline(cfg)
    viewer = PIVViewer()
    exporter = TxtExporter()

    jobs = pipe.build_jobs()
    results, names = pipe.compute_all(jobs)

    print(f"[PIV] DT_MS={cfg.dt_ms} ms (DT={cfg.dt_s():.6f} s)")
    print(f"[PIV] Pares procesados: {len(results)}")
    print(f"[PIV] Máscara: {_mask_mode_str(cfg)}")

    # Info opcional de tamaño si existe en cfg
    if hasattr(cfg, "width_px") and hasattr(cfg, "height_px"):
        print(f"[PIV] frame size: {cfg.width_px}x{cfg.height_px} px")

    print(
        f"[PIV] Validación: KEEP_PERCENTILE={cfg.keep_percentile} | "
        f"LocalMedian kernel={cfg.lm_kernel}, thresh={cfg.lm_thresh}, eps={cfg.lm_eps}"
    )

    if opt.show_viewers:
        viewer.show_initial(results, names, cfg)

    finals = pipe.validate_all(results)

    if opt.show_viewers:
        viewer.show_final(finals, names, cfg)

    if opt.clear_txt_before_export:
        deleted = clear_txt_in_out_dir(cfg.out_dir)
        if deleted:
            print(f"[PIV] Borrados {deleted} TXT antiguos en: {cfg.out_dir.resolve()}")

    exporter.export(finals, names, cfg)
    print(f"[PIV] Exportados {len(finals)} archivos TXT en: {cfg.out_dir.resolve()}")