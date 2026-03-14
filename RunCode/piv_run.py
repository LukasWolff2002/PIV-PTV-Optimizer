# piv_run.py
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIV.Codes.OpenPIV.config import PIVConfig
from PIV.Codes.OpenPIV.run import run_piv, PIVRunOptions


def main():
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    
    c = cfg["piv"]
    camera = cfg.get("camera", {})  # ← AGREGADO: leer sección camera

    piv_cfg = PIVConfig(
        images_dir=Path(c["images_dir"]),
        masks_dir=Path(c["masks_dir"]),
        out_dir=Path(c["out_dir"]),
        dt_ms=float(c["dt_ms"]),
        fps=float(camera.get("fps")),  # ← AGREGADO: FPS para timestamps
        px_per_mm=float(c["px_per_mm"]),
        window_sizes=list(c["window_sizes"]),
        overlaps=list(c["overlaps"]),
        search_area_factor=int(c["search_area_factor"]),
        sig2noise_method=str(c["sig2noise_method"]),
        mask_threshold=float(c["mask_threshold"]),
        apply_dynamic_mask=bool(c.get("apply_dynamic_mask", True)),
        default_quiver_scale=8.0,
        quiver_width=0.0025,
        keep_percentile=float(c.get("keep_percentile", 90.0)),
        lm_kernel=int(c.get("lm_kernel", 1)),
        lm_thresh=float(c.get("lm_thresh", 2.0)),
        lm_eps=float(c.get("lm_eps", 0.1)),
        # MEJORA #5: parametrizables desde config JSON
        replace_outliers_kernel=int(c.get("replace_outliers_kernel", 2)),
        replace_outliers_max_iter=int(c.get("replace_outliers_max_iter", 3)),
        # MEJORA #3: grilla completa por defecto
        export_full_grid=bool(c.get("export_full_grid", True)),
    )

    piv_opt = PIVRunOptions(
        show_viewers=bool(c["show_viewers"]),
        clear_txt_before_export=bool(c["clear_txt_before_export"]),
    )

    run_piv(piv_cfg, piv_opt)


if __name__ == "__main__":
    main()