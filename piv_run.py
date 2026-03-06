from __future__ import annotations
import json, sys
from pathlib import Path

from PIV.Codes.OpenPIV.config import PIVConfig
from PIV.Codes.OpenPIV.run import run_piv, PIVRunOptions


def main():
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    c = cfg["piv"]

    piv_cfg = PIVConfig(
        images_dir=Path(c["images_dir"]),
        masks_dir=Path(c["masks_dir"]),
        out_dir=Path(c["out_dir"]),
        dt_ms=float(c["dt_ms"]),
        px_per_mm=float(c["px_per_mm"]),
        window_sizes=list(c["window_sizes"]),
        overlaps=list(c["overlaps"]),
        search_area_factor=int(c["search_area_factor"]),
        sig2noise_method=str(c["sig2noise_method"]),
        mask_threshold=float(c["mask_threshold"]),
        keep_percentile=float(c["keep_percentile"]),
        lm_kernel=int(c["lm_kernel"]),
        lm_thresh=float(c["lm_thresh"]),
        lm_eps=float(c["lm_eps"]),
        default_quiver_scale=8.0,
        quiver_width=0.0025,
        # Si tu PIVConfig todavía incluye apply_dynamic_mask, lo puedes pasar.
        # Si no existe, simplemente bórralo también.
        apply_dynamic_mask=True,
    )

    piv_opt = PIVRunOptions(
        show_viewers=bool(c["show_viewers"]),
        clear_txt_before_export=bool(c["clear_txt_before_export"]),
    )

    run_piv(piv_cfg, piv_opt)


if __name__ == "__main__":
    main()