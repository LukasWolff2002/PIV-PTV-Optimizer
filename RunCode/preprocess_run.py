# preprocess_run.py
from __future__ import annotations
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIV.Codes.PreProcessing.blocks import run_block_sampling
from PIV.Codes.PreProcessing.masks import run_masks_yolo


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    pre = cfg["pre"]
    masks = cfg["masks"]

    preprocess_params = pre.get("preprocess_params", None)

    # 1) Preprocess (pares)
    run_block_sampling(
        input_dir=Path(pre["input_subdir"]),
        output_dir=Path(pre["dest_out_dir"]),
        blocks=pre["blocks"],
        block_size=int(pre["block_size"]),
        skip_inter=int(pre["skip_inter"]),
        skip_final=int(pre["skip_final"]),
        delete_existing=bool(pre["delete_existing"]),
        natural_sort=True,
        overwrite=True,
        preprocess_params=preprocess_params,
    )

    # 2) Masks
    apply_static = bool(masks.get("apply_static_mask", False))
    fixed_path = masks.get("fixed_mask_path", None)
    fixed_path = Path(fixed_path) if fixed_path else None

    run_masks_yolo(
        model_path=Path(masks["model_path"]),
        images_dir=Path(masks["images_dir"]),
        output_dir=Path(masks["output_dir"]),
        conf_thresh=float(masks["conf_thresh"]),
        device=str(masks["device"]),
        invert_mask=bool(masks["invert_mask"]),
        delete_existing=bool(masks["delete_existing"]),
        # NUEVO/CLAVE: respetar si quieres solo fija
        apply_dynamic_mask=bool(masks.get("apply_dynamic_mask", True)),
        apply_static_mask=apply_static,
        fixed_mask_path=fixed_path,
        fixed_mask_threshold=int(masks.get("fixed_mask_threshold", 127)),
        resize_fixed_mask_if_needed=bool(masks.get("resize_fixed_mask_if_needed", True)),
    )

    print("[OK] preprocess_run listo.")


if __name__ == "__main__":
    main()