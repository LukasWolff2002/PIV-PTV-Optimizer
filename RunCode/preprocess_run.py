# preprocess_run.py
from __future__ import annotations
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIV.Codes.PreProcessing.blocks import run_block_sampling, run_adaptive_block_sampling
from PIV.Codes.PreProcessing.masks import run_masks_yolo

# Importar TemporalRegion si está disponible
try:
    from PIV.Codes.PreProcessing.temporal_regions import TemporalRegion
    TEMPORAL_REGIONS_AVAILABLE = True
except ImportError:
    TEMPORAL_REGIONS_AVAILABLE = False
    TemporalRegion = None


def main() -> None:
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    pre = cfg["pre"]
    masks = cfg["masks"]
    camera = cfg.get("camera", {})

    preprocess_params = pre.get("preprocess_params", None)
    
    # ====================================================================
    # DECISIÓN: Muestreo adaptativo vs. legacy
    # ====================================================================
    use_adaptive = pre.get("use_temporal_regions", False)
    temporal_regions_config = pre.get("temporal_regions", None)
    
    if use_adaptive and temporal_regions_config and TEMPORAL_REGIONS_AVAILABLE:
        print("\n[MODE] Muestreo adaptativo multi-región ACTIVADO")
        
        # Reconstruir objetos TemporalRegion desde config
        regions = []
        for r_dict in temporal_regions_config:
            region = TemporalRegion(
                name=r_dict["name"],
                start_time=r_dict["start_time"],
                end_time=r_dict["end_time"],
                block_size=r_dict["block_size"],
                skip_inter=r_dict["skip_inter"],
                skip_final=r_dict["skip_final"],
                fps=r_dict["fps"],
            )
            regions.append(region)
        
        # Ejecutar muestreo adaptativo
        metadata_list = run_adaptive_block_sampling(
            input_dir=Path(pre["input_subdir"]),
            output_dir=Path(pre["dest_out_dir"]),
            regions=regions,
            delete_existing=bool(pre["delete_existing"]),
            natural_sort=True,
            overwrite=True,
            preprocess_params=preprocess_params,
            output_metadata=True,
        )
        
        print(f"[OK] Muestreo adaptativo completado: {len(metadata_list)} pares generados")
    
    else:
        # Modo legacy: un solo tipo de bloque
        print("\n[MODE] Muestreo legacy (bloque único) ACTIVADO")
        
        if use_adaptive and not TEMPORAL_REGIONS_AVAILABLE:
            print("[WARN] Muestreo adaptativo solicitado pero módulo no disponible")
            print("[WARN] Usando modo legacy")
        
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
        
        print(f"[OK] Muestreo legacy completado")

    # ====================================================================
    # 2) Máscaras (igual para ambos modos)
    # ====================================================================
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
        apply_dynamic_mask=bool(masks.get("apply_dynamic_mask", True)),
        apply_static_mask=apply_static,
        fixed_mask_path=fixed_path,
        fixed_mask_threshold=int(masks.get("fixed_mask_threshold", 127)),
        resize_fixed_mask_if_needed=bool(masks.get("resize_fixed_mask_if_needed", True)),
    )

    print("[OK] preprocess_run listo.")


if __name__ == "__main__":
    main()