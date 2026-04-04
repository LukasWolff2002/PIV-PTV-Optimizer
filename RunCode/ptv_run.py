"""
RunCode/ptv_run.py
==================
Punto de entrada del pipeline PTV.
Lee el JSON generado por pipeline_global.py y delega
toda la lógica a PTV/Codes/PTVCode/.

Prerequisito: preprocess_run_ptv.py ya debe haber corrido y
generado las imágenes en ptv.preprocessed_dir y las máscaras
en ptv.masks_dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PTV.Codes.PTVCode.config import build_tracking_config, validate_config
from PTV.Codes.PTVCode.runner import run_ptv


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python RunCode/ptv_run.py RunCode/pipeline_config.json")

    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe config JSON: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    run_cfg = build_tracking_config(cfg)
    validate_config(run_cfg)

    run_ptv(run_cfg, raw_cfg=cfg)


if __name__ == "__main__":
    main()