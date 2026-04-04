from __future__ import annotations
import json, sys, shutil
from pathlib import Path

def delete_safely(path_str):
    if path_str:
        p = Path(path_str)
        if p.exists() and p.is_dir():
            print(f"[CLEANUP] borrando: {p}")
            shutil.rmtree(p)

def main():
    if len(sys.argv) < 2: return
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    c = cfg.get("cleanup", {})

    # 1. Limpieza de PIV (solo si está habilitado)
    if c.get("delete_processed_subfolders", True):
        delete_safely(c.get("processed_dir_to_delete"))
        delete_safely(c.get("masks_dir_to_delete"))

    # 2. Limpieza de PTV (las carpetas temporales preprocesadas)
    delete_safely(c.get("ptv_preprocessed_to_delete"))
    delete_safely(c.get("ptv_masks_to_delete"))

    # 3. Limpieza de carpetas de predicción de YOLO (runs/segment)
    if c.get("delete_predict_folders", False):
        runs = c.get("runs_segment_dir")
        if runs and Path(runs).exists():
            for p in Path(runs).glob("predict*"):
                if p.is_dir():
                    print(f"[CLEANUP] borrando predict YOLO: {p}")
                    shutil.rmtree(p)

    print("[CLEANUP] listo.")

if __name__ == "__main__":
    main()