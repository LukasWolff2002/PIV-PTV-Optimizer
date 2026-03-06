from __future__ import annotations
import json, sys, shutil
from pathlib import Path

def main():
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    c = cfg["cleanup"]

    if not c.get("delete_processed_subfolders", True):
        print("[CLEANUP] delete_processed_subfolders=False")
        return

    proc = Path(c["processed_dir_to_delete"])
    msk  = Path(c["masks_dir_to_delete"])

    if proc.exists():
        print(f"[CLEANUP] borrando: {proc}")
        shutil.rmtree(proc)

    if msk.exists():
        print(f"[CLEANUP] borrando: {msk}")
        shutil.rmtree(msk)

    if c.get("delete_predict_folders", False):
        runs = Path(c["runs_segment_dir"])
        if runs.exists():
            for p in runs.glob("predict*"):
                if p.is_dir():
                    print(f"[CLEANUP] borrando: {p}")
                    shutil.rmtree(p)

    print("[CLEANUP] listo.")

if __name__ == "__main__":
    main()