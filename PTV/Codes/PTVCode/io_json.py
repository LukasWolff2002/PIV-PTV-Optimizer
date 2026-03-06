# io_json.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def to_json_compatible(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unsupported type: {type(obj)}")


def make_auto_json_name(images_dir: Path) -> str:
    """
    Nombre JSON automático según carpeta de imágenes.
    - default: <foldername>.json
    - regla extra: si foldername empieza con 'm', lo convierte a 'im' (ej: m72-... -> im72-...)
    """
    folder = images_dir.name.strip()

    if folder.lower().startswith("m") and len(folder) > 1:
        # m72-... -> im72-...
        folder = "i" + folder

    return f"{folder}.json"


def save_tracks_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=to_json_compatible)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_min_frames(data: dict, min_frames: int) -> dict:
    filtered = {}
    for k in ["ruta", "fibras_por_frame"]:
        if k in data:
            filtered[k] = data[k]

    for tid, tdata in data.items():
        if tid in ["ruta", "fibras_por_frame"]:
            continue
        frames = tdata.get("frame", [])
        if len(frames) >= min_frames:
            filtered[tid] = tdata
    return filtered