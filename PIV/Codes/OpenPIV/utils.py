# utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
import imageio.v3 as iio


def ensure_folder(path: Path, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"No existe la carpeta {name}: {path}")
    if not path.is_dir():
        raise RuntimeError(f"No es carpeta {name}: {path}")


def clear_txt_in_out_dir(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for p in out_dir.glob("*.txt"):
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] No pude borrar {p}: {e}")
    return deleted


def read_gray(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def pair_indices(n: int) -> Iterator[Tuple[int, int, int]]:
    pid = 0
    m = (n // 2) * 2
    for i in range(0, m, 2):
        yield pid, i, i + 1
        pid += 1


def whiten_masked_background(frame: np.ndarray, mask_union: np.ndarray, mask_threshold: float) -> np.ndarray:
    out = frame.astype(np.float32, copy=True)
    if np.nanmax(np.abs(out)) < 1e-6:
        return np.ones_like(out, dtype=np.float32)

    lo, hi = np.percentile(out, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(out)), float(np.nanmax(out))
        if hi <= lo:
            return np.ones_like(out, dtype=np.float32)

    out = (out - lo) / (hi - lo + 1e-12)
    out = np.clip(out, 0.0, 1.0)
    out[mask_union > mask_threshold] = 1.0
    return out