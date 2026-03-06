# exporter.py
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

from .models import PIVResultFinal
from .config import PIVConfig


class TxtExporter:
    def export(self, finals: List[PIVResultFinal], names: List[str], cfg: PIVConfig) -> None:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(tqdm(finals, desc="Exportando TXT", unit="archivo")):
            txt_path = cfg.out_dir / f"momento_{i:04d}.txt"

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)

            x = r.x_mm[valid].ravel()
            y = r.y_mm[valid].ravel()
            u = r.u_mms[valid].ravel()
            v = r.v_mms[valid].ravel()
            speed = np.sqrt(u*u + v*v)
            flag = np.ones_like(u, dtype=int)

            header = (
                f"# OpenPIV export (PIVlab-like)\n"
                f"# pair_id: {r.pair_id}\n"
                f"# source: {names[i]}\n"
                f"# DT_ms: {cfg.dt_ms}\n"
                f"# LM_KERNEL: {cfg.lm_kernel}, LM_THRESH: {cfg.lm_thresh}, LM_EPS: {cfg.lm_eps}\n"
                f"# KEEP_PERCENTILE: {cfg.keep_percentile}\n"
                f"# columns: x_mm y_mm u_mm_per_s v_mm_per_s speed_mm_per_s valid\n"
            )

            data = np.column_stack([x, y, u, v, speed, flag])

            np.savetxt(
                txt_path,
                data,
                fmt="%.6f %.6f %.6f %.6f %.6f %d",
                header=header,
                comments=""
            )