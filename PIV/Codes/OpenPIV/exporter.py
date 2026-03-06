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

            if cfg.export_full_grid:
                # MEJORA #3: exportar grilla completa (misma cantidad de filas en todos los TXT)
                # Permite apilar momentos directamente con np.loadtxt sin alinear índices.
                # flag=1 => válido, flag=0 => enmascarado o outlier
                x = r.x_mm.ravel()
                y = r.y_mm.ravel()
                u = r.u_mms.ravel()
                v = r.v_mms.ravel()
                speed = np.where(valid.ravel(), np.sqrt(u**2 + v**2), np.nan)
                flag = valid.ravel().astype(int)
            else:
                # Comportamiento anterior: solo filas válidas
                x = r.x_mm[valid].ravel()
                y = r.y_mm[valid].ravel()
                u = r.u_mms[valid].ravel()
                v = r.v_mms[valid].ravel()
                speed = np.sqrt(u**2 + v**2)
                flag = np.ones_like(u, dtype=int)

            # MEJORA #6: header completo con todos los parámetros PIV relevantes
            header = (
                f"# OpenPIV export (PIVlab-like)\n"
                f"# pair_id: {r.pair_id}\n"
                f"# source: {names[i]}\n"
                f"# DT_ms: {cfg.dt_ms}\n"
                f"# px_per_mm: {cfg.px_per_mm}\n"
                f"# window_sizes: {cfg.window_sizes}\n"
                f"# overlaps: {cfg.overlaps}\n"
                f"# search_area_factor: {cfg.search_area_factor}\n"
                f"# sig2noise_method: {cfg.sig2noise_method}\n"
                f"# keep_percentile: {cfg.keep_percentile}\n"
                f"# lm_kernel: {cfg.lm_kernel}  lm_thresh: {cfg.lm_thresh}  lm_eps: {cfg.lm_eps}\n"
                f"# replace_outliers_kernel: {cfg.replace_outliers_kernel}  max_iter: {cfg.replace_outliers_max_iter}\n"
                f"# export_full_grid: {cfg.export_full_grid}\n"
                f"# columns: x_mm y_mm u_mm_per_s v_mm_per_s speed_mm_per_s valid\n"
            )

            data = np.column_stack([x, y, u, v, speed, flag])

            np.savetxt(
                txt_path,
                data,
                fmt="%.6f %.6f %.6f %.6f %.6f %d",
                header=header,
                comments="",
            )