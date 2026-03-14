# exporter.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json
import numpy as np
from tqdm import tqdm

from .models import PIVResultFinal
from .config import PIVConfig


def _load_metadata_for_timestamp(images_dir: Path, img_a_name: str) -> Optional[float]:
    """
    Cargar timestamp desde block_metadata.json usando el nombre de img_a
    
    Returns:
        timestamp_s o None si no se encuentra
    """
    metadata_path = images_dir / "block_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Buscar par que contiene esta imagen
        for pair in data.get("pairs", []):
            if pair.get("img1_filename") == img_a_name:
                # Calcular timestamp desde índice original y fps
                img1_idx = pair.get("img1_original_idx")
                if img1_idx is not None:
                    # Encontrar región para obtener fps
                    region_idx = pair.get("region_idx")
                    if region_idx is not None:
                        regions = data.get("regions", [])
                        if region_idx < len(regions):
                            fps = regions[region_idx].get("fps", 220.0)
                            return img1_idx / fps
        
        return None
        
    except Exception as e:
        print(f"[WARN] Error leyendo metadata para timestamp: {e}")
        return None


class TxtExporter:
    def export(self, finals: List[PIVResultFinal], names: List[str], cfg: PIVConfig) -> None:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, r in enumerate(tqdm(finals, desc="Exportando TXT", unit="archivo")):
            txt_path = cfg.out_dir / f"momento_{i:04d}.txt"
            
            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
            
            if cfg.export_full_grid:
                # MEJORA #3: exportar grilla completa
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
            
            # ================================================================
            # NUEVO: Calcular timestamp desde metadata
            # ================================================================
            timestamp_s = None
            if hasattr(r, 'img_a') and r.img_a:
                timestamp_s = _load_metadata_for_timestamp(cfg.images_dir, r.img_a.name)
            
            # ================================================================
            # Header completo con timestamp y dt_ms
            # ================================================================
            header = (
                f"# OpenPIV export (PIVlab-like)\n"
                f"# pair_id: {r.pair_id}\n"
                f"# source: {names[i]}\n"
            )
            
            # Agregar timestamp si está disponible
            if timestamp_s is not None:
                header += f"# timestamp_s: {timestamp_s:.6f}\n"
            
            header += (
                f"# DT_ms: {r.dt_ms}\n"
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