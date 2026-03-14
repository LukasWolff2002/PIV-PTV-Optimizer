# timestamp_utils.py
"""
Utilidades para calcular timestamps correctos desde block_metadata.json
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import json


def load_timestamps_from_metadata(images_dir: Path) -> Dict[str, float]:
    """
    Cargar timestamps desde block_metadata.json
    
    Returns:
        Dict con key=img_filename -> timestamp_s
    """
    metadata_path = images_dir / "block_metadata.json"
    
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timestamps = {}
        
        # Mapear cada imagen a su timestamp
        for pair in data.get("pairs", []):
            img1_filename = pair.get("img1_filename")
            img2_filename = pair.get("img2_filename")
            img1_idx = pair.get("img1_original_idx")
            img2_idx = pair.get("img2_original_idx")
            region_idx = pair.get("region_idx")
            
            if img1_filename and img1_idx is not None and region_idx is not None:
                # Obtener fps de la región
                regions = data.get("regions", [])
                if region_idx < len(regions):
                    fps = regions[region_idx].get("fps", 220.0)
                    
                    # Calcular timestamps
                    timestamp1 = img1_idx / fps
                    timestamp2 = img2_idx / fps if img2_idx is not None else timestamp1
                    
                    timestamps[img1_filename] = timestamp1
                    if img2_filename:
                        timestamps[img2_filename] = timestamp2
        
        return timestamps
        
    except Exception as e:
        print(f"[WARN] Error cargando timestamps desde metadata: {e}")
        return {}


def get_timestamp_for_result(result, timestamps: Dict[str, float]) -> Optional[float]:
    """
    Obtener timestamp para un resultado PIV
    
    Args:
        result: PIVResult o PIVResultFinal
        timestamps: Dict de img_filename -> timestamp_s
    
    Returns:
        timestamp_s o None
    """
    if hasattr(result, 'img_a') and result.img_a:
        return timestamps.get(result.img_a.name)
    return None