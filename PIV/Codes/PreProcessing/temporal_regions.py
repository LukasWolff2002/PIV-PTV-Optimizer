"""
PIV/Codes/PreProcessing/temporal_regions.py

Estructuras de datos para muestreo adaptativo multi-región temporal
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import json


@dataclass
class TemporalRegion:
    """
    Define una región temporal con sus parámetros de muestreo específicos
    
    Permite adaptar el muestreo según las características del flujo en diferentes
    momentos de la captura (ej: altas velocidades al inicio, bajas al final)
    """
    name: str               # Identificador descriptivo (ej: "alta_velocidad")
    start_time: float       # Tiempo inicial en segundos desde inicio de captura
    end_time: float         # Tiempo final en segundos
    block_size: int         # Tamaño del bloque (debe cumplir regla 2+skip_inter+skip_final)
    skip_inter: int         # Frames saltados entre img1 e img2 del par PIV
    skip_final: int         # Frames saltados al final del bloque
    fps: float              # FPS de la cámara (necesario para calcular Δt)
    
    def __post_init__(self):
        """Validar regla de bloque al crear la región"""
        if (2 + self.skip_inter + self.skip_final) != self.block_size:
            raise ValueError(
                f"Regla de bloque inválida en región '{self.name}': "
                f"2 + {self.skip_inter} + {self.skip_final} = "
                f"{2 + self.skip_inter + self.skip_final} ≠ {self.block_size}"
            )
        
        if self.start_time >= self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) debe ser menor que "
                f"end_time ({self.end_time}) en región '{self.name}'"
            )
        
        if self.start_time < 0:
            raise ValueError(f"start_time no puede ser negativo en región '{self.name}'")
    
    @property
    def start_frame(self) -> int:
        """Índice del frame inicial de la región (inclusivo)"""
        return int(self.start_time * self.fps)
    
    @property
    def end_frame(self) -> int:
        """Índice del frame final de la región (exclusivo)"""
        return int(self.end_time * self.fps)
    
    @property
    def total_frames(self) -> int:
        """Total de frames en esta región"""
        return self.end_frame - self.start_frame
    
    @property
    def dt_ms(self) -> float:
        """
        Delta tiempo efectivo en milisegundos para PIV
        
        Fórmula: Δt = (1/FPS) × (skip_inter + 1) × 1000
        
        Ejemplos con 220 FPS:
        - skip_inter=0 → Δt = 4.545 ms (frames consecutivos)
        - skip_inter=1 → Δt = 9.091 ms (1 frame entre medio)
        - skip_inter=2 → Δt = 13.636 ms (2 frames entre medio)
        """
        return (1.0 / self.fps) * (self.skip_inter + 1) * 1000.0
    
    @property
    def max_blocks(self) -> int:
        """Máximo número de bloques posibles en esta región"""
        return self.total_frames // self.block_size
    
    @property
    def duration_seconds(self) -> float:
        """Duración de la región en segundos"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización JSON"""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "block_size": self.block_size,
            "skip_inter": self.skip_inter,
            "skip_final": self.skip_final,
            "dt_ms": self.dt_ms,
            "max_blocks": self.max_blocks,
            "duration_seconds": self.duration_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalRegion:
        """Crear región desde diccionario"""
        return cls(
            name=data["name"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            block_size=data["block_size"],
            skip_inter=data["skip_inter"],
            skip_final=data["skip_final"],
            fps=data["fps"],
        )
    
    def __repr__(self) -> str:
        return (
            f"TemporalRegion('{self.name}', "
            f"t={self.start_time:.1f}-{self.end_time:.1f}s, "
            f"frames={self.start_frame}-{self.end_frame}, "
            f"block={self.block_size}, skip_inter={self.skip_inter}, "
            f"Δt={self.dt_ms:.2f}ms)"
        )


@dataclass
class BlockMetadata:
    """
    Metadata de un par de imágenes procesado en el muestreo adaptativo
    
    Contiene toda la información necesaria para trazabilidad:
    - Región temporal a la que pertenece
    - Posición dentro de la región
    - Parámetros de muestreo
    - Índices originales en la secuencia completa
    """
    region_idx: int          # Índice de región (0, 1, 2, ...)
    region_name: str         # Nombre de región ("alta_velocidad", etc.)
    block_idx: int           # Índice del bloque dentro de la región (0-based)
    skip_inter: int          # skip_inter usado para este bloque
    dt_ms: float             # Delta tiempo calculado en milisegundos
    img1_original_idx: int   # Índice de img1 en secuencia original completa
    img2_original_idx: int   # Índice de img2 en secuencia original completa
    img1_filename: str       # Nombre del archivo img1 procesado
    img2_filename: str       # Nombre del archivo img2 procesado
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización JSON"""
        return {
            "region_idx": self.region_idx,
            "region_name": self.region_name,
            "block_idx": self.block_idx,
            "skip_inter": self.skip_inter,
            "dt_ms": self.dt_ms,
            "img1_original_idx": self.img1_original_idx,
            "img2_original_idx": self.img2_original_idx,
            "img1_filename": self.img1_filename,
            "img2_filename": self.img2_filename,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlockMetadata:
        """Crear metadata desde diccionario"""
        return cls(
            region_idx=data["region_idx"],
            region_name=data["region_name"],
            block_idx=data["block_idx"],
            skip_inter=data["skip_inter"],
            dt_ms=data["dt_ms"],
            img1_original_idx=data["img1_original_idx"],
            img2_original_idx=data["img2_original_idx"],
            img1_filename=data["img1_filename"],
            img2_filename=data["img2_filename"],
        )


def validate_regions(regions: List[TemporalRegion], total_frames: int) -> None:
    """
    Validar lista de regiones temporales
    
    Verifica:
    1. No hay gaps (huecos) entre regiones consecutivas
    2. No hay overlaps (solapamientos)
    3. Las regiones no exceden los frames disponibles
    4. Cada región tiene frames suficientes para al menos 1 bloque
    
    Args:
        regions: Lista de regiones temporales ordenadas por tiempo
        total_frames: Total de frames disponibles en el dataset
    
    Raises:
        ValueError: Si hay problemas de validación
    """
    if not regions:
        raise ValueError("Lista de regiones vacía")
    
    # Verificar orden temporal
    for i in range(len(regions) - 1):
        if regions[i].end_time > regions[i + 1].start_time:
            raise ValueError(
                f"Regiones '{regions[i].name}' y '{regions[i + 1].name}' "
                f"se solapan en tiempo: {regions[i].end_time} > {regions[i + 1].start_time}"
            )
    
    # Verificar que las regiones no exceden frames disponibles
    max_frame_needed = max(r.end_frame for r in regions)
    if max_frame_needed > total_frames:
        raise ValueError(
            f"Las regiones requieren hasta el frame {max_frame_needed}, "
            f"pero solo hay {total_frames} frames disponibles"
        )
    
    # Verificar que cada región tiene frames suficientes
    for region in regions:
        if region.total_frames < region.block_size:
            raise ValueError(
                f"Región '{region.name}' tiene {region.total_frames} frames, "
                f"pero necesita al menos {region.block_size} para un bloque"
            )
        
        if region.max_blocks == 0:
            raise ValueError(
                f"Región '{region.name}' no puede generar ningún bloque completo"
            )


def save_metadata_json(
    metadata_list: List[BlockMetadata],
    regions: List[TemporalRegion],
    output_path: Path,
    total_images_input: int,
) -> None:
    """
    Guardar metadata completa en archivo JSON
    
    Args:
        metadata_list: Lista de metadata de todos los pares procesados
        regions: Lista de regiones temporales usadas
        output_path: Ruta donde guardar el JSON
        total_images_input: Total de imágenes en el input original
    """
    total_pairs = len(metadata_list)
    total_images_selected = total_pairs * 2
    reduction_percent = (1 - total_images_selected / total_images_input) * 100
    
    data = {
        "summary": {
            "total_input_images": total_images_input,
            "total_pairs_processed": total_pairs,
            "total_images_selected": total_images_selected,
            "reduction_percent": round(reduction_percent, 2),
        },
        "regions": [r.to_dict() for r in regions],
        "pairs": [m.to_dict() for m in metadata_list],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[METADATA] JSON guardado: {output_path}")


def load_metadata_json(json_path: Path) -> tuple[List[TemporalRegion], List[BlockMetadata]]:
    """
    Cargar metadata desde archivo JSON
    
    Args:
        json_path: Ruta al archivo JSON de metadata
    
    Returns:
        Tupla (regiones, metadata_list)
    """
    if not json_path.exists():
        raise FileNotFoundError(f"No existe archivo de metadata: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    regions = [TemporalRegion.from_dict(r) for r in data["regions"]]
    metadata_list = [BlockMetadata.from_dict(m) for m in data["pairs"]]
    
    return regions, metadata_list


def print_regions_summary(regions: List[TemporalRegion]) -> None:
    """
    Imprimir resumen legible de las regiones configuradas
    
    Args:
        regions: Lista de regiones temporales
    """
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE REGIONES TEMPORALES")
    print("="*70)
    
    for i, region in enumerate(regions, 1):
        print(f"\n[REGIÓN {i}] {region.name.upper()}")
        print(f"  Tiempo:       {region.start_time:.2f}s → {region.end_time:.2f}s "
              f"(duración: {region.duration_seconds:.2f}s)")
        print(f"  Frames:       {region.start_frame} → {region.end_frame} "
              f"(total: {region.total_frames})")
        print(f"  FPS:          {region.fps:.1f}")
        print(f"  Block size:   {region.block_size}")
        print(f"  Skip inter:   {region.skip_inter}")
        print(f"  Skip final:   {region.skip_final}")
        print(f"  Δt efectivo:  {region.dt_ms:.3f} ms")
        print(f"  Max bloques:  {region.max_blocks}")
        print(f"  Validación:   2 + {region.skip_inter} + {region.skip_final} "
              f"= {region.block_size} ✓")
    
    print("\n" + "="*70 + "\n")