from __future__ import annotations
from pathlib import Path
import os, re, shutil
from typing import List, Optional

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_sorted_files(folder: Path, natural_sort: bool) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=(lambda p: natural_key(p.name)) if natural_sort else (lambda p: p.name))
    return files

def run_block_sampling(
    input_dir: Path,
    output_dir: Path,
    blocks: Optional[int],
    block_size: int,
    skip_inter: int,
    skip_final: int,
    delete_existing: bool,
    natural_sort: bool = True,
    overwrite: bool = True,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe input_dir: {input_dir}")

    if output_dir.exists() and delete_existing:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_sorted_files(input_dir, natural_sort)
    total = len(files)
    if total == 0:
        raise RuntimeError(f"No hay archivos en: {input_dir}")

    if (2 + skip_inter + skip_final) != block_size:
        raise ValueError(f"Regla inválida: 2 + {skip_inter} + {skip_final} != {block_size}")

    bloques_posibles = total // block_size
    blocks_max = bloques_posibles if blocks is None else min(int(blocks), bloques_posibles)

    i = 0
    hechos = 0
    while (i + block_size) <= total and hechos < blocks_max:
        idx1 = i
        idx2 = i + 1 + skip_inter

        for idx in (idx1, idx2):
            src = files[idx]
            dst = output_dir / src.name
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)

        i += block_size
        hechos += 1

    print(f"[PRE] {input_dir.name}: bloques {hechos}/{blocks_max} -> {output_dir}")