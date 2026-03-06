from pathlib import Path
import shutil
import re
from typing import List
from ..config import BlockConfig


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_files(folder: Path) -> List[Path]:
    files = [f for f in folder.iterdir() if f.is_file()]
    return sorted(files, key=lambda p: natural_key(p.name))


def process_subfolder(sub_dir: Path, cfg: BlockConfig):

    dst_sub = cfg.dest_root / sub_dir.name
    msk_sub = cfg.masks_root / sub_dir.name

    if dst_sub.exists() and cfg.delete_existing:
        shutil.rmtree(dst_sub)

    if msk_sub.exists() and cfg.delete_existing:
        shutil.rmtree(msk_sub)

    files = list_files(sub_dir)
    total = len(files)

    if total == 0:
        return

    block_size = cfg.block_size
    skip_inter = cfg.skip_inter

    blocks_possible = total // block_size

    if cfg.blocks is None:
        blocks_max = blocks_possible
    else:
        blocks_max = min(cfg.blocks, blocks_possible)

    i = 0
    blocks_done = 0

    while (i + block_size) <= total and blocks_done < blocks_max:

        idx1 = i
        idx2 = i + 1 + skip_inter

        for idx in (idx1, idx2):
            src = files[idx]
            dst = dst_sub / src.name

            dst_sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        i += block_size
        blocks_done += 1


def run_block_extraction(cfg: BlockConfig):

    subfolders = [p for p in cfg.base_dir.iterdir() if p.is_dir()]
    subfolders = sorted(subfolders, key=lambda p: natural_key(p.name))

    for sf in subfolders:
        process_subfolder(sf, cfg)