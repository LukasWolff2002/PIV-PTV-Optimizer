from pathlib import Path
import re


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path):

    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")

    files = []

    for ext in exts:
        files.extend(folder.glob(ext))

    return sorted(files, key=lambda p: natural_key(p.name))