from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm

from ..config import MaskConfig
from .postprocess import postprocess_mask
from .utils import list_images


def process_image(model, img_path: Path, out_dir: Path, cfg: MaskConfig):

    img = Image.open(img_path).convert("L")

    arr = np.array(img)
    arr3 = np.stack([arr, arr, arr], axis=-1)

    results = model.predict(
        source=arr3,
        imgsz=1024,
        conf=cfg.conf_thresh,
        device=cfg.device,
        verbose=False
    )

    mask = np.zeros_like(arr)

    if results and results[0].masks is not None:

        masks = results[0].masks.data.cpu().numpy()

        for m in masks:
            mask = np.maximum(mask, (m > 0).astype(np.uint8) * 255)

    mask = postprocess_mask(mask)

    if cfg.invert_mask:
        mask = 255 - mask

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{img_path.stem}_mask.tiff"
    Image.fromarray(mask).save(out_path)


def run_mask_generation(cfg: MaskConfig):

    model = YOLO(cfg.model_path)

    subfolders = [p for p in cfg.images_dir.iterdir() if p.is_dir()]

    for sf in subfolders:

        out_sf = cfg.output_dir / sf.name

        files = list_images(sf)

        for f in tqdm(files, desc=sf.name):
            process_image(model, f, out_sf, cfg)