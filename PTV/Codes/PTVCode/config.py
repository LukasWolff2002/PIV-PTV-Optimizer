"""
config.py
=========
TrackingConfig: configuración inmutable del pipeline PTV.
Incluye build_tracking_config (desde JSON) y validate_config.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackingConfig:
    images_dir: Path          # imágenes PREPROCESADAS (PTVPreprocesadas/<sub>/)
    original_images_dir: Path # imágenes originales (para referencia en summary)
    out_dir: Path
    weights_path: Path
    runs_segment_dir: Path | None

    fps: float
    px_per_mm: float
    width_px: int
    height_px: int

    # ── Máscaras ────────────────────────────────────────────────
    apply_dynamic_mask: bool
    apply_static_mask: bool
    masks_dir: Path | None        # PTVMascaras/<sub>/ (dinámicas, ya generadas)
    fixed_mask_path: Path | None  # máscara fija (estática)
    mask_threshold: float         # umbral para binarizar máscara (default 127)

    preprocess_params: dict | None

    max_images: int | None
    alpha: float
    beta: float
    gamma: float

    gate_x_px: float
    gate_y_px: float
    gate_angle_deg: float

    conf: float
    min_frames_keep: int
    annotate: bool

    device: str | int | None = None
    max_misses: int = 0

    # ── Similarity Search Scheme ────────────────────────────────
    sim_threshold: float = 0.85
    max_dist_px: float = 80.0
    feat_weights: tuple = (1.0, 1.0, 0.5, 1.5, 1.5)
    l_ref_px: float = 101.4

    # ── SAHI inference ──────────────────────────────────────────
    sahi_scale_factor: int    = 4
    sahi_tile_size: int       = 640
    sahi_overlap_ratio: float = 0.5
    sahi_iou_threshold: float = 0.3

    # ── Skip de imágenes iniciales ───────────────────────────────
    # Aplicado ANTES del preproceso; el runner ya recibe imágenes limpias.
    skip_first_images: int = 0

    # ── Regiones temporales PTV ──────────────────────────────────
    use_temporal_regions: bool = False
    temporal_regions: list | None = None   # lista de dicts (from_dict disponible)

    # ── Visualizador ─────────────────────────────────────────────
    viz_tail_length: int  = 0
    viz_update_every: int = 1

    # ── Guardar imágenes ─────────────────────────────────────────
    save_images: bool = True

    @property
    def dt(self) -> float:
        """dt base (frames consecutivos). El runner lo sobreescribe por región."""
        return 1.0 / self.fps


def build_tracking_config(cfg: dict) -> TrackingConfig:
    """Construye TrackingConfig desde el dict del JSON de pipeline."""
    ptv = cfg["ptv"]
    cam = cfg["camera"]

    # images_dir apunta a las imágenes PREPROCESADAS
    images_dir = Path(ptv["preprocessed_dir"])
    original_images_dir = Path(ptv["images_dir"])

    # Máscaras dinámicas
    masks_dir = Path(ptv["masks_dir"]) if ptv.get("masks_dir") else None

    # Regiones temporales
    temporal_regions = ptv.get("temporal_regions", None)
    use_tr = bool(ptv.get("use_temporal_regions", False)) and temporal_regions is not None

    return TrackingConfig(
        images_dir=images_dir,
        original_images_dir=original_images_dir,
        out_dir=Path(ptv["out_dir"]),
        weights_path=Path(ptv["weights_path"]),
        runs_segment_dir=Path(ptv["runs_segment_dir"]) if ptv.get("runs_segment_dir") else None,
        fps=float(ptv["fps"]),
        px_per_mm=float(cam["px_per_mm"]),
        width_px=int(ptv["width_px"]),
        height_px=int(ptv["height_px"]),
        apply_dynamic_mask=bool(ptv.get("apply_dynamic_mask", False)),
        apply_static_mask=bool(ptv.get("apply_static_mask", False)),
        masks_dir=masks_dir,
        fixed_mask_path=Path(ptv["fixed_mask_path"]) if ptv.get("fixed_mask_path") else None,
        mask_threshold=float(ptv.get("mask_threshold", 127.0)),
        preprocess_params=ptv.get("preprocess_params"),
        max_images=ptv.get("max_images"),
        alpha=float(ptv["alpha"]),
        beta=float(ptv["beta"]),
        gamma=float(ptv["gamma"]),
        gate_x_px=float(ptv["gate_x_px"]),
        gate_y_px=float(ptv["gate_y_px"]),
        gate_angle_deg=float(ptv["gate_angle_deg"]),
        conf=float(ptv["conf"]),
        min_frames_keep=int(ptv["min_frames_keep"]),
        annotate=bool(ptv.get("annotate", False)),
        device=ptv.get("device", None),
        max_misses=int(ptv.get("max_misses", 0)),
        sim_threshold=float(ptv.get("sim_threshold", 0.85)),
        max_dist_px=float(ptv.get("max_dist_px", 80.0)),
        feat_weights=tuple(ptv.get("feat_weights", [1.0, 1.0, 0.5, 1.5, 1.5])),
        l_ref_px=float(ptv.get("l_ref_px", 101.4)),
        sahi_scale_factor=int(ptv.get("sahi_scale_factor", 4)),
        sahi_tile_size=int(ptv.get("sahi_tile_size", 640)),
        sahi_overlap_ratio=float(ptv.get("sahi_overlap_ratio", 0.5)),
        sahi_iou_threshold=float(ptv.get("sahi_iou_threshold", 0.3)),
        skip_first_images=int(cfg.get("meta", {}).get("skip_first_images", 0)),
        use_temporal_regions=use_tr,
        temporal_regions=temporal_regions,
        viz_tail_length=int(ptv.get("viz_tail_length", 0)),
        viz_update_every=int(ptv.get("viz_update_every", 1)),
        save_images=bool(ptv.get("save_images", True)),
    )


def validate_config(cfg: TrackingConfig) -> None:
    """Valida existencia de rutas y valores lógicos."""
    if not cfg.images_dir.exists():
        raise FileNotFoundError(
            f"preprocessed_dir no existe: {cfg.images_dir}\n"
            f"Asegúrate de que preprocess_run_ptv.py se ejecutó correctamente."
        )
    if not cfg.images_dir.is_dir():
        raise NotADirectoryError(f"preprocessed_dir no es carpeta: {cfg.images_dir}")
    if not cfg.weights_path.exists():
        raise FileNotFoundError(f"weights_path no existe: {cfg.weights_path}")
    if cfg.apply_dynamic_mask:
        if cfg.masks_dir is None:
            raise ValueError("apply_dynamic_mask=True pero masks_dir es None")
        if not cfg.masks_dir.exists():
            raise FileNotFoundError(
                f"masks_dir no existe: {cfg.masks_dir}\n"
                f"Asegúrate de que preprocess_run_ptv.py generó las máscaras."
            )
    if cfg.apply_static_mask:
        if cfg.fixed_mask_path is None:
            raise ValueError("apply_static_mask=True pero fixed_mask_path es None")
        if not cfg.fixed_mask_path.exists():
            raise FileNotFoundError(f"fixed_mask_path no existe: {cfg.fixed_mask_path}")
    if cfg.fps <= 0:
        raise ValueError("fps debe ser > 0")
    if cfg.px_per_mm <= 0:
        raise ValueError("px_per_mm debe ser > 0")
    if cfg.width_px <= 0 or cfg.height_px <= 0:
        raise ValueError("width_px y height_px deben ser > 0")
    if cfg.use_temporal_regions and not cfg.temporal_regions:
        raise ValueError("use_temporal_regions=True pero temporal_regions está vacío")