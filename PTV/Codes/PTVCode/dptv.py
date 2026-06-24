"""
dptv.py — Depth-from-Defocus PTV (DPTV) for rod-like particles.

Uses the apparent widening of fibers of known physical width to estimate
the depth offset (Δz) of each fiber from the camera focal plane.

Physical model (thin-lens, paraxial):

    W_apparent² = W_ideal² + (k_blur × Δz)²

    W_apparent [px] : measured fiber width from YOLO segmentation + PCA
    W_ideal    [px] : w_ideal_px if set, else fiber_width_mm × px_per_mm
    k_blur [px/mm]  : blur growth rate — calibrated from known-depth images
    Δz     [mm]     : depth offset from focal plane (magnitude, always ≥ 0)

Outputs per detection (always computed, calibration not required):
    defocus_score   — W_apparent / W_ideal  [1.0 = in focus, >1 = defocused]
    blur_px         — sqrt(max(0, W_apparent² - W_ideal²))  [px]
    blur_mm         — blur_px / px_per_mm  [mm], proportional to Δz

Depth estimate (requires k_blur calibration):
    depth_mm        — |Δz| = blur_px / k_blur  [mm, ≥ 0], or None if uncalibrated
    depth_confidence — quality metric in [0, 1]:
                       tanh(blur_px / noise_px)
                       ≈0 when blur ≤ noise (in focus or indistinguishable from noise)
                       →1 for large blur (strongly defocused, reliable estimate)

Calibration of k_blur:
    Place fibers at known depths Δz_i from the focal plane, measure apparent
    widths W_i from YOLO/PCA, compute:
        σ_i = sqrt(W_i² - W_ideal²)
        k_blur = mean(σ_i / Δz_i)  [px/mm]
    Typical values: 0.5–10 px/mm depending on DOF and lens aperture.

Integration with PIV:
    The PIV laser sheet is at a fixed depth (typically close to the focal plane).
    For each fiber, `depth_mm` gives the distance from the focal plane; if the
    focal plane coincides with the laser sheet, depth_mm directly tells how far
    the fiber is from the PIV measurement plane.

    Use `depth_confidence` to weight PIV-fiber velocity correlations:
      - High confidence (strongly defocused) → fiber is far from PIV plane
        → velocity correlation is unreliable
      - Low confidence (near focus) → fiber is close to PIV plane
        → velocity correlation is most meaningful

    Suggested filter for PIV correlation:
        near_focus = (defocus_score < defocus_threshold) &
                     (depth_mm < max_depth_from_laser_mm)
"""
from __future__ import annotations
import math
from dataclasses import dataclass


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class DPTVConfig:
    """
    Configuration for the Depth-from-Defocus estimator.

    Attributes:
        fiber_width_mm    : known physical fiber width in mm (0.2 for standard)
        fiber_length_mm   : known physical fiber length in mm (13.0 for standard)
        noise_width_px    : expected PCA width measurement noise in pixels.
                            Governs the confidence metric; set from empirical
                            measurement or as ~10% of W_ideal_px.
        k_blur_px_per_mm  : blur growth rate [px of blur per mm of depth offset].
                            Set to None to disable depth_mm output (defocus_score
                            and blur_mm are always computed regardless).
    """
    fiber_width_mm:   float        = 0.2
    fiber_length_mm:  float        = 13.0
    noise_width_px:   float        = 1.0
    k_blur_px_per_mm: float | None = None
    w_ideal_px:       float | None = None  # override: empirical optical in-focus width [px]


# ─────────────────────────────────────────────
# ESTIMATOR
# ─────────────────────────────────────────────

class DPTVEstimator:
    """
    Per-detection depth estimator from defocus blur.

    Usage:
        cfg  = DPTVConfig(fiber_width_mm=0.2, k_blur_px_per_mm=2.5)
        dptv = DPTVEstimator(cfg)
        info = dptv.estimate(width_px=3.8, px_per_mm=8.0)
    """

    def __init__(self, cfg: DPTVConfig):
        self.cfg = cfg

    # ── Core estimation ───────────────────────────────────────────

    def estimate(self, width_px: float, px_per_mm: float) -> dict:
        """
        Compute depth metrics for one fiber detection.

        Args:
            width_px  : apparent fiber minor-axis width from PCA [px]
            px_per_mm : camera spatial calibration constant [px/mm]

        Returns:
            defocus_score    — W_apparent / W_ideal [dimensionless, ≥ 0]
            blur_px          — defocus blur contribution [px, ≥ 0]
            blur_mm          — defocus blur in physical units [mm, ≥ 0]
            depth_mm         — |Δz| from focal plane [mm] or None if uncalibrated
            depth_confidence — quality of the depth estimate [0, 1]
        """
        W_ideal_px = (self.cfg.w_ideal_px
                      if self.cfg.w_ideal_px is not None and self.cfg.w_ideal_px > 0
                      else self.cfg.fiber_width_mm * px_per_mm)

        if W_ideal_px <= 0:
            return _null_result()

        # Defocus ratio: > 1 means wider than ideal → defocused
        defocus_score = width_px / W_ideal_px

        # Blur width in quadrature subtraction
        w2_excess = max(0.0, width_px ** 2 - W_ideal_px ** 2)
        blur_px   = math.sqrt(w2_excess)
        blur_mm   = blur_px / px_per_mm if px_per_mm > 0 else 0.0

        # Depth estimate (requires calibration constant k_blur)
        k = self.cfg.k_blur_px_per_mm
        depth_mm = (blur_px / k) if (k is not None and k > 0) else None

        # Confidence: SNR-like, saturates to 1 for large blur
        noise = max(self.cfg.noise_width_px, 1e-9)
        depth_confidence = math.tanh(blur_px / noise)

        return {
            "defocus_score":    defocus_score,
            "blur_px":          blur_px,
            "blur_mm":          blur_mm,
            "depth_mm":         depth_mm,
            "depth_confidence": depth_confidence,
        }

    # ── Track-level depth statistics ─────────────────────────────

    @staticmethod
    def track_depth_stats(history: list) -> dict:
        """
        Aggregate depth metrics across all records in a track's history.

        Args:
            history: list of TrackRecord objects

        Returns:
            dict with per-track depth summary suitable for JSON export.
        """
        if not history:
            return _empty_track_stats()

        defocus_vals = [r.defocus_score    for r in history]
        blur_vals    = [r.depth_blur_mm    for r in history]
        conf_vals    = [r.depth_confidence for r in history]
        depth_vals   = [r.depth_mm for r in history if r.depth_mm is not None]

        n = len(history)
        n_depth = len(depth_vals)

        return {
            "n_observations":       n,
            "n_depth_estimated":    n_depth,
            "mean_defocus_score":   _mean(defocus_vals),
            "min_defocus_score":    min(defocus_vals),
            "max_defocus_score":    max(defocus_vals),
            "mean_blur_mm":         _mean(blur_vals),
            "std_blur_mm":          _std(blur_vals),
            "mean_depth_mm":        _mean(depth_vals) if depth_vals else None,
            "std_depth_mm":         _std(depth_vals)  if len(depth_vals) > 1 else None,
            "min_depth_mm":         min(depth_vals)   if depth_vals else None,
            "max_depth_mm":         max(depth_vals)   if depth_vals else None,
            "mean_depth_confidence": _mean(conf_vals),
        }

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "DPTVEstimator":
        """Build from a plain dict (e.g. parsed from pipeline_config.json)."""
        w_ideal = d.get("w_ideal_px")
        cfg = DPTVConfig(
            fiber_width_mm=float(d.get("fiber_width_mm", 0.2)),
            fiber_length_mm=float(d.get("fiber_length_mm", 13.0)),
            noise_width_px=float(d.get("noise_width_px", 1.0)),
            k_blur_px_per_mm=d.get("k_blur_px_per_mm"),
            w_ideal_px=float(w_ideal) if w_ideal is not None else None,
        )
        return cls(cfg)

    def to_dict(self) -> dict:
        """Serialize configuration for JSON export."""
        return {
            "enabled":           True,
            "fiber_width_mm":    self.cfg.fiber_width_mm,
            "fiber_length_mm":   self.cfg.fiber_length_mm,
            "noise_width_px":    self.cfg.noise_width_px,
            "k_blur_px_per_mm":  self.cfg.k_blur_px_per_mm,
            "w_ideal_px":        self.cfg.w_ideal_px,
            "depth_calibrated":  self.cfg.k_blur_px_per_mm is not None,
        }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _null_result() -> dict:
    return {
        "defocus_score":    1.0,
        "blur_px":          0.0,
        "blur_mm":          0.0,
        "depth_mm":         None,
        "depth_confidence": 0.0,
    }


def _empty_track_stats() -> dict:
    return {
        "n_observations":        0,
        "n_depth_estimated":     0,
        "mean_defocus_score":    None,
        "min_defocus_score":     None,
        "max_defocus_score":     None,
        "mean_blur_mm":          None,
        "std_blur_mm":           None,
        "mean_depth_mm":         None,
        "std_depth_mm":          None,
        "min_depth_mm":          None,
        "max_depth_mm":          None,
        "mean_depth_confidence": None,
    }


def _mean(vals: list) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(vals: list) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
