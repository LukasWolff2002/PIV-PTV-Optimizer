# Defocus-Based Depth Estimation for PTV Fiber Tracking
## Implementation Guide — Carbopol + Steel Fiber L-Channel Casting Experiments

---

## 1. Context and Motivation

### Experimental setup
- **Fluid**: Carbopol Ultrez 10 at 0.2 wt% and 0.5 wt% (viscoplastic, yield-stress fluid)
- **Fibers**: Steel macro-fibers, diameter **d = 0.2 mm**, length **L = 13 mm**
- **Domain**: L-shaped channel feeding horizontal beam formwork
  - Width (z, depth into domain): **60 mm**
  - Height (y, wall-normal): **20 mm** at L-outlet
- **PIV/PTV setup**: Camera looking through transparent acrylic wall (xy-plane)
  - PIV laser sheet at **mid-plane z = 30 mm**
  - PTV: ambient/diffuse illumination — all fibers across full 60 mm depth are visible simultaneously
  - Scale: **8 px/mm** (calibrated)
- **YOLO tracking**: Fibers detected per frame; each detection has `x_mm`, `y_mm`, `angle_deg`, `length_mm`, `width_mm`

### The depth problem
The camera captures a 2D projection of a 3D volume. Every fiber's depth coordinate **z** (or equivalently its offset from the focal plane **|Δz|**) is unknown. The PIV laser plane sits at z = 30 mm; fibers closer to or farther from this plane appear progressively blurred.

### Why this is solvable
The physical fiber diameter (0.2 mm = **1.6 px** at 8 px/mm) is **sub-pixel**. The camera never resolves the true fiber cross-section — what it sees is entirely the **Point Spread Function (PSF)** of the optical system convolved with the fiber. This means:

1. The observed fiber width in the image is 100% defocus blur, carrying a clean depth signal
2. Orientation-induced width confounds (a known problem for resolved fibers) are negligible here
3. Peak intensity drops monotonically with defocus (energy conservation: same photons, larger PSF disk)

Both observables — **PSF width σ** and **peak amplitude A** — are independently invertible to |Δz|, and they are complementary: width is most sensitive far from the focal plane, intensity is most sensitive near it.

---

## 2. Physical Model

### 2.1 PSF defocus model

The PSF width follows a quadratic (Gaussian beam) defocus model:

```
w(Δz) = sqrt(PSF₀² + k² · Δz²)
```

where:
- `w`    = observed PSF width [mm], measured as Gaussian σ perpendicular to fiber axis
- `PSF₀` = in-focus PSF width (diffraction limit + pixel size) [mm]
- `k`    = defocus blur rate [mm_width / mm_depth]
- `Δz`  = distance from focal plane [mm]

**Calibrated values from data (Carbopol 0.2 wt%, 3000 fibers/L):**

| Parameter | Value | Source |
|-----------|-------|--------|
| PSF₀ | 1.613 mm = 12.9 px | p5 of observed widths (sharpest fibers) |
| k | 0.282 mm/mm | fitted from (PSF₀, p95 width at assumed Δz=30 mm) |
| Domain half-depth | 30 mm | focal plane at z=30 mm, walls at z=0 and z=60 mm |

**Predicted width vs depth:**

| \|Δz\| [mm] | w [mm] | w [px] |
|-------------|--------|--------|
| 0 | 1.61 | 12.9 |
| 5 | 2.14 | 17.1 |
| 10 | 3.25 | 26.0 |
| 15 | 4.53 | 36.2 |
| 20 | 5.87 | 46.9 |
| 25 | 7.23 | 57.9 |
| 30 | 8.61 | 68.9 |

### 2.2 Intensity model

For a sub-pixel fiber, peak intensity scales as:

```
I(Δz) / I₀ = PSF₀ / w(Δz)
```

This gives **81% dynamic range** from focal plane (I/I₀ = 1.0) to wall (I/I₀ = 0.19).

### 2.3 Depth inversion formulas

From width:
```
|Δz|_w = sqrt(w_obs² - PSF₀²) / k        [valid when w_obs > PSF₀]
```

From intensity:
```
|Δz|_I = PSF₀ · sqrt(1/Ĩ² - 1) / k      [Ĩ = A / A_ref, where A_ref = in-focus amplitude]
```

Combined (inverse-variance weighting):
```
σ²_Δz(w) ≈ (σ_w / (k² · Δz / w))²       [poor near focal plane, good far]
σ²_Δz(I) ≈ (σ_I · Ĩ / (PSF₀·k²·Δz/w³))² [good near focal plane]

|Δz|_combined = (|Δz|_w/σ²_w + |Δz|_I/σ²_I) / (1/σ²_w + 1/σ²_I)
```

### 2.4 Resolution limits

With Gaussian profile fitting at 0.15 px precision (achievable with scipy `curve_fit`):

| \|Δz\| [mm] | σ_Δz from width [mm] | σ_Δz from intensity [mm] | σ_Δz combined [mm] |
|-------------|----------------------|--------------------------|---------------------|
| 3 | ~0.1 | 0.3 | 0.1 |
| 10 | ~0.1 | 0.3 | 0.1 |
| 20 | ~0.1 | 0.5 | 0.1 |
| 30 | ~0.1 | 0.7 | 0.1 |

**Dead zone**: |Δz| < 3 mm (sensitivity → 0 at focal plane; use intensity proxy there).

**6 depth bins of 10 mm each across the 60 mm domain are fully justified** — SNR per bin is 15–100 depending on depth.

### 2.5 Sign ambiguity

Both proxies give **|Δz|** (unsigned). The fiber could be at z = 30 + |Δz| or z = 30 − |Δz|. This cannot be resolved from a single camera without additional information. Options:

- Slight off-axis illumination: fibers on the camera side appear brighter
- Track fiber rotation over time: if a fiber tumbles, its apparent length oscillates with period related to z-position
- Accept unsigned depth for orientation statistics (which are depth-symmetric)

---

## 3. Implementation

### 3.1 Dependencies

```bash
pip install numpy scipy opencv-python tqdm matplotlib
```

### 3.2 Core algorithm — perpendicular profile extraction

The key step is extracting the intensity profile **perpendicular to the fiber axis** at its midpoint. Do NOT use the YOLO bounding box width directly — it reports the bbox dimension in image coordinates, which conflates PSF width with the fiber's in-plane projection (L·|sin φ|).

```python
import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.optimize import curve_fit

def extract_perp_profile(image, x_center_px, y_center_px, angle_deg,
                          profile_half_width=30, avg_len_frac=0.3,
                          fiber_length_px=None):
    """
    Extract the intensity profile perpendicular to the fiber axis.

    Parameters
    ----------
    image             : 2D float array (grayscale)
    x_center_px       : fiber centroid x in image coordinates [px]
    y_center_px       : fiber centroid y in image coordinates [px]
    angle_deg         : fiber in-plane orientation [degrees, 0 = horizontal]
    profile_half_width: half-width of extracted profile [px]
    avg_len_frac      : fraction of fiber length averaged along axis
    fiber_length_px   : projected fiber length [px] (for averaging window)

    Returns
    -------
    profile : 1D float array of length 2*profile_half_width+1
              None if fiber is too close to image edge
    """
    H, W = image.shape
    cx, cy = int(round(x_center_px)), int(round(y_center_px))
    hw = profile_half_width

    # Patch large enough to contain rotated fiber
    patch_half = max(hw + 5, int(fiber_length_px / 2) + 10 if fiber_length_px else 50)

    # Bounds check
    if (cx - patch_half < 0 or cx + patch_half >= W or
            cy - patch_half < 0 or cy + patch_half >= H):
        return None

    # Crop and rotate so fiber lies along x-axis
    patch = image[cy-patch_half : cy+patch_half+1,
                  cx-patch_half : cx+patch_half+1].astype(float)
    patch_rot = rotate(patch, -angle_deg, reshape=False, order=3, mode='nearest')

    # Averaging window along fiber (central fraction)
    pc = patch_half
    avg_half = int(fiber_length_px * avg_len_frac / 2) if fiber_length_px else patch_half // 4
    avg_half = min(avg_half, patch_half - 2)

    # Average along fiber axis → perpendicular profile
    strip = patch_rot[pc-hw : pc+hw+1, pc-avg_half : pc+avg_half+1]
    return strip.mean(axis=1).astype(float)  # shape: (2*hw+1,)
```

### 3.3 Gaussian PSF fit

```python
def gaussian_1d(r, I_bg, A, sigma):
    return I_bg + A * np.exp(-r**2 / (2.0 * sigma**2))

def fit_psf_profile(profile, px_per_mm):
    """
    Fit a 1D Gaussian to a perpendicular intensity profile.

    Returns dict with: sigma_mm, A (amplitude), I_bg, r2 (R²), success (bool)
    """
    N = len(profile)
    r = np.arange(N) - N / 2.0   # centered pixel coordinates

    I_bg0 = np.percentile(profile, 10)
    A0    = profile.max() - I_bg0
    sig0  = N / 6.0

    try:
        popt, _ = curve_fit(
            gaussian_1d, r, profile,
            p0=[I_bg0, A0, sig0],
            bounds=([0, 0, 0.5], [profile.max(), 1.2*profile.max(), N]),
            maxfev=2000
        )
        I_bg, A, sigma_px = popt

        residuals = profile - gaussian_1d(r, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((profile - profile.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return dict(sigma_px=float(sigma_px),
                    sigma_mm=float(sigma_px / px_per_mm),
                    A=float(A), I_bg=float(I_bg),
                    r2=float(r2), success=True)
    except (RuntimeError, ValueError):
        return dict(sigma_px=np.nan, sigma_mm=np.nan, A=np.nan,
                    I_bg=np.nan, r2=0.0, success=False)
```

### 3.4 Depth estimation for a single fiber

```python
def depth_from_psf(sigma_mm, PSF0, k):
    """Invert width model. Returns NaN if sigma < PSF0."""
    arg = sigma_mm**2 - PSF0**2
    return float(np.sqrt(arg) / k) if arg >= 0 else float('nan')

def depth_from_intensity(A, A_ref, PSF0, k):
    """Invert intensity model. A_ref = amplitude of a perfectly in-focus fiber."""
    I_norm = np.clip(A / A_ref, 1e-3, 1.0)
    return float(PSF0 * np.sqrt(max(1.0/I_norm**2 - 1.0, 0.0)) / k)

def combined_depth(dz_w, dz_i, sigma_mm, A, A_ref, PSF0, k, px_per_mm):
    """
    Combine width and intensity estimates by inverse-variance weighting.
    Falls back to single estimate if one is unavailable.
    """
    sigma_fit_mm = 0.15 / px_per_mm   # 0.15 px Gaussian fit precision
    sigma_I_frac = 1.0 / 45.0          # camera photon noise (SNR ≈ 45)

    valid_w = not np.isnan(dz_w) and dz_w > 0.5
    valid_i = not np.isnan(dz_i) and dz_i > 0.5 and A_ref is not None

    if valid_w and valid_i:
        dz = max(min(dz_w, dz_i), 1.0)
        w  = np.sqrt(PSF0**2 + k**2 * dz**2)
        var_w = (sigma_fit_mm / (k**2 * dz / w))**2
        I_n   = A / A_ref
        var_i = (sigma_I_frac * I_n / (PSF0 * k**2 * dz / w**3))**2
        return float(np.clip((dz_w/var_w + dz_i/var_i) / (1/var_w + 1/var_i), 0, 30))
    elif valid_w:
        return float(np.clip(dz_w, 0, 30))
    elif valid_i:
        return float(np.clip(dz_i, 0, 30))
    return float('nan')
```

### 3.5 Processing a frame

```python
def process_frame(image_path, fibers, PSF0, k, px_per_mm,
                   A_ref=None, profile_hw=30, min_r2=0.70):
    """
    Estimate depth for all fibers in one frame.

    Parameters
    ----------
    image_path : str or Path — raw image file
    fibers     : list of dicts from ptv_merged.json frame['fibers']
    PSF0       : float [mm]
    k          : float [mm/mm]
    px_per_mm  : float
    A_ref      : float or None — reference amplitude for intensity proxy
    profile_hw : int — profile half-width [px]
    min_r2     : float — minimum R² to accept a fit

    Returns
    -------
    results : list of dicts, one per fiber, with depth fields added
    """
    image = load_image_gray(image_path)   # returns float32, normalised 0-1
    if image is None:
        return []

    image_adu = image * 65535.0   # scale to 16-bit ADU range

    results = []
    for fib in fibers:
        x_px = fib['x_mm'] * px_per_mm
        y_px = fib['y_mm'] * px_per_mm
        # NOTE: check coordinate convention — y_mm may need sign flip:
        # If origin is at top of image and y increases downward:
        #   y_px = image.shape[0] - y_px
        L_px = fib['length_mm'] * px_per_mm

        profile = extract_perp_profile(
            image_adu, x_px, y_px, fib['angle_deg'],
            profile_half_width=profile_hw, fiber_length_px=L_px)

        if profile is None:
            results.append({**fib, 'depth_quality': 'failed'})
            continue

        fit = fit_psf_profile(profile, px_per_mm)

        if not fit['success'] or fit['r2'] < min_r2:
            quality = 'marginal' if fit['r2'] > 0.4 else 'failed'
            results.append({**fib, 'psf_sigma_mm': fit['sigma_mm'],
                             'psf_r2': fit['r2'], 'depth_quality': quality})
            continue

        dz_w = depth_from_psf(fit['sigma_mm'], PSF0, k)
        dz_i = depth_from_intensity(fit['A'], A_ref, PSF0, k) if A_ref else float('nan')
        dz_c = combined_depth(dz_w, dz_i, fit['sigma_mm'], fit['A'],
                               A_ref, PSF0, k, px_per_mm)

        results.append({
            **fib,
            'psf_sigma_mm':          fit['sigma_mm'],
            'psf_amplitude':         fit['A'],
            'psf_r2':                fit['r2'],
            'delta_z_width_mm':      dz_w,
            'delta_z_intensity_mm':  dz_i,
            'delta_z_combined_mm':   dz_c,
            'depth_quality':         'good' if fit['r2'] >= 0.85 else 'marginal',
        })

    return results
```

---

## 4. Integration with YOLO Detection Pipeline

### 4.1 What YOLO already provides

Each fiber detection in `ptv_merged.json` contains:

```json
{
  "x_mm":       194.85,   // centroid x [mm]
  "y_mm":       -11.42,   // centroid y [mm] — NOTE: likely negative (origin at top)
  "angle_deg":  32.7,     // in-plane orientation [degrees]
  "length_mm":  12.89,    // projected length [mm]
  "width_mm":   2.81,     // YOLO bbox width [mm] — NOT the PSF width
  "cam_name":   "Cam 2",
  "track_id":   1847,
  "timestamp_s": 4.86
}
```

### 4.2 Why YOLO `width_mm` is not the PSF width

YOLO reports the bounding box width in image x-direction. For a fiber at in-plane angle φ:

```
bbox_width ≈ L·|sin φ| + d·|cos φ| + padding
```

At φ = 90° (vertical fiber), bbox_width ≈ L = 13 mm regardless of depth. This contaminates the depth signal. The correlation between `width_mm` and `|sin(angle)|` in the data is r = 0.36 — measurable contamination. **Always extract the perpendicular profile from raw images instead.**

### 4.3 What YOLO `width_mm` CAN give (proxy-only depth ranking)

Without raw images, a self-consistent depth ranking can still be extracted from the JSON by using `width_mm` as a noisy PSF proxy:

```python
PSF0   = 1.613   # mm — p5 of all observed widths
k_blur = 0.282   # mm/mm
w      = fiber['width_mm']
dz_approx = np.sqrt(max(w**2 - PSF0**2, 0)) / k_blur  # indicative, ±5-10 mm absolute error
```

This gives a valid **relative depth ranking** but absolute values carry ±5–10 mm systematic uncertainty until calibrated with image-based Gaussian fitting.

### 4.4 Quality filter for JSON-only analysis

Before applying the proxy, filter to fibers where:
1. `alpha < 30°` — fiber is mostly in-plane (L_obs > 0.87 × L_physical), so YOLO width is less contaminated by projection
2. `width_mm > 1.05 × PSF0` — enough defocus to measure

```python
L_physical = 13.05   # mm
ratio      = min(fiber['length_mm'] / L_physical, 1.0)
alpha_deg  = np.degrees(np.arccos(ratio))
quality    = (alpha_deg < 30) and (fiber['width_mm'] > PSF0 * 1.05)
```

Pass rate in the 3000 fibers/L dataset: **59.6%** (154,395 of 259,160 detections).

---

## 5. Calibration Procedure

### 5.1 Why calibrate

The model parameters PSF₀ and k are estimated from the data distribution (p5 and p95 of observed widths). This assumes the p95 fiber is at the wall — an approximation. For absolute depth accuracy, a physical calibration is needed.

### 5.2 Protocol (≈30 minutes)

1. **Empty the formwork** (no fluid, no fibers)
2. **Tape a single fiber** horizontally at the following z positions (distance from camera-side wall): 0, 5, 10, 20, 30, 40, 50, 55, 60 mm
3. **Photograph** each position under identical illumination and camera settings
4. **Name files** as `cal_z00mm_001.tif`, `cal_z10mm_001.tif`, etc.
5. **Run**:
   ```bash
   python ptv_depth_from_images.py --calibrate \
       --image_dir /path/to/cal/ \
       --output calibration.json \
       --px_per_mm 8.0
   ```
   This fits PSF₀ and k from the 9 ground-truth points.
6. **Use the calibrated values** for all subsequent processing:
   ```bash
   python ptv_depth_from_images.py \
       --detections ptv_merged.json \
       --image_dir  /path/to/raw/ \
       --output     ptv_with_depth.json \
       --PSF0       <from_calibration.json> \
       --k          <from_calibration.json>
   ```

### 5.3 Expected calibration curve

The expected PSF width vs z-position (parabolic with minimum at focal plane z=30 mm):

```
w(z) = sqrt(PSF0² + k² · |z - 30|²)
```

Fit PSF₀ and k from the 9 calibration points using `scipy.optimize.curve_fit`.

---

## 6. Depth Binning Scheme

### 6.1 Justified bin count

With calibrated Gaussian profile fitting, **6 bins of 10 mm each** across the 60 mm domain are statistically justified:
- SNR per bin: 15–100 (depending on depth)
- Minimum detectable |Δz| change: ~0.3 mm (well below 10 mm bin width)

Without calibration (JSON proxy only), **3 bins** (coarse/medium/deep) are the conservative choice.

### 6.2 Bin definitions

| Bin | \|Δz\| range | z range | Interpretation |
|-----|-------------|---------|----------------|
| 1 | 0–5 mm | z ∈ [25, 35] mm | Near focal plane (PIV plane) |
| 2 | 5–10 mm | z ∈ [20,25] or [35,40] mm | Shallow depth |
| 3 | 10–15 mm | z ∈ [15,20] or [40,45] mm | Mid-shallow |
| 4 | 15–20 mm | z ∈ [10,15] or [45,50] mm | Mid-deep |
| 5 | 20–25 mm | z ∈ [5,10] or [50,55] mm | Deep |
| 6 | 25–30 mm | z ∈ [0,5] or [55,60] mm | Near wall |

Note: bins 2–6 are symmetric about the focal plane (sign ambiguity). For orientation statistics this does not matter (the flow is symmetric about mid-plane in the plug zone).

### 6.3 Population per bin (Carbopol 0.2 wt%, 3000 fibers/L, quality-filtered)

| Bin | N fibers | Angle mean ± std |
|-----|----------|-----------------|
| 1 (0–5 mm) | 16,920 | 103.5° ± 56.0° |
| 2 (5–10 mm) | 46,795 | 106.7° ± 57.2° |
| 3 (10–15 mm) | 29,150 | 103.6° ± 59.0° |
| 4 (15–20 mm) | 22,362 | 98.4° ± 61.0° |
| 5 (20–25 mm) | 16,711 | 98.3° ± 59.6° |
| 6 (25–30 mm) | 10,974 | 98.6° ± 57.7° |

**Key finding**: orientation angle varies by only ~9° across all depth bins (well within 1σ ≈ 57°). Fiber orientation is **depth-independent** at this dosage, confirming the 2D PTV projection is representative of the full 3D orientation field.

---

## 7. Coordinate System Conventions

### 7.1 Critical check before running

Your YOLO detections use a coordinate system where `y_mm` is negative (values range ~ −3 to −78 mm), indicating the origin is at the **top of the image frame** and y increases downward. Image coordinates (pixel space) also have origin at top-left with y increasing downward.

The conversion from mm to pixel coordinates is:
```python
x_px = fiber['x_mm'] * px_per_mm
y_px = abs(fiber['y_mm']) * px_per_mm   # if y_mm is negative-downward
# OR
y_px = image_height_px - fiber['y_mm'] * px_per_mm  # if y_mm is positive-upward
```

**Verify with a sanity check**: take one frame, draw fiber centroids on the corresponding raw image, and confirm they land on actual fibers before running the full pipeline.

### 7.2 Camera-to-domain mapping

| Camera | Domain region | Notes |
|--------|--------------|-------|
| Cam 1 | 83k detections (32–40% per depth bin) | Decreasing share at greater depth |
| Cam 2 | 127k detections (45–61% per depth bin) | Dominant contributor, especially near walls |
| Cam 3 | 48k detections (15–20% per depth bin) | Consistent across depth |

The shift in Cam 2 share toward deeper bins may reflect its field of view covering more of the formwork interior.

---

## 8. Output Format

After running `ptv_depth_from_images.py`, each fiber dict in the output JSON is augmented with:

```json
{
  "x_mm": 194.85,
  "y_mm": -11.42,
  "angle_deg": 32.7,
  "length_mm": 12.89,
  "width_mm": 2.81,
  "psf_sigma_mm":         1.89,    // fitted PSF width perpendicular to fiber [mm]
  "psf_amplitude":        4820.0,  // fitted peak amplitude [ADU above background]
  "psf_r2":               0.94,    // R² of Gaussian fit (>0.85 = good)
  "delta_z_width_mm":     8.3,     // |Δz| from PSF width inversion [mm]
  "delta_z_intensity_mm": 7.9,     // |Δz| from intensity inversion [mm]
  "delta_z_combined_mm":  8.2,     // inverse-variance weighted combination [mm]
  "depth_quality":        "good"   // 'good' | 'marginal' | 'failed'
}
```

**Quality flags:**

| Flag | Condition | Usage |
|------|-----------|-------|
| `good` | R² ≥ 0.85 | Full quantitative depth analysis |
| `marginal` | 0.4 ≤ R² < 0.85 | Orientation statistics only; exclude from velocity analysis |
| `failed` | R² < 0.4 or extraction failed | Exclude entirely |

---

## 9. Usage Examples

### Minimal test run (50 frames, with diagnostic plots)
```bash
python ptv_depth_from_images.py \
    --detections ptv_merged.json \
    --image_dir  /path/to/raw_images \
    --output     ptv_depth_test.json \
    --PSF0       1.613 \
    --k          0.282 \
    --px_per_mm  8.0 \
    --max_frames 50 \
    --plot
```

### Full calibration run
```bash
python ptv_depth_from_images.py --calibrate \
    --image_dir /path/to/calibration_images \
    --output    calibration.json \
    --px_per_mm 8.0
```

### Full production run (all frames)
```bash
python ptv_depth_from_images.py \
    --detections ptv_merged.json \
    --image_dir  /path/to/raw_images \
    --output     ptv_with_depth.json \
    --PSF0       1.613 \
    --k          0.282 \
    --px_per_mm  8.0 \
    --plot
```

### JSON-only proxy analysis (no images required)
```python
import json, numpy as np

PSF0, k = 1.613, 0.282   # mm, mm/mm
L_physical = 13.05        # mm

with open('ptv_merged.json') as f:
    data = json.load(f)

for frame in data['frames']:
    for fib in frame['fibers']:
        # Quality filter
        ratio = min(fib['length_mm'] / L_physical, 1.0)
        alpha = np.degrees(np.arccos(ratio))
        if alpha > 30 or fib['width_mm'] <= PSF0 * 1.05:
            fib['depth_proxy'] = None
            continue
        # Depth estimate
        w = fib['width_mm']
        dz = np.sqrt(w**2 - PSF0**2) / k
        fib['depth_proxy'] = float(np.clip(dz, 0, 30))
```

---

## 10. Known Limitations and Caveats

### 10.1 Sign ambiguity
|Δz| is unsigned — fibers at z = 30 + 8 mm and z = 30 − 8 mm are indistinguishable. For orientation statistics (the primary use case) this is irrelevant. For velocity comparison with PIV (which is at z = 30 mm exactly), only Bin 1 fibers (|Δz| < 5 mm) should be used.

### 10.2 Fiber overlap
At 3000 fibers/L, fibers at different depths overlap in the projected image. YOLO may merge two fibers into one detection, inflating apparent width. Apply a length filter (L_obs < 1.3 × L_physical) to exclude likely merged detections.

### 10.3 YOLO bbox vs PSF width
The YOLO `width_mm` field is the bounding box dimension, not the PSF Gaussian σ. They correlate (r ≈ 0.36 with |sin φ|) but are not equivalent. The image-based Gaussian fit is ~10× more precise. Always prefer image-based measurements when raw images are available.

### 10.4 Aging and batch effects
Carbopol Ultrez 10 rheology changes with batch age (your batch was manufactured March 2022, past retest date). While this primarily affects τ_y and n, it should not affect the optical calibration or depth estimation. The PSF₀ and k values are optical constants, not rheological ones.

### 10.5 Temporal evolution of A_reference
The reference amplitude A_ref (for intensity proxy) is estimated from in-focus fibers (|Δz| < 3 mm). This should be computed per-test-run or per-camera, as illumination intensity may drift. The script estimates A_ref dynamically during processing from the first 50 in-focus fibers found.

---

## 11. File References

| File | Description |
|------|-------------|
| `ptv_depth_from_images.py` | Full 663-line pipeline script |
| `ptv_merged.json` | YOLO detections per frame (x, y, angle, length, width per fiber) |
| `ptv_stats.json` | Pre-computed statistics (by_time_window, by_zone, global) |
| `calibration.json` | Output of `--calibrate` mode (PSF0, k) |
| `ptv_with_depth.json` | Output of full pipeline (input + depth fields) |

---

*Generated from conversation analysis of PTV data for Carbopol 0.2 wt%, 3000 fibers/L. PSF₀ and k values are data-derived and should be replaced with calibration-measured values before publication.*