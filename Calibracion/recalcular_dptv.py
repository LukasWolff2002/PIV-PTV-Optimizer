"""
recalcular_dptv.py — Re-procesa resultados PTV existentes con nueva calibración DPTV
======================================================================================

Aplica nuevos parámetros DPTV (k_blur, W_ideal) a los archivos ya procesados
por el pipeline sin necesidad de volver a correr YOLO ni el tracking.

Actualiza por carpeta de resultados:
  - detections.csv  — campos DPTV por detección
  - tracks.csv      — campos DPTV por frame de track
  - tracks.json     — historia + depth_stats por track
  - summary.json    — sección dptv con estadísticas globales

Uso rápido:
    python recalcular_dptv.py --k-blur 2.5
    python recalcular_dptv.py --k-blur 2.5 --w-ideal-px 6.0
    python recalcular_dptv.py --k-blur-cam1 2.5 --k-blur-cam2 2.3 --k-blur-cam3 2.3

    # Probar sin modificar archivos:
    python recalcular_dptv.py --k-blur 2.5 --dry-run

    # Sin crear backup (si ya tienes los originales en git):
    python recalcular_dptv.py --k-blur 2.5 --no-backup

Cómo calibrar k_blur con un experimento
-----------------------------------------
1. Colocar una fibra exactamente en el plano focal de la cámara (z = 0 mm).
   Medir el ancho aparente promedio → ese es W_ideal_px para esa cámara.
   Ej: W_ideal_px = 6.0 → --w-ideal-px 6.0

2. Colocar fibras a profundidades conocidas (z = 5, 10, 15, 20, 30 mm).
   Para cada z_i, medir width_px medio en las detecciones:
     σ_i = sqrt(width_px_i² − W_ideal_px²)
     k_blur = media(σ_i / z_i)   [px/mm]

3. Actualizar DPTV_K_BLUR_PX_PER_MM en RunCode/variables_ptv.py.

4. Ejecutar este script para re-procesar todos los resultados anteriores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "ResultadosPTV" / "ResultadosPTV"

# Umbral para marcar detecciones como "cerca del foco" en summary.json
NEAR_FOCUS_DEFOCUS_THR = 1.5
NEAR_FOCUS_CONF_THR    = 0.5


# ── Fórmulas DPTV (idénticas a dptv.py) ──────────────────────────────────────

def _dptv_estimate(
    width_px:     float,
    px_per_mm:    float,
    w_ideal_px:   float,
    k_blur:       Optional[float],
    noise_px:     float,
) -> dict:
    if w_ideal_px <= 0 or width_px <= 0:
        return dict(defocus_score=1.0, depth_blur_px=0.0,
                    depth_blur_mm=0.0, depth_mm=None, depth_confidence=0.0)

    defocus_score = width_px / w_ideal_px
    w2_excess     = max(0.0, width_px ** 2 - w_ideal_px ** 2)
    blur_px       = math.sqrt(w2_excess)
    blur_mm       = blur_px / px_per_mm if px_per_mm > 0 else 0.0
    depth_mm      = (blur_px / k_blur) if (k_blur is not None and k_blur > 0) else None
    depth_conf    = math.tanh(blur_px / max(noise_px, 1e-9))

    return dict(
        defocus_score    = defocus_score,
        depth_blur_px    = blur_px,
        depth_blur_mm    = blur_mm,
        depth_mm         = depth_mm,
        depth_confidence = depth_conf,
    )


def _mean(vals):
    return sum(vals) / len(vals) if vals else None

def _std(vals):
    if len(vals) < 2: return None
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ── Lectura de configuración de carpeta ──────────────────────────────────────

def _load_folder_meta(folder: Path) -> Optional[dict]:
    """Lee summary.json y retorna (cam, px_per_mm, fiber_width_mm, noise_px)."""
    p = folder / "summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    return {
        "cam":            s["camera"]["cam"],
        "px_per_mm":      s["camera"]["px_per_mm"],
        "fiber_width_mm": s.get("dptv", {}).get("config", {}).get("fiber_width_mm", 0.2),
        "noise_px":       s.get("dptv", {}).get("config", {}).get("noise_width_px", 1.0),
    }


# ── Backup ────────────────────────────────────────────────────────────────────

def _backup(folder: Path, filenames: list[str]) -> Path:
    backup_dir = folder / f"_backup_dptv_{date.today().isoformat()}"
    backup_dir.mkdir(exist_ok=True)
    for name in filenames:
        src = folder / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    return backup_dir


# ── Actualización de archivos ─────────────────────────────────────────────────

def _update_detections_csv(
    folder: Path, px_per_mm: float, w_ideal_px: float,
    k_blur: Optional[float], noise_px: float, dry_run: bool
) -> list[dict]:
    """Recomputa campos DPTV en detections.csv. Devuelve lista de resultados."""
    path = folder / "detections.csv"
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    results = []
    updated = []
    for row in rows:
        try:
            wpx = float(row["width_px"])
        except (KeyError, ValueError):
            updated.append(row)
            continue

        est = _dptv_estimate(wpx, px_per_mm, w_ideal_px, k_blur, noise_px)
        row["defocus_score"]    = f"{est['defocus_score']:.4f}"
        row["depth_blur_px"]    = f"{est['depth_blur_px']:.4f}"
        row["depth_blur_mm"]    = f"{est['depth_blur_mm']:.4f}"
        row["depth_mm"]         = f"{est['depth_mm']:.4f}" if est["depth_mm"] is not None else ""
        row["depth_confidence"] = f"{est['depth_confidence']:.4f}"
        updated.append(row)
        results.append(est)

    if not dry_run:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated)

    return results


def _update_tracks_csv(
    folder: Path, px_per_mm: float, w_ideal_px: float,
    k_blur: Optional[float], noise_px: float, dry_run: bool
) -> None:
    path = folder / "tracks.csv"
    if not path.exists():
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = []
    for row in rows:
        try:
            # width_mm es el ancho aparente de la fibra en mm
            wpx = float(row["width_mm"]) * px_per_mm
        except (KeyError, ValueError):
            updated.append(row)
            continue

        est = _dptv_estimate(wpx, px_per_mm, w_ideal_px, k_blur, noise_px)
        row["defocus_score"]    = f"{est['defocus_score']:.4f}"
        row["depth_blur_px"]    = f"{est['depth_blur_px']:.4f}"
        row["depth_blur_mm"]    = f"{est['depth_blur_mm']:.4f}"
        row["depth_mm"]         = f"{est['depth_mm']:.4f}" if est["depth_mm"] is not None else ""
        row["depth_confidence"] = f"{est['depth_confidence']:.4f}"
        updated.append(row)

    if not dry_run:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated)


def _recompute_track_depth_stats(history: list) -> dict:
    """Reagrupa estadísticas de profundidad de un track desde su historia."""
    defocus_vals = []
    blur_vals    = []
    conf_vals    = []
    depth_vals   = []

    for h in history:
        defocus_vals.append(h.get("defocus_score", 1.0) or 1.0)
        blur_vals.append(h.get("depth_blur_mm", 0.0) or 0.0)
        conf_vals.append(h.get("depth_confidence", 0.0) or 0.0)
        dm = h.get("depth_mm")
        if dm is not None:
            depth_vals.append(dm)

    n = len(history)
    return {
        "n_observations":        n,
        "n_depth_estimated":     len(depth_vals),
        "mean_defocus_score":    _mean(defocus_vals),
        "min_defocus_score":     min(defocus_vals) if defocus_vals else None,
        "max_defocus_score":     max(defocus_vals) if defocus_vals else None,
        "mean_blur_mm":          _mean(blur_vals),
        "std_blur_mm":           _std(blur_vals),
        "mean_depth_mm":         _mean(depth_vals) if depth_vals else None,
        "std_depth_mm":          _std(depth_vals)  if len(depth_vals) > 1 else None,
        "min_depth_mm":          min(depth_vals)   if depth_vals else None,
        "max_depth_mm":          max(depth_vals)   if depth_vals else None,
        "mean_depth_confidence": _mean(conf_vals),
    }


def _update_tracks_json(
    folder: Path, px_per_mm: float, w_ideal_px: float,
    k_blur: Optional[float], noise_px: float,
    fiber_width_mm: float, dry_run: bool
) -> None:
    path = folder / "tracks.json"
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Actualizar metadata DPTV
    data["metadata"]["dptv"] = {
        "enabled":          True,
        "fiber_width_mm":   fiber_width_mm,
        "fiber_length_mm":  data["metadata"].get("dptv", {}).get("fiber_length_mm", 13.0),
        "noise_width_px":   noise_px,
        "k_blur_px_per_mm": k_blur,
        "depth_calibrated": k_blur is not None,
    }

    for track in data["tracks"]:
        history = track.get("history", [])
        for h in history:
            # width_mm es el ancho aparente en mm
            wm = h.get("width_mm")
            if wm is None:
                continue
            wpx = float(wm) * px_per_mm
            est = _dptv_estimate(wpx, px_per_mm, w_ideal_px, k_blur, noise_px)
            h["defocus_score"]    = est["defocus_score"]
            h["depth_blur_px"]    = est["depth_blur_px"]
            h["depth_blur_mm"]    = est["depth_blur_mm"]
            h["depth_mm"]         = est["depth_mm"]
            h["depth_confidence"] = est["depth_confidence"]

        track["depth_stats"] = _recompute_track_depth_stats(history)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def _update_summary_json(
    folder: Path, cam: int, px_per_mm: float, w_ideal_px: float,
    k_blur: Optional[float], noise_px: float, fiber_width_mm: float,
    det_results: list[dict], dry_run: bool
) -> None:
    path = folder / "summary.json"
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        s = json.load(f)

    defocus_vals = [r["defocus_score"]    for r in det_results]
    blur_vals    = [r["depth_blur_mm"]    for r in det_results]
    conf_vals    = [r["depth_confidence"] for r in det_results]
    depth_vals   = [r["depth_mm"] for r in det_results if r["depth_mm"] is not None]

    n = len(det_results)
    near_focus_n = sum(
        1 for r in det_results
        if r["defocus_score"] < NEAR_FOCUS_DEFOCUS_THR
        and r["depth_confidence"] < NEAR_FOCUS_CONF_THR
    )

    s["dptv"] = {
        "enabled": True,
        "config": {
            "enabled":          True,
            "fiber_width_mm":   fiber_width_mm,
            "fiber_length_mm":  s.get("dptv", {}).get("config", {}).get("fiber_length_mm", 13.0),
            "noise_width_px":   noise_px,
            "k_blur_px_per_mm": k_blur,
            "depth_calibrated": k_blur is not None,
        },
        "n_records_total":      s.get("dptv", {}).get("n_records_total", n),
        "n_depth_estimated":    len(depth_vals),
        "mean_defocus_score":   _mean(defocus_vals),
        "std_defocus_score":    _std(defocus_vals),
        "mean_blur_mm":         _mean(blur_vals),
        "mean_depth_confidence": _mean(conf_vals),
        "pct_near_focus":       100.0 * near_focus_n / n if n > 0 else 0.0,
        "mean_depth_mm":        _mean(depth_vals) if depth_vals else None,
        "std_depth_mm":         _std(depth_vals)  if len(depth_vals) > 1 else None,
        "recalibrated": {
            "w_ideal_px":  w_ideal_px,
            "k_blur":      k_blur,
            "noise_px":    noise_px,
            "date":        date.today().isoformat(),
        },
    }

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False, allow_nan=False)


# ── Procesamiento de una carpeta ──────────────────────────────────────────────

def process_folder(
    folder: Path,
    k_blur_by_cam: dict,
    w_ideal_px_override: Optional[float],
    noise_px_global: float,
    dry_run: bool,
    no_backup: bool,
) -> bool:
    meta = _load_folder_meta(folder)
    if meta is None:
        print(f"  [skip] sin summary.json")
        return False

    cam           = meta["cam"]
    px_per_mm     = meta["px_per_mm"]
    fiber_width_mm = meta["fiber_width_mm"]
    noise_px      = noise_px_global if noise_px_global is not None else meta["noise_px"]

    k_blur = k_blur_by_cam.get(cam, k_blur_by_cam.get("global"))

    if w_ideal_px_override is not None:
        w_ideal_px = w_ideal_px_override
    else:
        w_ideal_px = fiber_width_mm * px_per_mm

    print(f"  cam-{cam}  px_per_mm={px_per_mm}  w_ideal={w_ideal_px:.2f}px"
          f"  k_blur={k_blur}  noise={noise_px}")

    if not no_backup and not dry_run:
        bd = _backup(folder, ["detections.csv", "tracks.csv", "tracks.json", "summary.json"])
        print(f"  Backup → {bd.name}/")

    det_results = _update_detections_csv(
        folder, px_per_mm, w_ideal_px, k_blur, noise_px, dry_run)

    _update_tracks_csv(
        folder, px_per_mm, w_ideal_px, k_blur, noise_px, dry_run)

    _update_tracks_json(
        folder, px_per_mm, w_ideal_px, k_blur, noise_px,
        fiber_width_mm, dry_run)

    if det_results:
        _update_summary_json(
            folder, cam, px_per_mm, w_ideal_px, k_blur, noise_px,
            fiber_width_mm, det_results, dry_run)

    # Resumen rápido
    if det_results:
        depth_vals = [r["depth_mm"] for r in det_results if r["depth_mm"] is not None]
        mean_d = _mean(depth_vals)
        pct_d  = 100.0 * len(depth_vals) / len(det_results)
        print(f"  → {len(det_results)} dets | depth estimado: {pct_d:.0f}% "
              + (f"| mean_depth={mean_d:.1f}mm" if mean_d else ""))

    action = "[DRY-RUN] no se modificaron archivos" if dry_run else "Archivos actualizados."
    print(f"  {action}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-procesa resultados PTV existentes con nueva calibración DPTV."
    )
    p.add_argument("--k-blur", type=float, default=None,
                   help="k_blur global (px/mm) para todas las cámaras")
    p.add_argument("--k-blur-cam1", type=float, default=None)
    p.add_argument("--k-blur-cam2", type=float, default=None)
    p.add_argument("--k-blur-cam3", type=float, default=None)
    p.add_argument("--k-blur-cam4", type=float, default=None)
    p.add_argument("--w-ideal-px", type=float, default=None,
                   help="Sobreescribir W_ideal_px (ancho en foco medido empíricamente). "
                        "Si no se provee, usa fiber_width_mm × px_per_mm.")
    p.add_argument("--noise-px", type=float, default=None,
                   help="Ruido de medición de ancho en px (default: el del summary.json)")
    p.add_argument("--folder", type=str, default=None,
                   help="Procesar solo esta carpeta (relativa a ResultadosPTV/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Mostrar resultados sin modificar archivos")
    p.add_argument("--no-backup", action="store_true",
                   help="No crear backup de archivos originales")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Construir mapa de k_blur por cámara
    k_blur_by_cam: dict = {}
    if args.k_blur is not None:
        k_blur_by_cam["global"] = args.k_blur
    for cam, val in [(1, args.k_blur_cam1), (2, args.k_blur_cam2),
                     (3, args.k_blur_cam3), (4, args.k_blur_cam4)]:
        if val is not None:
            k_blur_by_cam[cam] = val

    if not k_blur_by_cam:
        print("ERROR: especificar --k-blur o --k-blur-camX")
        return

    if not RESULTS_DIR.exists():
        print(f"No encontrado: {RESULTS_DIR}")
        return

    # Carpetas a procesar
    if args.folder:
        folders = [RESULTS_DIR / args.folder]
    else:
        folders = sorted(d for d in RESULTS_DIR.iterdir()
                         if d.is_dir() and (d / "summary.json").exists())

    if not folders:
        print("No se encontraron carpetas de resultados PTV.")
        return

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Recalculando DPTV en {len(folders)} carpeta(s)...")
    print(f"  k_blur_by_cam = {k_blur_by_cam}")
    print(f"  w_ideal_px_override = {args.w_ideal_px}")
    print()

    ok = 0
    for folder in folders:
        print(f"── {folder.name}")
        success = process_folder(
            folder        = folder,
            k_blur_by_cam = k_blur_by_cam,
            w_ideal_px_override = args.w_ideal_px,
            noise_px_global     = args.noise_px,
            dry_run             = args.dry_run,
            no_backup           = args.no_backup,
        )
        if success:
            ok += 1
        print()

    print(f"Completado: {ok}/{len(folders)} carpetas procesadas.")
    if not args.dry_run:
        print("Siguiente paso: re-ejecutar los scripts de análisis (Analisis/PTV/).")
    else:
        print("Ejecutar sin --dry-run para aplicar los cambios.")


if __name__ == "__main__":
    main()
