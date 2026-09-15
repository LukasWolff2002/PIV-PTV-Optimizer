from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from .models import PIVResult, PIVResultFinal
from .config import PIVConfig
from .validation import velocity_region_mask
from .timestamp_utils import load_timestamps_from_metadata, get_timestamp_for_result


# ===============================================================
# Estilo unificado
# ===============================================================

STYLE = {
    "figure_bg": "#ffffff",
    "panel_bg": "#f4f4f4",
    "card_bg": "#ffffff",
    "axes_bg": "#ffffff",
    "spine": "#000000",
    "text": "#000000",
    # Paleta Okabe–Ito: distinguible con daltonismo
    "valid": "#009E73",
    "invalid": "#D55E00",
    "accent": "#0072B2",
    "slider_track": "#d9d9d9",
    "button": "#e6e6e6",
    "button_hover": "#cccccc",
    "zero": "#000000",
    "vorticity_cmap": "RdBu_r",
    "speed_cmap": "viridis",
}

FONT_SERIF = ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"]

def _finite_values(a):
    """Devuelve un array normal 1D solo con valores válidos (sin máscara, sin NaN/inf)."""
    if np.ma.isMaskedArray(a):
        a = a.compressed()          # descarta los valores enmascarados
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]

def _setup_matplotlib_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": STYLE["figure_bg"],
        "axes.facecolor": STYLE["axes_bg"],
        "axes.edgecolor": STYLE["spine"],
        "axes.linewidth": 1.0,
        "axes.labelcolor": STYLE["text"],
        "axes.titlecolor": STYLE["text"],
        "axes.grid": False,
        "xtick.color": STYLE["text"],
        "ytick.color": STYLE["text"],
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#000000",
        "legend.framealpha": 0.92,
    })


def _style_axes(ax, equal: bool = False) -> None:
    ax.set_facecolor(STYLE["axes_bg"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(STYLE["spine"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(labelsize=10, colors=STYLE["text"], width=0.8, length=4)
    ax.grid(False)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def _style_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=8, fontsize=13,
                 fontweight="bold", color=STYLE["text"])


def _style_colorbar(cbar, label: str) -> None:
    cbar.set_label(label, color=STYLE["text"], fontsize=11)
    cbar.outline.set_edgecolor(STYLE["spine"])
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=10, colors=STYLE["text"], width=0.8)


def _time_label(timestamp_s: Optional[float], dt_ms: float, idx: int) -> str:
    """Tiempo real del par; si no hay metadata, estimación dt × índice."""
    t = timestamp_s if timestamp_s is not None else dt_ms * idx / 1000.0
    return f"t = {t:.3f} s"


def _background_limits(bg: np.ndarray) -> Tuple[float, float]:
    """Estira el contraste del fondo al rango p2–p98."""
    lo, hi = np.percentile(bg, (2, 98))
    if hi <= lo:
        return float(bg.min()), float(bg.max()) + 1e-6
    return float(lo), float(hi)


def _set_uv_limits(ax, uvals: np.ndarray, vvals: np.ndarray,
                   hull_closed: Optional[np.ndarray] = None) -> None:
    """
    Límites cuadrados que contienen el origen, el 99 % de los datos y la
    región de validación completa.
    """
    cu, cv = float(np.median(uvals)), float(np.median(vvals))
    half = max(np.percentile(np.abs(uvals - cu), 99),
               np.percentile(np.abs(vvals - cv), 99), 1e-6)
    u_lo, u_hi = min(cu - half, 0.0), max(cu + half, 0.0)
    v_lo, v_hi = min(cv - half, 0.0), max(cv + half, 0.0)
    if hull_closed is not None and hull_closed.size:
        u_lo, u_hi = min(u_lo, hull_closed[:, 0].min()), max(u_hi, hull_closed[:, 0].max())
        v_lo, v_hi = min(v_lo, hull_closed[:, 1].min()), max(v_hi, hull_closed[:, 1].max())
    span = max(u_hi - u_lo, v_hi - v_lo) * 1.08
    mu, mv = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0
    ax.set_xlim(mu - span / 2.0, mu + span / 2.0)
    ax.set_ylim(mv - span / 2.0, mv + span / 2.0)


def _draw_zero_axes(ax) -> None:
    ax.axhline(0, color=STYLE["zero"], linewidth=0.8, alpha=0.35, zorder=0)
    ax.axvline(0, color=STYLE["zero"], linewidth=0.8, alpha=0.35, zorder=0)


def _force_square_axes(*axes) -> None:
    for ax in axes:
        ax.set_box_aspect(1)


# ===============================================================
# Extracción de metadata de nombres de archivo
# ===============================================================

def _extract_metadata_from_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extrae región, bloque, timestamp y dt desde nombres como:
    'img_0000_r1b001s0.png' o 'pair_r1b001_t0.000s_dt4.545ms.txt'
    
    Returns:
        Dict con region_idx, block_idx, timestamp_s, dt_ms o None
    """
    import re
    
    # Patrón 1: img_XXXX_rRbBBBsS.ext
    pattern1 = r'_r(\d+)b(\d+)s(\d+)'
    match1 = re.search(pattern1, filename)
    
    if match1:
        region_idx = int(match1.group(1)) - 1  # r1 -> índice 0
        block_idx = int(match1.group(2)) - 1   # b001 -> índice 0
        skip_inter = int(match1.group(3))
        return {
            'region_idx': region_idx,
            'block_idx': block_idx,
            'skip_inter': skip_inter,
        }
    
    # Patrón 2: pair_rRbBBB_tT.TTTs_dtD.DDDms.ext
    pattern2 = r'pair_r(\d+)b(\d+)_t([\d.]+)s_dt([\d.]+)ms'
    match2 = re.search(pattern2, filename)
    
    if match2:
        region_idx = int(match2.group(1)) - 1
        block_idx = int(match2.group(2)) - 1
        timestamp_s = float(match2.group(3))
        dt_ms = float(match2.group(4))
        return {
            'region_idx': region_idx,
            'block_idx': block_idx,
            'timestamp_s': timestamp_s,
            'dt_ms': dt_ms,
        }
    
    return None


def _get_region_name(region_idx: int) -> str:
    """Mapear índice de región a nombre"""
    region_names = {
        0: "Alta Velocidad",
        1: "Media Velocidad",
        2: "Baja Velocidad",
    }
    return region_names.get(region_idx, f"Región {region_idx + 1}")


# ===============================================================
# Helpers numéricos
# ===============================================================

def _compute_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Calcula vorticidad omega = (dv/dx - du/dy) / 2
    usando diferencias finitas centrales.
    """
    omega = np.full_like(u, np.nan)

    if x.shape[1] > 1 and x.shape[0] > 1:
        dx = np.mean(np.diff(x[0, :]))
        dy = np.mean(np.diff(y[:, 0]))
    else:
        return omega

    dvdx = np.full_like(v, np.nan)
    dudy = np.full_like(u, np.nan)

    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)

    dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx
    dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx
    dudy[0, :] = (u[1, :] - u[0, :]) / dy
    dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy

    omega = (dvdx - dudy) / 2.0
    omega[~valid_mask] = np.nan
    return omega


def _precompute_hulls(
    results: List[PIVResult],
    keep_percentile: float,
) -> List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
    """
    Precalcula valid mask, hull y inside para cada resultado.
    """
    precomputed = []
    for r in results:
        valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
        uvals = r.u_mms[valid]
        vvals = r.v_mms[valid]

        if uvals.size >= 10:
            hull_closed, inside = velocity_region_mask(
                uvals, vvals, keep_percentile=keep_percentile
            )
        else:
            hull_closed = None
            inside = np.ones(uvals.size, dtype=bool)

        precomputed.append((valid, hull_closed, inside))
    return precomputed


# ===============================================================
# Panel lateral mejorado con info temporal
# ===============================================================

def _create_right_panel_enhanced(
    fig: plt.Figure,
    panel_spec,
    n_frames: int,
    frame_init: int = 0,
    scale_init: float = 1.0,
) -> Tuple[Any, Any, Slider, Slider, Button]:
    """Panel lateral con controles e información temporal dinámica."""
    ax_panel = fig.add_subplot(panel_spec)
    ax_panel.set_facecolor(STYLE["panel_bg"])
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    for s in ax_panel.spines.values():
        s.set_color(STYLE["spine"])
        s.set_linewidth(0.8)

    pos = ax_panel.get_position()
    x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height

    # Título
    ax_title = fig.add_axes([x0 + 0.07 * w, y0 + 0.91 * h, 0.86 * w, 0.05 * h], facecolor=STYLE["panel_bg"])
    ax_title.axis("off")
    ax_title.text(0.0, 0.5, "Controles", ha="left", va="center",
                  fontsize=13, fontweight="bold", color=STYLE["text"])

    def _slider(y_frac: float, label: str, **kw) -> Slider:
        ax_s = fig.add_axes([x0 + 0.30 * w, y0 + y_frac * h, 0.52 * w, 0.026 * h],
                            facecolor=STYLE["panel_bg"])
        s = Slider(ax=ax_s, label=label, color=STYLE["accent"],
                   track_color=STYLE["slider_track"], **kw)
        s.label.set_fontsize(11)
        s.label.set_color(STYLE["text"])
        s.valtext.set_fontsize(11)
        s.valtext.set_color(STYLE["text"])
        return s

    s_momento = _slider(0.83, "Par", valmin=0, valmax=max(0, n_frames - 1),
                        valinit=frame_init, valstep=1)
    s_scale = _slider(0.76, "Escala", valmin=0.2, valmax=4.0,
                      valinit=scale_init, valstep=0.05)

    # Botón reset
    ax_reset = fig.add_axes([x0 + 0.26 * w, y0 + 0.67 * h, 0.48 * w, 0.045 * h], facecolor=STYLE["panel_bg"])
    btn_reset = Button(ax=ax_reset, label="Reset",
                       color=STYLE["button"], hovercolor=STYLE["button_hover"])
    btn_reset.label.set_fontsize(11)
    btn_reset.label.set_color(STYLE["text"])
    for s in ax_reset.spines.values():
        s.set_color(STYLE["spine"])
        s.set_linewidth(0.8)

    # Área de información temporal (se rellena en _update_temporal_info)
    ax_temporal_info = fig.add_axes(
        [x0 + 0.07 * w, y0 + 0.31 * h, 0.86 * w, 0.33 * h],
        facecolor=STYLE["panel_bg"],
    )
    ax_temporal_info.axis("off")

    # Área de ayuda
    ax_info = fig.add_axes([x0 + 0.07 * w, y0 + 0.04 * h, 0.86 * w, 0.23 * h], facecolor=STYLE["panel_bg"])
    ax_info.axis("off")
    ax_info.add_patch(FancyBboxPatch(
        (0.004, 0.004), 0.992, 0.992, boxstyle="square,pad=0",
        transform=ax_info.transAxes, clip_on=False,
        facecolor=STYLE["card_bg"], edgecolor=STYLE["spine"], linewidth=0.8,
    ))
    ax_info.text(0.07, 0.86, "Navegación", ha="left", va="center",
                 fontsize=12, fontweight="bold", color=STYLE["text"],
                 transform=ax_info.transAxes)
    help_rows = [
        ("Par", "navega entre resultados"),
        ("Escala", "longitud de los vectores"),
        ("Reset", "valores iniciales"),
        ("← →", "par anterior / siguiente"),
    ]
    for i, (key, desc) in enumerate(help_rows):
        y = 0.66 - i * 0.17
        ax_info.text(0.07, y, key, ha="left", va="center", fontsize=10.5,
                     fontweight="bold", color=STYLE["text"], transform=ax_info.transAxes)
        ax_info.text(0.33, y, desc, ha="left", va="center", fontsize=10.5,
                     color=STYLE["text"], transform=ax_info.transAxes)

    return ax_panel, ax_temporal_info, s_momento, s_scale, btn_reset


def _update_temporal_info(
    ax_temporal_info,
    idx: int,
    n_frames: int,
    dt_ms: float,
    names: List[str],
    valid_count: int,
    total_count: int,
    max_speed: float,
    timestamp_s: Optional[float] = None,
) -> None:
    """Actualiza la tarjeta de información temporal."""
    ax_temporal_info.clear()
    ax_temporal_info.axis("off")

    name = names[idx] if idx < len(names) else ""
    metadata = _extract_metadata_from_filename(name)

    if timestamp_s is None:
        timestamp_s = idx * (dt_ms / 1000.0)

    if metadata and "region_idx" in metadata:
        region_info = _get_region_name(metadata["region_idx"])
        if "skip_inter" in metadata:
            region_info += f" · skip {metadata['skip_inter']}"
    else:
        region_info = "—"

    ax_temporal_info.add_patch(FancyBboxPatch(
        (0.004, 0.004), 0.992, 0.992, boxstyle="square,pad=0",
        transform=ax_temporal_info.transAxes, clip_on=False,
        facecolor=STYLE["card_bg"], edgecolor=STYLE["spine"], linewidth=0.8,
    ))
    ax_temporal_info.text(0.07, 0.89, "Información temporal", ha="left", va="center",
                          fontsize=12, fontweight="bold", color=STYLE["text"],
                          transform=ax_temporal_info.transAxes)
    ax_temporal_info.plot([0.07, 0.93], [0.79, 0.79], transform=ax_temporal_info.transAxes,
                          color=STYLE["spine"], linewidth=0.6)

    rows = [
        ("Par", f"{idx + 1} / {n_frames}", True),
        ("t", f"{timestamp_s:.3f} s", True),
        ("Δt", f"{dt_ms:.3f} ms", True),
        None,
        ("Vectores", f"{valid_count:,} / {total_count:,}", False),
        ("V máx", f"{max_speed:.1f} mm/s", False),
        ("Región", "", False),
    ]
    y = 0.69
    step = 0.084
    for row in rows:
        if row is not None:
            label, value, strong = row
            ax_temporal_info.text(0.07, y, label, ha="left", va="center", fontsize=10,
                                  color=STYLE["text"], transform=ax_temporal_info.transAxes)
            ax_temporal_info.text(0.93, y, value, ha="right", va="center", fontsize=10,
                                  fontweight="bold" if strong else "normal",
                                  color=STYLE["text"], transform=ax_temporal_info.transAxes)
        y -= step
    # La región puede ser larga: se escribe completa en la línea siguiente
    ax_temporal_info.text(0.07, y, region_info, ha="left", va="center", fontsize=10,
                          color=STYLE["text"], transform=ax_temporal_info.transAxes)


# ===============================================================
# Artist manager
# ===============================================================

class ArtistManager:
    """Gestiona artists para redraw limpio."""

    def __init__(self):
        self.artists: Dict[str, List[Any]] = {}

    def register(self, key: str, artist):
        if key not in self.artists:
            self.artists[key] = []
        if isinstance(artist, list):
            self.artists[key].extend(artist)
        else:
            self.artists[key].append(artist)

    def clear(self, key: str):
        if key in self.artists:
            for artist in self.artists[key]:
                try:
                    artist.remove()
                except Exception:
                    pass
            self.artists[key] = []

    def clear_all(self):
        for key in list(self.artists.keys()):
            self.clear(key)


# ===============================================================
# Viewer
# ===============================================================

class PIVViewer:

    def show_initial(self, results: List[PIVResult], names: List[str], cfg: PIVConfig) -> None:
        """Vista inicial: campo de velocidades y validación en el espacio u–v."""
        _setup_matplotlib_style()
        print("[PIV] Precalculando velocity regions para viewer...", flush=True)
        precomputed = _precompute_hulls(results, cfg.keep_percentile)

        timestamps = load_timestamps_from_metadata(cfg.images_dir)
        print(f"[PIV] Cargados {len(timestamps)} timestamps desde metadata", flush=True)

        fig = plt.figure(figsize=(17.0, 7.4), facecolor=STYLE["figure_bg"])
        fig.suptitle("Análisis PIV · Vista inicial", fontsize=16,
                     fontweight="bold", color=STYLE["text"], y=0.975)

        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1.0, 1.0, 0.42],
            wspace=0.24,
            left=0.045, right=0.985, top=0.88, bottom=0.08,
        )

        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])

        _, ax_temporal_info, s_momento, s_scale, btn_reset = _create_right_panel_enhanced(
            fig=fig,
            panel_spec=gs[0, 2],
            n_frames=len(results),
            frame_init=0,
            scale_init=1.0,
        )

        mm_per_px = cfg.mm_per_px()
        artist_mgr = ArtistManager()

        def draw(idx: int, scale: float) -> None:
            r = results[idx]
            valid, hull_closed, inside = precomputed[idx]

            artist_mgr.clear_all()
            ax_vel.clear()
            ax_uv.clear()

            _style_axes(ax_vel, equal=True)
            _style_axes(ax_uv, equal=False)

            bg = r.bg_display
            h_px, w_px = bg.shape
            extent = [0, w_px * mm_per_px, h_px * mm_per_px, 0]
            bg_lo, bg_hi = _background_limits(bg)

            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            total_points = valid.size
            valid_count = np.sum(inside) if inside is not None else uvals.size
            max_speed = float(np.nanmax(np.sqrt(r.u_mms**2 + r.v_mms**2))) if valid.any() else 0.0

            timestamp_s = get_timestamp_for_result(r, timestamps)

            _update_temporal_info(
                ax_temporal_info,
                idx=idx,
                n_frames=len(results),
                dt_ms=r.dt_ms,
                names=names,
                valid_count=valid_count,
                total_count=total_points,
                max_speed=max_speed,
                timestamp_s=timestamp_s,
            )

            # ---------------------------------------------------
            # Campo espacial
            # ---------------------------------------------------
            ax_vel.set_facecolor("black")
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent,
                          vmin=bg_lo, vmax=bg_hi, alpha=0.78, zorder=0)

            quiver_scale = max(scale * 0.12, 1e-6)
            arrow_style = dict(angles="xy", scale_units="xy", scale=quiver_scale,
                               width=cfg.quiver_width * 1.6, headwidth=3.6,
                               headlength=4.2, headaxislength=3.8)

            if uvals.size >= 10:
                inside_grid = np.zeros_like(valid, dtype=bool)
                inside_grid[valid] = inside

                ok = inside_grid
                bad = valid & (~inside_grid)

                speed_all = np.sqrt(r.u_mms**2 + r.v_mms**2)
                max_speed_norm = np.nanmax(speed_all)

                if max_speed_norm > 1e-6:
                    u_norm = r.u_mms / max_speed_norm
                    v_norm = r.v_mms / max_speed_norm
                else:
                    u_norm = r.u_mms
                    v_norm = r.v_mms

                q1 = ax_vel.quiver(r.x_mm[ok], r.y_mm[ok], u_norm[ok], v_norm[ok],
                                   color=STYLE["valid"], alpha=0.95, zorder=2, **arrow_style)
                q2 = ax_vel.quiver(r.x_mm[bad], r.y_mm[bad], u_norm[bad], v_norm[bad],
                                   color=STYLE["invalid"], alpha=0.95, zorder=3, **arrow_style)
                artist_mgr.register("vel", [q1, q2])

                ax_vel.legend(
                    handles=[
                        plt.Line2D([0], [0], color=STYLE["valid"], lw=2.5, label=f"Validados ({np.sum(ok):,})"),
                        plt.Line2D([0], [0], color=STYLE["invalid"], lw=2.5, label=f"Rechazados ({np.sum(bad):,})"),
                    ],
                    loc="upper right",
                )
            else:
                q = ax_vel.quiver(r.x_mm[valid], r.y_mm[valid], r.u_mms[valid], r.v_mms[valid],
                                  color=STYLE["invalid"], alpha=0.95, zorder=2, **arrow_style)
                artist_mgr.register("vel", q)

            _style_title(ax_vel, f"Campo de velocidades · {_time_label(timestamp_s, r.dt_ms, idx)}")
            ax_vel.set_xlabel("x [mm]")
            ax_vel.set_ylabel("y [mm]")

            # ---------------------------------------------------
            # Espacio u-v
            # ---------------------------------------------------
            _draw_zero_axes(ax_uv)
            if uvals.size >= 10:
                ax_uv.scatter(uvals[inside], vvals[inside], s=12, alpha=0.55,
                              c=STYLE["valid"], edgecolors="none", marker="o",
                              label=f"Validados ({np.sum(inside):,})", zorder=2)
                ax_uv.scatter(uvals[~inside], vvals[~inside], s=22, alpha=0.85,
                              c=STYLE["invalid"], marker="x", linewidths=0.9,
                              label=f"Rechazados ({np.sum(~inside):,})", zorder=3)
                if hull_closed is not None:
                    ax_uv.plot(hull_closed[:, 0], hull_closed[:, 1], color=STYLE["spine"],
                               linewidth=1.6, label="Región de validación", zorder=4)
                ax_uv.legend(loc="upper right")

            _style_title(ax_uv, f"Espacio de velocidades · Δt = {r.dt_ms:.3f} ms")
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")

            if uvals.size > 0:
                _set_uv_limits(ax_uv, uvals, vvals, hull_closed)

            _force_square_axes(ax_vel, ax_uv)
            fig.canvas.draw_idle()

        def update(_val=None) -> None:
            draw(int(s_momento.val), float(s_scale.val))

        def reset(_event) -> None:
            s_momento.reset()
            s_scale.reset()

        def on_key(event):
            """Navegación con teclado"""
            if event.key == 'right':
                new_val = min(s_momento.val + 1, s_momento.valmax)
                s_momento.set_val(new_val)
            elif event.key == 'left':
                new_val = max(s_momento.val - 1, s_momento.valmin)
                s_momento.set_val(new_val)

        s_momento.on_changed(update)
        s_scale.on_changed(update)
        btn_reset.on_clicked(reset)
        fig.canvas.mpl_connect('key_press_event', on_key)

        update()
        plt.show()
        plt.close(fig)

    def show_final(self, finals: List[PIVResultFinal], names: List[str], cfg: PIVConfig) -> None:
        """Vista final: velocidades, espacio u–v, vorticidad y su distribución."""
        _setup_matplotlib_style()

        timestamps = load_timestamps_from_metadata(cfg.images_dir)
        print(f"[PIV] Cargados {len(timestamps)} timestamps desde metadata", flush=True)

        fig = plt.figure(figsize=(19.0, 10.8), facecolor=STYLE["figure_bg"])
        fig.suptitle("Análisis PIV · Resultados finales", fontsize=17,
                     fontweight="bold", color=STYLE["text"], y=0.985)

        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.0, 1.0, 0.42],
            height_ratios=[1.0, 1.0],
            hspace=0.30,
            wspace=0.26,
            left=0.04, right=0.985, top=0.915, bottom=0.055,
        )

        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        ax_omega = fig.add_subplot(gs[1, 0])
        ax_omega_dist = fig.add_subplot(gs[1, 1])

        _, ax_temporal_info, s_momento, s_scale, btn_reset = _create_right_panel_enhanced(
            fig=fig,
            panel_spec=gs[:, 2],
            n_frames=len(finals),
            frame_init=0,
            scale_init=1.0,
        )

        mm_per_px = cfg.mm_per_px()
        artist_mgr = ArtistManager()
        omega_cache: Dict[int, np.ndarray] = {}
        cbar_refs: Dict[str, Any] = {}

        def draw(idx: int, scale: float) -> None:
            r = finals[idx]

            artist_mgr.clear_all()
            for ax in [ax_vel, ax_uv, ax_omega, ax_omega_dist]:
                ax.clear()

            _style_axes(ax_vel, equal=True)
            _style_axes(ax_uv, equal=False)
            _style_axes(ax_omega, equal=True)
            _style_axes(ax_omega_dist, equal=False)

            bg = r.bg_display
            h_px, w_px = bg.shape
            extent = [0, w_px * mm_per_px, h_px * mm_per_px, 0]
            bg_lo, bg_hi = _background_limits(bg)

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            total_points = valid.size
            valid_count = np.sum(valid)
            max_speed = float(np.nanmax(np.sqrt(r.u_mms[valid]**2 + r.v_mms[valid]**2))) if valid.any() else 0.0

            timestamp_s = get_timestamp_for_result(r, timestamps)
            t_label = _time_label(timestamp_s, r.dt_ms, idx)

            _update_temporal_info(
                ax_temporal_info,
                idx=idx,
                n_frames=len(finals),
                dt_ms=r.dt_ms,
                names=names,
                valid_count=valid_count,
                total_count=total_points,
                max_speed=max_speed,
                timestamp_s=timestamp_s,
            )

            if uvals.size == 0:
                for ax in [ax_vel, ax_omega]:
                    ax.text(0.5, 0.5, "Sin datos validados", ha="center", va="center",
                            transform=ax.transAxes, fontsize=14,
                            color=STYLE["text"], fontweight="bold")
                _style_title(ax_vel, f"Campo de velocidades · {t_label}")
                _style_title(ax_uv, f"Espacio de velocidades · Δt = {r.dt_ms:.3f} ms")
                _style_title(ax_omega, f"Campo de vorticidad · {t_label}")
                _style_title(ax_omega_dist, "Distribución de vorticidad")
                _force_square_axes(ax_vel, ax_uv, ax_omega)
                fig.canvas.draw_idle()
                return

            speed = np.sqrt(uvals**2 + vvals**2)
            speed_all = np.sqrt(r.u_mms[valid]**2 + r.v_mms[valid]**2)

            max_speed_norm = np.nanmax(speed_all)
            if max_speed_norm > 1e-6:
                u_norm = r.u_mms[valid] / max_speed_norm
                v_norm = r.v_mms[valid] / max_speed_norm
            else:
                u_norm = r.u_mms[valid]
                v_norm = r.v_mms[valid]

            vmin = float(np.nanpercentile(speed, 1))
            vmax = float(np.nanpercentile(speed, 99))
            if vmax <= vmin:
                vmax = vmin + 1e-6

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap_vel = plt.get_cmap(STYLE["speed_cmap"])

            # ---------------------------------------------------
            # Campo de velocidades + streamlines
            # ---------------------------------------------------
            ax_vel.set_facecolor("black")
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent,
                          vmin=bg_lo, vmax=bg_hi, alpha=0.78, zorder=0)

            try:
                valid_points = np.column_stack([r.x_mm[valid].ravel(), r.y_mm[valid].ravel()])
                u_valid_vals = r.u_mms[valid].ravel()
                v_valid_vals = r.v_mms[valid].ravel()

                if len(valid_points) > 10:
                    grid_points = np.column_stack([r.x_mm.ravel(), r.y_mm.ravel()])
                    u_interp = griddata(valid_points, u_valid_vals, grid_points, method="linear", fill_value=0.0)
                    v_interp = griddata(valid_points, v_valid_vals, grid_points, method="linear", fill_value=0.0)

                    u_for_stream = u_interp.reshape(r.x_mm.shape)
                    v_for_stream = v_interp.reshape(r.y_mm.shape)
                    speed_grid = np.sqrt(u_for_stream**2 + v_for_stream**2)

                    # streamplot no acepta alpha: se aplica sobre las colecciones
                    stream = ax_vel.streamplot(
                        r.x_mm[0, :], r.y_mm[:, 0],
                        u_for_stream, v_for_stream,
                        color=speed_grid,
                        cmap=cmap_vel,
                        norm=norm,
                        density=1.3,
                        linewidth=1.1,
                        arrowsize=0.9,
                        zorder=2,
                    )
                    stream.lines.set_alpha(0.6)
                    stream.arrows.set_alpha(0.6)
                    artist_mgr.register("stream", stream.lines)
            except Exception as e:
                print(f"[PIV] Advertencia streamlines: {e}")

            quiver_scale = max(scale * 0.12, 1e-6)

            q = ax_vel.quiver(
                r.x_mm[valid],
                r.y_mm[valid],
                u_norm,
                v_norm,
                speed_all,
                cmap=cmap_vel,
                norm=norm,
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=cfg.quiver_width * 1.2,
                headwidth=3.6,
                headlength=4.2,
                headaxislength=3.8,
                alpha=0.9,
                edgecolors="none",
                zorder=3,
            )

            if "vel" not in cbar_refs:
                cbar_refs["vel"] = fig.colorbar(q, ax=ax_vel, fraction=0.046, pad=0.03)
                _style_colorbar(cbar_refs["vel"], "Velocidad [mm/s]")
            else:
                cbar_refs["vel"].update_normal(q)

            artist_mgr.register("vel", q)

            _style_title(ax_vel, f"Campo de velocidades · {t_label}")
            ax_vel.set_xlabel("x [mm]")
            ax_vel.set_ylabel("y [mm]")

            # ---------------------------------------------------
            # Espacio u-v
            # ---------------------------------------------------
            _draw_zero_axes(ax_uv)
            sc = ax_uv.scatter(
                uvals, vvals,
                c=speed,
                cmap=cmap_vel,
                norm=norm,
                s=12,
                alpha=0.65,
                edgecolors="none",
                zorder=2,
            )
            artist_mgr.register("uv", sc)

            _style_title(ax_uv, f"Espacio de velocidades · Δt = {r.dt_ms:.3f} ms")
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")
            _set_uv_limits(ax_uv, uvals, vvals)

            # ---------------------------------------------------
            # Vorticidad
            # ---------------------------------------------------
            if idx not in omega_cache:
                omega_cache[idx] = _compute_vorticity(r.u_mms, r.v_mms, r.x_mm, r.y_mm, valid)
            omega = omega_cache[idx]

            ax_omega.set_facecolor("black")
            ax_omega.imshow(bg, cmap="gray", origin="upper", extent=extent,
                            vmin=bg_lo, vmax=bg_hi, alpha=0.78, zorder=0)

            omega_valid = omega[valid]
            omega_finite_all = omega_valid[np.isfinite(omega_valid)]
            if omega_finite_all.size > 0:
                # Normaliza por el p98 de |ω| en vez del máximo: un único vector
                # extremo dejaba casi todo el campo en tonos pálidos.
                omega_ref = float(np.percentile(np.abs(omega_finite_all), 98))
                if omega_ref <= 1e-6:
                    omega_ref = float(np.max(np.abs(omega_finite_all)))
                omega_norm = omega / omega_ref if omega_ref > 1e-6 else omega

                omega_norm_masked = omega_norm.copy()
                omega_norm_masked[~valid] = np.nan

                # Suavizado normalizado: no arrastra ceros hacia los bordes enmascarados
                weight = np.isfinite(omega_norm_masked).astype(float)
                num = gaussian_filter(np.nan_to_num(omega_norm_masked, nan=0.0), sigma=1.0)
                den = gaussian_filter(weight, sigma=1.0)
                omega_smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
                omega_smooth[~valid] = np.nan

                levels = np.linspace(-1.0, 1.0, 21)
                contf = ax_omega.contourf(
                    r.x_mm, r.y_mm, omega_smooth,
                    levels=levels,
                    cmap=STYLE["vorticity_cmap"],
                    extend="both",
                    zorder=1,
                )
                # Relleno opaco: con transparencia, los bordes entre niveles se
                # ven como líneas blancas. La estructura del dispositivo se
                # recupera con una capa tenue del fondo por encima del color.
                contf.set_edgecolor("face")
                ax_omega.imshow(bg, cmap="gray", origin="upper", extent=extent,
                                vmin=bg_lo, vmax=bg_hi, alpha=0.28, zorder=2)

                if "omega" not in cbar_refs:
                    cbar_refs["omega"] = fig.colorbar(contf, ax=ax_omega, fraction=0.046, pad=0.03)
                    _style_colorbar(cbar_refs["omega"], "ω / p98(|ω|)")
                else:
                    cbar_refs["omega"].update_normal(contf)

                artist_mgr.register("omega", contf)

            _style_title(ax_omega, f"Campo de vorticidad · {t_label}")
            ax_omega.set_xlabel("x [mm]")
            ax_omega.set_ylabel("y [mm]")

            # ---------------------------------------------------
            # Histograma de vorticidad
            # ---------------------------------------------------
            omega_vals = _finite_values(omega_finite_all)   # array normal, sin máscara ni NaN/inf

            if omega_vals.size > 0:
                abs_vals = np.abs(omega_vals)
                lim = float(np.percentile(abs_vals, 99.5))
                if not np.isfinite(lim) or lim <= 1e-6:
                    lim = float(abs_vals.max()) + 1e-6
                ax_omega_dist.hist(
                    omega_vals,
                    bins=np.linspace(-lim, lim, 41),
                    color=STYLE["accent"],
                    alpha=0.85,
                    edgecolor="#ffffff",
                    linewidth=0.5,
                    zorder=2,
                )
                ax_omega_dist.axvline(0.0, color=STYLE["zero"], linestyle="--",
                                      linewidth=1.0, label="ω = 0", zorder=3)
                median_val = float(np.median(omega_finite_all))
                ax_omega_dist.axvline(median_val, color=STYLE["invalid"], linestyle="-",
                                      linewidth=1.8, label=f"Mediana = {median_val:.2f}", zorder=4)
                ax_omega_dist.set_xlim(-lim * 1.03, lim * 1.03)
                ax_omega_dist.legend(loc="upper right")

            _style_title(ax_omega_dist, "Distribución de vorticidad")
            ax_omega_dist.set_xlabel("ω [1/s]")
            ax_omega_dist.set_ylabel("Frecuencia")

            _force_square_axes(ax_vel, ax_uv, ax_omega)
            fig.canvas.draw_idle()

        def update(_val=None) -> None:
            draw(int(s_momento.val), float(s_scale.val))

        def reset(_event) -> None:
            s_momento.reset()
            s_scale.reset()

        def on_key(event):
            """Navegación con teclado"""
            if event.key == 'right':
                new_val = min(s_momento.val + 1, s_momento.valmax)
                s_momento.set_val(new_val)
            elif event.key == 'left':
                new_val = max(s_momento.val - 1, s_momento.valmin)
                s_momento.set_val(new_val)

        s_momento.on_changed(update)
        s_scale.on_changed(update)
        btn_reset.on_clicked(reset)
        fig.canvas.mpl_connect('key_press_event', on_key)

        update()
        plt.show()
