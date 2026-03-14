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
    "panel_bg": "#f6f8fb",
    "axes_bg": "#ffffff",
    "spine": "#334155",
    "grid": "#cbd5e1",
    "text": "#0f172a",
    "muted": "#64748b",
    "valid": "#16a34a",
    "invalid": "#dc2626",
    "accent": "#2563eb",
    "accent_soft": "#93c5fd",
    "neutral": "#94a3b8",
    "zero": "#475569",
    "vorticity_cmap": "RdBu_r",
    "speed_cmap": "viridis",
    "info_bg": "#eff6ff",
    "info_border": "#93c5fd",
}


def _setup_matplotlib_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": STYLE["figure_bg"],
        "axes.facecolor": STYLE["axes_bg"],
        "axes.edgecolor": STYLE["spine"],
        "axes.labelcolor": STYLE["text"],
        "axes.titlecolor": STYLE["text"],
        "xtick.color": STYLE["muted"],
        "ytick.color": STYLE["muted"],
        "grid.color": STYLE["grid"],
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#cbd5e1",
        "legend.framealpha": 0.95,
    })


def _style_axes(ax, equal: bool = False) -> None:
    ax.set_facecolor(STYLE["axes_bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE["spine"])
    ax.spines["bottom"].set_color(STYLE["spine"])
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(labelsize=9, colors=STYLE["muted"])
    ax.grid(True)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def _style_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=10, fontweight="semibold", color=STYLE["text"])


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
    """Panel lateral con área de información temporal dinámica"""
    ax_panel = fig.add_subplot(panel_spec)
    ax_panel.set_facecolor(STYLE["panel_bg"])
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    for s in ax_panel.spines.values():
        s.set_color("#dbe3ec")
        s.set_linewidth(1.0)

    pos = ax_panel.get_position()
    x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height

    # Título
    ax_title = fig.add_axes([x0 + 0.08 * w, y0 + 0.90 * h, 0.84 * w, 0.05 * h], facecolor=STYLE["panel_bg"])
    ax_title.axis("off")
    ax_title.text(
        0.0, 0.5, "Controles",
        ha="left", va="center",
        fontsize=11, fontweight="semibold", color=STYLE["text"]
    )

    # Slider de momento/frame
    ax_momento = fig.add_axes([x0 + 0.12 * w, y0 + 0.82 * h, 0.76 * w, 0.028 * h], facecolor=STYLE["panel_bg"])
    s_momento = Slider(
        ax=ax_momento,
        label="Par",
        valmin=0,
        valmax=max(0, n_frames - 1),
        valinit=frame_init,
        valstep=1,
        color=STYLE["accent"],
        track_color="#dbeafe"
    )
    s_momento.label.set_fontsize(9)
    s_momento.valtext.set_fontsize(9)

    # Slider de escala
    ax_scale = fig.add_axes([x0 + 0.12 * w, y0 + 0.74 * h, 0.76 * w, 0.028 * h], facecolor=STYLE["panel_bg"])
    s_scale = Slider(
        ax=ax_scale,
        label="Escala",
        valmin=0.2,
        valmax=4.0,
        valinit=scale_init,
        valstep=0.05,
        color=STYLE["accent"],
        track_color="#dbeafe"
    )
    s_scale.label.set_fontsize(9)
    s_scale.valtext.set_fontsize(9)

    # Botón reset
    ax_reset = fig.add_axes([x0 + 0.22 * w, y0 + 0.66 * h, 0.56 * w, 0.045 * h], facecolor=STYLE["panel_bg"])
    btn_reset = Button(
        ax=ax_reset,
        label="Reset",
        color="#e2e8f0",
        hovercolor="#cbd5e1"
    )
    btn_reset.label.set_fontsize(9)
    btn_reset.label.set_color(STYLE["text"])

    # ================================================================
    # NUEVO: Área de información temporal dinámica
    # ================================================================
    ax_temporal_info = fig.add_axes(
        [x0 + 0.08 * w, y0 + 0.44 * h, 0.84 * w, 0.18 * h],
        facecolor=STYLE["panel_bg"]
    )
    ax_temporal_info.axis("off")

    # Card de fondo
    card_temporal = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax_temporal_info.transAxes,
        facecolor=STYLE["info_bg"],
        edgecolor=STYLE["info_border"],
        linewidth=1.2
    )
    ax_temporal_info.add_patch(card_temporal)

    # Área de uso/ayuda (más pequeña)
    ax_info = fig.add_axes([x0 + 0.08 * w, y0 + 0.10 * h, 0.84 * w, 0.28 * h], facecolor=STYLE["panel_bg"])
    ax_info.axis("off")

    card = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax_info.transAxes,
        facecolor="#ffffff",
        edgecolor="#dbe3ec",
        linewidth=1.0
    )
    ax_info.add_patch(card)

    ax_info.text(
        0.06, 0.88, "Navegación",
        ha="left", va="center",
        fontsize=10, fontweight="semibold", color=STYLE["text"],
        transform=ax_info.transAxes
    )

    info_text = (
        "• Par: navega entre resultados\n\n"
        "• Escala: ajusta longitud de vectores\n\n"
        "• Reset: valores iniciales\n\n"
        "• Teclado: ← → para navegar"
    )
    ax_info.text(
        0.06, 0.72, info_text,
        ha="left", va="top",
        fontsize=8.5, color=STYLE["muted"],
        transform=ax_info.transAxes, linespacing=1.5
    )

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
    timestamp_s: Optional[float] = None,  # ← NUEVO: recibir timestamp directamente
) -> None:
    """Actualiza el panel de información temporal"""
    ax_temporal_info.clear()
    ax_temporal_info.axis("off")
    
    # Extraer metadata del nombre si es posible
    name = names[idx] if idx < len(names) else ""
    metadata = _extract_metadata_from_filename(name)
    
    # Usar timestamp pasado como parámetro o estimación básica
    if timestamp_s is None:
        # Estimación básica (fallback)
        timestamp_s = idx * (dt_ms / 1000.0)
    
    # Información de región
    if metadata and 'region_idx' in metadata:
        region_name = _get_region_name(metadata['region_idx'])
        region_info = f"{region_name}"
        if 'skip_inter' in metadata:
            region_info += f" (skip={metadata['skip_inter']})"
    else:
        region_info = "—"
    
    # Card de fondo
    card = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax_temporal_info.transAxes,
        facecolor=STYLE["info_bg"],
        edgecolor=STYLE["info_border"],
        linewidth=1.2
    )
    ax_temporal_info.add_patch(card)
    
    # Título
    ax_temporal_info.text(
        0.06, 0.82, "Información Temporal",
        ha="left", va="center",
        fontsize=9.5, fontweight="semibold", color=STYLE["text"],
        transform=ax_temporal_info.transAxes
    )
    
    # Línea de separación
    ax_temporal_info.plot(
        [0.06, 0.94], [0.72, 0.72],
        transform=ax_temporal_info.transAxes,
        color=STYLE["info_border"], linewidth=1.0, alpha=0.5
    )
    
    # Información principal
    info_lines = [
        f"Par:  {idx + 1} / {n_frames}",
        f"t =  {timestamp_s:.3f} s",
        f"Δt = {dt_ms:.3f} ms",
        f"",
        f"Región:  {region_info}",
        f"Vectores:  {valid_count:,} / {total_count:,}",
        f"V_max:  {max_speed:.1f} mm/s",
    ]
    
    y_start = 0.62
    y_step = 0.10
    
    for i, line in enumerate(info_lines):
        if line == "":
            continue
        
        # Colorear diferente las primeras 3 líneas (datos temporales)
        if i < 3:
            color = STYLE["accent"]
            weight = "semibold"
            size = 9.0
        else:
            color = STYLE["text"]
            weight = "normal"
            size = 8.5
        
        ax_temporal_info.text(
            0.06, y_start - i * y_step,
            line,
            ha="left", va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
            transform=ax_temporal_info.transAxes,
            family="monospace"
        )


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
        """Vista inicial con información temporal mejorada."""
        _setup_matplotlib_style()
        print("[PIV] Precalculando velocity regions para viewer...", flush=True)
        precomputed = _precompute_hulls(results, cfg.keep_percentile)
        
        # ================================================================
        # NUEVO: Cargar timestamps desde metadata
        # ================================================================
        timestamps = load_timestamps_from_metadata(cfg.images_dir)
        print(f"[PIV] Cargados {len(timestamps)} timestamps desde metadata", flush=True)

        fig = plt.figure(figsize=(17.5, 7.5), facecolor=STYLE["figure_bg"])
        fig.suptitle(
            "Análisis PIV · Vista Inicial",
            fontsize=14,
            fontweight="semibold",
            color=STYLE["text"],
            y=0.97
        )

        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1.0, 1.0, 0.40],
            wspace=0.28,
            left=0.05, right=0.98, top=0.92, bottom=0.08
        )

        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])

        _, ax_temporal_info, s_momento, s_scale, btn_reset = _create_right_panel_enhanced(
            fig=fig,
            panel_spec=gs[0, 2],
            n_frames=len(results),
            frame_init=0,
            scale_init=1.0
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

            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]
            
            # Estadísticas para panel info
            total_points = valid.size
            valid_count = np.sum(inside) if inside is not None else uvals.size
            max_speed = float(np.nanmax(np.sqrt(r.u_mms**2 + r.v_mms**2))) if valid.any() else 0.0
            
            # Obtener timestamp correcto desde metadata
            timestamp_s = get_timestamp_for_result(r, timestamps)

            # Actualizar panel de información temporal
            _update_temporal_info(
                ax_temporal_info,
                idx=idx,
                n_frames=len(results),
                dt_ms=r.dt_ms,
                names=names,
                valid_count=valid_count,
                total_count=total_points,
                max_speed=max_speed,
                timestamp_s=timestamp_s,  # ← NUEVO
            )

            # ---------------------------------------------------
            # Campo espacial
            # ---------------------------------------------------
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.78)

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

                quiver_scale = max(scale * 0.12, 1e-6)

                q1 = ax_vel.quiver(
                    r.x_mm[ok], r.y_mm[ok],
                    u_norm[ok], v_norm[ok],
                    color=STYLE["valid"],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale,
                    width=cfg.quiver_width * 5,
                    alpha=0.90
                )

                q2 = ax_vel.quiver(
                    r.x_mm[bad], r.y_mm[bad],
                    u_norm[bad], v_norm[bad],
                    color=STYLE["invalid"],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale,
                    width=cfg.quiver_width * 1.15,
                    alpha=0.75
                )

                artist_mgr.register("vel", [q1, q2])

                ax_vel.legend(
                    handles=[
                        plt.Line2D([0], [0], color=STYLE["valid"], lw=2, label=f"Validados ({np.sum(ok):,})"),
                        plt.Line2D([0], [0], color=STYLE["invalid"], lw=2, label=f"Rechazados ({np.sum(bad):,})"),
                    ],
                    loc="upper right",
                    fontsize=9
                )

            else:
                quiver_scale = max(scale * 0.12, 1e-6)
                q = ax_vel.quiver(
                    r.x_mm[valid], r.y_mm[valid],
                    r.u_mms[valid], r.v_mms[valid],
                    color=STYLE["invalid"],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale,
                    width=cfg.quiver_width * 5,
                    alpha=0.75
                )
                artist_mgr.register("vel", q)

            _style_title(ax_vel, f"Campo de velocidades · t = {r.dt_ms * idx / 1000:.3f}s")
            ax_vel.set_xlabel("x [mm]")
            ax_vel.set_ylabel("y [mm]")

            # ---------------------------------------------------
            # Espacio u-v
            # ---------------------------------------------------
            if uvals.size >= 10:
                ax_uv.scatter(
                    uvals[inside], vvals[inside],
                    s=10, alpha=0.55,
                    c=STYLE["valid"], edgecolors="none",
                    label=f"Validados ({np.sum(inside):,})"
                )
                ax_uv.scatter(
                    uvals[~inside], vvals[~inside],
                    s=10, alpha=0.45,
                    c=STYLE["invalid"], edgecolors="none",
                    label=f"Rechazados ({np.sum(~inside):,})"
                )

                if hull_closed is not None:
                    ax_uv.plot(
                        hull_closed[:, 0], hull_closed[:, 1],
                        color=STYLE["spine"],
                        linewidth=1.6,
                        label="Región de validación"
                    )

                ax_uv.legend(loc="upper right", fontsize=9)

            ax_uv.axhline(0, color=STYLE["zero"], linewidth=1.0, alpha=0.6)
            ax_uv.axvline(0, color=STYLE["zero"], linewidth=1.0, alpha=0.6)

            _style_title(ax_uv, f"Espacio de velocidades · Δt = {r.dt_ms:.3f}ms")
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")

            if uvals.size > 0:
                umax = np.percentile(np.abs(uvals), 99)
                vmax = np.percentile(np.abs(vvals), 99)
                lim = max(umax, vmax, 1e-6)
                margin = lim * 0.06
                ax_uv.set_xlim(-lim - margin, lim + margin)
                ax_uv.set_ylim(-lim - margin, lim + margin)

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

    def show_final(self, finals: List[PIVResultFinal], names: List[str], cfg: PIVConfig) -> None:
        """Vista final con información temporal mejorada."""
        _setup_matplotlib_style()
        
        # ================================================================
        # NUEVO: Cargar timestamps desde metadata
        # ================================================================
        timestamps = load_timestamps_from_metadata(cfg.images_dir)
        print(f"[PIV] Cargados {len(timestamps)} timestamps desde metadata", flush=True)

        fig = plt.figure(figsize=(19.0, 10.5), facecolor=STYLE["figure_bg"])
        fig.suptitle(
            "Análisis PIV · Resultados Finales",
            fontsize=15,
            fontweight="semibold",
            color=STYLE["text"],
            y=0.975
        )

        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.0, 1.0, 0.40],
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.28,
            left=0.04, right=0.98, top=0.94, bottom=0.06
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
            scale_init=1.0
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

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]
            
            # Estadísticas
            total_points = valid.size
            valid_count = np.sum(valid)
            max_speed = float(np.nanmax(np.sqrt(r.u_mms[valid]**2 + r.v_mms[valid]**2))) if valid.any() else 0.0
            
            # Obtener timestamp correcto desde metadata
            timestamp_s = get_timestamp_for_result(r, timestamps)
            
            # Actualizar panel temporal
            _update_temporal_info(
                ax_temporal_info,
                idx=idx,
                n_frames=len(finals),
                dt_ms=r.dt_ms,
                names=names,
                valid_count=valid_count,
                total_count=total_points,
                max_speed=max_speed,
                timestamp_s=timestamp_s,  # ← NUEVO
            )

            if uvals.size == 0:
                for ax in [ax_vel, ax_omega]:
                    ax.text(
                        0.5, 0.5, "Sin datos validados",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=13,
                        color=STYLE["invalid"],
                        fontweight="semibold"
                    )
                _style_title(ax_vel, f"Campo de velocidades · t = {r.dt_ms * idx / 1000:.3f}s")
                _style_title(ax_uv, f"Espacio de velocidades · Δt = {r.dt_ms:.3f}ms")
                _style_title(ax_omega, f"Campo de vorticidad · t = {r.dt_ms * idx / 1000:.3f}s")
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
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.72)

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

                    stream = ax_vel.streamplot(
                        r.x_mm[0, :], r.y_mm[:, 0],
                        u_for_stream, v_for_stream,
                        color=speed_grid,
                        cmap=cmap_vel,
                        norm=norm,
                        density=1.35,
                        linewidth=1.4,
                        arrowsize=1.0,
                        alpha=0.85,
                        zorder=2
                    )
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
                width=cfg.quiver_width * 0.95,
                alpha=0.55,
                edgecolors="none",
                zorder=3
            )

            if "vel" not in cbar_refs:
                cbar_refs["vel"] = fig.colorbar(q, ax=ax_vel, fraction=0.046, pad=0.04)
                cbar_refs["vel"].set_label("Velocidad [mm/s]")
                cbar_refs["vel"].ax.tick_params(labelsize=9)
            else:
                cbar_refs["vel"].update_normal(q)

            artist_mgr.register("vel", q)

            _style_title(ax_vel, f"Campo de velocidades · t = {r.dt_ms * idx / 1000:.3f}s")
            ax_vel.set_xlabel("x [mm]")
            ax_vel.set_ylabel("y [mm]")

            # ---------------------------------------------------
            # Espacio u-v
            # ---------------------------------------------------
            sc = ax_uv.scatter(
                uvals, vvals,
                c=speed,
                cmap=cmap_vel,
                norm=norm,
                s=11,
                alpha=0.60,
                edgecolors="none"
            )
            artist_mgr.register("uv", sc)

            ax_uv.axhline(0, color=STYLE["zero"], linewidth=1.0, alpha=0.6)
            ax_uv.axvline(0, color=STYLE["zero"], linewidth=1.0, alpha=0.6)

            _style_title(ax_uv, f"Espacio de velocidades · Δt = {r.dt_ms:.3f}ms")
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")

            umax = np.percentile(np.abs(uvals), 99)
            vmax_ax = np.percentile(np.abs(vvals), 99)
            lim = max(umax, vmax_ax, 1e-6)
            margin = lim * 0.06
            ax_uv.set_xlim(-lim - margin, lim + margin)
            ax_uv.set_ylim(-lim - margin, lim + margin)

            # ---------------------------------------------------
            # Vorticidad
            # ---------------------------------------------------
            if idx not in omega_cache:
                omega_cache[idx] = _compute_vorticity(r.u_mms, r.v_mms, r.x_mm, r.y_mm, valid)
            omega = omega_cache[idx]

            ax_omega.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.65)

            omega_valid = omega[valid]
            if omega_valid.size > 0 and np.any(np.isfinite(omega_valid)):
                max_omega_abs = np.nanmax(np.abs(omega_valid[np.isfinite(omega_valid)]))
                if max_omega_abs > 1e-6:
                    omega_norm = omega / max_omega_abs
                else:
                    omega_norm = omega

                omega_norm_masked = omega_norm.copy()
                omega_norm_masked[~valid] = np.nan

                omega_smooth = gaussian_filter(np.nan_to_num(omega_norm_masked, 0), sigma=1.0)
                omega_smooth[~valid] = np.nan

                levels = np.linspace(-1.0, 1.0, 21)
                contf = ax_omega.contourf(
                    r.x_mm, r.y_mm, omega_smooth,
                    levels=levels,
                    cmap=STYLE["vorticity_cmap"],
                    alpha=0.82,
                    extend="both"
                )

                if "omega" not in cbar_refs:
                    cbar_refs["omega"] = fig.colorbar(contf, ax=ax_omega, fraction=0.046, pad=0.04)
                    cbar_refs["omega"].set_label("Vorticidad normalizada")
                    cbar_refs["omega"].ax.tick_params(labelsize=9)
                else:
                    cbar_refs["omega"].update_normal(contf)

                artist_mgr.register("omega", contf)

            _style_title(ax_omega, f"Campo de vorticidad · t = {r.dt_ms * idx / 1000:.3f}s")
            ax_omega.set_xlabel("x [mm]")
            ax_omega.set_ylabel("y [mm]")

            # ---------------------------------------------------
            # Histograma de vorticidad
            # ---------------------------------------------------
            if omega_valid.size > 0:
                omega_finite = omega_valid[np.isfinite(omega_valid)]
                if omega_finite.size > 0:
                    ax_omega_dist.hist(
                        omega_finite,
                        bins=40,
                        color=STYLE["accent"],
                        alpha=0.75,
                        edgecolor="#ffffff",
                        linewidth=0.6
                    )
                    ax_omega_dist.axvline(
                        0.0,
                        color=STYLE["zero"],
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.9,
                        label="ω = 0"
                    )
                    median_val = np.median(omega_finite)
                    ax_omega_dist.axvline(
                        median_val,
                        color=STYLE["invalid"],
                        linestyle="-",
                        linewidth=1.4,
                        alpha=0.85,
                        label=f"Mediana = {median_val:.2f}"
                    )
                    ax_omega_dist.legend(loc="upper right", fontsize=9)

            _style_title(ax_omega_dist, "Distribución de vorticidad")
            ax_omega_dist.set_xlabel("ω [1/s]")
            ax_omega_dist.set_ylabel("Frecuencia")
            ax_omega_dist.grid(True, axis="y")

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