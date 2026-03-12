from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter

from .models import PIVResult, PIVResultFinal
from .config import PIVConfig
from .validation import velocity_region_mask


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

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

    Args:
        u, v: campos de velocidad (grids 2D)
        x, y: coordenadas (grids 2D)
        valid_mask: máscara de puntos válidos

    Returns:
        omega: vorticidad [1/s], mismo shape que u/v
    """
    omega = np.full_like(u, np.nan)

    # Calcular espaciado (asumiendo grid regular)
    if x.shape[1] > 1 and x.shape[0] > 1:
        dx = np.mean(np.diff(x[0, :]))
        dy = np.mean(np.diff(y[:, 0]))
    else:
        return omega

    # Diferencias centrales con manejo de bordes
    dvdx = np.full_like(v, np.nan)
    dudy = np.full_like(u, np.nan)

    # Interior: diferencias centrales
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)

    # Bordes: diferencias hacia adelante/atrás
    dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx
    dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx
    dudy[0, :] = (u[1, :] - u[0, :]) / dy
    dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy

    # Calcular vorticidad solo donde ambos gradientes son válidos
    omega = (dvdx - dudy) / 2.0
    omega[~valid_mask] = np.nan

    return omega


def _precompute_hulls(
    results: List[PIVResult],
    keep_percentile: float,
) -> List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]]:
    """
    Precalcula valid mask, hull y inside para cada resultado.
    Evita recalcular en cada redibujado del slider.
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


def _force_square_axes(*axes) -> None:
    """
    Fuerza que todos los axes tengan caja cuadrada.
    Para ejes espaciales conviene además usar aspect='equal'.
    """
    for ax in axes:
        ax.set_box_aspect(1)


def _create_right_panel(
    fig: plt.Figure,
    panel_spec,
    n_frames: int,
    frame_init: int = 0,
    scale_init: float = 1.0,
    panel_facecolor: str = "#f8f9fa",
) -> Tuple[Any, Slider, Slider, Button]:
    """
    Crea panel derecho moderno y compacto con todos los elementos visibles.

    Devuelve:
      ax_panel, s_momento, s_scale, btn_reset
    """
    ax_panel = fig.add_subplot(panel_spec)
    ax_panel.set_facecolor(panel_facecolor)
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    for spine in ax_panel.spines.values():
        spine.set_edgecolor('#dee2e6')
        spine.set_linewidth(2.0)

    pos = ax_panel.get_position()
    x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height

    # Título
    ax_title = fig.add_axes(
        [x0 + 0.05 * w, y0 + 0.90 * h, 0.90 * w, 0.08 * h],
        facecolor=panel_facecolor
    )
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5, "Panel de Control",
        ha="center", va="center",
        fontsize=12, fontweight="600",
        color="#2c3e50"
    )

    # Slider Frame
    ax_momento = fig.add_axes(
        [x0 + 0.15 * w, y0 + 0.76 * h, 0.70 * w, 0.04 * h],
        facecolor=panel_facecolor
    )
    s_momento = Slider(
        ax=ax_momento,
        label="Frame",
        valmin=0,
        valmax=max(0, n_frames - 1),
        valinit=frame_init,
        valstep=1,
        color="#3498db",
        track_color="#e9ecef"
    )
    s_momento.label.set_fontsize(10)
    s_momento.label.set_fontweight("500")
    s_momento.valtext.set_fontsize(9)

    # Slider Escala
    ax_scale = fig.add_axes(
        [x0 + 0.15 * w, y0 + 0.66 * h, 0.70 * w, 0.04 * h],
        facecolor=panel_facecolor
    )
    s_scale = Slider(
        ax=ax_scale,
        label="Escala",
        valmin=0.1,
        valmax=10.0,
        valinit=scale_init,
        valstep=0.1,
        color="#e74c3c",
        track_color="#e9ecef"
    )
    s_scale.label.set_fontsize(10)
    s_scale.label.set_fontweight("500")
    s_scale.valtext.set_fontsize(9)

    # Botón reset
    ax_reset = fig.add_axes(
        [x0 + 0.20 * w, y0 + 0.56 * h, 0.60 * w, 0.06 * h],
        facecolor=panel_facecolor
    )
    btn_reset = Button(
        ax=ax_reset,
        label="Restablecer",
        color="#e9ecef",
        hovercolor="#ced4da"
    )
    btn_reset.label.set_fontsize(10)
    btn_reset.label.set_fontweight("500")
    btn_reset.label.set_color("#2c3e50")

    # Separador visual
    ax_sep = fig.add_axes(
        [x0 + 0.10 * w, y0 + 0.52 * h, 0.80 * w, 0.005 * h],
        facecolor=panel_facecolor
    )
    ax_sep.axis("off")
    ax_sep.axhline(0.5, color='#ced4da', linewidth=1.5, alpha=0.7)

    # Bloque informativo compacto
    ax_info = fig.add_axes(
        [x0 + 0.08 * w, y0 + 0.15 * h, 0.84 * w, 0.35 * h],
        facecolor=panel_facecolor
    )
    ax_info.axis("off")
    
    # Rectángulo de fondo
    from matplotlib.patches import FancyBboxPatch
    fancy_box = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.02",
        transform=ax_info.transAxes,
        facecolor="#ffffff",
        edgecolor="#ced4da",
        linewidth=1.2,
        alpha=0.9
    )
    ax_info.add_patch(fancy_box)
    
    ax_info.text(
        0.5, 0.88,
        "Instrucciones",
        ha="center", va="top",
        fontsize=10, fontweight="600",
        color="#2c3e50",
        transform=ax_info.transAxes
    )
    ax_info.text(
        0.12, 0.68,
        "• Frame\n  Navegación temporal",
        va="top",
        fontsize=8.5,
        color="#495057",
        transform=ax_info.transAxes,
        linespacing=1.5
    )
    ax_info.text(
        0.12, 0.38,
        "• Escala\n  Longitud de vectores",
        va="top",
        fontsize=8.5,
        color="#495057",
        transform=ax_info.transAxes,
        linespacing=1.5
    )
    ax_info.text(
        0.12, 0.08,
        "• Restablecer\n  Valores iniciales",
        va="top",
        fontsize=8.5,
        color="#495057",
        transform=ax_info.transAxes,
        linespacing=1.5
    )

    return ax_panel, s_momento, s_scale, btn_reset


# ---------------------------------------------------------------
# Optimized Artist Manager
# ---------------------------------------------------------------

class ArtistManager:
    """Gestiona artists de matplotlib para blitting eficiente"""

    def __init__(self):
        self.artists: Dict[str, List[Any]] = {}

    def register(self, key: str, artist):
        """Registra un artist para tracking"""
        if key not in self.artists:
            self.artists[key] = []
        if isinstance(artist, list):
            self.artists[key].extend(artist)
        else:
            self.artists[key].append(artist)

    def clear(self, key: str):
        """Limpia artists de una clave"""
        if key in self.artists:
            for artist in self.artists[key]:
                try:
                    artist.remove()
                except Exception:
                    pass
            self.artists[key] = []

    def clear_all(self):
        """Limpia todos los artists"""
        for key in list(self.artists.keys()):
            self.clear(key)


# ---------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------

class PIVViewer:

    def show_initial(self, results: List[PIVResult], names: List[str], cfg: PIVConfig) -> None:
        """Visualización inicial con panel de control mejorado."""
        print("[PIV] Precalculando velocity regions para viewer...", flush=True)
        precomputed = _precompute_hulls(results, cfg.keep_percentile)

        # Configurar estilo moderno
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Layout mejorado con panel más ancho
        fig = plt.figure(figsize=(17.5, 7.2), facecolor='white')
        fig.suptitle("Análisis PIV - Vista Inicial", 
                     fontsize=15, fontweight='600', 
                     color="#2c3e50", y=0.98)
        
        gs = fig.add_gridspec(
            nrows=1,
            ncols=3,
            width_ratios=[1.0, 1.0, 0.45],  # Panel más ancho
            wspace=0.28,
            left=0.05, right=0.98, top=0.93, bottom=0.08
        )

        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])

        _, s_momento, s_scale, btn_reset = _create_right_panel(
            fig=fig,
            panel_spec=gs[0, 2],
            n_frames=len(results),
            frame_init=0,
            scale_init=1.0,
        )

        mm_per_px = cfg.mm_per_px()
        artist_mgr = ArtistManager()

        def draw(idx: int, scale: float) -> None:
            """Dibuja velocidades lineales para el frame idx."""
            r = results[idx]
            valid, hull_closed, inside = precomputed[idx]

            artist_mgr.clear_all()

            for ax in [ax_vel, ax_uv]:
                ax.clear()
                # Estilo mejorado para los ejes
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_color('#34495e')
                ax.spines['bottom'].set_color('#34495e')

            bg = r.bg_display
            h_px, w_px = bg.shape
            extent = [0, w_px * mm_per_px, h_px * mm_per_px, 0]

            # ----- VELOCIDADES LINEALES -----
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.75)

            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            if uvals.size >= 10:
                inside_grid = np.zeros_like(valid, dtype=bool)
                inside_grid[valid] = inside

                ok = inside_grid
                bad = valid & (~inside_grid)

                # Normalización por frame
                speed_all = np.sqrt(r.u_mms**2 + r.v_mms**2)
                max_speed = np.nanmax(speed_all)

                if max_speed > 1e-6:
                    u_norm = r.u_mms / max_speed
                    v_norm = r.v_mms / max_speed
                else:
                    u_norm = r.u_mms
                    v_norm = r.v_mms

                subsample = max(1, r.x_mm[ok].size // 2000)

                # Vectores válidos con mejor color
                q1 = ax_vel.quiver(
                    r.x_mm[ok][::subsample],
                    r.y_mm[ok][::subsample],
                    u_norm[ok][::subsample],
                    v_norm[ok][::subsample],
                    color="#27ae60",
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                    width=cfg.quiver_width * 1.1,
                    alpha=0.85,
                    edgecolors='#1e8449',
                    linewidths=0.3
                )
                
                # Vectores rechazados
                q2 = ax_vel.quiver(
                    r.x_mm[bad][::subsample],
                    r.y_mm[bad][::subsample],
                    u_norm[bad][::subsample],
                    v_norm[bad][::subsample],
                    color="#e67e22",
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                    width=cfg.quiver_width * 1.1,
                    alpha=0.75,
                    edgecolors='#d35400',
                    linewidths=0.3
                )

                artist_mgr.register("vel", [q1, q2])

                # Leyenda mejorada
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='>', color='w', 
                           markerfacecolor='#27ae60', markersize=10,
                           label=f'Validados ({np.sum(ok):,})'),
                    Line2D([0], [0], marker='>', color='w',
                           markerfacecolor='#e67e22', markersize=10,
                           label=f'Rechazados ({np.sum(bad):,})')
                ]
                ax_vel.legend(handles=legend_elements, loc='upper right',
                             framealpha=0.95, edgecolor='#bdc3c7',
                             fontsize=9)

            else:
                q = ax_vel.quiver(
                    r.x_mm[valid],
                    r.y_mm[valid],
                    r.u_mms[valid],
                    r.v_mms[valid],
                    color="#e67e22",
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                    width=cfg.quiver_width
                )
                artist_mgr.register("vel", q)

            ax_vel.set_title(f"Campo de Velocidades — Frame {idx}", 
                           fontweight="600", fontsize=12, 
                           color="#2c3e50", pad=12)
            ax_vel.set_xlabel("x [mm]", fontsize=10, fontweight='500', color='#34495e')
            ax_vel.set_ylabel("y [mm]", fontsize=10, fontweight='500', color='#34495e')
            ax_vel.set_aspect("equal", adjustable="box")
            ax_vel.grid(True, alpha=0.25, linestyle="--", linewidth=0.8, color='#95a5a6')
            ax_vel.tick_params(labelsize=9, colors='#34495e')

            # Scatter u-v mejorado
            if uvals.size >= 10:
                # Puntos validados
                scatter_valid = ax_uv.scatter(
                    uvals[inside], vvals[inside],
                    s=12, alpha=0.65,
                    color="#27ae60",
                    edgecolors="#1e8449",
                    linewidths=0.5,
                    label=f'Validados ({np.sum(inside):,})'
                )
                
                # Puntos rechazados
                scatter_invalid = ax_uv.scatter(
                    uvals[~inside], vvals[~inside],
                    s=12, alpha=0.55,
                    color="#e67e22",
                    edgecolors="#d35400",
                    linewidths=0.5,
                    label=f'Rechazados ({np.sum(~inside):,})'
                )

                # Hull con estilo mejorado
                if hull_closed is not None:
                    ax_uv.plot(
                        hull_closed[:, 0], hull_closed[:, 1],
                        color="#2c3e50",
                        linewidth=2.8,
                        linestyle="-",
                        alpha=0.85,
                        label='Región validación'
                    )
                    ax_uv.scatter(
                        hull_closed[:-1, 0], hull_closed[:-1, 1],
                        s=50,
                        color="#27ae60",
                        zorder=5,
                        edgecolors="#2c3e50",
                        linewidths=1.5,
                        marker='o'
                    )

                ax_uv.legend(loc='upper right', framealpha=0.95, 
                           edgecolor='#bdc3c7', fontsize=9)

            ax_uv.set_title("Espacio de Velocidades (u-v)", 
                          fontweight="600", fontsize=12,
                          color="#2c3e50", pad=12)
            ax_uv.set_xlabel("u [mm/s]", fontsize=10, fontweight='500', color='#34495e')
            ax_uv.set_ylabel("v [mm/s]", fontsize=10, fontweight='500', color='#34495e')
            ax_uv.grid(True, alpha=0.25, linestyle="--", linewidth=0.8, color='#95a5a6')
            ax_uv.axhline(0, color="#7f8c8d", linewidth=1.2, alpha=0.5, linestyle='-')
            ax_uv.axvline(0, color="#7f8c8d", linewidth=1.2, alpha=0.5, linestyle='-')
            ax_uv.tick_params(labelsize=9, colors='#34495e')

            if uvals.size > 0:
                umax = np.percentile(np.abs(uvals), 99)
                vmax = np.percentile(np.abs(vvals), 99)
                lim = max(umax, vmax, 1e-6)
                margin = lim * 0.05
                ax_uv.set_xlim(-lim - margin, lim + margin)
                ax_uv.set_ylim(-lim - margin, lim + margin)

            # Forzar cajas cuadradas
            _force_square_axes(ax_vel, ax_uv)

            fig.canvas.draw_idle()

        def update(_val=None) -> None:
            draw(int(s_momento.val), float(s_scale.val))

        def reset(_event) -> None:
            s_momento.reset()
            s_scale.reset()

        s_momento.on_changed(update)
        s_scale.on_changed(update)
        btn_reset.on_clicked(reset)

        update()
        plt.show()

    def show_final(self, finals: List[PIVResultFinal], names: List[str], cfg: PIVConfig) -> None:
        """Visualización final mejorada con streamlines visibles."""

        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig = plt.figure(figsize=(19.5, 11), facecolor='white')
        fig.suptitle("Análisis PIV — Resultados Finales", 
                     fontsize=16, fontweight='600',
                     color="#2c3e50", y=0.985)
        
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            width_ratios=[1.0, 1.0, 0.45],  # Panel más ancho
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.26,
            left=0.04, right=0.98, top=0.95, bottom=0.05
        )

        # Grilla principal
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        ax_omega = fig.add_subplot(gs[1, 0])
        ax_omega_dist = fig.add_subplot(gs[1, 1])

        # Aplicar estilo a todos los ejes
        for ax in [ax_vel, ax_uv, ax_omega, ax_omega_dist]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_color('#34495e')
            ax.spines['bottom'].set_color('#34495e')

        # Panel derecho ocupando ambas filas
        _, s_momento, s_scale, btn_reset = _create_right_panel(
            fig=fig,
            panel_spec=gs[:, 2],
            n_frames=len(finals),
            frame_init=0,
            scale_init=1.0,
        )

        mm_per_px = cfg.mm_per_px()
        artist_mgr = ArtistManager()

        # Caché para vorticidad
        omega_cache: Dict[int, np.ndarray] = {}

        # Referencias para colorbars persistentes
        cbar_refs: Dict[str, Any] = {}

        def draw(idx: int, scale: float) -> None:
            """Dibuja velocidades lineales, streamlines y vorticidad para el frame idx."""
            r = finals[idx]

            artist_mgr.clear_all()

            for ax in [ax_vel, ax_uv, ax_omega, ax_omega_dist]:
                ax.clear()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_color('#34495e')
                ax.spines['bottom'].set_color('#34495e')

            bg = r.bg_display
            h_px, w_px = bg.shape
            extent = [0, w_px * mm_per_px, h_px * mm_per_px, 0]

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            if uvals.size == 0:
                ax_vel.text(
                    0.5, 0.5, "⚠ Sin datos validados",
                    ha="center", va="center",
                    transform=ax_vel.transAxes,
                    fontsize=14,
                    color="#e74c3c",
                    fontweight="600"
                )
                ax_vel.set_title(f"Campo de Velocidades Validado — Frame {idx}",
                                 fontweight="600", fontsize=12, color="#2c3e50")
                ax_uv.set_title("Espacio de Velocidades (u-v)", 
                              fontweight="600", fontsize=12, color="#2c3e50")
                ax_omega.set_title(f"Campo de Vorticidad — Frame {idx}",
                                   fontweight="600", fontsize=12, color="#2c3e50")
                ax_omega_dist.set_title("Distribución de Vorticidad",
                                        fontweight="600", fontsize=12, color="#2c3e50")

                _force_square_axes(ax_vel, ax_uv, ax_omega, ax_omega_dist)
                fig.canvas.draw_idle()
                return

            # ----- FILA 1: VELOCIDADES LINEALES CON STREAMLINES -----
            # Primero el fondo
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.65)

            speed = np.sqrt(uvals ** 2 + vvals ** 2)
            speed_all = np.sqrt(r.u_mms[valid] ** 2 + r.v_mms[valid] ** 2)

            # Normalización por frame para longitud de flecha
            max_speed = np.nanmax(speed_all)
            if max_speed > 1e-6:
                u_norm = r.u_mms[valid] / max_speed
                v_norm = r.v_mms[valid] / max_speed
            else:
                u_norm = r.u_mms[valid]
                v_norm = r.v_mms[valid]

            # Colormap mejorado
            vmin = float(np.nanpercentile(speed, 1))
            vmax = float(np.nanpercentile(speed, 99))
            if vmax <= vmin:
                vmax = vmin + 1e-6

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap_vel = plt.get_cmap("turbo")

            # STREAMLINES MEJORADAS - Hacerlas más visibles
            try:
                # Preparar datos para interpolación
                u_for_stream = r.u_mms.copy()
                v_for_stream = r.v_mms.copy()
                
                # Reemplazar valores no válidos con interpolación de vecinos
                from scipy.interpolate import griddata
                
                # Obtener puntos válidos
                valid_points = np.column_stack([r.x_mm[valid].ravel(), r.y_mm[valid].ravel()])
                u_valid_vals = r.u_mms[valid].ravel()
                v_valid_vals = r.v_mms[valid].ravel()
                
                # Crear grid completo
                x_grid = r.x_mm
                y_grid = r.y_mm
                grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
                
                # Interpolar para tener campo completo sin NaNs
                if len(valid_points) > 10:
                    u_interpolated = griddata(valid_points, u_valid_vals, grid_points, 
                                             method='linear', fill_value=0.0)
                    v_interpolated = griddata(valid_points, v_valid_vals, grid_points,
                                             method='linear', fill_value=0.0)
                    
                    u_for_stream = u_interpolated.reshape(x_grid.shape)
                    v_for_stream = v_interpolated.reshape(x_grid.shape)
                    
                    # Calcular magnitud para colorear
                    speed_grid = np.sqrt(u_for_stream**2 + v_for_stream**2)
                    
                    # Crear streamplot con alta visibilidad
                    stream = ax_vel.streamplot(
                        r.x_mm[0, :], r.y_mm[:, 0],
                        u_for_stream, v_for_stream,
                        color=speed_grid,
                        cmap=cmap_vel,
                        norm=norm,
                        density=1.8,  # Más denso
                        linewidth=2.0,  # Líneas más gruesas
                        arrowsize=1.5,  # Flechas más grandes
                        arrowstyle='->',
                        alpha=0.85,  # Más opaco
                        zorder=2  # Por encima del fondo
                    )
                    artist_mgr.register("stream", stream.lines)
                    
            except Exception as e:
                print(f"[PIV] Advertencia streamlines: {e}")

            # Vectores quiver MUY subsampleados para no ocultar streamlines
            subsample = max(1, r.x_mm[valid].size // 800)  # Menos vectores

            q = ax_vel.quiver(
                r.x_mm[valid][::subsample],
                r.y_mm[valid][::subsample],
                u_norm[::subsample],
                v_norm[::subsample],
                speed_all[::subsample],
                cmap=cmap_vel,
                norm=norm,
                angles="xy",
                scale_units="xy",
                scale=scale,
                width=cfg.quiver_width * 0.8,  # Más delgados
                alpha=0.4,  # Más transparentes
                edgecolors='none',
                zorder=3  # Por encima de streamlines
            )

            if "vel" not in cbar_refs:
                cbar_refs["vel"] = fig.colorbar(
                    q, ax=ax_vel, label="Velocidad [mm/s]",
                    fraction=0.046, pad=0.04
                )
                cbar_refs["vel"].ax.tick_params(labelsize=9)
            else:
                cbar_refs["vel"].update_normal(q)

            artist_mgr.register("vel", q)

            ax_vel.set_title(f"Campo de Velocidades + Streamlines — Frame {idx}",
                             fontweight="600", fontsize=12, color="#2c3e50", pad=12)
            ax_vel.set_xlabel("x [mm]", fontsize=10, fontweight='500', color='#34495e')
            ax_vel.set_ylabel("y [mm]", fontsize=10, fontweight='500', color='#34495e')
            ax_vel.set_aspect("equal", adjustable="box")
            ax_vel.grid(True, alpha=0.20, linestyle="--", linewidth=0.8, color='#95a5a6', zorder=1)
            ax_vel.tick_params(labelsize=9, colors='#34495e')

            # Scatter u-v con gradiente de color
            scatter = ax_uv.scatter(
                uvals, vvals,
                c=speed,
                cmap=cmap_vel,
                norm=norm,
                s=14,
                alpha=0.70,
                edgecolors="#2c3e50",
                linewidths=0.3
            )
            
            ax_uv.set_title("Espacio de Velocidades (u-v)", 
                          fontweight="600", fontsize=12, color="#2c3e50", pad=12)
            ax_uv.set_xlabel("u [mm/s]", fontsize=10, fontweight='500', color='#34495e')
            ax_uv.set_ylabel("v [mm/s]", fontsize=10, fontweight='500', color='#34495e')
            ax_uv.grid(True, alpha=0.25, linestyle="--", linewidth=0.8, color='#95a5a6')
            ax_uv.axhline(0, color="#7f8c8d", linewidth=1.2, alpha=0.5)
            ax_uv.axvline(0, color="#7f8c8d", linewidth=1.2, alpha=0.5)
            ax_uv.tick_params(labelsize=9, colors='#34495e')

            umax = np.percentile(np.abs(uvals), 99)
            vmax_ax = np.percentile(np.abs(vvals), 99)
            lim = max(umax, vmax_ax, 1e-6)
            margin = lim * 0.05
            ax_uv.set_xlim(-lim - margin, lim + margin)
            ax_uv.set_ylim(-lim - margin, lim + margin)

            # ----- FILA 2: VORTICIDAD -----
            if idx not in omega_cache:
                omega = _compute_vorticity(r.u_mms, r.v_mms, r.x_mm, r.y_mm, valid)
                omega_cache[idx] = omega
            else:
                omega = omega_cache[idx]

            ax_omega.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.6)

            omega_valid = omega[valid]
            if omega_valid.size > 0 and np.any(np.isfinite(omega_valid)):
                omega_finite = np.isfinite(omega)
                if np.sum(omega_finite) > 10:
                    max_omega_abs = np.nanmax(np.abs(omega_valid[np.isfinite(omega_valid)]))

                    if max_omega_abs > 1e-6:
                        omega_norm = omega / max_omega_abs
                    else:
                        omega_norm = omega

                    omega_norm_masked = omega_norm.copy()
                    omega_norm_masked[~valid] = np.nan

                    omega_smooth = gaussian_filter(np.nan_to_num(omega_norm_masked, 0), sigma=1.2)
                    omega_smooth[~valid] = np.nan

                    vmin_om = -1.0
                    vmax_om = 1.0
                    levels = np.linspace(vmin_om, vmax_om, 25)

                    # Usar colormap divergente mejorado
                    contf = ax_omega.contourf(
                        r.x_mm, r.y_mm, omega_smooth,
                        levels=levels,
                        cmap="RdBu_r",
                        alpha=0.80,
                        extend="both"
                    )

                    if "omega" not in cbar_refs:
                        cbar_refs["omega"] = plt.colorbar(
                            contf, ax=ax_omega,
                            label="Vorticidad normalizada (ω/ω_max)",
                            fraction=0.046, pad=0.04
                        )
                        cbar_refs["omega"].ax.tick_params(labelsize=9)
                    else:
                        cbar_refs["omega"].update_normal(contf)

                    artist_mgr.register("omega", contf)

            ax_omega.set_title(f"Campo de Vorticidad — Frame {idx}",
                               fontweight="600", fontsize=12, color="#2c3e50", pad=12)
            ax_omega.set_xlabel("x [mm]", fontsize=10, fontweight='500', color='#34495e')
            ax_omega.set_ylabel("y [mm]", fontsize=10, fontweight='500', color='#34495e')
            ax_omega.set_aspect("equal", adjustable="box")
            ax_omega.grid(True, alpha=0.25, linestyle="--", linewidth=0.8, color='#95a5a6')
            ax_omega.tick_params(labelsize=9, colors='#34495e')

            # Distribución de vorticidad mejorada
            if omega_valid.size > 0:
                omega_finite_vals = omega_valid[np.isfinite(omega_valid)]
                if omega_finite_vals.size > 0:
                    # Histograma con mejor diseño
                    n, bins, patches = ax_omega_dist.hist(
                        omega_finite_vals,
                        bins=45,
                        color="#3498db",
                        alpha=0.75,
                        edgecolor="#2c3e50",
                        linewidth=1.2
                    )
                    
                    # Colorear barras según valor
                    cm = plt.get_cmap('RdBu_r')
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
                    for c, p in zip(col, patches):
                        plt.setp(p, 'facecolor', cm(c))
                    
                    # Líneas de referencia
                    ax_omega_dist.axvline(
                        0,
                        color="#e74c3c",
                        linestyle="--",
                        linewidth=2.5,
                        alpha=0.85,
                        label="ω = 0"
                    )
                    median_val = np.median(omega_finite_vals)
                    ax_omega_dist.axvline(
                        median_val,
                        color="#f39c12",
                        linestyle="--",
                        linewidth=2.5,
                        alpha=0.85,
                        label=f"Mediana = {median_val:.2f}"
                    )
                    
                    ax_omega_dist.legend(fontsize=9, framealpha=0.95, 
                                       edgecolor='#bdc3c7')

            ax_omega_dist.set_title("Distribución de Vorticidad",
                                    fontweight="600", fontsize=12, color="#2c3e50", pad=12)
            ax_omega_dist.set_xlabel("ω [1/s]", fontsize=10, fontweight='500', color='#34495e')
            ax_omega_dist.set_ylabel("Frecuencia", fontsize=10, fontweight='500', color='#34495e')
            ax_omega_dist.grid(True, alpha=0.25, axis="y", linestyle="--", 
                             linewidth=0.8, color='#95a5a6')
            ax_omega_dist.tick_params(labelsize=9, colors='#34495e')

            # Forzar que todos los paneles sean cuadrados
            _force_square_axes(ax_vel, ax_uv, ax_omega, ax_omega_dist)

            fig.canvas.draw_idle()

        def update(_val=None) -> None:
            draw(int(s_momento.val), float(s_scale.val))

        def reset(_event) -> None:
            s_momento.reset()
            s_scale.reset()

        s_momento.on_changed(update)
        s_scale.on_changed(update)
        btn_reset.on_clicked(reset)

        update()
        plt.show()