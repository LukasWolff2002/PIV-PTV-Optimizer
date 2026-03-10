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
            hull_closed, inside = velocity_region_mask(uvals, vvals, keep_percentile=keep_percentile)
        else:
            hull_closed = None
            inside = np.ones(uvals.size, dtype=bool)

        precomputed.append((valid, hull_closed, inside))
    return precomputed


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
                except:
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
        """Visualización inicial con grid 1x2 (solo velocidades lineales)"""
        print("[PIV] Precalculando velocity regions para viewer...", flush=True)
        precomputed = _precompute_hulls(results, cfg.keep_percentile)

        # Crear figura con grid 1x2
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(nrows=1, ncols=2, 
                             width_ratios=[1.4, 1.0],
                             hspace=0.30, wspace=0.25)
        
        # Fila 1: Velocidades lineales
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        
        plt.subplots_adjust(bottom=0.12)

        # Sliders
        ax_momento = plt.axes([0.15, 0.06, 0.55, 0.025])
        s_momento = Slider(ax_momento, "Frame", 0, len(results) - 1, 
                          valinit=0, valstep=1, color='steelblue')

        ax_scale = plt.axes([0.15, 0.02, 0.55, 0.025])
        s_scale = Slider(ax_scale, "Escala", 0.1, 10.0, 
                        valinit=1.0, valstep=0.1,
                        color='coral')
        
        # Botón de reset
        ax_reset = plt.axes([0.75, 0.02, 0.08, 0.055])
        btn_reset = Button(ax_reset, 'Reset', color='lightgray', hovercolor='gray')

        mm_per_px = cfg.mm_per_px()
        artist_mgr = ArtistManager()

        def draw(idx: int, scale: float, force_redraw: bool = False) -> None:
            """Dibuja velocidades lineales para el frame idx"""
            r = results[idx]
            valid, hull_closed, inside = precomputed[idx]
            
            # Limpiar artists previos
            artist_mgr.clear_all()
            
            # Limpiar ejes
            for ax in [ax_vel, ax_uv]:
                ax.clear()

            bg = r.bg_display
            h_px, w_px = bg.shape
            extent = [0, w_px * mm_per_px, h_px * mm_per_px, 0]

            # ----- VELOCIDADES LINEALES -----
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.7)

            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            if uvals.size >= 10:
                inside_grid = np.zeros_like(valid, dtype=bool)
                inside_grid[valid] = inside

                ok = inside_grid
                bad = valid & (~inside_grid)

                # NORMALIZACIÓN: calcular el máximo de velocidad en este timestep
                speed_all = np.sqrt(r.u_mms**2 + r.v_mms**2)
                max_speed = np.nanmax(speed_all)
                
                if max_speed > 1e-6:
                    # Normalizar u, v por el máximo
                    u_norm = r.u_mms / max_speed
                    v_norm = r.v_mms / max_speed
                else:
                    u_norm = r.u_mms
                    v_norm = r.v_mms
                
                # Subsampling si hay muchos vectores
                subsample = max(1, len(r.x_mm[ok].flatten()) // 2000)
                
                q1 = ax_vel.quiver(r.x_mm[ok][::subsample], r.y_mm[ok][::subsample],
                                   u_norm[ok][::subsample], v_norm[ok][::subsample],
                                   color="limegreen", angles="xy", scale_units="xy",
                                   scale=scale, width=cfg.quiver_width, alpha=0.8)
                q2 = ax_vel.quiver(r.x_mm[bad][::subsample], r.y_mm[bad][::subsample],
                                   u_norm[bad][::subsample], v_norm[bad][::subsample],
                                   color="orange", angles="xy", scale_units="xy",
                                   scale=scale, width=cfg.quiver_width, alpha=0.8)
                
                artist_mgr.register('vel', [q1, q2])
            else:
                q = ax_vel.quiver(r.x_mm[valid], r.y_mm[valid], 
                                 r.u_mms[valid], r.v_mms[valid],
                                 color="orange", angles="xy", scale_units="xy",
                                 scale=scale, width=cfg.quiver_width)
                artist_mgr.register('vel', q)

            ax_vel.set_title(f"Campo de Velocidades - Frame {idx}", fontweight='bold', fontsize=12)
            ax_vel.set_xlabel("x [mm]")
            ax_vel.set_ylabel("y [mm]")
            ax_vel.set_aspect("equal")
            ax_vel.grid(True, alpha=0.2, linestyle='--')

            # Scatter u-v
            if uvals.size >= 10:
                ax_uv.scatter(uvals[inside], vvals[inside], s=8, alpha=0.5, 
                            color="limegreen", edgecolors='darkgreen', linewidths=0.3)
                ax_uv.scatter(uvals[~inside], vvals[~inside], s=8, alpha=0.5, 
                            color="orange", edgecolors='darkorange', linewidths=0.3)

                if hull_closed is not None:
                    ax_uv.plot(hull_closed[:, 0], hull_closed[:, 1], 
                             color="black", linewidth=2.5, linestyle='-', alpha=0.8)
                    ax_uv.scatter(hull_closed[:-1, 0], hull_closed[:-1, 1],
                                s=40, color="limegreen", zorder=5, 
                                edgecolors='black', linewidths=1)

            ax_uv.set_title("Espacio de Velocidades (u-v)", fontweight='bold', fontsize=12)
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")
            ax_uv.grid(True, alpha=0.3)
            ax_uv.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax_uv.axvline(0, color='k', linewidth=0.5, alpha=0.3)

            if uvals.size > 0:
                umax = np.percentile(np.abs(uvals), 99)
                vmax = np.percentile(np.abs(vvals), 99)
                lim = max(umax, vmax, 1e-6)
                ax_uv.set_xlim(-lim, lim)
                ax_uv.set_ylim(-lim, lim)

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
        """Visualización final con grid 2x2 (velocidades + vorticidad)"""
        
        # Crear figura con grid 2x2 - campos más grandes
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(nrows=2, ncols=2,
                             width_ratios=[2.0, 1.0],  # ← campos 2× más anchos
                             height_ratios=[1.0, 1.0],
                             hspace=0.30, wspace=0.30)
        
        # Fila 1: Velocidades lineales
        ax_vel = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        
        # Fila 2: Velocidades angulares
        ax_omega = fig.add_subplot(gs[1, 0])
        ax_omega_dist = fig.add_subplot(gs[1, 1])
        
        plt.subplots_adjust(bottom=0.12)

        # Sliders - mismo rango y valor inicial que show_initial
        ax_momento = plt.axes([0.15, 0.06, 0.55, 0.025])
        s_momento = Slider(ax_momento, "Frame", 0, len(finals) - 1,
                          valinit=0, valstep=1, color='steelblue')

        ax_scale = plt.axes([0.15, 0.02, 0.55, 0.025])
        s_scale = Slider(ax_scale, "Escala", 0.1, 10.0,
                        valinit=1.0, valstep=0.1,
                        color='coral')
        
        # Botón de reset
        ax_reset = plt.axes([0.75, 0.02, 0.08, 0.055])
        btn_reset = Button(ax_reset, 'Reset', color='lightgray', hovercolor='gray')

        mm_per_px = cfg.mm_per_px()
        artist_mgr = ArtistManager()
        
        # Caché para vorticidad
        omega_cache: Dict[int, np.ndarray] = {}
        
        # Referencias para colorbars persistentes
        cbar_refs: Dict[str, Any] = {}

        def draw(idx: int, scale: float) -> None:
            """Dibuja velocidades lineales y vorticidad para el frame idx"""
            r = finals[idx]
            
            # Limpiar artists previos
            artist_mgr.clear_all()
            
            # Limpiar ejes
            for ax in [ax_vel, ax_uv, ax_omega, ax_omega_dist]:
                ax.clear()

            bg = r.bg_display
            h_px, w_px = bg.shape
            extent = [0, w_px * mm_per_px, h_px * mm_per_px, 0]

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            if uvals.size == 0:
                ax_vel.text(0.5, 0.5, "Sin datos validados", 
                          ha="center", va="center", transform=ax_vel.transAxes,
                          fontsize=14, color='red')
                fig.canvas.draw_idle()
                return

            # ----- FILA 1: VELOCIDADES LINEALES -----
            ax_vel.imshow(bg, cmap="gray", origin="upper", extent=extent, alpha=0.7)

            speed = np.sqrt(uvals ** 2 + vvals ** 2)
            speed_all = np.sqrt(r.u_mms[valid] ** 2 + r.v_mms[valid] ** 2)

            # NORMALIZACIÓN: calcular el máximo de velocidad en este timestep
            max_speed = np.nanmax(speed_all)
            
            if max_speed > 1e-6:
                # Normalizar u, v por el máximo
                u_norm = r.u_mms[valid] / max_speed
                v_norm = r.v_mms[valid] / max_speed
            else:
                u_norm = r.u_mms[valid]
                v_norm = r.v_mms[valid]

            # Colormap por magnitud (sin normalizar para color)
            vmin = float(np.nanpercentile(speed, 1))
            vmax = float(np.nanpercentile(speed, 99))
            if vmax <= vmin:
                vmax = vmin + 1e-6

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap_vel = plt.get_cmap("plasma")  # Mismo colormap que vorticidad

            # Subsampling inteligente
            subsample = max(1, len(r.x_mm[valid].flatten()) // 2000)

            q = ax_vel.quiver(
                r.x_mm[valid][::subsample], r.y_mm[valid][::subsample],
                u_norm[::subsample], v_norm[::subsample],
                speed_all[::subsample],
                cmap=cmap_vel, norm=norm,
                angles="xy", scale_units="xy",
                scale=scale, width=cfg.quiver_width,
                alpha=0.85
            )

            # Colorbar persistente
            if 'vel' not in cbar_refs:
                cbar_refs['vel'] = fig.colorbar(q, ax=ax_vel, label="|v| [mm/s]",
                                               fraction=0.046, pad=0.04)
            else:
                cbar_refs['vel'].update_normal(q)

            artist_mgr.register('vel', q)

            ax_vel.set_title(f"Campo de Velocidades Validado - Frame {idx}", 
                           fontweight='bold', fontsize=12)
            ax_vel.set_xlabel("x [mm]")
            ax_vel.set_ylabel("y [mm]")
            ax_vel.set_aspect("equal")
            ax_vel.grid(True, alpha=0.2, linestyle='--')

            # Scatter u-v coloreado
            sc = ax_uv.scatter(uvals, vvals, c=speed, cmap=cmap_vel, norm=norm,
                             s=10, alpha=0.6, edgecolors='black', linewidths=0.2)
            ax_uv.set_title("Espacio de Velocidades (u-v)", fontweight='bold', fontsize=12)
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")
            ax_uv.grid(True, alpha=0.3)
            ax_uv.axhline(0, color='k', linewidth=0.5, alpha=0.3)
            ax_uv.axvline(0, color='k', linewidth=0.5, alpha=0.3)

            umax = np.percentile(np.abs(uvals), 99)
            vmax_ax = np.percentile(np.abs(vvals), 99)
            lim = max(umax, vmax_ax, 1e-6)
            ax_uv.set_xlim(-lim, lim)
            ax_uv.set_ylim(-lim, lim)

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
                    # NORMALIZACIÓN DE VORTICIDAD: normalizar por el máximo absoluto
                    max_omega_abs = np.nanmax(np.abs(omega_valid[np.isfinite(omega_valid)]))
                    
                    if max_omega_abs > 1e-6:
                        omega_norm = omega / max_omega_abs
                    else:
                        omega_norm = omega
                    
                    # IMPORTANTE: poner NaN en zonas enmascaradas para que aparezcan blancas
                    omega_norm_masked = omega_norm.copy()
                    omega_norm_masked[~valid] = np.nan
                    
                    # Suavizar para mejor visualización
                    omega_smooth = gaussian_filter(np.nan_to_num(omega_norm_masked, 0), sigma=1.0)
                    
                    # Restaurar NaN después del suavizado para que aparezcan blancas
                    omega_smooth[~valid] = np.nan
                    
                    # Usar rango simétrico normalizado
                    vmin_om = -1.0
                    vmax_om = 1.0
                    
                    levels = np.linspace(vmin_om, vmax_om, 21)
                    contf = ax_omega.contourf(r.x_mm, r.y_mm, omega_smooth,
                                              levels=levels, cmap='plasma',  # Mismo colormap
                                              alpha=0.7, extend='neither')
                    
                    if 'omega' not in cbar_refs:
                        cbar_refs['omega'] = plt.colorbar(contf, ax=ax_omega, 
                                                         label='ω/ω_max',
                                                         fraction=0.046, pad=0.04)
                    
                    artist_mgr.register('omega', contf)

            ax_omega.set_title(f"Campo de Vorticidad - Frame {idx}", 
                             fontweight='bold', fontsize=12)
            ax_omega.set_xlabel("x [mm]")
            ax_omega.set_ylabel("y [mm]")
            ax_omega.set_aspect("equal")
            ax_omega.grid(True, alpha=0.2, linestyle='--')

            # Distribución de vorticidad (valores reales, no normalizados)
            if omega_valid.size > 0:
                omega_finite_vals = omega_valid[np.isfinite(omega_valid)]
                if omega_finite_vals.size > 0:
                    ax_omega_dist.hist(omega_finite_vals, bins=40, color='steelblue',
                                      alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax_omega_dist.axvline(0, color='red', linestyle='--',
                                         linewidth=2, alpha=0.7, label='ω=0')
                    ax_omega_dist.axvline(np.median(omega_finite_vals),
                                         color='orange', linestyle='--',
                                         linewidth=2, alpha=0.7, label='Mediana')
                    ax_omega_dist.legend(fontsize=9)

            ax_omega_dist.set_title("Distribución de Vorticidad", fontweight='bold', fontsize=12)
            ax_omega_dist.set_xlabel("ω [1/s]")
            ax_omega_dist.set_ylabel("Frecuencia")
            ax_omega_dist.grid(True, alpha=0.3, axis='y')

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