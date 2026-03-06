# viewer.py
from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .models import PIVResult, PIVResultFinal
from .config import PIVConfig
from .validation import velocity_region_mask


class PIVViewer:
    def show_initial(self, results: List[PIVResult], names: List[str], cfg: PIVConfig) -> None:
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.55, 1.0])
        ax = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        plt.subplots_adjust(bottom=0.18, wspace=0.25)

        ax_momento = plt.axes([0.15, 0.08, 0.62, 0.03])
        s_momento = Slider(ax_momento, "Momento", 0, len(results) - 1, valinit=0, valstep=1)

        ax_scale = plt.axes([0.15, 0.03, 0.62, 0.03])
        s_scale = Slider(ax_scale, "Escala", 0.5, 80.0, valinit=cfg.default_quiver_scale, valstep=0.5)

        mm_per_px = cfg.mm_per_px()

        def draw(idx: int, scale: float) -> None:
            r = results[idx]
            ax.clear()
            ax_uv.clear()

            bg = r.bg_display
            h_px, w_px = bg.shape
            ax.imshow(bg, cmap="gray", origin="upper",
                      extent=[0, w_px * mm_per_px, h_px * mm_per_px, 0])

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)
            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            if uvals.size < 10:
                ax.quiver(r.x_mm[valid], r.y_mm[valid], r.u_mms[valid], r.v_mms[valid],
                          color="orange", angles="xy", scale_units="xy", scale=scale, width=cfg.quiver_width)
                ax.set_title(f"PIV (pocos datos): {names[idx]}")
                ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")
                ax_uv.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
                fig.canvas.draw_idle()
                return

            hull_closed, inside = velocity_region_mask(uvals, vvals, keep_percentile=cfg.keep_percentile)
            inside_grid = np.zeros_like(valid, dtype=bool)
            inside_grid[valid] = inside

            ok = inside_grid
            bad = valid & (~inside_grid)

            ax.quiver(r.x_mm[ok],  r.y_mm[ok],  r.u_mms[ok],  r.v_mms[ok],
                      color="limegreen", angles="xy", scale_units="xy", scale=scale, width=cfg.quiver_width)
            ax.quiver(r.x_mm[bad], r.y_mm[bad], r.u_mms[bad], r.v_mms[bad],
                      color="orange", angles="xy", scale_units="xy", scale=scale, width=cfg.quiver_width)

            ax.set_title(f"PIV: {names[idx]}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")

            ax_uv.scatter(uvals[inside], vvals[inside], s=6, alpha=0.45, color="limegreen")
            ax_uv.scatter(uvals[~inside], vvals[~inside], s=6, alpha=0.45, color="orange")

            if hull_closed is not None:
                ax_uv.plot(hull_closed[:, 0], hull_closed[:, 1], color="black", linewidth=2)
                ax_uv.scatter(hull_closed[:-1, 0], hull_closed[:-1, 1], s=30, color="limegreen", zorder=5)

            ax_uv.set_title("Velocity-based validation (región)")
            ax_uv.set_xlabel("u [mm/s]")
            ax_uv.set_ylabel("v [mm/s]")
            ax_uv.grid(True, alpha=0.2)

            umax = np.percentile(np.abs(uvals), 99)
            vmax = np.percentile(np.abs(vvals), 99)
            lim = max(umax, vmax, 1e-6)
            ax_uv.set_xlim(-lim, lim)
            ax_uv.set_ylim(-lim, lim)

            fig.canvas.draw_idle()

        def update(_val=None) -> None:
            draw(int(s_momento.val), float(s_scale.val))

        s_momento.on_changed(update)
        s_scale.on_changed(update)
        update()
        plt.show()

    def show_final(self, finals: List[PIVResultFinal], names: List[str], cfg: PIVConfig) -> None:
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.55, 1.0])
        ax = fig.add_subplot(gs[0, 0])
        ax_uv = fig.add_subplot(gs[0, 1])
        plt.subplots_adjust(bottom=0.18, wspace=0.25)

        ax_momento = plt.axes([0.15, 0.08, 0.62, 0.03])
        s_momento = Slider(ax_momento, "Momento", 0, len(finals) - 1, valinit=0, valstep=1)

        ax_scale = plt.axes([0.15, 0.03, 0.62, 0.03])
        s_scale = Slider(ax_scale, "Escala", 0.5, 80.0, valinit=cfg.default_quiver_scale, valstep=0.5)

        mm_per_px = cfg.mm_per_px()

        def draw(idx: int, scale: float) -> None:
            r = finals[idx]
            ax.clear()
            ax_uv.clear()

            bg = r.bg_display
            h_px, w_px = bg.shape
            ax.imshow(bg, cmap="gray", origin="upper",
                      extent=[0, w_px * mm_per_px, h_px * mm_per_px, 0])

            valid = np.isfinite(r.u_mms) & np.isfinite(r.v_mms) & (~r.in_mask)

            ax.quiver(r.x_mm[valid], r.y_mm[valid], r.u_mms[valid], r.v_mms[valid],
                      color="limegreen", angles="xy", scale_units="xy", scale=scale, width=cfg.quiver_width)

            ax.set_title(f"FINAL (validado): {names[idx]}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")

            uvals = r.u_mms[valid]
            vvals = r.v_mms[valid]

            if uvals.size > 0:
                ax_uv.scatter(uvals, vvals, s=6, alpha=0.45, color="limegreen")
                ax_uv.set_title("u-v final (scatter)")
                ax_uv.set_xlabel("u [mm/s]")
                ax_uv.set_ylabel("v [mm/s]")
                ax_uv.grid(True, alpha=0.2)

                umax = np.percentile(np.abs(uvals), 99)
                vmax = np.percentile(np.abs(vvals), 99)
                lim = max(umax, vmax, 1e-6)
                ax_uv.set_xlim(-lim, lim)
                ax_uv.set_ylim(-lim, lim)
            else:
                ax_uv.text(0.5, 0.5, "Sin datos", ha="center", va="center")

            fig.canvas.draw_idle()

        def update(_val=None) -> None:
            draw(int(s_momento.val), float(s_scale.val))

        s_momento.on_changed(update)
        s_scale.on_changed(update)
        update()
        plt.show()