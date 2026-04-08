"""
VISUALIZACIÓN Y RENDERING DE ANIMACIONES PIV
============================================
Genera videos a partir de datos pre-computados.
"""

from pathlib import Path
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

from Analisis.PIV.InterpolarPIV.piv_config import (
    CACHE_DIR, VISUALIZATION_SMOOTH_SIGMA, nan_gaussian
)


# ============================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================

# Animación
ANIMATION_FPS = 30
TIME_SPEEDUP = 1.0

# Subsampling temporal (para mantener velocidad real del tiempo)
# Si True, solo muestra frames necesarios para mantener ANIMATION_FPS en tiempo real
# Si False, muestra todos los frames disponibles
REALTIME_PLAYBACK = True  # Mantener velocidad real del experimento

# Visualización
SHOW_STATISTICS = True
COLORMAP = "turbo"

# Escalas dinámicas - percentiles calculados POR FRAME
VMIN_PERCENTILE = 5
VMAX_PERCENTILE = 95

# Salida
OUTPUT_VIDEO = Path("piv_animation.mp4")
VIDEO_DPI = 150
VIDEO_BITRATE = 5000

# Hardware encoding
USE_HARDWARE_ENCODER = True


# ============================================================
# UTILIDADES FFMPEG
# ============================================================

def check_ffmpeg_available():
    """Verifica si FFmpeg está disponible"""
    return shutil.which('ffmpeg') is not None


def check_videotoolbox_available():
    """Verifica si VideoToolbox (GPU encoding) está disponible"""
    if not check_ffmpeg_available():
        return False
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                              capture_output=True, text=True, timeout=5)
        return 'h264_videotoolbox' in result.stdout
    except:
        return False


# ============================================================
# UTILIDADES DE TIMELINE
# ============================================================

def subsample_timeline_for_realtime(timeline: np.ndarray, target_fps: float) -> tuple:
    """
    Subsamplea timeline para mantener velocidad de reproducción en tiempo real.
    Selecciona frames cada 1/fps segundos, no uniformemente.
    
    Args:
        timeline: Array de timestamps en segundos
        target_fps: FPS objetivo del video
    
    Returns:
        (indices_subsampled, timeline_subsampled)
    """
    if len(timeline) == 0:
        return np.array([]), np.array([])
    
    # Calcular intervalo de tiempo entre frames
    dt = 1.0 / target_fps  # segundos entre frames
    
    # Construir timeline objetivo (cada dt segundos)
    t_start = timeline[0]
    t_end = timeline[-1]
    duration_s = t_end - t_start
    
    target_times = np.arange(t_start, t_end + dt, dt)
    
    # Para cada tiempo objetivo, encontrar el frame más cercano
    indices = []
    used_frames = set()
    
    for target_t in target_times:
        # Encontrar el índice del frame más cercano
        idx = np.argmin(np.abs(timeline - target_t))
        
        # Evitar usar el mismo frame dos veces
        if idx not in used_frames:
            indices.append(idx)
            used_frames.add(idx)
    
    indices = np.array(indices)
    subsampled_timeline = timeline[indices]
    
    print(f"\n⏱️  Subsampling temporal (basado en tiempo real):")
    print(f"   Duración real: {duration_s:.2f} s")
    print(f"   FPS objetivo: {target_fps}")
    print(f"   Intervalo entre frames: {dt*1000:.2f} ms")
    print(f"   Frames disponibles: {len(timeline)}")
    print(f"   Frames seleccionados: {len(indices)} ({len(timeline) - len(indices)} descartados)")
    
    # Análisis de distribución
    if len(indices) > 1:
        time_diffs = np.diff(subsampled_timeline)
        print(f"   Tiempo entre frames: min={time_diffs.min()*1000:.2f}ms, "
              f"max={time_diffs.max()*1000:.2f}ms, "
              f"promedio={time_diffs.mean()*1000:.2f}ms")
    
    return indices, subsampled_timeline


# ============================================================
# SUAVIZADO PARA VISUALIZACIÓN
# ============================================================

def smooth_field_for_visualization(field, sigma=None):
    """Aplica suavizado adicional solo para visualización"""
    if sigma is None:
        sigma = VISUALIZATION_SMOOTH_SIGMA
    if sigma <= 0:
        return field
    valid_mask = np.isfinite(field)
    if not np.any(valid_mask):
        return field
    return nan_gaussian(field, sigma=sigma, allowed_mask=valid_mask)


# ============================================================
# CLASE DE ANIMACIÓN
# ============================================================

class PIVAnimator:
    """Renderiza animación leyendo archivos pre-computados"""
    
    def __init__(self):
        # Verificar que existe caché
        if not CACHE_DIR.exists() or not (CACHE_DIR / "metadata.npz").exists():
            raise FileNotFoundError(
                f"No se encontró caché en {CACHE_DIR}. "
                "Ejecuta primero piv_interpolate.py"
            )
        
        # Cargar metadatos
        meta = np.load(CACHE_DIR / "metadata.npz")
        self.X = meta['X']
        self.Y = meta['Y']
        self.dx = float(meta['dx'])
        self.dy = float(meta['dy'])
        timeline_full = meta['timeline']
        
        # Aplicar subsampling si está habilitado
        if REALTIME_PLAYBACK:
            self.frame_indices, self.timeline = subsample_timeline_for_realtime(
                timeline_full, ANIMATION_FPS * TIME_SPEEDUP
            )
        else:
            self.frame_indices = np.arange(len(timeline_full))
            self.timeline = timeline_full
        
        self.x_min = float(self.X.min())
        self.x_max = float(self.X.max())
        self.y_min = float(self.Y.min())
        self.y_max = float(self.Y.max())
        
        print("\n" + "="*70)
        print("RENDERING DE ANIMACIÓN (escalas dinámicas + layout vertical)")
        print("="*70)
        print(f"Frames totales: {len(timeline_full)}")
        if REALTIME_PLAYBACK:
            print(f"Frames a renderizar: {len(self.timeline)} (realtime @ {ANIMATION_FPS} FPS)")
        else:
            print(f"Frames a renderizar: {len(self.timeline)} (todos los frames)")
        print(f"Dominio: X=[{self.x_min:.1f}, {self.x_max:.1f}], Y=[{self.y_min:.1f}, {self.y_max:.1f}] mm")
        
        # Calcular aspect ratio del dominio
        domain_width = self.x_max - self.x_min
        domain_height = self.y_max - self.y_min
        domain_aspect = domain_width / domain_height
        
        print(f"Aspect ratio del dominio: {domain_aspect:.2f} (ancho/alto)")
        
        # LAYOUT VERTICAL: un gráfico arriba, otro abajo
        # Cada subplot mantiene el aspect ratio real del dominio
        fig_width = 12
        subplot_height = fig_width / domain_aspect
        fig_height = 2 * subplot_height + 1.5  # Espacio para títulos
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1,  # 2 filas, 1 columna
            figsize=(fig_width, fig_height)
        )
        
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax1.set_facecolor('#2a2a2a')
        self.ax2.set_facecolor('#2a2a2a')
        
        self.pcolormesh_velocity = None
        self.pcolormesh_vorticity = None
        self.cbar_velocity = None
        self.cbar_vorticity = None
        self.time_text = None
        self.stats_text = None
    
    def init_frame(self):
        """Inicializa visualización"""
        for ax in [self.ax1, self.ax2]:
            ax.set_xlabel('X (mm)', color='white', fontsize=12)
            ax.set_ylabel('Y (mm)', color='white', fontsize=12)
            ax.tick_params(colors='white', labelsize=10)
            ax.set_aspect('equal', adjustable='box')  # Mantener aspecto real
            ax.grid(True, alpha=0.2, color='white', linestyle=':', linewidth=0.5)
            ax.invert_yaxis()
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_max, self.y_min)
        
        dummy = np.full_like(self.X, np.nan)
        
        # Panel superior: Velocidad
        self.pcolormesh_velocity = self.ax1.pcolormesh(
            self.X, self.Y, dummy, shading="auto", cmap=COLORMAP,
            vmin=0, vmax=100, zorder=1  # Valores iniciales dummy
        )
        self.cbar_velocity = self.fig.colorbar(
            self.pcolormesh_velocity, ax=self.ax1, 
            label='Velocidad (mm/s)', pad=0.02, fraction=0.046
        )
        self.cbar_velocity.ax.tick_params(colors='white', labelsize=9)
        self.cbar_velocity.set_label('Velocidad (mm/s)', color='white', fontsize=11)
        self.ax1.set_title('Magnitud de Velocidad', color='white', fontsize=14, pad=10)
        
        # Panel inferior: Vorticidad
        self.pcolormesh_vorticity = self.ax2.pcolormesh(
            self.X, self.Y, dummy, shading="auto", cmap=COLORMAP,
            vmin=-10, vmax=10, zorder=1  # Valores iniciales dummy
        )
        self.cbar_vorticity = self.fig.colorbar(
            self.pcolormesh_vorticity, ax=self.ax2, 
            label='Vorticidad (1/s)', pad=0.02, fraction=0.046
        )
        self.cbar_vorticity.ax.tick_params(colors='white', labelsize=9)
        self.cbar_vorticity.set_label('Vorticidad (1/s)', color='white', fontsize=11)
        self.ax2.set_title('Vorticidad (ω = ∂v/∂x - ∂u/∂y)', color='white', fontsize=14, pad=10)
        
        # Texto de tiempo (arriba de todo)
        self.time_text = self.fig.text(
            0.5, 0.98, '', transform=self.fig.transFigure,
            fontsize=16, fontweight='bold', color='white', ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        
        # Estadísticas en panel superior
        if SHOW_STATISTICS:
            self.stats_text = self.ax1.text(
                0.02, 0.98, '', transform=self.ax1.transAxes,
                fontsize=10, color='white', verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
            )
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Dejar espacio para texto superior
        
        return []
    
    def update_frame(self, anim_frame_idx):
        """Actualiza frame con ESCALAS DINÁMICAS"""
        # Mapear índice de animación a índice de caché real
        cache_frame_idx = self.frame_indices[anim_frame_idx]
        cache_file = CACHE_DIR / f"frame_{cache_frame_idx:05d}.npz"
        
        if not cache_file.exists():
            return []
        
        # Leer archivo
        data = np.load(cache_file)
        speed = data['speed']
        vorticity = data['vorticity']
        t = float(data['timestamp'])
        
        # Aplicar suavizado de visualización
        speed_smooth = smooth_field_for_visualization(speed)
        vorticity_smooth = smooth_field_for_visualization(vorticity)
        
        # ESCALAS DINÁMICAS POR FRAME
        valid_speed = speed_smooth[np.isfinite(speed_smooth)]
        valid_vort = vorticity_smooth[np.isfinite(vorticity_smooth)]
        
        if len(valid_speed) > 10:
            vmin_vel = np.percentile(valid_speed, VMIN_PERCENTILE)
            vmax_vel = np.percentile(valid_speed, VMAX_PERCENTILE)
        else:
            vmin_vel, vmax_vel = 0, 100
        
        if len(valid_vort) > 10:
            vmax_vort = np.percentile(np.abs(valid_vort), VMAX_PERCENTILE)
            vmin_vort = -vmax_vort
        else:
            vmin_vort, vmax_vort = -10, 10
        
        # Actualizar datos
        self.pcolormesh_velocity.set_array(speed_smooth.ravel())
        self.pcolormesh_vorticity.set_array(vorticity_smooth.ravel())
        
        # Actualizar escalas de colores
        self.pcolormesh_velocity.set_clim(vmin=vmin_vel, vmax=vmax_vel)
        self.pcolormesh_vorticity.set_clim(vmin=vmin_vort, vmax=vmax_vort)
        
        # Textos
        self.time_text.set_text(f't = {t:.4f} s  |  Frame {anim_frame_idx+1}/{len(self.timeline)}')
        
        if SHOW_STATISTICS and self.stats_text:
            stats_str = (
                f'Cámaras: {int(data["n_cams"])}\n'
                f'Puntos: {int(data["n_valid"])}\n'
                f'V_mean: {data["v_mean"]:.1f} mm/s\n'
                f'V_max: {data["v_max"]:.1f} mm/s\n'
                f'V_std: {data["v_std"]:.1f} mm/s\n'
                f'Scale: [{vmin_vel:.1f}, {vmax_vel:.1f}]'
            )
            self.stats_text.set_text(stats_str)
        
        return []
    
    def render(self, output_path: Path = OUTPUT_VIDEO):
        """Genera video desde caché"""
        print("\n📹 Renderizando video...")
        
        has_ffmpeg = check_ffmpeg_available()
        has_videotoolbox = check_videotoolbox_available()
        
        if not has_ffmpeg:
            print("⚠️  FFmpeg no disponible - no se puede generar video")
            return None
        
        interval = 1000.0 / ANIMATION_FPS / TIME_SPEEDUP
        
        anim = FuncAnimation(
            self.fig, self.update_frame, init_func=self.init_frame,
            frames=len(self.timeline), interval=interval, blit=False, repeat=True
        )
        
        if USE_HARDWARE_ENCODER and has_videotoolbox:
            print("🚀 Usando VideoToolbox (GPU)")
            writer = FFMpegWriter(
                fps=ANIMATION_FPS, codec='h264_videotoolbox', bitrate=VIDEO_BITRATE,
                extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-allow_sw', '1']
            )
        else:
            print("🎬 Usando encoder de software")
            writer = FFMpegWriter(fps=ANIMATION_FPS, bitrate=VIDEO_BITRATE)
        
        with tqdm(total=len(self.timeline), desc="Renderizando") as pbar:
            def progress_callback(i, n):
                pbar.update(1)
            anim.save(str(output_path), writer=writer, dpi=VIDEO_DPI,
                     progress_callback=progress_callback)
        
        print(f"✓ Video guardado: {output_path}")
        print(f"  Tamaño: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return anim
    
    def show_preview(self):
        """Muestra una preview interactiva (útil para desarrollo)"""
        print("\n👁️  Mostrando preview interactivo...")
        print("   Cierra la ventana para continuar")
        
        anim = FuncAnimation(
            self.fig, self.update_frame, init_func=self.init_frame,
            frames=len(self.timeline), interval=1000.0 / ANIMATION_FPS,
            blit=False, repeat=True
        )
        plt.show()
        return anim


# ============================================================
# FUNCIÓN PRINCIPAL DE RENDERING
# ============================================================

def render_animation(output_path: Path = OUTPUT_VIDEO, show_preview: bool = False):
    """
    Renderiza animación desde caché pre-computado.
    
    Args:
        output_path: Ruta donde guardar el video
        show_preview: Si True, muestra preview interactivo antes de renderizar
    
    Returns:
        Animation object de matplotlib
    """
    animator = PIVAnimator()
    
    if show_preview:
        animator.show_preview()
    
    return animator.render(output_path)


if __name__ == "__main__":
    render_animation()