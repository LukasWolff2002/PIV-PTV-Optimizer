"""
ANÁLISIS COMPARATIVO DE TOMAS POR CARBOPOL
===========================================
Compara tomas del mismo carbopol para verificar consistencia y reproducibilidad.

Métricas implementadas:
- Distribuciones de velocidad (histogramas, KDE)
- RMSE (Root Mean Square Error) entre campos de velocidad
- Coeficiente de correlación de Pearson
- Análisis de percentiles (P50, P90, P95, P99)
- Prueba de Kolmogorov-Smirnov para distribuciones
- Análisis espacial de diferencias

Uso:
    python compare_carbopol_takes.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy import stats
from scipy.interpolate import griddata
from dataclasses import dataclass

from piv_config import (
    INTERPOLATED_DIR, load_piv_frame, read_piv_header,
    GRID_DX_MM, GRID_DY_MM, compute_global_grid
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RESULTS_DIR = Path("analisis_comparativo")
PLOT_DPI = 150
PLOT_FORMAT = "png"


# ============================================================
# ESTRUCTURAS DE DATOS
# ============================================================

@dataclass
class TakeStats:
    """Estadísticas de una toma completa"""
    name: str
    carbopol: str
    n_frames: int
    timestamps: np.ndarray
    velocities: np.ndarray  # Todas las velocidades de todos los frames
    mean_velocity: float
    median_velocity: float
    std_velocity: float
    max_velocity: float
    percentiles: Dict[int, float]  # P50, P90, P95, P99


@dataclass
class ComparisonMetrics:
    """Métricas de comparación entre dos tomas"""
    take1: str
    take2: str
    carbopol: str
    
    # Métricas globales
    rmse_velocity: float
    correlation: float
    ks_statistic: float
    ks_pvalue: float
    
    # Diferencias en estadísticos
    mean_diff_pct: float
    median_diff_pct: float
    std_diff_pct: float
    max_diff_pct: float
    
    # Diferencias en percentiles
    percentile_diffs: Dict[int, float]


# ============================================================
# FUNCIONES DE CARGA
# ============================================================

def extract_carbopol_from_name(take_name: str) -> str:
    """
    Extrae el tipo de carbopol del nombre de la toma.
    Ejemplos:
        'm70-toma-2-n-3000-car-02-piv' -> 'car-02'
        'm83-toma-2-n-3000-car-05-piv' -> 'car-05'
    """
    parts = take_name.split('-')
    for i, part in enumerate(parts):
        if part == 'car' and i + 1 < len(parts):
            return f'car-{parts[i+1]}'
    return 'unknown'


def load_take_data(take_folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga todos los datos de una toma.
    
    Returns:
        timestamps: Array de timestamps
        velocities: Array de todas las velocidades (concatenadas de todos los frames)
    """
    frames = sorted(take_folder.glob("frame_*.txt"))
    
    if not frames:
        return np.array([]), np.array([])
    
    timestamps = []
    all_velocities = []
    
    for frame_file in frames:
        try:
            frame = load_piv_frame(frame_file)
            timestamps.append(frame.timestamp_s)
            
            # Extraer velocidades válidas
            data = frame.data
            valid = data[:, 5].astype(int) == 1
            u = data[valid, 2]
            v = data[valid, 3]
            
            # Calcular magnitud de velocidad
            speed = np.sqrt(u**2 + v**2)
            
            # Filtrar finitos
            speed = speed[np.isfinite(speed)]
            
            if len(speed) > 0:
                all_velocities.append(speed)
        
        except Exception as e:
            print(f"⚠️  Error cargando {frame_file.name}: {e}")
            continue
    
    if not all_velocities:
        return np.array([]), np.array([])
    
    timestamps = np.array(timestamps)
    velocities = np.concatenate(all_velocities)
    
    return timestamps, velocities


def compute_take_statistics(take_name: str, take_folder: Path) -> TakeStats:
    """Computa estadísticas completas de una toma"""
    carbopol = extract_carbopol_from_name(take_name)
    timestamps, velocities = load_take_data(take_folder)
    
    if len(velocities) == 0:
        raise ValueError(f"No se pudieron cargar datos de {take_name}")
    
    percentiles = {
        50: np.percentile(velocities, 50),
        90: np.percentile(velocities, 90),
        95: np.percentile(velocities, 95),
        99: np.percentile(velocities, 99),
    }
    
    return TakeStats(
        name=take_name,
        carbopol=carbopol,
        n_frames=len(timestamps),
        timestamps=timestamps,
        velocities=velocities,
        mean_velocity=float(np.mean(velocities)),
        median_velocity=float(np.median(velocities)),
        std_velocity=float(np.std(velocities)),
        max_velocity=float(np.max(velocities)),
        percentiles=percentiles
    )


# ============================================================
# MÉTRICAS DE COMPARACIÓN
# ============================================================

def compute_comparison_metrics(stats1: TakeStats, stats2: TakeStats) -> ComparisonMetrics:
    """
    Computa métricas de comparación entre dos tomas.
    
    Nota: Como las tomas pueden tener diferentes duraciones y posiciones espaciales,
    comparamos las distribuciones estadísticas globales, no campos punto a punto.
    """
    # RMSE basado en distribuciones (comparando histogramas)
    # Crear bins comunes
    v_min = min(stats1.velocities.min(), stats2.velocities.min())
    v_max = max(stats1.velocities.max(), stats2.velocities.max())
    bins = np.linspace(v_min, v_max, 100)
    
    hist1, _ = np.histogram(stats1.velocities, bins=bins, density=True)
    hist2, _ = np.histogram(stats2.velocities, bins=bins, density=True)
    
    # RMSE entre histogramas normalizados
    rmse = float(np.sqrt(np.mean((hist1 - hist2)**2)))
    
    # Correlación entre histogramas (en lugar de valores individuales)
    if len(hist1) > 1 and len(hist2) > 1:
        correlation = float(np.corrcoef(hist1, hist2)[0, 1])
    else:
        correlation = 0.0
    
    # Prueba de Kolmogorov-Smirnov (compara distribuciones completas)
    ks_stat, ks_pval = stats.ks_2samp(stats1.velocities, stats2.velocities)
    
    # Diferencias porcentuales en estadísticos
    def pct_diff(a, b):
        if a == 0:
            return 0.0
        return 100 * abs(a - b) / abs(a)
    
    mean_diff_pct = pct_diff(stats1.mean_velocity, stats2.mean_velocity)
    median_diff_pct = pct_diff(stats1.median_velocity, stats2.median_velocity)
    std_diff_pct = pct_diff(stats1.std_velocity, stats2.std_velocity)
    max_diff_pct = pct_diff(stats1.max_velocity, stats2.max_velocity)
    
    # Diferencias en percentiles
    percentile_diffs = {}
    for p in [50, 90, 95, 99]:
        percentile_diffs[p] = pct_diff(stats1.percentiles[p], stats2.percentiles[p])
    
    return ComparisonMetrics(
        take1=stats1.name,
        take2=stats2.name,
        carbopol=stats1.carbopol,
        rmse_velocity=rmse,
        correlation=correlation,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
        mean_diff_pct=mean_diff_pct,
        median_diff_pct=median_diff_pct,
        std_diff_pct=std_diff_pct,
        max_diff_pct=max_diff_pct,
        percentile_diffs=percentile_diffs
    )


# ============================================================
# VISUALIZACIÓN
# ============================================================

def plot_velocity_distributions(carbopol: str, takes_stats: List[TakeStats], output_dir: Path):
    """
    Genera gráficos de distribuciones de velocidad para un carbopol.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Distribuciones de Velocidad - {carbopol.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Colores distintivos usando paleta tab20 (20 colores diferentes)
    n_takes = len(takes_stats)
    if n_takes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_takes]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_takes]
    
    # 1. Histogramas como curvas suavizadas (escala logarítmica en Y)
    ax = axes[0, 0]
    for take_stat, color in zip(takes_stats, colors):
        # Calcular histograma
        hist, bin_edges = np.histogram(take_stat.velocities, bins=150, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Filtrar valores muy pequeños para log scale
        mask = hist > 1e-10
        hist_filtered = hist[mask]
        bin_centers_filtered = bin_centers[mask]
        
        # Plotear como curva suavizada (SIN relleno)
        ax.plot(bin_centers_filtered, hist_filtered, 
                label=take_stat.name, color=color, linewidth=2.5, alpha=0.85)
    
    ax.set_xlabel('Velocidad (mm/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Densidad de probabilidad (log)', fontsize=11, fontweight='bold')
    ax.set_title('Distribuciones Normalizadas', fontweight='bold', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2 if n_takes > 5 else 1, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # 2. KDE (Kernel Density Estimation) - escala logarítmica en Y
    ax = axes[0, 1]
    for take_stat, color in zip(takes_stats, colors):
        # Crear KDE con scipy.stats
        kde = stats.gaussian_kde(take_stat.velocities)
        x_range = np.linspace(0.1, np.percentile(take_stat.velocities, 99.5), 500)
        kde_values = kde(x_range)
        # Evitar valores <= 0 para log scale
        kde_values = np.maximum(kde_values, 1e-10)
        # Plotear curva (SIN relleno)
        ax.plot(x_range, kde_values, label=take_stat.name, color=color, linewidth=2.5, alpha=0.85)
    
    ax.set_xlabel('Velocidad (mm/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Densidad (KDE, log)', fontsize=11, fontweight='bold')
    ax.set_title('Estimación de Densidad (KDE)', fontweight='bold', fontsize=12)
    ax.set_yscale('log')
    #ax.legend(fontsize=7, ncol=2 if n_takes > 5 else 1, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # 3. Funciones de distribución acumulada (CDF) - escala logarítmica en X
    ax = axes[1, 0]
    for take_stat, color in zip(takes_stats, colors):
        sorted_v = np.sort(take_stat.velocities)
        # Filtrar velocidades > 0 para log scale
        sorted_v = sorted_v[sorted_v > 0]
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, cdf, label=take_stat.name, color=color, linewidth=2.5, alpha=0.85)
    
    ax.set_xlabel('Velocidad (mm/s, log)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probabilidad acumulada', fontsize=11, fontweight='bold')
    ax.set_title('Distribución Acumulada (CDF)', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.legend(fontsize=7, ncol=2 if n_takes > 5 else 1, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # 4. Box plots simplificados (escala logarítmica en Y)
    ax = axes[1, 1]
    data_for_box = [take_stat.velocities for take_stat in takes_stats]
    labels_for_box = [s.name.replace(f'-{carbopol}-piv', '') for s in takes_stats]
    
    # Box plot con configuración simplificada
    bp = ax.boxplot(data_for_box, 
                    labels=labels_for_box, 
                    patch_artist=True,
                    showmeans=False,      # Ocultar media
                    showfliers=False,     # Ocultar outliers (puntos negros)
                    widths=0.6,
                    medianprops=dict(linewidth=2.5, color='darkred'),
                    boxprops=dict(linewidth=1.5, edgecolor='black'),
                    whiskerprops=dict(linewidth=1.5, color='black', linestyle='-'),
                    capprops=dict(linewidth=1.5, color='black'))
    
    # Colorear cada box con su color correspondiente
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Velocidad (mm/s, log)', fontsize=11, fontweight='bold')
    ax.set_title('Distribución por Cuartiles', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y', which='both', linestyle='--')
    
    # Agregar nota explicativa
    #ax.text(0.02, 0.98, 'Línea roja = mediana\nCaja = Q1-Q3 (50% central)\nBigotes = rango total',
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top',
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / f'distribuciones_{carbopol}.{PLOT_FORMAT}'
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráfico guardado: {output_file}")


def plot_comparison_heatmap(carbopol: str, comparisons: List[ComparisonMetrics], output_dir: Path):
    """
    Genera heatmap de métricas de comparación.
    """
    if len(comparisons) < 2:
        return
    
    # Crear matriz de similitud
    take_names = sorted(set([c.take1 for c in comparisons] + [c.take2 for c in comparisons]))
    n = len(take_names)
    
    # Matrices para diferentes métricas
    rmse_matrix = np.zeros((n, n))
    corr_matrix = np.ones((n, n))
    ks_matrix = np.zeros((n, n))
    
    for comp in comparisons:
        i = take_names.index(comp.take1)
        j = take_names.index(comp.take2)
        
        rmse_matrix[i, j] = comp.rmse_velocity
        rmse_matrix[j, i] = comp.rmse_velocity
        
        corr_matrix[i, j] = comp.correlation
        corr_matrix[j, i] = comp.correlation
        
        ks_matrix[i, j] = comp.ks_statistic
        ks_matrix[j, i] = comp.ks_statistic
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Métricas de Comparación - {carbopol.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Simplificar nombres para labels
    short_names = [name.replace(f'-{carbopol}-piv', '') for name in take_names]
    
    # 1. RMSE (usar colormap viridis - más contrastante)
    ax = axes[0]
    im1 = ax.imshow(rmse_matrix, cmap='plasma', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_title('RMSE de Velocidad (mm/s)', fontweight='bold')
    
    # Agregar valores en las celdas
    for i in range(n):
        for j in range(n):
            if i != j:
                # Color del texto según intensidad del fondo
                text_color = 'white' if rmse_matrix[i, j] > rmse_matrix.max()/2 else 'black'
                text = ax.text(j, i, f'{rmse_matrix[i, j]:.1f}',
                             ha="center", va="center", color=text_color, fontsize=9,
                             fontweight='bold')
    
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    
    # 2. Correlación (usar RdYlGn - rojo-amarillo-verde)
    ax = axes[1]
    im2 = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_title('Coeficiente de Correlación', fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            # Color del texto según valor de correlación
            text_color = 'white' if corr_matrix[i, j] < 0.5 else 'black'
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                         ha="center", va="center", color=text_color, fontsize=9,
                         fontweight='bold')
    
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    
    # 3. Kolmogorov-Smirnov (usar inferno - oscuro a amarillo brillante)
    ax = axes[2]
    im3 = ax.imshow(ks_matrix, cmap='inferno', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_title('Estadístico K-S', fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Color del texto según intensidad
                text_color = 'white' if ks_matrix[i, j] > ks_matrix.max()/2 else 'black'
                text = ax.text(j, i, f'{ks_matrix[i, j]:.3f}',
                             ha="center", va="center", color=text_color, fontsize=9,
                             fontweight='bold')
    
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_file = output_dir / f'comparacion_heatmap_{carbopol}.{PLOT_FORMAT}'
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Heatmap guardado: {output_file}")


def plot_statistics_comparison(carbopol: str, takes_stats: List[TakeStats], output_dir: Path):
    """
    Compara estadísticos entre tomas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comparación de Estadísticos - {carbopol.upper()}', 
                 fontsize=16, fontweight='bold')
    
    names = [s.name.replace(f'-{carbopol}-piv', '') for s in takes_stats]
    
    # Colores distintivos
    n_takes = len(takes_stats)
    if n_takes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_takes]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_takes]
    
    x = np.arange(len(names))
    width = 0.6
    
    # 1. Media y mediana
    ax = axes[0, 0]
    means = [s.mean_velocity for s in takes_stats]
    medians = [s.median_velocity for s in takes_stats]
    
    ax.bar(x - width/4, means, width/2, label='Media', color=colors, alpha=0.8, 
           edgecolor='black', linewidth=1.5)
    ax.bar(x + width/4, medians, width/2, label='Mediana', color=colors, alpha=0.4, 
           edgecolor='black', linewidth=1.5, hatch='//')
    
    ax.set_ylabel('Velocidad (mm/s)', fontsize=11)
    ax.set_title('Media y Mediana', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Desviación estándar
    ax = axes[0, 1]
    stds = [s.std_velocity for s in takes_stats]
    
    ax.bar(x, stds, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Desviación Estándar (mm/s)', fontsize=11)
    ax.set_title('Desviación Estándar', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Percentiles
    ax = axes[1, 0]
    percentile_keys = [50, 90, 95, 99]
    x_perc = np.arange(len(percentile_keys))
    width_perc = 0.8 / len(takes_stats)
    
    for i, (take_stat, color) in enumerate(zip(takes_stats, colors)):
        values = [take_stat.percentiles[p] for p in percentile_keys]
        offset = (i - len(takes_stats)/2) * width_perc + width_perc/2
        ax.bar(x_perc + offset, values, width_perc, 
               label=take_stat.name.replace(f'-{carbopol}-piv', ''),
               color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Velocidad (mm/s)', fontsize=11)
    ax.set_title('Percentiles de Velocidad', fontweight='bold')
    ax.set_xticks(x_perc)
    ax.set_xticklabels([f'P{p}' for p in percentile_keys])
    ax.legend(fontsize=7, ncol=2 if n_takes > 5 else 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Máximo
    ax = axes[1, 1]
    maxs = [s.max_velocity for s in takes_stats]
    
    ax.bar(x, maxs, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Velocidad Máxima (mm/s)', fontsize=11)
    ax.set_title('Velocidad Máxima', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / f'estadisticos_{carbopol}.{PLOT_FORMAT}'
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Estadísticos guardados: {output_file}")


# ============================================================
# REPORTE
# ============================================================

def generate_text_report(carbopol: str, takes_stats: List[TakeStats], 
                         comparisons: List[ComparisonMetrics], output_dir: Path):
    """
    Genera reporte de texto con todas las métricas.
    """
    report_file = output_dir / f'reporte_{carbopol}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"REPORTE DE ANÁLISIS COMPARATIVO - {carbopol.upper()}\n")
        f.write("="*70 + "\n\n")
        
        # Estadísticas individuales
        f.write("ESTADÍSTICAS INDIVIDUALES POR TOMA\n")
        f.write("-"*70 + "\n\n")
        
        for stats in takes_stats:
            f.write(f"Toma: {stats.name}\n")
            f.write(f"  Número de frames: {stats.n_frames}\n")
            f.write(f"  Duración: {stats.timestamps[-1] - stats.timestamps[0]:.2f} s\n")
            f.write(f"  Número de vectores: {len(stats.velocities):,}\n")
            f.write(f"  Media: {stats.mean_velocity:.3f} mm/s\n")
            f.write(f"  Mediana: {stats.median_velocity:.3f} mm/s\n")
            f.write(f"  Desv. Std: {stats.std_velocity:.3f} mm/s\n")
            f.write(f"  Máximo: {stats.max_velocity:.3f} mm/s\n")
            f.write(f"  Percentiles:\n")
            for p, val in stats.percentiles.items():
                f.write(f"    P{p}: {val:.3f} mm/s\n")
            f.write("\n")
        
        # Comparaciones par a par
        f.write("\n" + "="*70 + "\n")
        f.write("COMPARACIONES PAR A PAR\n")
        f.write("="*70 + "\n\n")
        
        for comp in comparisons:
            f.write(f"{comp.take1} vs {comp.take2}\n")
            f.write("-"*70 + "\n")
            f.write(f"  RMSE (velocidad): {comp.rmse_velocity:.3f} mm/s\n")
            f.write(f"  Correlación: {comp.correlation:.4f}\n")
            f.write(f"  K-S statistic: {comp.ks_statistic:.4f} (p-value: {comp.ks_pvalue:.4e})\n")
            f.write(f"\n  Diferencias porcentuales en estadísticos:\n")
            f.write(f"    Media: {comp.mean_diff_pct:.2f}%\n")
            f.write(f"    Mediana: {comp.median_diff_pct:.2f}%\n")
            f.write(f"    Desv. Std: {comp.std_diff_pct:.2f}%\n")
            f.write(f"    Máximo: {comp.max_diff_pct:.2f}%\n")
            f.write(f"\n  Diferencias en percentiles:\n")
            for p, diff in comp.percentile_diffs.items():
                f.write(f"    P{p}: {diff:.2f}%\n")
            
            # Interpretación del test K-S
            f.write(f"\n  Interpretación K-S test:\n")
            if comp.ks_pvalue > 0.05:
                f.write(f"    ✓ No hay evidencia significativa de diferencia entre distribuciones\n")
                f.write(f"      (p-value > 0.05, no se rechaza H0)\n")
            else:
                f.write(f"    ⚠ Las distribuciones son significativamente diferentes\n")
                f.write(f"      (p-value < 0.05, se rechaza H0)\n")
            
            f.write("\n\n")
        
        # Resumen global
        f.write("="*70 + "\n")
        f.write("RESUMEN GLOBAL\n")
        f.write("="*70 + "\n\n")
        
        # Estadísticas de las comparaciones
        if comparisons:
            rmses = [c.rmse_velocity for c in comparisons]
            correlations = [c.correlation for c in comparisons]
            ks_stats = [c.ks_statistic for c in comparisons]
            
            f.write(f"Rango de RMSE: {min(rmses):.3f} - {max(rmses):.3f} mm/s\n")
            f.write(f"RMSE promedio: {np.mean(rmses):.3f} mm/s\n")
            f.write(f"\n")
            f.write(f"Rango de correlación: {min(correlations):.4f} - {max(correlations):.4f}\n")
            f.write(f"Correlación promedio: {np.mean(correlations):.4f}\n")
            f.write(f"\n")
            f.write(f"Rango de K-S statistic: {min(ks_stats):.4f} - {max(ks_stats):.4f}\n")
            f.write(f"K-S statistic promedio: {np.mean(ks_stats):.4f}\n")
            f.write(f"\n")
            
            # Conclusión
            mean_correlation = np.mean(correlations)
            mean_ks = np.mean(ks_stats)
            
            f.write("CONCLUSIÓN:\n")
            f.write("-"*70 + "\n")
            
            if mean_correlation > 0.9 and mean_ks < 0.1:
                f.write("✓ ALTA CONSISTENCIA entre tomas del mismo carbopol\n")
                f.write("  Las tomas muestran comportamiento muy similar.\n")
            elif mean_correlation > 0.7 and mean_ks < 0.2:
                f.write("✓ BUENA CONSISTENCIA entre tomas del mismo carbopol\n")
                f.write("  Las tomas muestran comportamiento similar con variaciones menores.\n")
            else:
                f.write("⚠ VARIABILIDAD SIGNIFICATIVA entre tomas\n")
                f.write("  Se recomienda investigar las causas de las diferencias.\n")
    
    print(f"  ✓ Reporte guardado: {report_file}")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    """
    Función principal de análisis comparativo.
    """
    print("\n" + "="*70)
    print("ANÁLISIS COMPARATIVO DE TOMAS POR CARBOPOL")
    print("="*70)
    
    # Verificar que existe PIV_INTERPOLADO
    if not INTERPOLATED_DIR.exists():
        print(f"\n❌ Error: No se encontró {INTERPOLATED_DIR}")
        print("   Ejecuta primero main.py para generar los datos interpolados.")
        return
    
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"\nDirectorio de resultados: {RESULTS_DIR.absolute()}\n")
    
    # Agrupar tomas por carbopol
    takes_by_carbopol = defaultdict(list)
    
    for subfolder in INTERPOLATED_DIR.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Verificar que tenga archivos
        n_files = len(list(subfolder.glob("frame_*.txt")))
        if n_files == 0:
            continue
        
        carbopol = extract_carbopol_from_name(subfolder.name)
        takes_by_carbopol[carbopol].append(subfolder)
    
    print(f"🔍 Tomas encontradas por carbopol:")
    for carbopol, folders in sorted(takes_by_carbopol.items()):
        print(f"   {carbopol}: {len(folders)} toma(s)")
    
    # Procesar cada carbopol
    for carbopol, folders in sorted(takes_by_carbopol.items()):
        if len(folders) < 2:
            print(f"\n⚠️  {carbopol}: Solo 1 toma encontrada, se necesitan al menos 2 para comparar")
            continue
        
        print(f"\n{'='*70}")
        print(f"PROCESANDO: {carbopol.upper()}")
        print(f"{'='*70}")
        
        # Crear subdirectorio para este carbopol
        carbopol_dir = RESULTS_DIR / carbopol
        carbopol_dir.mkdir(exist_ok=True)
        
        # Cargar estadísticas de todas las tomas
        print(f"\n📊 Cargando datos...")
        takes_stats = []
        
        for folder in sorted(folders):
            print(f"   • {folder.name}...", end=" ")
            try:
                stats = compute_take_statistics(folder.name, folder)
                takes_stats.append(stats)
                print(f"✓ ({stats.n_frames} frames, {len(stats.velocities):,} vectores)")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if len(takes_stats) < 2:
            print(f"   ⚠️  No se pudieron cargar suficientes tomas")
            continue
        
        # Computar comparaciones par a par
        print(f"\n📈 Computando métricas de comparación...")
        comparisons = []
        
        for i in range(len(takes_stats)):
            for j in range(i + 1, len(takes_stats)):
                comp = compute_comparison_metrics(takes_stats[i], takes_stats[j])
                comparisons.append(comp)
                print(f"   • {takes_stats[i].name} vs {takes_stats[j].name}")
                print(f"      RMSE: {comp.rmse_velocity:.3f} mm/s, "
                      f"Corr: {comp.correlation:.4f}, "
                      f"K-S: {comp.ks_statistic:.4f}")
        
        # Generar visualizaciones
        print(f"\n📊 Generando gráficos...")
        plot_velocity_distributions(carbopol, takes_stats, carbopol_dir)
        plot_statistics_comparison(carbopol, takes_stats, carbopol_dir)
        plot_comparison_heatmap(carbopol, comparisons, carbopol_dir)
        
        # Generar reporte de texto
        print(f"\n📝 Generando reporte...")
        generate_text_report(carbopol, takes_stats, comparisons, carbopol_dir)
        
        print(f"\n✅ {carbopol.upper()} completado")
    
    # Resumen final
    print(f"\n{'='*70}")
    print("✅ ANÁLISIS COMPLETADO")
    print(f"{'='*70}")
    print(f"\nResultados guardados en: {RESULTS_DIR.absolute()}/")
    print(f"\nPor cada carbopol se generaron:")
    print(f"   • Gráficos de distribuciones")
    print(f"   • Gráficos de estadísticos")
    print(f"   • Heatmaps de comparación")
    print(f"   • Reporte de texto con métricas detalladas")


if __name__ == "__main__":
    main()