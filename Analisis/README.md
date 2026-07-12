# Analisis/ — Post-análisis de resultados PIV y PTV

Herramientas que consumen las salidas del pipeline (`ResultadosPIV/`, `ResultadosPTV/`) para
interpolar, comparar y visualizar. No forman parte del cómputo principal; son de exploración y
control de calidad.

## `PIV/`

| Archivo | Rol |
|---|---|
| `compare_piv.py` | **Comparación de tomas** del mismo carbopol (reproducibilidad): distribuciones de velocidad (hist/KDE), RMSE entre campos, correlación de Pearson, percentiles, test de Kolmogorov–Smirnov, mapas de diferencias y reporte de texto. |
| `InterpolarPIV/main.py` | Pipeline automático de **interpolación + visualización** de todas las tomas. |
| `InterpolarPIV/piv_interpolate_simple.py` | Interpola los `.txt` crudos del PIV a una **grilla regular** por cámara → `PIV_INTERPOLADO/`. |
| `InterpolarPIV/piv_generate_cache.py` | Genera caché `.npz` desde los `.txt` interpolados para renderizado rápido. |
| `InterpolarPIV/piv_visualize.py` | Renderiza **animaciones** (videos) desde los datos pre-computados. |
| `InterpolarPIV/piv_config.py` | Config y utilidades compartidas del sistema de animación PIV. |

> `piv_interpolate_simple.py` / `piv_config.py` son la misma familia de interpolación que usa el repo
> de tesis (`Analisis_PIV-PTV_Final/global_piv_ptv/PIV/`) para llevar el campo PIV a grilla regular.

## `PTV/`

Todos leen `tracks.json` + `schedule.csv` (la **grilla temporal real**, con `dt_s` variable por
región) y reconstruyen trayectorias en unidades físicas.

| Archivo | Rol |
|---|---|
| `analisis_ptv_por_camara.py` | Visualizador PTV por cámara: interpola tracks a una grilla temporal uniforme, suaviza velocidades, dibuja fibras como varillas y anima. |
| `analisis_ptv_profundidad.py` | Análisis estándar **+ profundidad DPTV**: heatmaps de velocidad lineal/angular, velocidad vs tiempo, distribución espacial de profundidad, histogramas de *defocus/blur*, correlación profundidad–velocidad, profundidad vs tiempo. |
| `analisis_ptv_3d.py` | **Animación 3D** de las fibras con Z = distancia al plano focal (`depth_blur_mm` o `depth_mm` si `k_blur` calibrado); el plano PIV se muestra en Z=0. |

## Uso

```bash
python Analisis/PIV/compare_piv.py
python Analisis/PIV/InterpolarPIV/main.py
python Analisis/PTV/analisis_ptv_profundidad.py
python Analisis/PTV/analisis_ptv_3d.py
```
