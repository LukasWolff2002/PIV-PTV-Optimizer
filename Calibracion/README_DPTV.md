# DPTV — Depth-from-Defocus Particle Tracking Velocimetry

Módulo para estimar la posición en profundidad (eje Z) de fibras de acero a partir del **desenfoque óptico** medido por PCA en las detecciones YOLO, sin necesidad de láser estructurado ni cámaras estereoscópicas.

---

## Principio físico

Una fibra a distancia |Δz| del plano focal aparece más ancha de lo esperado.
El modelo de thin-lens en aproximación paraxial da:

```
W_apparent² = W_ideal² + (k_blur × Δz)²
```

| Símbolo | Descripción | Valor cam-1 |
|---------|-------------|-------------|
| `W_apparent` [px] | Ancho aparente medido por PCA | variable |
| `W_ideal` [px] | Ancho óptico mínimo en foco | **6.1 px** (empírico) |
| `k_blur` [px/mm] | Tasa de ensanchamiento por desenfoque | **2.5 px/mm** |
| `Δz` [mm] | Distancia al plano focal (siempre ≥ 0) | estimado |

De donde se obtiene:

```
blur_px  = sqrt(max(0, W_apparent² - W_ideal²))
depth_mm = blur_px / k_blur          →  |Δz| desde el plano focal
depth_confidence = tanh(blur_px / noise_px)   →  calidad del estimado [0, 1]
```

### Simetría del eje Z

El plano focal está centrado en el contenedor (profundidad nominal 62 mm).
Por lo tanto `depth_mm` mide la **distancia al centro** y cubre **ambas mitades**:

```
depth_mm ∈ [0, 31 mm]   →  mitad trasera  (z < 31 mm)
                         →  mitad delantera (z > 31 mm)
```
Ambas son indistinguibles con una sola cámara. Un `depth_mm` de 10 mm significa
que la fibra está 10 mm por delante **o** 10 mm por detrás del plano focal.

---

## Parámetros calibrados

Todos los parámetros se configuran en [`RunCode/variables_ptv.py`](../RunCode/variables_ptv.py):

```python
DPTV_ENABLED          = True
DPTV_K_BLUR_PX_PER_MM = 2.5   # px/mm — estimación geométrica (viga 62 mm, cam-1)
DPTV_W_IDEAL_PX       = 6.1   # px  — ancho óptico mínimo empírico (p1 de distribución)
DPTV_NOISE_WIDTH_PX   = 6.8   # px  — ruido de medición intra-trayectoria (std empírico)
```

### Historia de la calibración

| Parámetro | Valor anterior | Valor actual | Método |
|-----------|---------------|--------------|--------|
| `DPTV_NOISE_WIDTH_PX` | 1.0 px | **6.8 px** | Std intra-trayectoria en 589 tracks (cam-1) |
| `DPTV_W_IDEAL_PX` | 1.6 px (teórico) | **6.1 px** | p1 de distribución de `width_px` en detections.csv |
| `DPTV_K_BLUR_PX_PER_MM` | — | **2.5 px/mm** | Estimación geométrica; calibrar con fibras en posiciones conocidas |

**Por qué W_ideal = 6.1 px y no 1.6 px:**
La fibra mide 0.2 mm × 8 px/mm = 1.6 px teórico, pero YOLO+PCA no puede resolver
menos de ~6 px (límite de la segmentación de instancias). Usar 1.6 px produce
`blur_px` artificialmente grande para todas las fibras. El percentil 1 de la
distribución de anchos detectados (6.1 px) representa el mínimo óptico real.

**Por qué noise = 6.8 px y no 1.0 px:**
La varianza de `width_px` dentro de una misma trayectoria (frames consecutivos,
misma fibra) tiene una std mediana de 6.78 px. Con noise=1 px,
`tanh(blur/1) ≈ 1.0` para todas las fibras → confianzas artificialmente altas.

---

## Archivos de análisis

### `calibracion_dptv.py`
**Propósito:** Estima `k_blur` a partir de los datos PTV existentes usando la geometría conocida del contenedor.

```bash
python Calibracion/calibracion_dptv.py
```

**Metodología:** Asumiendo distribución uniforme de fibras en profundidad:
- `E[depth] = max_depth / 2 = 31 mm` (viga 62 mm)
- `k_blur = E[blur_px] / E[depth]`

**Outputs:** Figuras en `Calibracion/DPTV/` con distribución de blur y estimación de k_blur por cámara.

---

### `recalcular_dptv.py`
**Propósito:** Re-procesa los resultados PTV existentes con nuevos parámetros DPTV, sin necesidad de re-correr el pipeline completo.

```bash
# Recalcular con los parámetros actuales calibrados:
python Calibracion/recalcular_dptv.py --k-blur 2.5 --w-ideal-px 6.1 --noise-px 6.8

# Ver qué cambiaría sin modificar archivos:
python Calibracion/recalcular_dptv.py --dry-run

# Especificar carpeta:
python Calibracion/recalcular_dptv.py --folder ResultadosPTV/ResultadosPTV/m70-toma-1-cam-1-n-3000-car-02-ptv
```

**Genera backup automático** en `_backup_dptv_YYYY-MM-DD/` antes de modificar.

---

### `analisis_flujo_dptv.py`
**Propósito:** Analiza el perfil de flujo 3D usando la profundidad estimada por DPTV. Genera 6 gráficos:

1. **Histograma de profundidad** — distribución de |Δz| con marcador de pared
2. **Velocidad vs profundidad** — banda IQR + media + mediana + referencia Poiseuille
3. **Rosa de dirección** — ángulo del vector velocidad medio por bin de profundidad
4. **Confianza vs profundidad** — scatter + bins (valida parámetro noise_px)
5. **Mapa 2D de velocidad** — grilla 8×8 en plano X-Y
6. **Rotación de ángulo** — gráfico del ángulo medio del flujo vs profundidad

```bash
# Mostrar en pantalla (primer resultado encontrado):
python Calibracion/analisis_flujo_dptv.py

# Carpeta específica:
python Calibracion/analisis_flujo_dptv.py --folder ResultadosPTV/.../m70-toma-1-cam-1-n-3000-car-02-ptv

# Guardar figuras en Calibracion/DPTV/:
python Calibracion/analisis_flujo_dptv.py --save-figures
```

**Hallazgos principales (cam-1, m70-toma-1, carbopol):**
- El ángulo de flujo rota ~45° con la profundidad: 20° cerca del foco → 65° lejos
- La velocidad **aumenta** con la profundidad (opuesto a Poiseuille clásico)
- Ambos efectos se explican por la geometría del **dispositivo L**:
  - Fibras cercanas al plano focal están en la zona horizontal (flujo frenado)
  - Fibras lejanas están en la zona vertical (flujo acelerado por gravedad)

---

## Cadena de código

```
RunCode/variables_ptv.py          ← parámetros (DPTV_K_BLUR_PX_PER_MM, etc.)
        ↓
RunCode/pipeline_global.py        ← construye JSON de configuración del pipeline
        ↓
PTV/Codes/PTVCode/config.py       ← TrackingConfig (dptv_k_blur_px_per_mm, dptv_w_ideal_px, ...)
        ↓
PTV/Codes/PTVCode/runner.py       ← crea DPTVConfig y DPTVEstimator
        ↓
PTV/Codes/PTVCode/dptv.py         ← modelo físico + estimación por detección
        ↓
PTV/Codes/PTVCode/detector.py     ← llama dptv.estimate() en cada detección YOLO
        ↓
ResultadosPTV/.../tracks.json     ← salida: depth_mm, depth_confidence por frame
```

---

## Calibración futura (recomendada)

Para mejorar `k_blur` más allá de la estimación geométrica:

1. Sumergir una fibra sujeta a profundidades conocidas: z = 0, 5, 10, 20, 30 mm
2. Medir `width_px` en las detecciones para cada posición
3. Ajustar por regresión lineal: `k_blur = slope(blur_px vs Δz)`
4. Repetir por cámara (cada una tiene diferente apertura numérica efectiva)
5. Actualizar `DPTV_K_BLUR_PX_PER_MM` en `variables_ptv.py` y re-correr con `recalcular_dptv.py`

### Calibración de px_per_mm (validada)
La longitud conocida de la fibra (13 mm) valida la escala espacial:
- Cam-1: p50 de `length_px` = 103 px → 103/13 = **7.92 px/mm** (error 0.8% vs 8.0 px/mm nominal) ✓

---

## Estructura de salida en tracks.json

Cada frame de cada trayectoria incluye:

```json
{
  "frame": 42,
  "x_mm": 45.3,
  "y_mm": 67.1,
  "width_px": 24.6,
  "defocus_score": 4.03,
  "depth_blur_mm": 2.46,
  "depth_mm": 9.84,
  "depth_confidence": 0.997
}
```

Y a nivel de trayectoria:

```json
{
  "track_id": 85,
  "depth_stats": {
    "mean_depth_mm": 17.3,
    "std_depth_mm": 3.8,
    "mean_depth_confidence": 1.0,
    "n_observations": 17,
    "n_depth_estimated": 17
  }
}
```
