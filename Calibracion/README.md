# Calibracion/ — Calibración DPTV (profundidad por desenfoque)

Calibra y aplica el modelo de **depth-from-defocus** que estima la profundidad Δz de cada fibra a
partir de cuánto se ensancha su imagen al alejarse del plano focal.

**Modelo físico:**
```
W_apparent² = W_ideal² + (k_blur · Δz)²
   W_ideal  = FIBER_WIDTH_MM × px_per_mm   (ancho de fibra en foco perfecto)
   k_blur   = tasa de desenfoque [px/mm]   ← el parámetro a calibrar
```

## Scripts

| Script | Rol |
|---|---|
| `calibracion_dptv.py` | Estima **`k_blur`** por cámara a partir de (1) la distribución de anchos detectados por YOLO+PCA en los resultados PTV y (2) la geometría conocida del contenedor (viga 62 mm → profundidad máx. ~31 mm). Genera figuras de calibración. |
| `recalcular_dptv.py` | **Re-aplica** nuevos parámetros DPTV (`k_blur`, `W_ideal`) a resultados PTV **ya procesados**, sin re-correr YOLO ni el tracking. Actualiza `detections.csv`, `tracks.csv`, `tracks.json` y la sección `dptv` de `summary.json` (con backup previo). |

## Datos

- `Cam 1`…`Cam 4`, `Cam 1_ant`: imágenes/tomas de calibración por cámara.
- `info.txt`: escalas medidas — `cam1/1ant = 80 px/cm`, `cam2/cam3 = 78 px/cm`, `cam4 = 107 px/cm`
  (≈ 8 px/mm salvo cam4 ≈ 10.7 px/mm).

## Uso típico

```bash
# 1) estimar k_blur con los resultados PTV existentes
python Calibracion/calibracion_dptv.py
# 2) fijar el valor en RunCode/variables_ptv.py (DPTV_K_BLUR_PX_PER_MM) para futuras corridas,
#    o re-procesar resultados ya generados:
python Calibracion/recalcular_dptv.py
```

El valor calibrado alimenta `DPTV_K_BLUR_PX_PER_MM` en
[`RunCode/variables_ptv.py`](../RunCode/variables_ptv.py); el estimador vive en
[`PTV/Codes/PTVCode/dptv.py`](../PTV/README.md). Fundamento completo:
[`Contextos/dptv.md`](../Contextos/dptv.md).
