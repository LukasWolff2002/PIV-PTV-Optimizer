# PTV/ — Motor de Particle Tracking Velocimetry (fibras)

Detecta y **sigue cada fibra individual** a lo largo del tiempo, entregando su trayectoria completa
(posición, velocidad, ángulo, geometría) más su **profundidad Δz** por *depth-from-defocus* (DPTV).

```
PTV/
├── Tomas/                       # fotogramas crudos por toma (no versionado)
├── Codes/
│   ├── Segmentation-Models/     # pesos YOLO: camN-ptv-yolo26.pt (máscaras) + best.pt (tracking)
│   ├── PreProcessing/           # filtros de imagen (comparte técnicas con PIV)
│   └── PTVCode/                 # librería de detección + tracking + DPTV
```

Orquestado desde [`RunCode/`](../RunCode/README.md) (`preprocess_run_ptv.py` → `ptv_run.py`).

---

## `Codes/PTVCode/` — Librería PTV

### Detección y tracking

| Módulo | Rol |
|---|---|
| `runner.py` | **Loop principal** (`run_ptv`): recorre los frames según las regiones temporales (con `dt_s` y `max_dist_px` variables por región), con **prefetch asíncrono** de imágenes (CPU solapado con GPU); orquesta detector → tracker → export. |
| `detector.py` | `FiberYOLODetector`: segmentación YOLO con **SAHI** (tiling para fibras pequeñas), paralelismo CPU+GPU (rasterización de máscaras en hilos, **PCA** por fibra en procesos), NMS por IoU de máscaras. Entrega centroide, ángulo, largo y **ancho** por fibra. |
| `tracker.py` | `Tracker`: asociación multi-objeto con **Similarity Search Scheme (SSS)** vectorizado + **gate espacial** (x, y, ángulo). Respeta la convención de máscara (blanco = ignorar). |
| `features.py` | Vector de **17 features** por fibra (posición, geometría, orientación con simetría ±θ, forma —solidity/extent/Hu—, textura) y su escalado; `compute_similarity_scores` / matriz de costo para la asignación. |
| `filters.py` | Filtro **alpha-beta-gamma (ABG)**: `predict_state_abg` / `update_state_abg`, con cinemática lineal y **angular desacoplada** (ganancias y *decay* separados; ver `variables_ptv.py`). |

### Profundidad (DPTV)

| Módulo | Rol |
|---|---|
| `dptv.py` | `DPTVEstimator`: invierte el modelo de desenfoque **W_apparent² = W_ideal² + (k_blur·Δz)²** para estimar `|Δz|` de cada fibra desde su ancho aparente. `estimate()` por detección; `track_depth_stats()` agrega por track. Fundamento en [`Contextos/dptv.md`](../Contextos/dptv.md). |

### Dominio, E/S y visualización

| Módulo | Rol |
|---|---|
| `models.py` | Dataclasses: `Detection`, `TrackState`, `TrackRecord`, `Track`. **Unidades de salida: mm, mm/s, mm/s², grados.** |
| `config.py` | `TrackingConfig` (inmutable) + `build_tracking_config(json)` + `validate_config`. |
| `exporters.py` | Export a `detections.csv`, `tracks.csv`, `tracks.json`, `schedule.csv`. |
| `image_utils.py` | Carga/normalización de imágenes, máscaras booleanas, geometría de contornos, *wrap* de ángulos de fibra (simetría 180°). |
| `annotator.py` | Dibuja detecciones/tracks sobre los frames para revisión. |
| `visualizer.py` | Visualizador **HTML interactivo** (slider temporal, trayectorias, vectores de velocidad). |

---

## Esquema de tracking (SSS + ABG)

1. **Detección** (por frame): YOLO-seg + SAHI → máscaras → PCA → `Detection` con geometría.
2. **Predicción** (ABG): cada track predice su próximo estado (posición, velocidad, ángulo, `omega`).
3. **Gate espacial:** descarta candidatos fuera de `gate_x/y/angle`.
4. **Asociación (SSS):** vector de 17 features → similitud → matriz de costo → asignación.
5. **Actualización** (ABG): corrige el estado; `omega` espurio decae (`OMEGA_DECAY`).
6. **Profundidad:** por detección se estima `|Δz|` (DPTV); se agrega por track.

El desacople angular evita que el ruido PCA del ángulo (≈5–10° para fibras de aspect ratio ~65) se
amplifique por `1/dt` a `omega` no físicos (una fibra en flujo laminar rota ≤50–100 °/s).

---

## Salidas (`ResultadosPTV/<toma>/`)

| Archivo | Contenido |
|---|---|
| `tracks.json` | historia por track (mm, mm/s, °, largo/ancho mm, `depth`/`blur`) + `depth_stats` |
| `tracks.csv` | fila por (track, frame) |
| `detections.csv` | detecciones crudas por frame |
| `schedule.csv` | grilla temporal real: qué frames se analizaron, `timestamp_s`, `dt_s` por frame |
| `summary.json` | metadata de la toma + estadísticas globales (sección `dptv`) |

Todos los parámetros (SAHI, ABG, gates, DPTV, regiones temporales, modelos) se definen en
[`RunCode/variables_ptv.py`](../RunCode/variables_ptv.py) y llegan vía `pipeline_config.json`.
