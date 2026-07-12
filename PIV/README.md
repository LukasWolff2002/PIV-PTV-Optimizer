# PIV/ — Motor de Particle Image Velocimetry

Calcula **campos de velocidad del fluido** a partir de pares de fotogramas, usando correlación cruzada
por ventanas (OpenPIV) con preprocesamiento adaptativo y enmascarado YOLO + estático.

```
PIV/
├── Tomas/                       # fotogramas crudos por toma (no versionado)
├── Codes/
│   ├── Segmentation-Models/     # pesos YOLO camN-piv-yolo26.pt (no versionado)
│   ├── PreProcessing/           # muestreo por bloques, filtros, máscaras, regiones temporales
│   ├── OpenPIV/                 # motor de cómputo PIV modular
│   └── Others/                  # herramientas de máscaras standalone
```

Las etapas se orquestan desde [`RunCode/`](../RunCode/README.md) (`preprocess_run.py` → `piv_run.py`);
esta carpeta contiene la **librería**.

---

## `Codes/PreProcessing/` — Preparación de imágenes

| Módulo | Rol |
|---|---|
| `temporal_regions.py` | `TemporalRegion` / `BlockMetadata`: dataclasses del muestreo adaptativo por fase de velocidad (`block_size`, `skip_inter`, `skip_final`, `fps`), con validación y (de)serialización a JSON. |
| `blocks.py` | Muestreo por bloques: `run_block_sampling` (uniforme) y `run_adaptive_block_sampling` (multi-región temporal). Selecciona qué fotogramas forman cada par y escribe `block_metadata.json` con los timestamps reales. |
| `filters.py` | Filtros estilo **PIVlab**: ajuste de histograma, *intensity capping* (n·σ), **CLAHE**, high-pass, Wiener, ROI. `apply_preprocessing(img, params)` los encadena según los `preprocess_params` de la cámara. |
| `masks.py` | Máscaras **dinámicas YOLO** por imagen (`run_masks_yolo`): segmentación → postproceso morfológico → intersección con la **máscara fija** (`fixed_mask_path`) → binaria `*_mask.tiff`. |

## `Codes/OpenPIV/` — Cómputo PIV (modular)

Arquitectura por responsabilidades; el punto de entrada es `run.run_piv(cfg)`:

| Módulo | Rol |
|---|---|
| `config.py` | `PIVConfig` (dataclass inmutable): rutas, `dt_ms`, `px_per_mm`, `window_sizes`, `overlaps`, umbrales de validación, flags de máscara. Propiedades `dt_s`, `mm_per_px`. |
| `models.py` | `PairJob`, `PIVResult`, `PIVResultFinal`: el trabajo por par y sus resultados (campos crudos → validados). |
| `pipeline.py` | `PIVPipeline`: `build_jobs()` (empareja imágenes con sus máscaras), `compute_all()` (paralelo con `ProcessPoolExecutor`), `validate_all()`. Lee `block_metadata.json` para los timestamps. |
| `workers.py` | Trabajo por par: `compute_pair_worker` (correlación por ventanas, con blanqueo del fondo enmascarado) y `validate_pair_worker`. |
| `validation.py` | Rechazo de vectores: **convex hull** de la región con flujo, región circular/**Mahalanobis**, y **mediana local robusta** (`local_median_flags`) para outliers. |
| `exporter.py` | `TxtExporter`: escribe los `.txt` finales (`x,y,u,v`) con metadata temporal por par. |
| `viewer.py` | `PIVViewer`: visor interactivo matplotlib (slider temporal, vorticidad, hulls) para inspección. |
| `naming.py`, `timestamp_utils.py` | Nombres de archivo con metadata y cálculo de timestamps desde `block_metadata.json`. |
| `run.py` | `run_piv(cfg, opt)`: valida entradas de máscara, orquesta pipeline → viewer → exporter. |
| `utils.py` | E/S en escala de grises, emparejado de índices, blanqueo de fondo enmascarado. |

## `Codes/Others/` — Máscaras standalone

`dynamic_mask.py`, `masked.py` (genera máscaras y/o imágenes "masked" a partir de YOLO),
`masks/` (subpaquete `generator`/`postprocess`/`utils`), `ExtractPIVPhotos.py`, `openpiv-code.py`,
`piv_comparison.py`, `image_blocks/extractor.py`. Utilidades independientes del pipeline principal;
la ruta de producción usa `PreProcessing/masks.py`.

---

## Flujo de datos

```
PIV/Tomas/<toma>/*.tiff
   → (blocks) TomasProcesadas/<toma>/  + block_metadata.json
   → (masks)  Masks/<toma>/*_mask.tiff
   → (OpenPIV) ResultadosPIV/<toma>/*.txt   (x, y, u[mm/s], v[mm/s])
```

Los parámetros por cámara/carbopol (ventanas, solapamientos, percentiles, filtros, regiones
temporales) se definen en [`RunCode/variables_piv.py`](../RunCode/variables_piv.py) y llegan al motor
vía `pipeline_config.json`.
