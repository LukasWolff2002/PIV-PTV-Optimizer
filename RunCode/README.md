# RunCode/ — Orquestación del pipeline

Coordina todo el flujo PIV/PTV: descubre las tomas, decide cuáles procesar, construye la
configuración por toma y ejecuta cada etapa en su entorno conda. **Aquí vive toda la configuración
del pipeline.**

## Ejecución

```bash
python RunCode/pipeline_global.py
```

## Archivos

| Archivo | Rol |
|---|---|
| `pipeline_global.py` | **Orquestador maestro.** Descubre carpetas, consulta la Google Sheet, arma `pipeline_config.json` y lanza las etapas por entorno. |
| `variables_piv.py` | **Config PIV:** directorios, `PIV_METODO`, perfiles de cámara, preprocesamiento por cámara (CLAHE, capping…), **regiones temporales**, ventanas de correlación, percentiles de validación, máscaras. |
| `variables_ptv.py` | **Config PTV:** directorios, modelos, dispositivo, filtro **ABG** (lineal + angular desacoplado), gates de tracking, **SAHI**, y parámetros **DPTV**. |
| `preprocess_run.py` | Etapa PIV-1 (env `yolov11`): block sampling + preprocesamiento + máscaras dinámicas YOLO. |
| `preprocess_run_ptv.py` | Etapa PTV-1 (env `yolov11`): preprocesa **solo** los frames que el tracker necesitará + sus máscaras. |
| `piv_run.py` | Etapa PIV-2 (env `piv`): corre OpenPIV desde el JSON. |
| `ptv_run.py` | Etapa PTV-2 (env `yolox4`): corre el tracker de fibras desde el JSON. |
| `cleanup_run.py` | Borra intermedios (preprocesadas, máscaras, `runs/segment`) según el JSON. |

## `pipeline_global.py` en detalle

### Modo de ejecución
```python
RUN_MODE = "piv" | "ptv" | "both"
ALLOW_BOTH_WITHOUT_PTV = True   # en "both", omite PTV si no hay carpetas PTV
```

### Entornos por etapa
```python
ENV_YOLO           = "yolov11"   # máscaras dinámicas (PIV) + preprocesamiento PTV
ENV_PIV            = "piv"        # cómputo PIV
ENV_PREPROCESS_PTV = "yolov11"
ENV_PTV            = "yolox4"     # tracking PTV
```
Cada etapa se ejecuta como subproceso: `conda run -n <env> python <script> pipeline_config.json`.
La ruta a `conda.bat` se autodetecta de una lista de candidatos (`CONDA_BAT_OPTIONS`) — **ajústala a
tu máquina**.

### Google Sheets (gating de experimentos)
`load_experiment_config()` descarga una hoja pública (CSV) con una fila por
`Mezcla · Toma · Tipo(carbopol) · Metodo`. De ahí sale:
- `should_process_folder()` → procesa solo si la columna `Usar ∈ {si, sí, s, yes, y}`.
- `get_skip_images_for_folder()` → cuántos fotogramas saltar al inicio (`Fotos Saltar`), ajustado por
  `Razon FPS` para `cam4` (cámara rápida).

> Si la hoja no carga, todo se procesa por defecto y `skip=0`.

### Convención de nombres de carpeta
```
m{mezcla}-toma-{toma}-cam-{cam}-n-{n}-car-{car}-{metodo}
```
`parse_subfolder_name()` la valida con regex; `list_matching_subfolders()` filtra por `metodo`
(`piv`/`ptv`) y ordena naturalmente.

### Perfiles de cámara
`CAM_PROFILES_PIV` / `CAM_PROFILES_PTV` (en `variables_piv/ptv.py`) definen por cámara: `fps`,
`dt_ms`, `px_per_mm`, `width_px`, `height_px`, y los flags `apply_dynamic_mask` / `apply_static_mask`.
`_validate_mask_setup()` verifica que exista el modelo YOLO (si dynamic) y la máscara fija (si static)
antes de correr.

### `pipeline_config.json`
`write_cfg()` compila **todos** los parámetros de una toma en un JSON con secciones:
`meta`, `camera`, `pre`, `masks`, `piv`, `pre_ptv`, `masks_ptv`, `ptv` (incluye SAHI, ABG y DPTV) y
`cleanup`. Es el único contrato entre el orquestador y cada script de etapa (cada `*_run.py` solo lee
este JSON).

## Rutas (definidas en `variables_*.py`)

| Variable | Ruta |
|---|---|
| `PRE_BASE_DIR` / `PTV_BASE_DIR` | `PIV/Tomas` / `PTV/Tomas` (entrada cruda) |
| `PROCESSED_ROOT`, `MASKS_ROOT` | `TomasProcesadas/`, `Masks/` (intermedios PIV) |
| `PTV_PREPROCESSED`, `PTV_MASKS_ROOT` | `PTVPreprocesadas/`, máscaras PTV (intermedios) |
| `RESULTS_PIV_ROOT` / `RESULTS_PTV_ROOT` | `ResultadosPIV/` / `ResultadosPTV/` (salida) |
| `PIV_MODELS_DIR` / `PTV_MODELS_DIR` | `PIV|PTV/Codes/Segmentation-Models/` |

## Regiones temporales (muestreo adaptativo)

`variables_piv.py::TEMPORAL_REGIONS_CAR02/05` y su equivalente PTV segmentan cada toma por **fase de
velocidad** (alta / media / baja…) con distinto `block_size`, `skip_inter`, `skip_final` y `fps`. Así
se muestrea denso cuando el flujo es rápido y ralo cuando ya casi no hay movimiento — clave para PIV
(pares con desplazamiento adecuado) y para no procesar frames redundantes en PTV.

## Filtro ABG del tracker (resumen)

`variables_ptv.py` documenta a fondo el filtro **alpha-beta-gamma** con cinemática **lineal** y
**angular desacoplada**: como `dt = 1/220 s` es muy chico, una ganancia angular alta amplificaría el
ruido PCA del ángulo (×1/dt); por eso `BETA_ANG ≪ BETA` y hay términos de amortiguamiento
(`OMEGA_DECAY`, `ALPHA_ANG_DECAY`) que hacen decaer `omega` espurio si no lo sostienen residuos
consistentes.
