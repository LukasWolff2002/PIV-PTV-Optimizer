# PIV-PTV-Optimizer

Toolchain de **visión por computador** para procesar filmaciones de alta velocidad de un experimento
de colado de **hormigón autocompactante con fibras de acero** (análogo transparente: Carbopol +
macrofibras), y extraer:

- **PIV** (*Particle Image Velocimetry*) → campos de **velocidad del fluido**.
- **PTV** (*Particle Tracking Velocimetry*) → **trayectorias individuales de cada fibra** (posición,
  velocidad, ángulo, largo, ancho y **profundidad Δz** por defocus).

Estos resultados son la materia prima del análisis de tesis (repo `Analisis_PIV-PTV_Final`), que
modela la ley de saturación de orientación η₁(Γ) y el factor de orientación para UHPC.

---

## 1. El experimento

- **Fluido:** Carbopol Ultrez 10 al 0.2 % y 0.5 % en peso (viscoplástico, con esfuerzo de fluencia).
- **Fibras:** macrofibras de acero, diámetro **d = 0.2 mm**, largo **L = 13 mm**.
- **Geometría:** canal en **L** que alimenta un molde de **viga** horizontal.
- **Cámaras:** filman a través de una pared de acrílico (plano *xy*). 4 cámaras (`cam1`–`cam4`),
  escala calibrada ≈ **8 px/mm** (cam1–3) y **10.7 px/mm** (cam4), a 220/660 fps.
- **PIV:** hoja láser en el plano medio (z = 30 mm) → 2D del fluido.
- **PTV:** iluminación difusa → todas las fibras del volumen (60 mm de profundidad) visibles a la vez;
  la profundidad de cada fibra se estima por **desenfoque** (*depth-from-defocus*, DPTV).

El fundamento físico del DPTV está documentado en [`Contextos/dptv.md`](Contextos/dptv.md).

---

## 2. Arquitectura del pipeline

Un orquestador ([`RunCode/pipeline_global.py`](RunCode/README.md)) recorre las carpetas de tomas
crudas, consulta una **Google Sheet** para saber cuáles procesar, escribe un `pipeline_config.json`
por toma y ejecuta cada etapa en su **entorno conda** correspondiente:

```
                      PIV/Tomas/  ó  PTV/Tomas/           (fotogramas crudos por cámara)
                              │
        ┌─────────────────────┴─────────────────────┐
        │  pipeline_global.py  (lee Google Sheet +   │
        │  variables_piv/ptv → pipeline_config.json) │
        └─────────────────────┬─────────────────────┘
                              │
   ╔══════════ PIV ══════════╪══════════ PTV ══════════╗
   ║  preprocess_run.py       │   preprocess_run_ptv.py ║  (env: yolov11)
   ║   · block sampling       │    · selecciona frames  ║
   ║   · CLAHE / capping      │    · CLAHE / capping     ║
   ║   · máscaras YOLO+fija    │    · máscaras YOLO+fija  ║
   ║          │               │          │              ║
   ║  piv_run.py (env: piv)   │   ptv_run.py (env yolox4)║
   ║   · OpenPIV por pares    │    · YOLO-seg + SAHI     ║
   ║   · validación regional  │    · tracker SSS + ABG   ║
   ║   · export .txt          │    · DPTV (profundidad)  ║
   ║          │               │          │              ║
   ║  cleanup_run.py          │   cleanup_run.py         ║
   ╚══════════╪═══════════════╧══════════╪══════════════╝
              ▼                          ▼
        ResultadosPIV/              ResultadosPTV/
     (campos u,v por .txt)     (tracks.json/csv, schedule.csv)
```

Modelos de segmentación (YOLO) → entrenados con [`Trainers/`](Trainers/README.md).
Máscaras estáticas → dibujadas con [`FixMasks/`](FixMasks/README.md).
Calibración de profundidad → [`Calibracion/`](Calibracion/README.md).
Ajuste de filtros de preprocesamiento → [`Filters/`](Filters/README.md).
Análisis posterior de los resultados → [`Analisis/`](Analisis/README.md).

---

## 3. Entornos (conda)

Tres entornos, exportados en [`Entornos/`](Entornos/README.md). El orquestador invoca cada etapa con
`conda run -n <env> …`:

| Entorno | Uso | Notas |
|---|---|---|
| `yolov11` | Preprocesamiento + **máscaras dinámicas YOLO** (PIV y PTV) | PyTorch + CUDA |
| `piv` | **Cómputo PIV** (OpenPIV) | Python 3.11, ligero (sin GPU) |
| `yolox4` | **Detección + tracking PTV** (YOLO-seg + SAHI) | PyTorch + CUDA |

```bash
conda env create -f Entornos/yolov11.yml
conda env create -f Entornos/piv.yml
conda env create -f Entornos/yolox4.yml
```

---

## 4. Estructura del repositorio

| Carpeta | Contenido | README |
|---|---|---|
| `RunCode/` | Orquestación: `pipeline_global.py`, config (`variables_piv/ptv.py`), runners por etapa | [ver](RunCode/README.md) |
| `PIV/` | Motor PIV (`Codes/OpenPIV`, `Codes/PreProcessing`), modelos y tomas crudas | [ver](PIV/README.md) |
| `PTV/` | Motor PTV (`Codes/PTVCode`: detector, tracker, DPTV), modelos y tomas | [ver](PTV/README.md) |
| `Trainers/` | Entrenamiento de modelos de segmentación (YOLOv8-seg, DINOv2, DINO+RAFT) | [ver](Trainers/README.md) |
| `Calibracion/` | Calibración DPTV (`k_blur`) y re-cálculo sobre resultados existentes | [ver](Calibracion/README.md) |
| `Filters/` | GUI interactiva para tunear el preprocesamiento PIV por cámara | [ver](Filters/README.md) |
| `FixMasks/` | Editor interactivo de máscaras estáticas (polígonos) → `cam-N.tiff` | [ver](FixMasks/README.md) |
| `Analisis/` | Post-análisis: interpolación PIV, comparación de tomas, 3D/profundidad PTV | [ver](Analisis/README.md) |
| `Others/` | Utilidades (filtrar resultados PTV, muestreo aleatorio de fotos) | [ver](Others/README.md) |
| `Contextos/` | `dptv.md` — guía física del depth-from-defocus | — |
| `Entornos/` | Exports de los entornos conda | [ver](Entornos/README.md) |
| `Paper/` | Manuscrito LaTeX (IOP) del método | — |
| `BasePhotos/` | Fotogramas de ejemplo | — |
| `ResultadosPIV/`, `ResultadosPTV/` | Salidas (no versionadas; ver su `README.md`) | — |

---

## 5. Cómo correr el pipeline

1. **Coloca los datos crudos** en `PIV/Tomas/` y/o `PTV/Tomas/`. Cada toma es una carpeta con el
   nombre canónico:

   ```
   m{mezcla}-toma-{toma}-cam-{cam}-n-{n}-car-{car}-{piv|ptv}
   ej:  m70-toma-2-cam-1-n-3000-car-02-piv
   ```

2. **Coloca los modelos** en `PIV/Codes/Segmentation-Models/` (`camN-piv-yolo26.pt`) y
   `PTV/Codes/Segmentation-Models/` (`camN-ptv-yolo26.pt`, `best.pt`), y las **máscaras estáticas**
   en `FixMasks/cam-N.tiff` (solo si `apply_static_mask=True`).

3. **Configura** el modo en [`RunCode/pipeline_global.py`](RunCode/pipeline_global.py):

   ```python
   RUN_MODE = "ptv"   # "piv" | "ptv" | "both"
   ```

   Los parámetros por cámara/carbopol están en `RunCode/variables_piv.py` y `variables_ptv.py`
   (perfiles de cámara, regiones temporales, filtros, tracking, DPTV).

4. **Ejecuta:**

   ```bash
   python RunCode/pipeline_global.py
   ```

   El orquestador procesa cada carpeta que la Google Sheet marque con `Usar=si`, escribe
   `pipeline_config.json` y lanza las etapas en sus entornos. Los resultados quedan en
   `ResultadosPIV/<toma>/` y `ResultadosPTV/<toma>/`.

> **Sin Google Sheet:** si la hoja no está disponible, el pipeline procesa todo por defecto
> (`should_process_folder` → `True`).

---

## 6. Salidas

**PIV** (`ResultadosPIV/<toma>/`): archivos `.txt` por par de fotogramas con columnas
`x, y, u, v` (mm y mm/s) tras validación regional (convex hull + mediana local) y metadata temporal.

**PTV** (`ResultadosPTV/<toma>/`):

| Archivo | Contenido |
|---|---|
| `tracks.json` | historia completa por track (posición mm, velocidad mm/s, ángulo °, largo/ancho mm, `depth`/`blur`) + `depth_stats` |
| `tracks.csv` | una fila por (track, frame) |
| `detections.csv` | detecciones crudas por frame |
| `schedule.csv` | **grilla temporal real** (qué frames se analizaron, `timestamp_s`, `dt_s` por frame) |
| `summary.json` | metadata de la toma + estadísticas globales (incl. sección `dptv`) |

Unidades de salida: posición **mm**, velocidad **mm/s**, aceleración **mm/s²**, ángulo **grados**.

---

## 7. Datos y `.gitignore`

No se versionan (grandes o descargables): `PIV/Tomas/`, `PTV/Tomas/`, `ResultadosPIV/`,
`ResultadosPTV/`, `Masks/`, `TomasProcesadas/`, `PTVPreprocesadas/`, pesos `*.pt` de los modelos, y
`pipeline_config.json` (se regenera). Los `ResultadosPIV/README.md` y `ResultadosPTV/README.md`
(con el enlace de descarga) sí se conservan.
