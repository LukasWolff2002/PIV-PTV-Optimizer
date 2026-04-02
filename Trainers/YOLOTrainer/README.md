# Pipeline de detección de fibras — YOLOv8-seg + SAHI

## Setup del entorno

```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install ultralytics sahi opencv-python tqdm matplotlib pyyaml pillow
```

---

## Paso 1: Exportar dataset desde Roboflow

En Roboflow:
1. Ir al dataset → **Export Dataset**
2. Formato: **YOLOv8 Oriented Segmentation** (o "YOLOv8 Segmentation")
3. Descargar y descomprimir en la carpeta `dataset/`

Estructura esperada:
```
dataset/
├── train/
│   ├── images/    (*.jpg / *.png / *.tiff)
│   └── labels/    (*.txt — polígonos YOLO)
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## Paso 2: Preparar dataset con tiles + entrenar

```bash
python 1_prepare_and_train.py
```

Esto hace automáticamente:
- Upscalea cada imagen ×3 (fibra pasa de ~1.5 px a ~4.7 px de ancho)
- Divide en tiles 512×512 con 50% de solapamiento
- Ajusta todas las máscaras de instancia a cada tile
- Entrena YOLOv8n-seg (o s/m según `MODEL_SIZE`) con augmentaciones rotacionales

**Parámetros clave a ajustar en el script:**
| Variable | Default | Cuándo cambiar |
|---|---|---|
| `SCALE_FACTOR` | 3 | Subir a 4 si las fibras son muy tenues |
| `MODEL_SIZE` | "n" | Cambiar a "s" o "m" si tienes >500 imágenes |
| `BATCH` | 16 | Bajar a 8 si aparece OOM en GPU |
| `EPOCHS` | 150 | Bajar a 80 si el dataset es grande |

El mejor modelo queda en: `runs/fiber_seg/train/weights/best.pt`

---

## Paso 3: Inferencia sobre imagen nueva

```bash
python 2_infer_sahi.py \
    --img  cam-1.tiff \
    --model runs/fiber_seg/train/weights/best.pt \
    --out  output/
```

Genera:
- `output/cam-1_detections.png` — visualización con rectas coloreadas
- `output/cam-1_fibers.csv` — tabla con centroide, ángulo y largo por fibra

### Formato CSV de salida (entrada para PTV)

| columna | descripción |
|---|---|
| `id` | índice de la fibra |
| `cx_px`, `cy_px` | centroide en píxeles (espacio original 1024×1024) |
| `angle_deg` | ángulo del eje mayor (0°=horizontal, 90°=vertical) |
| `length_px` | largo proyectado en píxeles |
| `length_mm` | largo proyectado en mm (÷ 7.8 px/mm) |
| `width_px` | ancho estimado en píxeles |
| `conf` | confianza del modelo (0–1) |

---

## Notas sobre el entrenamiento

### Augmentaciones configuradas
- `degrees=180` — rotación completa (fibras en cualquier orientación)
- `flipud=0.5`, `fliplr=0.5` — espejo horizontal y vertical
- `hsv_s=0.0` — sin augmentación de saturación (imágenes grayscale)
- `hsv_v=0.3` — variación de brillo (simula distintas iluminaciones)

### Qué esperar
- Con 200–400 imágenes tileadas: ~800–1600 tiles de entrenamiento
- Tiempo de entrenamiento (GPU local, 150 epochs): ~1–3 horas
- mAP50 esperado en fibras individuales bien visibles: 0.4–0.7
  (fibras con mucho overlap son difíciles incluso para humanos)

### Si el modelo sobreajusta rápido
Bajar `lr0` a `5e-4` y subir `patience` a 50 en `1_prepare_and_train.py`

### Si hay pocas detecciones en inferencia
Bajar `CONF_THRESHOLD` a `0.15` en `2_infer_sahi.py`
