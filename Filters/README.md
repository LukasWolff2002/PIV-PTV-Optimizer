# Filters/ — Ajuste interactivo del preprocesamiento PIV

GUI (Tkinter) para **tunear a mano** los parámetros de preprocesamiento de imágenes PIV por cámara,
viendo el efecto en vivo, y así fijar los valores que luego van a
[`RunCode/variables_piv.py`](../RunCode/variables_piv.py) (`CAM_PREPROCESS_PARAMS`).

## `PIV/`

| Archivo | Rol |
|---|---|
| `funciones.py` | Biblioteca de filtros estilo **PIVlab** (ajuste de histograma, *intensity capping*, **CLAHE**, high-pass, Wiener, ROI) + la clase `ImageTuner` (panel interactivo con sliders) + `detect_camera()`. No ejecuta nada. |
| `uso.py` | Lanzador: abre la GUI sobre un conjunto de imágenes y permite explorar/guardar la configuración por cámara. |

## Uso

```bash
python Filters/PIV/uso.py
```

Mueve los sliders hasta que las partículas/fibras queden bien realzadas y el fondo limpio; copia los
valores resultantes al perfil de la cámara correspondiente en `variables_piv.py`. Los mismos filtros
se aplican en producción vía `PIV/Codes/PreProcessing/filters.py`.
