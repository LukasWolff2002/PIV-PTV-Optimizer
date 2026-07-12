# Entornos/ — Entornos conda

Exports de los tres entornos conda que usa el pipeline. Cada etapa se ejecuta en el suyo (el
orquestador invoca `conda run -n <env> …`).

| Archivo | Entorno | Uso | Stack |
|---|---|---|---|
| `piv.yml` | `piv` | Cómputo **PIV** (OpenPIV) | Python 3.11, CPU (ligero) |
| `yolov11.yml` | `yolov11` | Preprocesamiento + **máscaras dinámicas YOLO** (PIV y PTV) | PyTorch + CUDA (NVIDIA) |
| `yolox4.yml` | `yolox4` | **Detección + tracking PTV** (YOLO-seg + SAHI) | PyTorch + CUDA (NVIDIA) |

## Crear los entornos

```bash
conda env create -f Entornos/piv.yml
conda env create -f Entornos/yolov11.yml
conda env create -f Entornos/yolox4.yml
```

> Los `.yml` son *exports* completos (con builds fijados) generados en Windows. Si los recreas en
> otra plataforma o hay conflictos de build, puede convenir un `environment.yml` mínimo con solo las
> dependencias de alto nivel (`pytorch`, `ultralytics`, `sahi`, `openpiv`, `opencv`, `numpy`,
> `pandas`, `scipy`, `matplotlib`, `imageio`, `tifffile`, `requests`).

Los nombres de entorno que el pipeline espera se configuran en
[`RunCode/pipeline_global.py`](../RunCode/README.md) (`ENV_YOLO`, `ENV_PIV`, `ENV_PTV`,
`ENV_PREPROCESS_PTV`).
