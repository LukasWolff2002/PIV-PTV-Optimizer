# Trainers/ — Entrenamiento de modelos de segmentación de fibras

Entrena los modelos que el pipeline usa para segmentar fibras (máscaras dinámicas PIV/PTV y detección
PTV). Dos familias: **YOLOv8-seg** (producción) y **DINOv2** (experimental / comparación).

## `YOLOTrainer/` — YOLOv8-seg + SAHI (producción)

| Script | Rol |
|---|---|
| `1_prepare_and_train.py` | Prepara el dataset (exportado de **Roboflow** como *YOLOv8 Segmentation*), lo **tilea** (recorta en mosaicos con reescalado de los polígonos) para captar fibras pequeñas, y entrena YOLOv8-seg. |
| `2_infer_sahi.py` | Inferencia con **SAHI** (*Slicing Aided Hyper Inference*): segmenta por tiles, reescala las máscaras ÷scale, extrae parámetros de fibra por **PCA** (más robusto que `fitEllipse`), visualiza y exporta CSV. |

Ver [`YOLOTrainer/README.md`](YOLOTrainer/README.md) para el paso a paso (Roboflow → dataset →
entrenamiento → inferencia). Dependencias: `ultralytics sahi opencv-python tqdm matplotlib pyyaml pillow`.

Los pesos resultantes se copian a `PIV|PTV/Codes/Segmentation-Models/` con el nombre que espera el
pipeline (`camN-piv-yolo26.pt`, `camN-ptv-yolo26.pt`, `best.pt`).

## `DINOTrainer/` — DINOv2 (+ RAFT) (experimental)

| Script | Rol |
|---|---|
| `dino_seg_train.py` | Entrena una cabeza de segmentación sobre features congeladas de **DINOv2 ViT-S/14** (73×73×384 → upsample → conv 1×1 → máscara P(fibra)). Pérdida BCE + Dice. |
| `dino_infer.py` | Inferencia del modelo DINO sobre la secuencia: máscara binaria PNG + JSON de fibras (centroide, largo, ángulo) + MP4 comparativo, con fusión de segmentos colineales. |
| `dino_raft.py` | Pipeline combinado **DINOv2 + RAFT** (optical flow): la máscara DINO restringe dónde hay fibra y RAFT (sobre el frame original) aporta el movimiento; extrae fibras con dirección de flujo. |

> YOLOv8-seg es la ruta usada por el pipeline; DINO/RAFT es exploratorio (comparación de enfoques de
> segmentación/flujo).
