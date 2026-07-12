# FixMasks/ — Editor de máscaras estáticas

Herramienta interactiva (OpenCV) para **dibujar la máscara fija** de cada cámara: el polígono que
delimita las zonas a **ignorar** de forma permanente (paredes del molde, reflejos, bordes del acrílico)
que no cambian entre fotogramas.

## `Code/make_fix_masks.py`

Editor de polígonos sobre una imagen de referencia (`.tiff`), con:
- Zoom/paneo y ajuste de visualización (exposición, gamma, *clip* por percentiles) para ver bien fibras
  sub-pixel.
- Edición de vértices del polígono.
- Exportación a **máscara binaria TIFF**.

```bash
python FixMasks/Code/make_fix_masks.py
```

## Salida

Las máscaras se guardan como **`cam-N.tiff`** en `FixMasks/` (una por cámara). El pipeline las toma
desde ahí cuando el perfil de cámara tiene `apply_static_mask=True`
(`fixed_mask_path_for_cam(cam)` en [`RunCode/pipeline_global.py`](../RunCode/README.md)).

**Convención:** blanco = zona a **ignorar**, negro = zona a **analizar** (consistente con el tracker
PTV y el enmascarado PIV).
