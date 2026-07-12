# Others/ — Utilidades sueltas

Scripts auxiliares independientes del pipeline principal.

| Archivo | Rol |
|---|---|
| `ptv_results_filter.py` | Recorre recursivamente una carpeta de resultados PTV y **copia solo lo esencial** (`tracks.json`, `schedule.csv`) a un destino, replicando la estructura. Útil para compartir/archivar resultados sin los intermedios pesados. |
| `random_fotos.py` | Toma una **muestra aleatoria** de fotogramas de las tomas, los preprocesa y convierte, y los copia a una carpeta destino (p. ej. para armar un dataset de anotación en Roboflow o inspección rápida). Detecta la cámara desde la ruta y respeta el *bit depth* original. |

## Uso

```bash
python Others/ptv_results_filter.py     # editar rutas origen/destino en el script
python Others/random_fotos.py <origen> <destino> <cantidad>
```
