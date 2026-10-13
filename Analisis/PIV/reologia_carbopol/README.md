# Reología Carbopol — Análisis PIV + Reómetro HR30

Scripts de análisis para la caracterización reológica de Carbopol 0.2 % mediante PIV y reómetro HR30 (TA Instruments) en geometría L-beam 30°.

## Scripts

| Script | Descripción |
|--------|-------------|
| `boxplot_shear_rate.py` | Boxplot inicial γ̇ — Z1, Carbopol 0.2 %, un run |
| `boxplot_v2.py` | Versión mejorada: violin + box, p99 cutoff |
| `boxplot_z2.py` | γ̇ en Z2 (codo) usando dirección local de velocidad |
| `boxplot_z3.py` | γ̇ en Z3 (salida, 30° horizontal) con ejes fijos |
| `comparativo_zonas.py` | Pasada única sobre todos los frames: Z1 + Z2 + Z3 simultáneo |
| `comparativo_tmin.py` | Efecto del corte temporal t > 0.5s vs t > 1.8s en Z1/Z2/Z3 |
| `viga_shear_rate.py` | γ̇ por celda en la viga horizontal (grilla 2×3, t > 1.8 s) |
| `analisis_reometro_slip.py` | Corrección Mooney (3 gaps) + ajuste HB + slip power-law + comparación PIV |
| `diagrama_zonas.py` | Diagrama de zonas Z1/Z2/Z3 en la geometría L-beam |
| `generar_informe_matias.py` | Genera informe Word para el estudiante Matías (python-docx) |

## Zonas de análisis

- **Z1** — Brazo inclinado (60° desde horizontal); T=(−0.5, 0.866), N=(−0.866, −0.5)  
- **Z2** — Codo; dirección local de velocidad por punto  
- **Z3** — Salida (30° desde horizontal); T=(0.866, −0.5), N=(0.5, 0.866)  
- **Viga** — Beam horizontal; grilla 2 filas × 3 columnas (Vf1c1…Vf2c3); T=(1,0), N=(0,1)

## Corrección de slip (Mooney)

Medición con HR30, geometría M105 (placa paralela 40 mm), 3 gaps: 0.5 / 1.0 / 2.0 mm.  
Método: interpolación PCHIP en log-log, regresión γ̇_app vs 2/h a estrés constante.

**Resultado Carbopol 0.2 %:**
- HB corregido: τ₀ ≈ 5.4 Pa, K ≈ 38.3 Pa·sⁿ, n ≈ 0.20
- Slip power-law: v_slip = 1.77×10⁻⁵ × τ^0.672 m/s
- τ₀ geométrico PIV (Balmforth): 40.6 Pa (yield stress estático)

## Dependencias

```
numpy, scipy, matplotlib, xlrd==1.2.0, python-docx
```

Los archivos de datos PIV deben estar en:
```
C:/Users/patri/OneDrive/Desktop/PIV-PTV-Optimizer/PIV/PIV_INTERPOLADO/PIV_INTERPOLADO/
```
Los scripts de zona y extracción en:
```
C:/Users/patri/OneDrive/Desktop/PIV_analisis_carbopol_fibers/
```
