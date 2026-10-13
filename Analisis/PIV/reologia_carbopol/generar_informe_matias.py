"""
Genera informe Word con resumen de la sesión de trabajo:
Reología Carbopol — mediciones rheómetro + simulaciones Fluent
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Estilos globales ────────────────────────────────────────────────────────
style_normal = doc.styles['Normal']
style_normal.font.name = 'Calibri'
style_normal.font.size = Pt(11)

def h1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p

def h2(text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    return p

def h3(text):
    return doc.add_heading(text, level=3)

def body(text):
    return doc.add_paragraph(text)

def bullet(text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Cm(0.5 * (level + 1))
    return p

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr = t.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), '2E74B5')
        shd.set(qn('w:color'), 'FFFFFF')
        shd.set(qn('w:val'), 'clear')
        tcPr.append(shd)
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # Data rows
    for r, row_data in enumerate(rows):
        row = t.rows[r + 1]
        for c, val in enumerate(row_data):
            cell = row.cells[c]
            cell.text = str(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t

def code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    return p

# ══════════════════════════════════════════════════════════════════════════════
# PORTADA
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('Protocolo de Medición Reológica y Configuración CFD\npara Fluidos Carbopol')
run.bold = True
run.font.size = Pt(18)
run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(f'Fecha: {datetime.date.today().strftime("%d de %B de %Y")}').font.size = Pt(12)

doc.add_paragraph()
intro = doc.add_paragraph()
intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
intro.add_run(
    'Resumen de la sesión de trabajo — Rheómetro Discovery HR2 (TA Instruments)\n'
    'Modelo Herschel-Bulkley · Corrección de deslizamiento · Implementación en ANSYS Fluent'
).italic = True

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTEXTO
# ══════════════════════════════════════════════════════════════════════════════
h1('1. Contexto del estudio')
body(
    'Se trabaja con geles de Carbopol a concentraciones 0,2 % y 0,5 % en peso, '
    'cuyo comportamiento reológico se describe mediante el modelo Herschel-Bulkley (HB):'
)
body('τ = τ₀ + K · γ̇ⁿ')
body(
    'Los parámetros de referencia provienen del ajuste PIV (método geométrico de Balmforth et al. 2007):'
)
add_table(
    ['Concentración', 'τ₀ (Pa)', 'K (Pa·sⁿ)', 'n'],
    [
        ['0,2 %', '40,6', '7,70', '0,532'],
        ['0,5 %', '84,0', '23,96', '0,442'],
    ],
    col_widths=[4, 3, 3, 3]
)
body(
    'El objetivo de las mediciones en el rheómetro es confirmar K y n en el rango de tasas de corte '
    'accesible y medir la velocidad de deslizamiento en pared para incorporarla en las simulaciones Fluent.'
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUCIÓN PIV DE γ̇
# ══════════════════════════════════════════════════════════════════════════════
h1('2. Distribución de tasa de corte medida con PIV (Zona Z1)')
body(
    'La distribución de γ̇ extraída de los campos PIV en la zona Z1 (brazo inclinado 60°) '
    'para Carbopol 0,2 % (run m71-toma-1, 88 805 pares τ–γ̇) muestra:'
)
add_table(
    ['Percentil', 'γ̇ (s⁻¹)'],
    [
        ['p5',  '0,052'],
        ['Q1',  '0,261'],
        ['Mediana', '0,589'],
        ['Q3',  '1,318'],
        ['p95', '5,889'],
        ['Máx', '23,09'],
    ],
    col_widths=[5, 5]
)
body(
    'La distribución es fuertemente sesgada a la derecha: la mayor parte del flujo ocurre '
    'a tasas < 1 s⁻¹, con una cola hasta ~23 s⁻¹ cerca de la pared durante el peak transitorio. '
    'Este rango define la ventana de comparación para las mediciones en el rheómetro.'
)
BOXPLOT_PATH = r'C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\boxplot_shear_rate_z1.png'
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_img = p.add_run()
run_img.add_picture(BOXPLOT_PATH, width=Cm(10))
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.add_run(
    'Figura 1. Distribución de tasa de corte γ̇ (s⁻¹) en escala logarítmica — '
    'Zona Z1, Carbopol 0,2 % (run m71-toma-1, N = 88 805 pares τ–γ̇). '
    'Whiskers en p5/p95. Puntos semitransparentes: muestra aleatoria de 3 000 valores.'
).italic = True

# ══════════════════════════════════════════════════════════════════════════════
# 3. RHEÓMETRO — LÍMITE INFERIOR
# ══════════════════════════════════════════════════════════════════════════════
h1('3. Límite inferior del Discovery HR2')
body(
    'El HR2 usa rodamiento magnético (sin fricción). En rotación continua, la limitación práctica '
    'es el torque mínimo de ~5 nN·m.'
)
add_table(
    ['Límite', 'γ̇ (s⁻¹)'],
    [
        ['Teórico (vel. angular mín.)', '~10⁻⁶'],
        ['Práctico — estado estacionario', '~0,001 – 0,01'],
        ['Recomendado para Carbopol', '0,01'],
    ],
    col_widths=[7, 5]
)
body(
    'Con Carbopol, la alta viscosidad aparente cerca del punto de fluencia permite acercarse '
    'a 0,01 s⁻¹ con buena relación señal/ruido, pero el deslizamiento en pared domina '
    'por debajo de ~0,05 s⁻¹ en placas lisas.'
)

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIGURACIÓN DE PLACAS
# ══════════════════════════════════════════════════════════════════════════════
h1('4. Configuración de placas disponibles')
body(
    'Se dispone de dos placas de distinto diámetro y acabado superficial:'
)
add_table(
    ['Placa', 'Diámetro', 'Superficie'],
    [
        ['M103', '40 mm', 'Rugosa (crosshatched)'],
        ['M100', '50 mm', 'Lisa (plana)'],
    ],
    col_widths=[3, 4, 5]
)
body(
    'Al ser de diámetros distintos, NO pueden usarse en la misma configuración sin corrección. '
    'La solución es: montar la placa rugosa (40 mm) arriba y la lisa (50 mm) abajo, '
    'cargar la muestra y recortarla al borde de la placa de 40 mm. '
    'Configurar el software TRIOS para diámetro = 40 mm. '
    'La placa de 50 mm actúa como base extendida; todo el cálculo usa R = 20 mm.'
)
body(
    'Esta configuración asimétrica (una rugosa + una lisa) reduce el deslizamiento a la mitad '
    'respecto al par liso-liso, y permite aplicar la corrección por variación de gap.'
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. PROTOCOLO DE MEDICIÓN
# ══════════════════════════════════════════════════════════════════════════════
h1('5. Protocolo de medición — corrección de deslizamiento por variación de gap')

h2('5.1 Fundamento')
body(
    'Con una superficie con deslizamiento (placa lisa) y una sin deslizamiento (placa rugosa), '
    'la tasa de corte aparente medida es:'
)
body('γ̇_app = γ̇_real + v_desliz(τ) / h')
body(
    'Al repetir el barrido a distintos gaps h, se obtiene un sistema sobredeterminado que '
    'permite resolver γ̇_real y v_desliz independientemente, sin necesidad de un segundo par de placas.'
)

h2('5.2 Rango y distribución de puntos')
body('Rango recomendado: 0,01 a 30 s⁻¹ (escala logarítmica, barrido ascendente).')
add_table(
    ['Zona', 'Rango (s⁻¹)', 'N puntos', 'Justificación'],
    [
        ['Cerca de fluencia', '0,01 – 0,5',  '12', 'Máxima sensibilidad a τ₀'],
        ['Rodilla / transición', '0,5 – 10',  '12', 'Máxima sensibilidad a n'],
        ['Ley de potencia', '10 – 30',         '6',  'Confirmar pendiente K·γ̇ⁿ'],
    ],
    col_widths=[4, 4, 3, 6]
)
body('Total: 30 puntos por gap. No es necesario llegar a 100 s⁻¹.')

h2('5.3 Protocolo por run')
bullet('Pre-cizallamiento: 50 s⁻¹ durante 60 s (borrar historia térmica y estructural)')
bullet('Reposo: 3 minutos')
bullet('Barrido ascendente 0,01 → 30 s⁻¹, modo estado estacionario (TRIOS: tolerancia 5 % sobre 10 s)')
bullet('Repetir para cada gap: h = 0,5 mm, 1,0 mm y 2,0 mm')
bullet('Recargar muestra fresca antes de cada gap (nunca reusar muestra ya cizallada)')

h2('5.4 Post-procesamiento')
body(
    'A cada nivel de tensión τ, graficar γ̇_app vs 1/h. '
    'La regresión lineal da:'
)
bullet('Intercepto = γ̇_real')
bullet('Pendiente = v_desliz(τ)')
body('Ajustar la ley de deslizamiento: v_desliz = α · τᵐ (típicamente m ≈ 0,88 para Carbopol 0,2 %).')
body('Ajustar HB sobre los pares (τ, γ̇_real) corregidos, fijando τ₀ desde PIV.')

h2('5.5 Curva ascendente vs. descendente')
body(
    'Para simulaciones de colada (flujo iniciando desde reposo), usar la CURVA ASCENDENTE. '
    'El material comienza con la red de gel intacta, análogo al estado inicial en el ensayo de colada. '
    'La curva ascendente proporciona el τ₀ estático (más alto) y los K, n consistentes con ese estado.'
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONFIABILIDAD DE τ₀
# ══════════════════════════════════════════════════════════════════════════════
h1('6. Confiabilidad de la tensión de fluencia τ₀')
body(
    'La corrección por variación de gap mejora significativamente los estimados de K y n, '
    'pero τ₀ sigue siendo difícil de obtener con placas paralelas:'
)
add_table(
    ['Método', 'Confiabilidad τ₀'],
    [
        ['Placas lisas (sin corrección)', '★☆☆☆☆'],
        ['Corrección gap (asimétrico)',    '★★★☆☆'],
        ['Barrido de amplitud oscilatoria (G\'/G\'\')', '★★★★☆'],
        ['Geometría vane',                 '★★★★★'],
    ],
    col_widths=[8, 5]
)
body(
    'Recomendación: usar τ₀ del método PIV geométrico (Balmforth et al. 2007) '
    'como valor fijo en el ajuste HB. El rheómetro aporta K y n; el PIV aporta τ₀. '
    'Esta combinación es la más robusta disponible con el equipamiento actual.'
)

# ══════════════════════════════════════════════════════════════════════════════
# 7. IMPLEMENTACIÓN EN ANSYS FLUENT
# ══════════════════════════════════════════════════════════════════════════════
h1('7. Implementación en ANSYS Fluent')

h2('7.1 Modelo Herschel-Bulkley (built-in)')
body(
    'En Materials → Viscosity → Herschel-Bulkley, ingresar los siguientes parámetros:'
)
add_table(
    ['Parámetro Fluent', 'Símbolo', '0,2 %', '0,5 %'],
    [
        ['Consistency index',  'K',    '7,70 Pa·sⁿ',  '23,96 Pa·sⁿ'],
        ['Power-law index',    'n',    '0,532',         '0,442'],
        ['Yield stress',       'τ₀',   '40,6 Pa',       '84,0 Pa'],
        ['Yielding viscosity', 'η₀',   '~42 600 Pa·s',  '~107 000 Pa·s'],
        ['Critical shear rate','γ̇_c',  '0,001 s⁻¹',     '0,001 s⁻¹'],
    ],
    col_widths=[5, 3, 4, 4]
)
body('Cálculo de η₀ (debe ser consistente con γ̇_c):')
code_block('η₀ = K · γ̇_c^(n-1) + τ₀ / γ̇_c')
body(
    'Este valor alto de η₀ aproxima el comportamiento rígido dentro de la zona de tapón (plug). '
    'Fluent usa un modelo bi-viscosidad: para γ̇ < γ̇_c, la viscosidad se fija en η₀.'
)

h2('7.2 Estrategia de convergencia')
bullet('Comenzar con τ₀ = 0 (fluido HB sin fluencia), converger')
bullet('Luego aumentar τ₀ gradualmente hasta el valor real')
bullet('Usar factores de sub-relajación de momento ≈ 0,3')
bullet('Refinar malla cerca de la superficie de fluencia: mínimo 3-4 celdas en la zona de transición')

h2('7.3 Condición de borde de deslizamiento (UDF)')
body(
    'La velocidad de deslizamiento medida en el rheómetro sigue la ley potencial:'
)
body('v_desliz = α · τ_pared^m')
body('Para Carbopol 0,2 % (literatura): α = 5,15×10⁻⁴ mm/s/Paᵐ, m = 0,876')
body('Implementación como UDF en Fluent:')
code_block(
    'DEFINE_PROFILE(wall_slip_velocity, thread, index)\n'
    '{\n'
    '    face_t f;\n'
    '    real tau_w, alpha, m;\n'
    '    alpha = 5.15e-7;   /* m/s / Pa^m */\n'
    '    m     = 0.876;\n'
    '    begin_f_loop(f, thread)\n'
    '    {\n'
    '        tau_w = F_STORAGE_R(f, thread, SV_WALL_SHEAR);\n'
    '        F_PROFILE(f, thread, index) = alpha * pow(tau_w, m);\n'
    '    }\n'
    '    end_f_loop(f, thread)\n'
    '}'
)
body(
    'Con los valores medidos en el propio rheómetro (variación de gap), reemplazar α y m '
    'con los coeficientes ajustados para cada concentración.'
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. REFERENCIAS
# ══════════════════════════════════════════════════════════════════════════════
h1('8. Referencias clave')
refs = [
    ('Yoshimura & Prud\'homme (1988)',
     'Wall Slip Corrections for Couette and Parallel Disk Viscometers. '
     'J. Rheology, 32(1):53–67. — Método Y-P para corrección de deslizamiento.'),
    ('Poumaere et al. (2014)',
     'Unsteady laminar flows of a Carbopol gel in the presence of wall slip. '
     'J. Non-Newtonian Fluid Mech., 205:28–40. — Carbopol + deslizamiento en flujo confinado.'),
    ('Piau (2007)',
     'Carbopol gels: Elastoviscoplastic and slippery glasses. '
     'J. Non-Newtonian Fluid Mech., 144:1–29. — Referencia central sobre reología Carbopol.'),
    ('Balmforth et al. (2007)',
     'Viscoplastic dam breaks and the Bostwick consistometer. '
     'J. Non-Newtonian Fluid Mech., 142:63–78. — Método geométrico para τ₀.'),
    ('TA Instruments App Note',
     'Preventing Wall Slip in Rheology Experiments. '
     'tainstruments.com/applications-notes/preventing-wall-slip — Guía práctica HR2.'),
    ('Slippery flows of Carbopol in a micro-channel (2020)',
     'arxiv.org/pdf/2002.00856 — Ley de deslizamiento: v_s = 5,15×10⁻⁴ τ^0,876 para 0,2 % Carbopol.'),
]
for authors, text in refs:
    p = doc.add_paragraph(style='List Bullet')
    run_bold = p.add_run(authors + ': ')
    run_bold.bold = True
    p.add_run(text)

# ── Guardar ─────────────────────────────────────────────────────────────────
out = r'C:\Users\patri\OneDrive\Desktop\PIV_analisis_carbopol_fibers\Informe_Matias_Reologia_Carbopol.docx'
doc.save(out)
print(f'Guardado: {out}')
