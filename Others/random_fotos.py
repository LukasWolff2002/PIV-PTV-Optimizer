import os
import random
from PIL import Image

def copiar_y_convertir_fotos_random(carpeta_origen, carpeta_destino, cantidad):
    # 1. Crear la carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"Carpeta creada: {carpeta_destino}")

    # 2. Filtrar solo los archivos que sean imágenes (incluyendo .tiff y .tif)
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif')
    todas_las_fotos = [
        archivo for archivo in os.listdir(carpeta_origen) 
        if archivo.lower().endswith(extensiones_validas)
    ]

    total_disponible = len(todas_las_fotos)
    print(f"Se encontraron {total_disponible} fotos en el origen.")

    # 3. Validar la cantidad
    if total_disponible == 0:
        print("No hay fotos en la carpeta de origen.")
        return
    
    if cantidad > total_disponible:
        print(f"Ojo: Pediste {cantidad} fotos pero solo hay {total_disponible}. Procesando todas...")
        cantidad = total_disponible

    # 4. Selección TOTALMENTE random y sin repetir
    fotos_seleccionadas = random.sample(todas_las_fotos, cantidad)

    # 5. Abrir, convertir a PNG y guardar en la nueva carpeta
    for foto in fotos_seleccionadas:
        ruta_origen = os.path.join(carpeta_origen, foto)
        
        # Separar el nombre del archivo de su extensión original (.tiff)
        nombre_sin_ext, _ = os.path.splitext(foto)
        # Crear el nuevo nombre con extensión .png
        nuevo_nombre = nombre_sin_ext + ".png"
        ruta_destino = os.path.join(carpeta_destino, nuevo_nombre)
        
        try:
            # Abrir la imagen y guardarla como PNG
            with Image.open(ruta_origen) as img:
                img.save(ruta_destino, "PNG")
            print(f"Convertida a PNG y guardada -> {nuevo_nombre}")
        except Exception as e:
            print(f"Error al procesar la imagen {foto}: {e}")

    print(f"\n¡Proceso terminado! Se convirtieron y guardaron {cantidad} fotos aleatorias en '{carpeta_destino}'.")

# ==========================================
# CONFIGURA TUS VARIABLES AQUÍ
# ==========================================
RUTA_ORIGEN = r"PIV\Tomas\m72-toma-1-cam-1-n-0-car-02-piv"   # Tu ruta origen
RUTA_DESTINO = r"fotos_random"                               # Tu ruta destino
NUMERO_DE_FOTOS = 1000                                         # El número X de fotos

# Ejecutar el script
copiar_y_convertir_fotos_random(RUTA_ORIGEN, RUTA_DESTINO, NUMERO_DE_FOTOS)