import os
import shutil

def filtrar_resultados(origen, destino):
    # Definimos los archivos que estamos buscando
    archivos_objetivo = ['tracks.json', 'schedule.csv']
    
    # Recorremos la carpeta origen de forma recursiva
    for root, dirs, files in os.walk(origen):
        # Determinamos la ruta relativa para replicarla en el destino
        rel_path = os.path.relpath(root, origen)
        target_dir = os.path.join(destino, rel_path)
        
        # Bandera para saber si necesitamos crear la carpeta en el destino
        carpeta_creada = False
        
        for file in files:
            if file in archivos_objetivo:
                # Si encontramos uno de los archivos, aseguramos que la carpeta destino exista
                if not carpeta_creada:
                    os.makedirs(target_dir, exist_ok=True)
                    carpeta_creada = True
                
                # Definimos rutas completas
                ruta_completa_origen = os.path.join(root, file)
                ruta_completa_destino = os.path.join(target_dir, file)
                
                # Copiamos el archivo
                shutil.copy2(ruta_completa_origen, ruta_completa_destino)
                print(f"Copiado: {os.path.join(rel_path, file)}")

# --- CONFIGURACIÓN ---
ruta_origen = 'ResultadosPTV'
ruta_destino = 'ResultadosPTVFiltrado'

if __name__ == "__main__":
    if os.path.exists(ruta_origen):
        print(f"Iniciando filtrado de {ruta_origen}...")
        filtrar_resultados(ruta_origen, ruta_destino)
        print("\n¡Proceso finalizado con éxito!")
    else:
        print(f"Error: No se encontró la carpeta '{ruta_origen}'")