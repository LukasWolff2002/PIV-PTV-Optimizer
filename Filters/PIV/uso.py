"""
CÓDIGO 2: main.py
Script principal que usa las funciones de piv_functions.py
"""

import tkinter as tk
import glob
import os
from funciones import (
    ImageTuner, 
    detect_camera, 
    load_image, 
    save_image, 
    apply_preprocessing
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Carpeta con las imágenes
CARPETA_IMAGENES = "FOTOS/"

# Parámetros iniciales por cámara (todos deshabilitados)
CAMERA_PARAMS = {
    'cam1': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 200,
        'clahe_clip_limit': 0.0010,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0395,
        'max_intensity': 1.0000,
    },
    'cam2': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 17,
        'clahe_clip_limit': 0.0492,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 14,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0000,
        'max_intensity': 1.0000,
    },
    'cam3': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 10,
        'clahe_clip_limit': 0.1000,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0000,
        'max_intensity': 1.0000,
    },
    'cam4': {
        'roi_enabled': False,
        'roi_x': 0,
        'roi_y': 0,
        'roi_width': 100,
        'roi_height': 100,
        'clahe_enabled': True,
        'clahe_tile_size': 10,
        'clahe_clip_limit': 0.0100,
        'intensity_capping': True,
        'capping_n_std': 5.0000,
        'highpass_enabled': False,
        'highpass_size': 15,
        'wiener_enabled': False,
        'wiener_size': 3,
        'gaussian_size': 3,
        'min_intensity': 0.0000,
        'max_intensity': 0.7237,
    },
}

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    
    print("\n" + "="*70)
    print("PIV PARAMETER TUNER")
    print("="*70)
    print(f"\nCarpeta: {CARPETA_IMAGENES}")
    
    # Buscar imágenes
    images = glob.glob(os.path.join(CARPETA_IMAGENES, '*.tif')) + \
             glob.glob(os.path.join(CARPETA_IMAGENES, '*.tiff'))
    
    if not images:
        print("ERROR: No se encontraron imágenes")
        return
    
    images.sort()
    print(f"Encontradas: {len(images)} imágenes\n")
    print("="*70 + "\n")
    
    # Procesar cada imagen
    results = {}
    
    for i, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        camera = detect_camera(filename)
        
        if camera is None:
            print(f"[{i}/{len(images)}] {filename} - ⚠ Sin cámara, saltando...")
            continue
        
        print(f"[{i}/{len(images)}] {filename} - {camera.upper()}")
        
        # Obtener parámetros iniciales
        initial_params = CAMERA_PARAMS[camera].copy()
        
        # Abrir tuner (llama a la clase de piv_functions.py)
        root = tk.Tk()
        tuner = ImageTuner(root, img_path, camera, initial_params)
        root.mainloop()
        
        # Guardar parámetros finales
        results[camera] = tuner.get_params()
        print(f"  ✓ Ajustado\n")
    
    # Mostrar resultados finales
    print("\n" + "="*70)
    print("PARÁMETROS FINALES - COPIAR Y PEGAR")
    print("="*70 + "\n")
    
    print("CAMERA_PARAMS = {")
    for cam in ['cam1', 'cam2', 'cam3', 'cam4']:
        if cam in results:
            params = results[cam]
        else:
            params = CAMERA_PARAMS[cam]
        
        print(f"    '{cam}': {{")
        for key, value in params.items():
            if isinstance(value, bool):
                print(f"        '{key}': {value},")
            elif isinstance(value, int):
                print(f"        '{key}': {value},")
            elif isinstance(value, float):
                print(f"        '{key}': {value:.4f},")
        print(f"    }},")
    print("}\n")
    
    print("="*70)
    print("✓ COMPLETADO")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()