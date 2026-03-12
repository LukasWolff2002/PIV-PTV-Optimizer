"""
CÓDIGO 1: piv_functions.py
Biblioteca con TODAS las funciones de filtros y visualización
NO ejecuta nada, solo define funciones
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from PIL import Image
import cv2
import os


# ============================================================================
# FUNCIONES DE CARGA/GUARDADO
# ============================================================================

def load_image(filepath):
    """
    Cargar y normalizar imagen a rango 0-1
    
    Args:
        filepath: ruta al archivo de imagen
    
    Returns:
        numpy array (float64, rango 0-1)
    """
    img = Image.open(filepath)
    img_array = np.array(img)
    
    # Convertir a grayscale si es RGB
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Normalizar según tipo de dato
    if img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float64) / 255.0
    elif img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float64) / 65535.0
    else:
        img_array = img_array.astype(np.float64)
        if img_array.max() > 1.0:
            img_array = img_array / img_array.max()
    
    return img_array


def save_image(image, filepath, bit_depth=16):
    """
    Guardar imagen procesada
    
    Args:
        image: numpy array (float64, rango 0-1)
        filepath: ruta donde guardar
        bit_depth: 8 o 16 bits
    """
    if bit_depth == 8:
        img_save = (image * 255).astype(np.uint8)
    else:
        img_save = (image * 65535).astype(np.uint16)
    
    Image.fromarray(img_save).save(filepath)


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def apply_histogram_adjustment(img, min_intensity, max_intensity):
    """
    Ajuste de histograma
    
    Args:
        img: imagen (float64, 0-1)
        min_intensity: intensidad mínima (0-1)
        max_intensity: intensidad máxima (0-1)
    
    Returns:
        Imagen ajustada
    """
    if min_intensity < max_intensity and (min_intensity > 0 or max_intensity < 1):
        return np.clip((img - min_intensity) / (max_intensity - min_intensity), 0, 1)
    return img


def apply_intensity_capping(img, n_std):
    """
    Intensity Capping - Limitar puntos brillantes
    
    Args:
        img: imagen (float64, 0-1)
        n_std: número de desviaciones estándar
    
    Returns:
        Imagen con capping aplicado
    """
    upper_limit = np.median(img) + n_std * np.std(img)
    result = np.clip(img, 0, upper_limit)
    
    # Renormalizar
    if result.max() > 0:
        result = result / result.max()
    
    return result


def apply_clahe(img, tile_size, clip_limit):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization
    
    Args:
        img: imagen (float64, 0-1)
        tile_size: tamaño de tiles en píxeles
        clip_limit: límite de clip
    
    Returns:
        Imagen con CLAHE aplicado
    """
    # Convertir a uint8
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Calcular grid de tiles
    h, w = img.shape
    grid_h = max(2, int(h / tile_size))
    grid_w = max(2, int(w / tile_size))
    
    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_w, grid_h))
    result = clahe.apply(img_uint8).astype(np.float64) / 255.0
    
    return result


def apply_highpass(img, kernel_size):
    """
    Filtro Highpass - Eliminar componentes de baja frecuencia
    
    Args:
        img: imagen (float64, 0-1)
        kernel_size: tamaño del kernel gaussiano
    
    Returns:
        Imagen con highpass aplicado
    """
    # Asegurar tamaño impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Filtro gaussiano para baja frecuencia
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), kernel_size/3)
    
    # Restar baja frecuencia
    result = img - blurred
    
    # Normalizar
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    
    return result


def apply_wiener(img, wiener_size, gaussian_size):
    """
    Filtro Wiener + Gaussian - Reducción de ruido
    
    Args:
        img: imagen (float64, 0-1)
        wiener_size: tamaño ventana wiener
        gaussian_size: tamaño kernel gaussian
    
    Returns:
        Imagen filtrada
    """
    # Asegurar tamaños impares
    if wiener_size % 2 == 0:
        wiener_size += 1
    if gaussian_size % 2 == 0:
        gaussian_size += 1
    
    # Convertir a uint8 para Wiener
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Aplicar filtro Non-local Means (aproximación a Wiener)
    result = cv2.fastNlMeansDenoising(img_uint8, None, h=10,
                                     templateWindowSize=wiener_size,
                                     searchWindowSize=wiener_size*2+1)
    result = result.astype(np.float64) / 255.0
    
    # Aplicar Gaussian
    result = cv2.GaussianBlur(result, (gaussian_size, gaussian_size), gaussian_size/2)
    
    return result


def apply_roi(img, x, y, width, height):
    """
    Extraer Region of Interest (ROI)
    
    Args:
        img: imagen (float64, 0-1)
        x, y: coordenadas esquina superior izquierda
        width, height: dimensiones del ROI
    
    Returns:
        ROI extraído
    """
    h, w = img.shape
    
    # Validar límites
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    return img[y:y+height, x:x+width]


def apply_preprocessing(img, params):
    """
    Aplicar todos los filtros de preprocesamiento según parámetros
    
    Args:
        img: imagen original (float64, 0-1)
        params: diccionario con parámetros
    
    Returns:
        Imagen procesada (float64, 0-1)
    """
    result = img.copy()
    
    # 0. ROI (si está habilitado)
    if params.get('roi_enabled', False):
        x = int(params.get('roi_x', 0))
        y = int(params.get('roi_y', 0))
        width = int(params.get('roi_width', 100))
        height = int(params.get('roi_height', 100))
        roi = apply_roi(result, x, y, width, height)
    else:
        roi = result
        x, y = 0, 0
    
    # 1. Ajuste de histograma
    if params.get('min_intensity', 0.0) > 0 or params.get('max_intensity', 1.0) < 1.0:
        roi = apply_histogram_adjustment(
            roi, 
            params.get('min_intensity', 0.0),
            params.get('max_intensity', 1.0)
        )
    
    # 2. Intensity Capping
    if params.get('intensity_capping', False):
        roi = apply_intensity_capping(
            roi,
            params.get('capping_n_std', 2.0)
        )
    
    # 3. CLAHE
    if params.get('clahe_enabled', False):
        roi = apply_clahe(
            roi,
            int(params.get('clahe_tile_size', 50)),
            params.get('clahe_clip_limit', 0.01)
        )
    
    # 4. Highpass
    if params.get('highpass_enabled', False):
        roi = apply_highpass(
            roi,
            int(params.get('highpass_size', 15))
        )
    
    # 5. Wiener
    if params.get('wiener_enabled', False):
        roi = apply_wiener(
            roi,
            int(params.get('wiener_size', 3)),
            int(params.get('gaussian_size', 3))
        )
    
    # Reconstruir imagen completa si se usó ROI
    if params.get('roi_enabled', False):
        result[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        return result
    
    return roi


# ============================================================================
# INTERFAZ GRÁFICA
# ============================================================================

class ImageTuner:
    """Interfaz gráfica para ajustar parámetros"""
    
    def __init__(self, root, image_path, camera_id, initial_params):
        """
        Inicializar tuner
        
        Args:
            root: ventana raíz de tkinter
            image_path: ruta a la imagen
            camera_id: ID de cámara (cam1, cam2, etc)
            initial_params: diccionario con parámetros iniciales
        """
        self.root = root
        self.root.title(f"{camera_id.upper()} - {os.path.basename(image_path)}")
        self.root.geometry("1200x700")
        
        self.image_path = image_path
        self.camera_id = camera_id
        self.original_image = load_image(image_path)
        self.final_params = initial_params.copy()
        
        # Variables de control
        self.vars = {
            'roi_enabled': tk.BooleanVar(value=initial_params.get('roi_enabled', False)),
            'roi_x': tk.IntVar(value=initial_params.get('roi_x', 0)),
            'roi_y': tk.IntVar(value=initial_params.get('roi_y', 0)),
            'roi_width': tk.IntVar(value=initial_params.get('roi_width', 100)),
            'roi_height': tk.IntVar(value=initial_params.get('roi_height', 100)),
            'clahe_enabled': tk.BooleanVar(value=initial_params.get('clahe_enabled', False)),
            'clahe_tile_size': tk.IntVar(value=initial_params.get('clahe_tile_size', 50)),
            'clahe_clip_limit': tk.DoubleVar(value=initial_params.get('clahe_clip_limit', 0.01)),
            'intensity_capping': tk.BooleanVar(value=initial_params.get('intensity_capping', False)),
            'capping_n_std': tk.DoubleVar(value=initial_params.get('capping_n_std', 2.0)),
            'highpass_enabled': tk.BooleanVar(value=initial_params.get('highpass_enabled', False)),
            'highpass_size': tk.IntVar(value=initial_params.get('highpass_size', 15)),
            'wiener_enabled': tk.BooleanVar(value=initial_params.get('wiener_enabled', False)),
            'wiener_size': tk.IntVar(value=initial_params.get('wiener_size', 3)),
            'gaussian_size': tk.IntVar(value=initial_params.get('gaussian_size', 3)),
            'min_intensity': tk.DoubleVar(value=initial_params.get('min_intensity', 0.0)),
            'max_intensity': tk.DoubleVar(value=initial_params.get('max_intensity', 1.0)),
        }
        
        self.setup_ui()
        self.update_preview()
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        
        # Panel de controles (con scrollbar)
        ctrl_canvas = tk.Canvas(main, width=300)
        scrollbar = ttk.Scrollbar(main, orient="vertical", command=ctrl_canvas.yview)
        ctrl = ttk.Frame(ctrl_canvas, padding="5")
        
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)
        
        ctrl_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        ctrl_window = ctrl_canvas.create_window((0, 0), window=ctrl, anchor="nw")
        
        def configure_scroll(event):
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
            ctrl_canvas.itemconfig(ctrl_window, width=event.width)
        
        ctrl.bind("<Configure>", configure_scroll)
        
        row = 0
        
        # Título
        ttk.Label(ctrl, text=f"Cámara: {self.camera_id.upper()}", 
                 font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # ROI
        ttk.Label(ctrl, text="ROI (Region of Interest)", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        ttk.Checkbutton(ctrl, text="Habilitar ROI", variable=self.vars['roi_enabled'],
                       command=self.update_preview).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(ctrl, text="X:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=0, to=1000, variable=self.vars['roi_x'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Label(ctrl, text="Y:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=0, to=1000, variable=self.vars['roi_y'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Label(ctrl, text="Ancho:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=10, to=1000, variable=self.vars['roi_width'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Label(ctrl, text="Alto:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=10, to=1000, variable=self.vars['roi_height'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Separator(ctrl, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.EW, pady=10)
        row += 1
        
        # Ajuste de histograma
        ttk.Label(ctrl, text="Ajuste Histograma", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        ttk.Label(ctrl, text="Min:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.vars['min_intensity'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Label(ctrl, text="Max:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.vars['max_intensity'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Separator(ctrl, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.EW, pady=10)
        row += 1
        
        # Intensity Capping
        ttk.Label(ctrl, text="Intensity Capping", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        ttk.Checkbutton(ctrl, text="Habilitar", variable=self.vars['intensity_capping'],
                       command=self.update_preview).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(ctrl, text="N×Std:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=1.0, to=5.0, variable=self.vars['capping_n_std'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Separator(ctrl, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.EW, pady=10)
        row += 1
        
        # CLAHE
        ttk.Label(ctrl, text="CLAHE", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        ttk.Checkbutton(ctrl, text="Habilitar", variable=self.vars['clahe_enabled'],
                       command=self.update_preview).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(ctrl, text="Tile Size:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=10, to=200, variable=self.vars['clahe_tile_size'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Label(ctrl, text="Clip Limit:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=0.001, to=0.1, variable=self.vars['clahe_clip_limit'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Separator(ctrl, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.EW, pady=10)
        row += 1
        
        # Highpass
        ttk.Label(ctrl, text="Highpass", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        ttk.Checkbutton(ctrl, text="Habilitar", variable=self.vars['highpass_enabled'],
                       command=self.update_preview).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(ctrl, text="Size:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=3, to=50, variable=self.vars['highpass_size'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Separator(ctrl, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.EW, pady=10)
        row += 1
        
        # Wiener + Gaussian
        ttk.Label(ctrl, text="Wiener + Gaussian", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        ttk.Checkbutton(ctrl, text="Habilitar", variable=self.vars['wiener_enabled'],
                       command=self.update_preview).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        
        ttk.Label(ctrl, text="Wiener Size:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=3, to=15, variable=self.vars['wiener_size'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Label(ctrl, text="Gaussian Size:").grid(row=row, column=0, sticky=tk.W)
        ttk.Scale(ctrl, from_=3, to=15, variable=self.vars['gaussian_size'],
                 command=lambda x: self.update_preview()).grid(row=row, column=1, sticky=tk.EW)
        row += 1
        
        ttk.Separator(ctrl, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.EW, pady=10)
        row += 1
        
        # Botón cerrar
        ttk.Button(ctrl, text="Cerrar y Continuar", 
                  command=self.close).grid(row=row, column=0, columnspan=2, pady=20, sticky=tk.EW)
        
        # Panel de visualización
        visual = ttk.Frame(main, padding="5")
        visual.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        main.columnconfigure(2, weight=1)
        
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Imagen original
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Imagen procesada
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Histograma original
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Histograma procesado
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=visual)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_preview(self):
        """Actualizar preview de imágenes y histogramas"""
        params = {k: v.get() for k, v in self.vars.items()}
        processed = apply_preprocessing(self.original_image, params)
        
        # Limpiar todos los ejes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Imagen original
        self.ax1.imshow(self.original_image, cmap='gray', vmin=0, vmax=1)
        self.ax1.set_title('Imagen Original')
        self.ax1.axis('off')
        
        # Imagen procesada
        self.ax2.imshow(processed, cmap='gray', vmin=0, vmax=1)
        self.ax2.set_title('Imagen Preprocesada')
        self.ax2.axis('off')
        
        # Histograma original
        self.ax3.hist(self.original_image.ravel(), bins=256, range=(0, 1), 
                     color='blue', alpha=0.7)
        self.ax3.set_title('Histograma Original')
        self.ax3.set_xlim(0, 1)
        
        # Histograma procesado
        self.ax4.hist(processed.ravel(), bins=256, range=(0, 1), 
                     color='green', alpha=0.7)
        self.ax4.set_title('Histograma Preprocesado')
        self.ax4.set_xlim(0, 1)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def close(self):
        """Cerrar ventana y guardar parámetros finales"""
        self.final_params = {k: v.get() for k, v in self.vars.items()}
        self.root.destroy()
    
    def get_params(self):
        """Obtener parámetros finales"""
        return self.final_params


# ============================================================================
# UTILIDADES
# ============================================================================

def detect_camera(filename):
    """
    Detectar cámara del nombre del archivo
    
    Args:
        filename: nombre del archivo
    
    Returns:
        'cam1', 'cam2', 'cam3', 'cam4', o None
    """
    filename_lower = filename.lower()
    for cam in ['cam1', 'cam2', 'cam3', 'cam4']:
        if cam in filename_lower:
            return cam
    return None