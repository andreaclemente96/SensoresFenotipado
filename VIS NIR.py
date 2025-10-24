import os  # Para manejar rutas y operaciones del sistema de archivos
import tifffile  # Para leer y mostrar imágenes TIFF, especialmente hiperespectrales
import numpy as np  # Para manejo de arrays, cálculos numéricos y operaciones matriciales
from matplotlib import pyplot as plt  # Para crear gráficas, mostrar espectros y visualización general
import tkinter as tk  # Para crear interfaces gráficas simples (ventanas)
from tkinter import filedialog  # Para abrir ventanas de selección de archivos
from scipy.interpolate import make_interp_spline  # Para suavizar curvas mediante interpolación spline
from scipy.signal import savgol_filter  # Para suavizado de señales (Savitzky-Golay)

# Longitudes de onda asociadas
wavelen = np.array([
    350, 354, 358, 362, 366, 370, 374, 378, 382, 386, 390, 394, 398, 402, 406, 410,
    414, 418, 422, 426, 430, 434, 438, 442, 446, 450, 454, 458, 462, 466, 470, 474,
    478, 482, 486, 490, 494, 498, 502, 506, 510, 514, 518, 522, 526, 530, 534, 538,
    542, 546, 550, 554, 558, 562, 566, 570, 574, 578, 582, 586, 590, 594, 598, 602,
    606, 610, 614, 618, 622, 626, 630, 634, 638, 642, 646, 650, 654, 658, 662, 666,
    670, 674, 678, 682, 686, 690, 694, 698, 702, 706, 710, 714, 718, 722, 726, 730,
    734, 738, 742, 746, 750, 754, 758, 762, 766, 770, 774, 778, 782, 786, 790, 794,
    798, 802, 806, 810, 814, 818, 822, 826, 830, 834, 838, 842, 846, 850, 854, 858,
    862, 866, 870, 874, 878, 882, 886, 890, 894, 898, 902, 906, 910, 914, 918, 922,
    926, 930, 934, 938, 942, 946, 950, 954, 958, 962, 966, 970, 974, 978, 982, 986,
    990, 994, 998, 1002
])

# Selección de imagen con ventana emergente
root = tk.Tk()
root.withdraw()  # Oculta la ventana principal

img_path = filedialog.askopenfilename(
    title="Selecciona la imagen hiperespectral",
    filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
)

if not img_path:
    print("No se seleccionó ninguna imagen.")
    exit()

# Cargar imagen
img = tifffile.imread(img_path).astype(float)
img = img[:, :, 12:]  # Eliminar las 12 primeras bandas: Ultravioleta
wavelen = wavelen[12:]

# Guardar imagen en reflectancia real
img_reflectance = img / 10000.0 * 100  # [0–100 %]

# Crear imagen escalada para visualización con mejor contraste
img_display = np.zeros_like(img, dtype=np.uint8)
start_band = 125
end_band = 140

for i in range(img.shape[2]):
    band = img[:, :, i] / 10000.0  # Normalizar 0-1
    if start_band <= i <= end_band:
        band = np.clip((band * 1.2) - 0.05, 0, 1)
    else:
        p1, p99 = np.percentile(band, [1, 99])
        band = np.clip((band - p1) / (p99 - p1), 0, 1)
    img_display[:, :, i] = (band * 255).astype(np.uint8)

# Clase de control de eventos
class EventController():
    def __init__(self):
        self.shift_is_held = False

    def on_key_press(self, event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if not self.shift_is_held:
            return

        x_plot, y_plot = int(event.xdata), int(event.ydata)
        spectrum = img_reflectance[y_plot, x_plot, :].copy()

        # Suavizado Savitzky-Golay
        spectrum_smooth = savgol_filter(spectrum, window_length=9, polyorder=2)

        # Interpolación spline cúbica
        wavelen_smooth = np.linspace(wavelen.min(), wavelen.max(), 500)
        spline = make_interp_spline(wavelen, spectrum_smooth, k=3)
        spectrum_smooth = spline(wavelen_smooth)

        plt.figure(figsize=(8,5))
        plt.plot(wavelen_smooth, spectrum_smooth, label="Reflectancia suavizada")

        # Líneas para bandas de interés
        for band in [670, 798]:
            idx = np.argmin(np.abs(wavelen - band))
            plt.axvline(wavelen[idx], color="red", linestyle="--", label=f"{band} nm")
            plt.scatter(wavelen[idx], spectrum[idx], color="red")
            plt.text(wavelen[idx], spectrum[idx]+1, f"{spectrum[idx]:.2f}%", color='red', fontsize=9)

        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Reflectance [%]")
        plt.title(f"Píxel seleccionado (X={x_plot}, Y={y_plot})")
        plt.grid(True)

        # Cálculo NDVI
        red_val = spectrum[np.argmin(np.abs(wavelen - 670))]
        nir_val = spectrum[np.argmin(np.abs(wavelen - 798))]
        NDVI = (nir_val - red_val) / (nir_val + red_val)

        # Clasificación por salud
        if NDVI > 0.8:
            status = "Muy sana"
        elif NDVI > 0.4:
            status = "Moderadamente sana"
        elif NDVI > 0.2:
            status = "Posible estrés"
        else:
            status = "Estrés severo"

        # Texto en gráfica
        info_text = (
            f"Píxel (X={x_plot}, Y={y_plot})\n"
            f"Reflectancia roja (670 nm): {red_val:.2f}%\n"
            f"Reflectancia NIR (798 nm): {nir_val:.2f}%\n"
            f"NDVI: {NDVI:.2f} → {status}"
        )
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7))

        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\n" + "="*30)
        print(info_text)
        print("="*30 + "\n")

# Mostrar imagen y activar controles
fig, ax = plt.subplots()
controller = EventController()
tifffile.imshow(np.rollaxis(img_display, 2, 0), figure=fig)
fig.canvas.mpl_connect('button_press_event', controller.onclick)
fig.canvas.mpl_connect('key_press_event', controller.on_key_press)
fig.canvas.mpl_connect('key_release_event', controller.on_key_release)

plt.show()


# Bibliografía general:

# Harris, C. R., Millman, K. J., van der Walt, S. J., et al. Array programming with NumPy. https://numpy.org/
# (Uso de NumPy para manipulación de arrays y normalización de datos)

# Hunter, J. D. Matplotlib: A 2D graphics environment. https://matplotlib.org/
# (Uso de Matplotlib para visualización de imágenes y colorbar)

# van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., et al. scikit-image: Image processing in Python. https://scikit-image.org/
# (Uso de scikit-image para procesamiento de imágenes)

