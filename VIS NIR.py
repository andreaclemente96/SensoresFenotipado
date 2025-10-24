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
        if self.shift_is_held and not event.dblclick and event.button == 1 and (event.xdata is not None) and (event.ydata is not None):
            self.shift_is_held = False

            # Coordenadas del píxel seleccionado
            point = [round(event.xdata), round(event.ydata)]
            coord_x = point[0]  # Coordenada X (Columna)
            coord_y = point[1]  # Coordenada Y (Fila)

            spectrum = img_reflectance[coord_y, coord_x, :]  # Reflectancia en %

            # Suavizar la curva espectral
            # Suavizado del espectro usando filtro Savitzky–Golay
            # Basado en: Luo, J., Ying, K., & Bai, J. (2005). 
            # "Savitzky–Golay smoothing and differentiation filter for even number data". Signal Processing, 85(7), 1429–1434.
            spectrum_soft = savgol_filter(spectrum, window_length=9, polyorder=2)

            # Interpolación spline cúbica: Mészáros, Sz., & Allende Prieto, C. (2013). 
            # On the interpolation of model atmospheres and high-resolution synthetic stellar spectra. MNRAS, 430(4), 3285–3292.
            # Usado para suavizar la curva espectral
            wavelen_smooth = np.linspace(wavelen.min(), wavelen.max(), 500)
            spline = make_interp_spline(wavelen, spectrum_soft, k=3)
            spectrum_smooth = spline(wavelen_smooth)


            # Curva espectral
            plt.figure()
            plt.plot(wavelen_smooth, spectrum_smooth)

            # Líneas discontinuas rojas para las bandas NDVI
            plt.axvline(x=670, linestyle='--', color= "red", label='Red (670 nm)')
            plt.axvline(x=798, linestyle='--', color= "red", label='NIR (798 nm)')

            plt.xticks(rotation='vertical')
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("Reflectance [%]")
            plt.grid(True)
            plt.title("Espectro del Píxel Seleccionado")

            # Cálculo de NDVI
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

            # --- MODIFICACIÓN 2: INCLUIR COORDENADAS EN EL TEXTO ---
            info_text = f"Píxel (Columna X, Fila Y): ({coord_x}, {coord_y})\n"
            info_text += f"Reflectancia roja (670 nm): {red_val:.2f}%\n"
            info_text += f"Reflectancia NIR (798 nm): {nir_val:.2f}%\n"
            info_text += f"Índice NDVI: {NDVI:.2f} → {status}"

            # Mostrar texto en la gráfica
            plt.text(
                0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7)
            )

            plt.legend()
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)

            # También por consola
            print("\n" + "=" * 30)
            print(f"Píxel seleccionado: (X={coord_x}, Y={coord_y})")
            print(info_text)
            print("=" * 30 + "\n")

# Guardar imagen en reflectancia real
img_reflectance = img / 10000.0 * 100  # [0–100 %]

# Crear imagen escalada para visualización con mejor contraste
img_display = np.zeros_like(img, dtype=np.uint8)

# Rango de bandas a aclarar y aumentar sutilmente el contraste
start_band = 125
end_band = 140

for i in range(img.shape[2]):
    band = img[:, :, i] / 10000.0  # Normalizar 0-1

    # Ajuste específico para bandas del rango definido
    if start_band <= i <= end_band:
        band = np.clip((band * 1.2) - 0.05, 0, 1)
    else:
        p1, p99 = np.percentile(band, [1, 99])
        band = np.clip((band - p1) / (p99 - p1), 0, 1)

    img_display[:, :, i] = (band * 255).astype(np.uint8)

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
