import numpy as np
import spectral as sp
from spectral import open_image
from tkinter import Tk
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from tqdm import tqdm

# Carga imagen
Tk().withdraw()

sample = sp.envi.open(askopenfilename(title="Selecciona una imagen hdr SAMPLE")).load()
dark = sp.envi.open(askopenfilename(title="Selecciona una imagen hdr DARK")).load()
white = sp.envi.open(askopenfilename(title="Selecciona una imagen hdr WHITE")).load()

# Redimensionar cubo de datos
resized_cube = []
for i in range(sample.shape[2]):
    band = sample.read_band(i).T
    resized_band = cv2.resize(band, (band.shape[0], 500), interpolation=cv2.INTER_LINEAR)
    resized_cube.append(resized_band)
resized_cube = np.array(resized_cube)

# Obtener dimensiones 
sample_bands, cols, rows = resized_cube.shape

# Obtener longitudes de onda
wavelengths = sample.metadata.get('wavelength')
if wavelengths:
    wavelengths = np.array([float(w) for w in wavelengths])
else:
    wavelengths = np.arange(sample_bands)

# Calcular media de blanco y negro
dark_mean = np.mean(dark, axis=0)
white_mean = np.mean(white, axis=0)

# Mostrar imagen de una banda para seleccionar píxeles
banda_vista = 30
img = resized_cube[banda_vista]
plt.figure()
plt.imshow(img, cmap='gray')
plt.title(f"Banda {banda_vista} (Sample)")
plt.axis('off')

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x_plot, y_plot = int(event.xdata), int(event.ydata)
        ref = np.zeros(sample_bands, dtype=np.float32)
        epsilon = 1e-6
        for b in range(sample_bands):
            s = resized_cube[b, y_plot, x_plot]
            d = dark_mean[x_plot, b]
            w = white_mean[x_plot, b]
            denominator = w - d
            if abs(denominator) > epsilon:
                value = ((s - d) / denominator) * 100
                ref[b] = np.clip(value, 0, 100)
            else:
                ref[b] = 0

        plt.figure()
        plt.plot(wavelengths, ref, label='Reflectancia')

        # Marcar longitudes de onda específicas
        for target_wl in [1510, 1680]:
            idx = (np.abs(wavelengths - target_wl)).argmin()
            plt.axvline(wavelengths[idx], color='r', linestyle='--', label=f'{wavelengths[idx]:.1f} nm')
            plt.scatter(wavelengths[idx], ref[idx], color='r')
            plt.text(wavelengths[idx], ref[idx] + 1, f'{ref[idx]:.1f}%', color='r', fontsize=9)

        # Calcular NDNI
        R_1510 = ref[np.argmin(np.abs(wavelengths - 1510))]
        R_1680 = ref[np.argmin(np.abs(wavelengths - 1680))]
        if R_1510 > 0 and R_1680 > 0:
            #NDNI = (np.log10(1 / R_1510) - np.log10(1 / R_1680)) / (np.log10(1 / R_1510) + np.log10(1 / R_1680))
            NDNI = (np.log(R_1680) - np.log(R_1510)) / (np.log(R_1680) + np.log(R_1510))
        else:
            NDNI = 0  # evitar log(0)

        # Clasificación según NDNI
        if NDNI > 0.08:
            status = "Muy sana"
        elif NDNI > 0.05:
            status = "Moderadamente sana"
        elif NDNI > 0.02:
            status = "Posible estrés"
        else:
            status = "Estrés severo"

        # Mostrar info en el gráfico
        info_text = f"Reflectancia en 1510nm: {R_1510:.2f}\n"
        info_text += f"Reflectancia en 1680nm: {R_1680:.2f}\n"
        info_text += f"Índice NDNI: {NDNI:.2f} → {status}"
        plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Mostrar en consola también
        print(info_text)

        plt.title(f"Reflectancia en pixel (x={x_plot}, y={y_plot})")
        plt.xlabel("Longitud de onda (nm)")
        plt.ylabel("Reflectancia (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
