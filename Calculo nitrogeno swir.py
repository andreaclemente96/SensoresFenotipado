# numpy para manipulación de arrays
# spectral para manejo de imágenes hiperespectrales ENVI
# tkinter para selección de archivos con ventana emergente
# matplotlib para graficar
# cv2 para redimensionar imágenes
import numpy as np
import spectral as sp
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt
import cv2


# Crear ventana oculta de tkinter y seleccionar archivos

Tk().withdraw()
sample_file = askopenfilename(title="Selecciona una imagen hdr SAMPLE")
dark_file   = askopenfilename(title="Selecciona una imagen hdr DARK")
white_file  = askopenfilename(title="Selecciona una imagen hdr WHITE")


# Abrir y cargar las imágenes ENVI
sample = sp.envi.open(sample_file).load()
dark   = sp.envi.open(dark_file).load()
white  = sp.envi.open(white_file).load()


# Redimensionar cubo de muestra 

# Definir número de filas objetivo
# Recorrer cada banda y redimensionar usando cv2
# Apilar bandas en un nuevo array con shape (bands, rows, cols)
target_rows = 500
bands = sample.shape[2]
resized_cube = []
for b in range(bands):
    band = sample[:, :, b]
    resized_band = cv2.resize(band, (band.shape[1], target_rows), interpolation=cv2.INTER_LINEAR)
    resized_cube.append(resized_band)
resized_cube = np.stack(resized_cube, axis=0)
rows, cols = resized_cube.shape[1], resized_cube.shape[2]

# Obtener longitudes de onda
wavelengths = sample.metadata.get('wavelength')
if wavelengths:
    wavelengths = np.array([float(w) for w in wavelengths])
else:
    wavelengths = np.arange(bands)


# Calcular media global de dark y white por banda
# Para normalización posterior
dark_mean  = np.mean(dark, axis=(0,1))
white_mean = np.mean(white, axis=(0,1))


# Mostrar banda de ejemplo para seleccionar pixel

banda_vista = 30 # Se usa la banda 30 como ejemplo
plt.figure()
plt.imshow(resized_cube[banda_vista], cmap='gray')
plt.title(f"Banda {banda_vista} (Sample)")
plt.axis('off')


# Función de clic para calcular reflectancia

def onclick(event):
    # Evitar clicks fuera de la imagen
    if event.xdata is None or event.ydata is None:
        return
    
    # Coordenadas del pixel seleccionado
    x_plot, y_plot = int(event.xdata), int(event.ydata)
    
    # Array para almacenar reflectancia
    ref = np.zeros(bands, dtype=np.float32)
    epsilon = 1e-6

 
    # Calcular reflectancia por banda
  
    white_reflectance = 0.99 # Valor de reflectancia conocida del panel blanco
    for b in range(bands):
        s = resized_cube[b, y_plot, x_plot]
        d = dark[y_plot, x_plot, b]
        w = white[y_plot, x_plot, b]
        denom = w - d
        ref[b] = np.clip((s - d) / (denom + epsilon) * white_reflectance * 100, 0, 100)

    
    # Graficar reflectancia
    
    plt.figure()
    plt.plot(wavelengths, ref, label='Reflectancia (%)')

    # Marcar longitudes clave
    for target_wl in [1510, 1680]:
        idx = (np.abs(wavelengths - target_wl)).argmin()
        plt.axvline(wavelengths[idx], color='r', linestyle='--', label=f'{wavelengths[idx]:.1f} nm')
        plt.scatter(wavelengths[idx], ref[idx], color='r')
        plt.text(wavelengths[idx], ref[idx]+1, f'{ref[idx]:.1f}%', color='r', fontsize=9)

  
    # Calcular NDNI
    
    R_1510 = ref[np.argmin(np.abs(wavelengths - 1510))] / 100
    R_1680 = ref[np.argmin(np.abs(wavelengths - 1680))] / 100
    R_1510 = max(R_1510, epsilon)
    R_1680 = max(R_1680, epsilon)
    NDNI = (np.log(1 / R_1510) - np.log(1 / R_1680)) / (np.log(1 / R_1510) + np.log(1 / R_1680))

   
    # Clasificación de NDNI

    if NDNI > 0.18:
        status = "Muy sana"
    elif NDNI > 0.12:
        status = "Moderadamente sana"
    elif NDNI > 0.05:
        status = "Posible estrés"
    else:
        status = "Estrés severo"

    
    
    info_text = f"Reflectancia en 1510nm: {R_1510*100:.2f}%\n"
    info_text += f"Reflectancia en 1680nm: {R_1680*100:.2f}%\n"
    info_text += f"Índice NDNI: {NDNI:.2f} → {status}"
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"Reflectancia pixel (x={x_plot}, y={y_plot})")
    plt.xlabel("Longitud de onda (nm)")
    plt.ylabel("Reflectancia (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    print(info_text)

# Evento clic para la figura
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()


