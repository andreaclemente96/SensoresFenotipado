import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Selección de la imagen
Tk().withdraw()
img_path = askopenfilename(title="Selecciona la imagen LWIR BMP",
                           filetypes=[("Archivos BMP", "*.bmp"), ("Todos los archivos", "*.*")])

with Image.open(img_path) as img:
    img_array = np.array(img)

# Normalizar los valores a 0-1
img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min())

# Mostrar imagen en escala de grises y colorbar también en gris
plt.figure()
plt.imshow(img_norm, cmap='gray')  # escala de grises
plt.colorbar(label="Intensidad relativa")
plt.title("Imagen LWIR (normalizada)")

plt.axis('off')
plt.show()
