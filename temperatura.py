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

# Bibliografía utilizada para justificar el código:
# 
# Harris, C. R., Millman, K. J., van der Walt, S. J., et al. Array programming with NumPy. Nature. https://numpy.org/
# (Uso de NumPy para manipulación de arrays y normalización de datos)

# Hunter, J. D. Matplotlib: A 2D graphics environment. Computing in Science & Engineering. https://matplotlib.org/
# (Uso de Matplotlib para visualización de la imagen y colorbar)

# Clark, A., & Contributors. Pillow (PIL fork). https://hugovk.github.io/pillow/
# (Uso de Pillow para abrir imágenes BMP y convertirlas a arrays)

# Van Rossum, G. Tkinter — Python interface to Tcl/Tk. https://docs.python.org/3/library/tkinter.html
# (Uso de Tkinter para crear la ventana emergente de selección de archivo)

