import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Elegir imagen
Tk().withdraw()
img_path = askopenfilename(title="Selecciona imagen LWIR (8bit)",
                           filetypes=[("BMP/PNG", "*.bmp;*.png"), ("Todos", "*.*")])

# Cargar a array
img = np.array(Image.open(img_path))

# Usar un canal si es RGB
if img.ndim == 3:
    img = img[:, :, 0]

# Mostrar con paleta térmica
plt.figure(figsize=(8,6))
im = plt.imshow(img, cmap='inferno')  
plt.axis('off')
plt.title("Imagen térmica LWIR")

# Añadir colorbar personalizada
cbar = plt.colorbar(im)
cbar.set_ticks([0, 255])
cbar.set_ticklabels(['0 = Frío', '255 = Caliente'])

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

