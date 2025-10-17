import spectral as sp
from matplotlib import pyplot as plt
import cv2
import numpy as np

# Longitudes de onda asociadas
wavelen = np.array([350, 354, 358, 362, 366, 370, 374, 378, 382, 386, 390, 394, 398, 402, 406, 410, 414, 418, 422, 426,
                    430, 434, 438, 442, 446, 450, 454, 458, 462, 466, 470, 474, 478, 482, 486, 490, 494, 498, 502, 506,
                    510, 514, 518, 522, 526, 530, 534, 538, 542, 546, 550, 554, 558, 562, 566, 570, 574, 578, 582, 586,
                    590, 594, 598, 602, 606, 610, 614, 618, 622, 626, 630, 634, 638, 642, 646, 650, 654, 658, 662, 666,
                    670, 674, 678, 682, 686, 690, 694, 698, 702, 706, 710, 714, 718, 722, 726, 730, 734, 738, 742, 746,
                    750, 754, 758, 762, 766, 770, 774, 778, 782, 786, 790, 794, 798, 802, 806, 810, 814, 818, 822, 826,
                    830, 834, 838, 842, 846, 850, 854, 858, 862, 866, 870, 874, 878, 882, 886, 890, 894, 898, 902, 906,
                    910, 914, 918, 922, 926, 930, 934, 938, 942, 946, 950, 954, 958, 962, 966, 970, 974, 978, 982, 986,
                    990, 994, 998, 1002])

# Cargar imagen hiperespectral
hdr = sp.envi.open("C:/Users/andea/Documents/hyspex_image_inspector/resources/vignatestigo1_Baldur_S-384N_SN12022_2832us_2025-06-24T104209_raw.hdr")
wvl = hdr.bands.centers
rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
meta = hdr.metadata

class EventController():
    def __init__(self, ax, img_shape, resize_shape):
        self.shift_is_held = False
        self.ax = ax
        self.img_shape = img_shape  # (rows, cols)
        self.resize_shape = resize_shape  # (height, width)

    def on_key_press(self, event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def onclick(self, event):
        if self.shift_is_held and not event.dblclick and event.button == 1 and (event.xdata is not None) and (event.ydata is not None):
            self.shift_is_held = False
            # Mapear coordenadas del plot a la imagen original
            x_plot, y_plot = int(event.xdata), int(event.ydata)
            orig_x = int(x_plot * self.img_shape[1] / self.resize_shape[1])
            orig_y = int(y_plot * self.img_shape[0] / self.resize_shape[0])
            orig_x = np.clip(orig_x, 0, self.img_shape[1] - 1)
            orig_y = np.clip(orig_y, 0, self.img_shape[0] - 1)
            spectrum = hdr[orig_y, orig_x, :]  # Reflectancia en %

            plt.figure()
            plt.plot(wavelen, spectrum)
            plt.xticks(rotation='vertical')
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("Reflectance")
            plt.grid(True)

            red_val = spectrum[np.argmin(np.abs(wavelen - 670))]
            nir_val = spectrum[np.argmin(np.abs(wavelen - 798))]
            NDVI = (nir_val - red_val) / (nir_val + red_val) if (nir_val + red_val) != 0 else 0

            if NDVI > 0.8:
                status = "Muy sana"
            elif NDVI > 0.4:
                status = "Moderadamente sana"
            elif NDVI > 0.2:
                status = "Posible estrés"
            else:
                status = "Estrés severo"

            info_text = f"Reflectancia roja (670 nm): {red_val:.2f}\n"
            info_text += f"Reflectancia NIR (798 nm): {nir_val:.2f}\n"
            info_text += f"Índice NDVI: {NDVI:.2f} → {status}"
            plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes,
                     fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            print(info_text)
            plt.tight_layout()
            plt.show()

# Mostrar imagen y activar controles
fig, ax = plt.subplots()
band_index = 150  # Cambia el índice si quieres otra banda

# Normalizar para visualizar mejor (0-255)
band = hdr.read_band(band_index)
band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Redimensionar manteniendo proporción (ancho fijo a 500 px)
fixed_width = 500
aspect_ratio = band_norm.shape[0] / band_norm.shape[1]
new_height = int(fixed_width * aspect_ratio)
resize_shape = (fixed_width, new_height)
resized_band = cv2.resize(band_norm, resize_shape, interpolation=cv2.INTER_LINEAR)

im = ax.imshow(resized_band, cmap='gray')
ax.set_title(f"Banda {band_index} ({wavelen[band_index]} nm)")
ax.axis('off')

controller = EventController(ax, band_norm.shape, resized_band.shape)
fig.canvas.mpl_connect('button_press_event', controller.onclick)
fig.canvas.mpl_connect('key_press_event', controller.on_key_press)
fig.canvas.mpl_connect('key_release_event', controller.on_key_release)

plt.show()