
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise

plt.close('all')

x=np.float64(io.imread('Immagini/Lena.jpg'))

plt.figure(1)
plt.imshow(x, cmap='gray', clim=[0,255])
plt.show()

rumore=random_noise(x, mode='s&p')*255

plt.figure(2)
plt.imshow(rumore, cmap='gray', clim=[0,255])
plt.show()