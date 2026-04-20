import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import equalize_hist 
from mylib import FSHS
plt.close('all')
x = np.float64(io.imread('Immagini/marte.jpg'))

plt.figure(1)
plt.subplot(1, 2,1)
plt.imshow(x, clim=[0,255], cmap='gray')

plt.subplot(1,2,2)
plt.hist(x.flatten(), 256)
plt.xlim([0,255])
plt.ylim([0,25000])
plt.show()

y=equalize_hist(x)
y=FSHS(y) #EQUALIZE restituisce tra 0 e 1 quindi conviene riportare tra 0 e 255
plt.figure(2)
plt.subplot(1, 2,1)
plt.imshow(y, clim=[0,255], cmap='gray')

plt.subplot(1,2,2)
plt.hist(y.flatten(), 256)
plt.xlim([0,255])
plt.ylim([0,25000])
plt.show()