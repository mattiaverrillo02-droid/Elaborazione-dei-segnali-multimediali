import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')


x = np.float64(io.imread('Immagini/test.jpg'))
k = 7; 
h = np.ones((k,k))/(k**2) #matrice di tutti 1 di dimensione k*k. Dividiamo poi per k^2
y = ndi.correlate(x, h, mode='constant') #mode è l'estensione ai bordi. 
plt.figure(1); 
plt.subplot(1,2,1)
plt.imshow(x,cmap='gray', clim=[0,255])
plt.subplot(1,2,2)
plt.imshow(y,clim=[0,255],cmap='gray')
plt.show()