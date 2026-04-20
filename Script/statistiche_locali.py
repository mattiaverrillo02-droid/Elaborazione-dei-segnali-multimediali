#enhancement su una parte nascosta mediante statistiche locali
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')
x= io.imread('immagini/filamento.jpg')
plt.figure(3)
plt.imshow(x, cmap=('gray'))
plt.show()

k0 = 0.4 #costante per la media
k1 = 0.02 #costante inferiore per la varianza
k2 = 0.4  #costante superiore per la varianza
E = 4 #fattore di scala per la zona da modificare

med= np.mean(x)
dev= np.std(x)
MED= ndi.generic_filter(x, np.mean, (3,3))
DEV= ndi.generic_filter(x, np.std, (3,3))

#mask è la matrice binaria che evidenza la parte da elaborare
mask = (MED<=0.4*med) & (DEV<=0.4*dev) & (DEV>=0.02*dev)
plt.figure(1)
plt.imshow(mask, cmap=('gray'))
plt.show()

z=4*x
y= z*mask + (1-mask)*x #modifico i valori che la maschera ha messo a 1, e rimango inalterato le parti a 0 con 1-mask
plt.figure(2)
plt.imshow(y, cmap=('gray'),clim=  [0,255])
plt.show()

