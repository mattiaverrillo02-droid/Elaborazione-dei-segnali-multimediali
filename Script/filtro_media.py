import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')
x = np.float64(io.imread('Immagini/space.jpg'))
k = 15;
h = np.ones((k,k))/(k**2) #matrice di tutti 1 e poi diviso k^2 (mask per la media)
y = ndi.correlate(x, h, mode='reflect')
plt.figure(1); 
plt.imshow(y,clim=[0,255],cmap='gray');

plt.figure(2); 
plt.imshow(x,clim=[0,255],cmap='gray');

xmax= np.max(y)
soglia= 25*xmax/100
maschera= y>soglia

plt.figure(3); 
plt.imshow(maschera,cmap='gray');

z= x*maschera #metto a 0 quelli che la maschera ha messo a 0,  rimango inalterati quelli che la maschera h amesso a 1
plt.figure(4); 
plt.imshow(z,cmap='gray');