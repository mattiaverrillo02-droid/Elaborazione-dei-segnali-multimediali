import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')
x = np.float64(io.imread('Immagini/Lena.jpg'))
M,N=x.shape

d=20
#d è la deviazione standard della gaussiana
n = d*np.random.randn(M,N) #stiamo creando una gaussiana

noisy=x+n

plt.figure(1)
plt.subplot(2,2, 1)
plt.hist(x.flatten(), 256)

plt.subplot(2,2, 2)
plt.imshow(n, cmap='gray')

plt.subplot(2,2, 3)
plt.imshow(noisy, cmap='gray', clim=[0,255])

k=7
z=ndi.uniform_filter(noisy, (k,k))

plt.subplot(2,2, 4)
plt.imshow(z, cmap='gray', clim=[0,255])


plt.show()

mse= np.mean((x-z)**2)