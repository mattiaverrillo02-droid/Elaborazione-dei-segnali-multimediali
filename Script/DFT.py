import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise


plt.close('all')

x = np.float64(io.imread('Immagini/volto.tif'))
X = np.fft.fft2(x)
X = np.log(1+np.abs((np.fft.fftshift(X))))


plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(x, cmap='gray', clim=[0,255], )

plt.subplot(1,2,2)
plt.imshow(X, clim=None, cmap='gray',  extent=(-0.5,+0.5,+0.5,-0.5)); #abs perchè stiamo vedendo il modulo
#extent serve solo per vedere gli assi da -1/2 e 1/2 cosi da dare un significato frequentistico. 


m = np.fft.fftshift(np.fft.fftfreq(X.shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(X.shape[1]))
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure()); # crea una figura per i grafici 3d
l,k = np.meshgrid(n,m)
ax.plot_surface(l,k,X, linewidth=0, cmap='jet')

plt.show()