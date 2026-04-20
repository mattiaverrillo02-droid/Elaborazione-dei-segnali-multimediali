import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise


plt.close('all')

x = np.float64(io.imread('Immagini/anelli.tif'))
X = np.fft.fftshift(np.fft.fft2(x) )
modulo_X = np.abs(X)
fase_X=np.angle(X)


plt.figure(3)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.figure(1)
plt.imshow(np.log(1+modulo_X), clim=None, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

plt.figure(2)
plt.imshow(fase_X, clim=[-np.pi, np.pi], cmap='gray')


m = np.fft.fftshift(np.fft.fftfreq(np.log(1+modulo_X).shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(np.log(1+modulo_X).shape[1]))
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure()); # crea una figura per i grafici 3d
l,k = np.meshgrid(n,m)
ax.plot_surface(l,k,np.log(1+modulo_X), linewidth=0, cmap='jet')

plt.show()
