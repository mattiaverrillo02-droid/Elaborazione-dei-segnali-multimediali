import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
from skimage.util import random_noise
import skimage.io as io

plt.close('all')
v0=u0=0.25
B=0.20

x=np.float64(io.imread('Immagini/fiori.jpg'))/255

plt.figure(1)
plt.imshow(x)
plt.title('immagine originale')
M,N,L=x.shape

m=np.fft.fftshift(np.fft.fftfreq(M))
n=np.fft.fftshift(np.fft.fftfreq(N))
l,k=np.meshgrid(n,m)

D =2*np.abs(np.sqrt(l**2 + k**2)-0.25)
H = D<B

plt.figure(2)
plt.imshow(H,clim=[0,1], cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure(3), auto_add_to_figure=True) # crea una figura per i grafici 3d
ax.plot_surface(l,k,H, linewidth=0, cmap='jet')

r=x[:,:,0]
g=x[:,:,1]
b=x[:,:,2]

R=np.fft.fftshift(np.fft.fft2(r))
G=np.fft.fftshift(np.fft.fft2(g))
B=np.fft.fftshift(np.fft.fft2(b))

Rn=R*H
Gn=G*H
Bn=B*H

rn=np.real(np.fft.ifft2(np.fft.ifftshift(Rn)))
gn=np.real(np.fft.ifft2(np.fft.ifftshift(Gn)))
bn=np.real(np.fft.ifft2(np.fft.ifftshift(Bn)))

y=np.stack((rn,gn,bn), 2)

plt.figure(4)
plt.imshow(y)
plt.title('Immagine filtrata')

plt.show()







