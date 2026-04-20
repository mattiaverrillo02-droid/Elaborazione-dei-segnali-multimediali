import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist
from skimage.util import random_noise
plt.close('all')
x=np.float64(io.imread('Immagini/foto_originale.tif'))/255
plt.figure(1)
plt.imshow(x)

r=x[:,:,0]
g=x[:,:,1]
b=x[:,:,2]

R=np.fft.fftshift(np.fft.fft2(r))
G=np.fft.fftshift(np.fft.fft2(g))
B=np.fft.fftshift(np.fft.fft2(b))

M,N,L=x.shape

m= np.fft.fftshift(np.fft.fftfreq(M))
n= np.fft.fftshift(np.fft.fftfreq(N))
l,k= np.meshgrid(n,m) #l,k sono come gli assi cartesiani: x e y. quindi l sono le colonne perchè si sposta 
#sull'asse x quindi nell'argomento mettiamo n,m cioè colonne, righe

H= (np.abs(l)<=0.10) & (np.abs(k)<=0.25)

plt.figure(2)
plt.imshow(H, cmap='gray')


R=R*H
G=G*H
B=B*H

r=np.real(np.fft.ifft2(np.fft.fftshift(R)))
g=np.real(np.fft.ifft2(np.fft.fftshift(G)))
b=np.real(np.fft.ifft2(np.fft.fftshift(B)))

y=np.stack((r,g,b), 2)

plt.figure(3)
plt.imshow(y)
plt.show()



plt.show()