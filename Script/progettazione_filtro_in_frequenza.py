import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import equalize_hist 
from scipy.ndimage import correlate


plt.close('all')

x=np.float64(io.imread('Immagini/lena.jpg'))
X=np.fft.fft2(x)
X=np.fft.fftshift(X)

plt.figure(1)
plt.imshow(np.log(1+np.abs(X)), cmap='gray')

plt.figure(2)
plt.imshow(x, cmap='gray', clim=[0,255])

#filtro che fa passare le basse frequenze

M,N=x.shape
m=np.fft.fftshift(np.fft.fftfreq(M)) #frequenze verticali
n=np.fft.fftshift(np.fft.fftfreq(M)) #frequenze orizzontali
l,k=np.meshgrid(n,m)

D=np.sqrt(l**2+k**2) #man mano che ci allontaniamo da (0,0) diventa sempre più grande
H=D<0.1 #avremo un cerchio che è T per tutte le coordinate che hanno D <0.1

plt.figure(3)
plt.imshow(H, cmap='gray', clim=[0,1], extent=(-0.5,+0.5,+0.5,-0.5))

Y=H*X
plt.figure(4)
plt.imshow(np.log(1+np.abs(Y)), cmap='gray')
y=np.real(np.fft.ifft2(np.fft.ifftshift(Y)))

plt.figure(5)
plt.imshow(y, cmap='gray')
plt.show()