import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import equalize_hist 
from scipy.ndimage import correlate


plt.close('all')


k=5
#h=np.ones((k,k))/k**2 #filtro media -> passa basso

x=np.float64(io.imread('Immagini/test.jpg'))
h=np.array([[1,0,-1], [2,0,-2],[1,0,-1]])
M,N=x.shape

X=(np.fft.fft2(x,(M,N)))
H=(np.fft.fft2(h,(M,N)))

#non serve fare fftshift perchè non la voglio visualizzare, 
#sto facendo un'analisi

Y=H*X
y=np.real(np.fft.ifft2(Y))


plt.figure(1)
plt.imshow(y, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

y1=correlate(x,h,mode='reflect')
plt.figure(2)
plt.imshow(y1, cmap='gray', clim=None)
