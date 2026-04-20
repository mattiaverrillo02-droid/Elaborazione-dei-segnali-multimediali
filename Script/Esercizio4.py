import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import equalize_hist 
from scipy.ndimage import correlate

plt.close('all')
x=io.imread('Immagini/volto4.png', as_gray=True)
x=np.float64(x)/255 #la traccia chiede tra 0 e 1



X=np.fft.fft2(x)
modulo_X=np.abs(X)
H=np.log(modulo_X)

tau=0.35

#l'esercizio dice di calcolare la somma di H a delle frequenze specifiche.
#per avere l'asse delle frequenze ci serve fftfreq
M,N = H.shape
m = np.fft.fftfreq(M)
n = np.fft.fftfreq(N)
#poi per avere le coordinate facciamo meshgrid
l,k = np.meshgrid(n, m)
mask = (np.abs(l)<=tau) & (np.abs(k)<=tau)
d = np.sum(H[mask])/ np.sum(H)

plt.figure(1)
plt.imshow(x, cmap='gray')
plt.show()
