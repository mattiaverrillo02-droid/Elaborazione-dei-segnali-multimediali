import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram
from skimage.transform import rescale
from skimage.transform import AffineTransform,warp
plt.close('all')
x=np.float64(io.imread('Immagini/Lena.jpg'))
plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')

A=AffineTransform(translation=(100,50))
y=warp(x,A,order=1)
#y=rescale(x,3.0, order=1)
#y= x[::2, ::2] #decimazione
plt.figure(2)
plt.imshow(y, clim=[0,255], cmap='gray')
