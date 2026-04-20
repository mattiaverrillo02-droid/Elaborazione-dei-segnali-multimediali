import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.color as cl
from sklearn.cluster import k_means
from skimage.exposure import equalize_hist
from seg_utils import zero_crossing


plt.close('all')


x=cl.rgb2gray(np.float64(io.imread('Immagini/ala_ape.jpg'))/255)

plt.figure(1)
plt.imshow(x, clim=[0,1], cmap='gray')

M,N=x.shape

y=ndi.gaussian_filter(x, (7,7))

h=np.array([[0,1,0], [1,-4,1],[0,1,0]])

z=ndi.correlate(y,h)
plt.figure(3)
plt.imshow(z, cmap='gray')

plt.figure(2)
plt.imshow(y, cmap='gray')



plt.show()