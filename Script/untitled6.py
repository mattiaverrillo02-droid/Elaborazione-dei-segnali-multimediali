import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.color as cl
from sklearn.cluster import k_means
from skimage.exposure import equalize_hist
from seg_utils import zero_crossing


plt.close('all')


x=np.float64(io.imread('Immagini/ala_ape.jpg'))/255

x=cl.rgb2hsv(x)

S=x[:,:,1]

plt.figure(1)
plt.imshow(S, clim=[0,1], cmap='gray')

y=ndi.uniform_filter(S,(5,5))
max_x=np.max(y)
soglia=22/100*max_x
y=y>soglia


z=y*S
plt.figure(2)
plt.imshow(y, cmap='gray', clim=[0,1])


h=np.array([[0,1,0], [1,-4,1],[0,1,0]])

mappa=ndi.correlate(y,h)
plt.figure(3)
plt.imshow(mappa, cmap='gray')
