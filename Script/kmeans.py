import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist
plt.close('all')
x=np.float64(io.imread('Immagini/Fiori256.bmp'))/255
plt.figure(1)
plt.imshow(x)

N,M,L=x.shape
K=5

from sklearn.cluster import k_means
d = np.reshape(x, (M*N, 3))
centroid, idx, sum_var = k_means(d, K)
y = np.reshape(idx, (M,N))

plt.figure(2)
plt.imshow(y, clim=[0,K-1], cmap='jet')
plt.show()