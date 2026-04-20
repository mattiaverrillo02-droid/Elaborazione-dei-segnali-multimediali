import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram
from seg_utils import zero_crossing

plt.close('all')

x=np.float64(io.imread('immagini/circuito_rumoroso.jpg'))

y=ndi.gaussian_laplace(x, (4, 4))
h=np.array([[0,1,0],[1,-4,1],[0,1,0]])

bordi=zero_crossing(y)
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(y, cmap='gray', clim=[0,1])

plt.subplot(1,2,2)
plt.imshow(bordi, cmap='gray', clim=[0,1])

plt.show()