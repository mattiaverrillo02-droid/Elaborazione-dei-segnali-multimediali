import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')
x = np.float64(io.imread('Immagini/luna.jpg'))
h=np.array(([[0,1,0],[1,-4,1],[0,1,0]]))

y=ndi.correlate(x, h)

y=x-y

plt.subplot(1, 2,1)
plt.imshow(x, clim=[0,255], cmap='gray')

plt.subplot(1, 2,2)
plt.imshow(y, cmap='gray')

plt.show()

