import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')
x = np.float64(io.imread('Immagini/turbina.jpg'))

plt.figure(1)
plt.subplot(1,2,1)

plt.imshow(x, cmap='gray', clim=[0,255])


mask=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

y=ndi.correlate(x, mask)

y=np.abs(y)
soglia= 0.9*np.max(y)

mask= y>soglia
plt.subplot(1,2,2)

plt.imshow(mask, cmap='gray', clim=[0,1])

plt.show()
