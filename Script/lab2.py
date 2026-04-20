import numpy as np
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
import skimage.io as io


plt.close('all')


x= io.imread('Immagini/granelli.jpg')
x=np.float64(x)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(x, cmap='gray', clim=[0,255])

plt.subplot(1,2,2)
plt.hist(x.flatten(), 256)
plt.show()

y=x-50

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(y, cmap='gray', clim=[0,255])

plt.subplot(1,2,2)
plt.hist(y.flatten(), 256)
plt.show()

