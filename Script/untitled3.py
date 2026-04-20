import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
from skimage.util import random_noise
import skimage.io as io

plt.close('all')

x=np.float64(np.fromfile('Immagini/target_rumorosa.raw', np.float32))
x=np.reshape(x, (256,256))


plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('Immagine originale')

#edge detection con tecnica standard. Possiamo fare sia con derivata prima che con derivata seconda
#con la derivata seconda, soffriamo molto il rumore. 
#h=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
h=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

y=ndi.median_filter(x, (3,3))
plt.figure(2)
plt.imshow(y, cmap='gray', clim=[0,255])
plt.title('Immagine filtrata')

y=ndi.correlate(y,h)>10


plt.figure(3)
plt.imshow(y, cmap='gray', clim=[0,1])
plt.title('Bordi derivata seconda')



plt.show()