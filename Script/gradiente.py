import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')

x=np.fromfile('Immagini/house.y',np.uint8)
x=np.reshape(x, (512,512))
x=np.float64(x)

#la definizione di derivata che sta usando è x(n+1)-x(n-1)
derivata_vert=np.array([[0,-1,0],[0,0,0],[0,1,0]])
derivata_oriz=np.array([[0,0,0],[-1,0,1],[0,0,0]])

y_verticale=ndi.correlate(x, derivata_vert)
y_orizz=ndi.correlate(x, derivata_oriz)

gradiente= np.sqrt(y_verticale**2+y_orizz**2)

plt.figure(1)
plt.imshow(x,cmap='gray', clim=[0,255])
plt.title('Originale')

plt.figure(2)
plt.imshow(gradiente, cmap='gray')
plt.title('Gradiente')

plt.figure(3)
plt.imshow(np.abs(y_verticale), cmap='gray')
plt.title('Derivata verticale')

plt.figure(4)
plt.imshow(np.abs(y_orizz), cmap='gray')
plt.title('Derivata Orizzontale')
plt.show()


