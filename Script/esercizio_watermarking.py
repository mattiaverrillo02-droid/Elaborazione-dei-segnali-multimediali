import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from bitop import bitget,bitset
plt.close('all')

x=np.fromfile('Immagini/Lena.y', np.uint8)
x=np.reshape(x, (512,512))

m=np.fromfile('Immagini/marchio.y', np.uint8)
m=np.reshape(m, (350,350))

x= x[50:400,50:400]

y=bitset(x,6,m) #sostituiamo al bit meno significativo il marchio

b=bitget(y,6)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(y,cmap='gray', clim=[0,255])

plt.subplot(1,2,2)
plt.imshow(b ,cmap='gray', clim=[0,1])

plt.show()