import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram
from bitop import bitget
from bitop import bitset

plt.close('all')
x=io.imread('Immagini/frattale.jpg')

plt.figure(1)
plt.imshow(x, clim=[0,255], cmap='gray')

plt.figure(2)

for index in range (0,8):
    b=bitget(x, index)
    plt.subplot(2,4,1+index)
    plt.imshow(b, cmap='gray', clim=[0,1])
    plt.title("bit" + str(index))
    
    
#cancelliamo man mano i bit plane
plt.figure(3)
y=np.copy(x)
for index in range (0,8): 
    y=bitset(y,index,0)
    plt.subplot(2,4,1+index)
    plt.imshow(y, cmap='gray', clim=[0,255])
    plt.title("bit cancellato" + str(index))
    

plt.show()