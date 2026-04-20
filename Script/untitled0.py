import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
from skimage.util import random_noise


plt.close('all')
x=np.float64(io.imread('Immagini/Lena.jpg'))

def smf(x,k,T):
    mediana=ndi.median_filter(x, (k,k))
    mask=np.abs(mediana-x)>T
    y= mask*mediana +(1-mask)*x
    return y



lena_rumorosa=random_noise(x/255,mode='s&p', amount=0.2 )*255


plt.figure(0)
plt.subplot(1,2,1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('Lena originale')


plt.subplot(1,2,2)
plt.imshow(lena_rumorosa, clim=[0,255], cmap='gray')
plt.title('Lena rumorosa')
K=[3,5,7,9,11]
PSNR=[]
MSE=0
for num in K: 
    y=smf(lena_rumorosa,num,30)
    MSE=np.mean(np.abs((x-y)**2))
    print('dioporco', MSE)
    PSNR.append(10*np.log10(255**2/MSE))
    print('dioporco', PSNR)
    
plt.figure(10)
plt.plot(K, PSNR)

plt.show()




    
    
    
    