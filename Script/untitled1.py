import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
from skimage.util import random_noise
import skimage.io as io

plt.close('all')

def filtro_sigma(x,K,sigma):
    im_medie=ndi.generic_filter(x, fun_media, (K,K), extra_arguments=(sigma,))
    
    return im_medie
    
    
                
    

def fun_media(x, sigma):
    central_index=len(x)//2
    lower=x[central_index]-2*sigma
    upper=x[central_index]+2*sigma
    y=x[ (x>lower) & (x<upper)]
    if len(y)<4:
        media=np.mean(x)
    else: 
        media=np.mean(y)
    
    return media
    


x=np.float64(io.imread('Immagini/barbara.png'))

sigma=20
x_noisy= random_noise(x/255, mode='gaussian', var=(sigma/255)**2)*255

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('Originale')

plt.subplot(1,2,2)
plt.imshow(x_noisy, cmap='gray', clim=[0,255])

K=7
y=filtro_sigma(x_noisy, K, sigma)

plt.figure(2)
plt.imshow(y, clim=[0,255], cmap='gray')

mse_noisy=np.mean(np.abs(x-x_noisy)**2)
psnr_orig_noisy = 10*np.log10((255**2)/mse_noisy)
mse_filtrata=np.mean(np.abs(x-y)**2)
psnr_filtrata = 10*np.log10((255**2)/mse_filtrata)
