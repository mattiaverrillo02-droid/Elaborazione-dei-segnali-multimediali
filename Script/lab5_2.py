import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist
from scipy.ndimage import gaussian_filter
plt.close('all')
x=np.float64(io.imread('Immagini/azzurro.jpg'))/255
plt.figure(1)
plt.imshow(x)

y=rgb2hsv(x)
H=y[:,:,0]
S=y[:,:,1]
V=y[:,:,2]

mask_saturazione= S>0.4
mask_v=V>0.5
mask=mask_saturazione&mask_v

H=0.0*mask + H*(1-mask)
S=1*mask + S*(1-mask)

y=np.stack((H,S,V), 2)

z=hsv2rgb(y)


           
plt.figure(2)
plt.imshow(z)





