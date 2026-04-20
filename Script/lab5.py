import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist
from scipy.ndimage import uniform_filter
plt.close('all')
x=np.float64(io.imread('Immagini/fragole.jpg'))/255
plt.figure(1)
plt.imshow(x)

k=11
y =uniform_filter(x, (k,k,1))

w=rgb2hsv(x)
H=w[:,:,0]
S=w[:,:,1]
V=w[:,:,2]

#filtriamo solo l'intensità
fV=uniform_filter(V, (k,k))

W=np.stack((H,S,fV), 2)

z=hsv2rgb(W)

plt.figure(2)
plt.imshow(y)

plt.figure(3)
plt.imshow(z)



