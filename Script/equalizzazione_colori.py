import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist
plt.close('all')
x=np.float64(io.imread('Immagini/volto.tiff'))/255
plt.figure(1)
plt.imshow(x)

#y=x**4 #in automatico eleva alla quarta tutte e 3 le matrici

x_hsv=rgb2hsv(x)
H=x_hsv[:,:,0]
S=x_hsv[:,:,1]
V=x_hsv[:,:,2]

#Vnew=V**4

y=equalize_hist(x)
Vnew=equalize_hist(V)

z_hsv=np.stack((H,S,Vnew), 2)

z=hsv2rgb(z_hsv)


plt.figure(2)
plt.imshow(y)

plt.figure(3)
plt.imshow(z)



plt.show()