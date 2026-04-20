import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

plt.close('all')
x=np.float64(io.imread('Immagini/fragole.jpg'))/255
plt.figure(1)
plt.imshow(x)


R=x[:,:,0]
G=x[:,:,1]
B=x[:,:,2]

plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(R,cmap='gray', clim=[0,1])
plt.title('Componente rossa')

plt.subplot(1,3,2)
plt.imshow(G,cmap='gray', clim=[0,1])
plt.title('Componente verde')

plt.subplot(1,3,3)
plt.imshow(B,cmap='gray', clim=[0,1])
plt.title('Componente blu')

#CMY
C=1-R
M=1-G
Y=1-B

plt.figure(3)
plt.subplot(1,3,1)
plt.imshow(C,cmap='gray', clim=[0,1])
plt.title('Componente ciano')

plt.subplot(1,3,2)
plt.imshow(M,cmap='gray', clim=[0,1])
plt.title('Componente magenta')

plt.subplot(1,3,3)
plt.imshow(Y,cmap='gray', clim=[0,1])
plt.title('Componente giallo')

#CMYK

z=np.stack((C,M,Y),2)
k=np.min(z,2)

Cn=C-k
Mn=M-k
Yn=Y-k

plt.figure(4)
plt.subplot(1,4,1)
plt.imshow(Cn,cmap='gray', clim=[0,1])
plt.title('Componente ciano')

plt.subplot(1,4,2)
plt.imshow(Mn,cmap='gray', clim=[0,1])
plt.title('Componente magenta')

plt.subplot(1,4,3)
plt.imshow(Yn,cmap='gray', clim=[0,1])
plt.title('Componente giallo')

plt.subplot(1,4,4)
plt.imshow(k,cmap='gray', clim=[0,1])
plt.title('Componente black')

#terna HSI
x_hsv=rgb2hsv(x)
H=x_hsv[:,:,0]
S=x_hsv[:,:,1]
V=x_hsv[:,:,2]

plt.figure(5)
plt.subplot(1,3,1)
plt.imshow(H,cmap='gray', clim=[0,1])
plt.title('Tinta')

plt.subplot(1,3,2)
plt.imshow(S,cmap='gray', clim=[0,1])
plt.title('Saturazione')

plt.subplot(1,3,3)
plt.imshow(V,cmap='gray', clim=[0,1])
plt.title('Intensità luminosa')

plt.show()

