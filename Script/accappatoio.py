import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist
plt.close('all')
x=np.float64(io.imread('Immagini/foto.jpg'))/255
plt.figure(1)
plt.imshow(x)

x_cmy=1-x #invece che farlo singolarmente sulle 3 componenti

C= x_cmy[:,:,0]
M= x_cmy[:,:,1]
Y= x_cmy[:,:,2]

#color_balancing
#vogliamo ridurre il ciano quindi a>1
#gli altri li rimaniamo uguali
Cn=C**1.5
Mn=M**1.0
Yn=Y**1.0

#cmin=np.min(C)
#new=C-cmin

#y_cmy=np.stack((Cnew,M,Y),2)
#y=1-y_cmy


#Mn=M+0.1
#z_cmy=np.stack((Cnew,Mn,Y),2)
#z=1-z_cmy

Y_cmy=np.stack((Cn,Mn,Yn),2)
y=1-Y_cmy
plt.figure(2)
plt.imshow(y)


