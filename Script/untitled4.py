import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.color as cl

plt.close('all')

x1 = cl.rgb2gray(np.float64(io.imread('Immagini/disk1.jpg'))/255)
x2 = cl.rgb2gray(np.float64(io.imread('immagini/disk2.jpg'))/255)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(x1, clim=[0,1], cmap='gray')
plt.title('Disk1')

plt.subplot(1,2,2)
plt.imshow(x2, clim=[0,1], cmap='gray')
plt.title('Disk2')

h=np.array([[0,1,0], [1,-4,1],[0,1,0]])


#laplaciano immagine 1
lap1=ndi.correlate(x1,h)**2

#laplaciano immagine 2
lap2=ndi.correlate(x2,h)**2


#media e varianza immagine 1
u1=ndi.generic_filter(lap1, np.mean, (5,5))
sigma_quadro_1=ndi.generic_filter(lap1, np.var, (5,5))

#media e varianza immagine 2
u2=ndi.generic_filter(lap2, np.mean, (5,5))
sigma_quadro_2=ndi.generic_filter(lap2, np.var, (5,5))

a1=u1*sigma_quadro_1
a2=u2*sigma_quadro_2

a1n=a1/(a1+a2+1e-15)
a2n=1-a1n

y=a1n*x1+ a2n*x2

plt.figure(2)
plt.imshow(y, clim=[0,1], cmap='gray')
plt.title('Fusione')




