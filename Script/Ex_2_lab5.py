import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import equalize_hist 
from scipy.ndimage import correlate


plt.close('all')

x=np.float64(io.imread('Immagini/car.tif'))
X=np.fft.fft2(x)
X=np.fft.fftshift(X)

plt.figure(1)
plt.imshow(x, cmap='gray', clim=[0,255])
plt.figure(2)
plt.imshow(np.log(1+np.abs(X)), cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5) )



m = np.fft.fftshift(np.fft.fftfreq(np.log(1+np.abs(X)).shape[0]))
n = np.fft.fftshift(np.fft.fftfreq(np.log(1+np.abs(X)).shape[1]))
l,k = np.meshgrid(n,m)

x= 0.158
y1=0.338
y2=0.174

#Definizione dei cerchietti per togliere i picchi
D1=np.sqrt((l-x)**2+(k-y1)**2)
D2=np.sqrt((l+x)**2+(k-y1)**2)
D3=np.sqrt((l-x)**2+(k+y1)**2)
D4=np.sqrt((l+x)**2+(k+y1)**2)
D5=np.sqrt((l-x)**2+(k-y2)**2)
D6=np.sqrt((l+x)**2+(k-y2)**2)
D7=np.sqrt((l-x)**2+(k+y2)**2)
D8=np.sqrt((l+x)**2+(k+y2)**2)

H1= (D1>0.025) & (D2 >0.025) & (D3 >0.025) & (D4 >0.025)
H2= (D5>0.05) & (D6 >0.05) & (D7 >0.05) & (D8 >0.05)

H= H1 & H2


plt.figure(3)
plt.imshow(H, cmap='gray', clim=[0,1], extent=(-0.5,+0.5,+0.5,-0.5))
plt.title('Filtro')

Y=H*X
plt.figure(4)
plt.imshow(np.log(1+np.abs(Y)), cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
plt.title('Trasformata immagine filtrata')


y=np.real(np.fft.ifft2(np.fft.ifftshift(Y)))

plt.figure(5)
plt.imshow(y, cmap='gray', clim=[0,255])
plt.title('Immagine filtrata')




plt.show()