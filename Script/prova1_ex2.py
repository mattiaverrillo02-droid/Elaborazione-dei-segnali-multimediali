import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_hist


plt.close('all')

x=np.float64(io.imread('Immagini/pears_noise.png'))/255
plt.figure(1)
plt.imshow(x)

#si deve operare solo sulla luminanza quindi spazio hsv
y=rgb2hsv(x)
H=y[:,:,0]
S=y[:,:,1]
V=y[:,:,2]

plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(H, cmap='gray', clim=[0,1])
plt.title('Tinta')

plt.subplot(1,3,2)
plt.imshow(S, cmap='gray', clim=[0,1])
plt.title('Saturazione')

plt.subplot(1,3,3)
plt.imshow(V, cmap='gray', clim=[0,1])
plt.title('Luminanza')

trasf_V=np.fft.fftshift(np.fft.fft2(V))

plt.figure(3)
plt.imshow(np.log(1+np.abs(trasf_V)), cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

M,N=np.log(1+np.abs(trasf_V)).shape
m=np.fft.fftshift(np.fft.fftfreq(M))
n=np.fft.fftshift(np.fft.fftfreq(N))
l,k= np.meshgrid(n,m)

u=v=0.100
D1=np.sqrt((l-u)**2+(k+v)**2)
D2=np.sqrt((l+u)**2+(k-v)**2)

filtro=(D1>0.03) & (D2>0.03)
plt.figure(4)
plt.imshow(filtro, cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))

trasf_Vnew=trasf_V*filtro

Vnew=np.real(np.fft.ifft2(np.fft.fftshift(trasf_Vnew)))

plt.figure(5)
plt.imshow(Vnew, cmap='gray', clim=[0,1])

#ricostruiamo l'immagine
z=np.stack((H,S,Vnew),2)

#dobbiamo sempre tornare nello spazio rgb
z=hsv2rgb(z)
plt.figure(6)
plt.imshow(z)

mse=np.mean((V-Vnew)**2)

#enhancement

S=equalize_hist(S)

#oppure con la potenza. Se voglio aumentare la saturazione scelgo un valore <1, se voglio diminuire >1
y=np.stack((H,S,Vnew),2)
y=hsv2rgb(y)


plt.figure(9)
plt.imshow(y)


plt.show()