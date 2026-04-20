import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import equalize_hist 


plt.close('all')

x = np.float64(io.imread('Immagini/volto.tif'))
X = np.fft.fftshift(np.fft.fft2(x) )
modulo_X = np.abs(X)
fase_X=np.angle(X)

#X= |X|*e^(j*theta)
X1=modulo_X*(np.exp(1j*0)) #metto la fase a 0
X2=1*np.exp(1j*fase_X) #metto il modulo a 1


#antitrasformate
senza_fase= np.real(np.fft.ifft2(np.fft.ifftshift(X1)))
senza_modulo=np.real(np.fft.ifft2(np.fft.ifftshift(X2)))

#se quando abbiamo trasformato abbiamo trasformato e shiftato, ora dobbiamo 
#shiftare al contrario e antitrasformare 
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(x, cmap='gray', clim=[0,255])
plt.title('Originale')

#quando antitrasformo possono uscire valori negativi e quindi posso avere poi 
#logaritmo di una cosa negativo e ci da NULL e vengono visualizzati bianchi
plt.subplot(1,2,2)
plt.imshow(np.log(1+modulo_X), cmap='gray', clim=None)
plt.title('Modulo trasformata')


plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(np.log(1+senza_fase-np.min(senza_fase)), cmap='gray')
#per evitare i valori negativi sottraggo prima il minimo e poi faccio il log di 1+
plt.title('Immagine senza fase')

plt.subplot(1,2,2)
plt.imshow(equalize_hist(senza_modulo), cmap='gray', clim=None)
plt.title('Immagine senza modulo')



plt.show()