import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
from skimage.exposure import histogram

plt.close('all')
x= io.imread('immagini/Lena.jpg')

n, b= histogram(x.flatten(), nbins=256) #flatten lo trasforma in vettore, numero barrette=256
plt.figure(1)
#plt.bar(b,n) #plt.bar crea un grafico a barre. passiamo praticamente gli assi. sull'asse x che tipo 
#di barretta, quindi valore 0, valore 1 etc. asse y l'altezza della baretta che è il vettore n
#plt.plot(b,n) #interpolazione lineare
plt.stem(b,n) #discreto
plt.axis([0,255, 0, 1.1*np.max(n)])

         
plt.figure(2)
plt.imshow(x, cmap=('gray'), clim=[0,255])
plt.show()