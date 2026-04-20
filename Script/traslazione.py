import numpy as np
# importa Numpy
import matplotlib.pyplot as plt # importa Matplotlib
import scipy.ndimage as ndi
# importa Scipy per le immagini
import skimage.io as io
# importa il modulo Input/Output di SK-Image

plt.close('all')
x=np.float64(io.imread('Immagini/granelli.jpg'))

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(x, cmap='gray', clim=[0,255])

plt.subplot(1,2,2)
plt.hist(x.flatten(), 256) #possiamo fare anche cosi con l'istogramma, invece che crearlo e poi scegliere l
#l'interpolazione
plt.xlim([0,255])
plt.show()

y=x-50
Y=2*y

#saturazione
mask = y<0 # crea una matrice di vero (True) e falso (False)
x[mask] = 0 # si annullano tutti i valori minori di 0, cioè si portano a 0
x[x>255] = 255 

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(y, cmap='gray',clim=[0,255] )

plt.subplot(1,2,2)
plt.hist(y.flatten(), 256) #possiamo fare anche cosi con l'istogramma, invece che crearlo e poi scegliere l
#l'interpolazione
plt.xlim([0,255])
plt.show()