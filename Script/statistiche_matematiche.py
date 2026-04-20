import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
plt.close('all')
x= io.imread('immagini/test.jpg')

x=np.float64(x) 

xavg=np.mean(x) #media di tutti i valori della matrice di x
xstd=np.std(x) #deviazione standard
xvar= np.var(x) #varianza

y= ndi.generic_filter(x, np.mean, (11,11)) #immagine delle medie
z= ndi.generic_filter(x, np.var, (11,11)) #immagine delle varianze
plt.figure(1)

plt.subplot(1,3,1) #1 riga, 3 colonne. la foto normale la metto sulla prima colonna
plt.imshow(x,cmap='gray', clim=[0,255] )
plt.title('Foto normale')

plt.subplot(1,3,2) #media delle foto sulla seconda colonna
plt.imshow(y,cmap='gray', clim=[0,255] )
plt.title('Immagine delle medie')

plt.subplot(1,3,3) #media delle foto sulla seconda colonna
plt.imshow(z,cmap='gray' )#dobbiamo togliere clim altrimenti schiaccia su 255
plt.title('Immagine delle varianze')





