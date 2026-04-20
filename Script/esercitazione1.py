import numpy as np
import skimage.io as io #io è una libreria pèer leggere e scrivere immagini in formato standar
import matplotlib.pyplot as plt #serve per visualizzare le immagini

plt.close("all") #chiude tutte le figure aperte precedentemente. serve per pulire
x= io.imread('immagini/dorian.jpg')

plt.figure(1) #crea una finestra, posso inserire un numero per indicare una specifica figura, 
#cosi posso fare più figure

#all'interno della finestra devo visualizzare l'immagine
plt.imshow(x, cmap='gray', clim=[0,255]) #scala grigi, 0=nero, 255=bianco
#se non specifico la scala, di default usa quella a colori. se non specifico i limit di default
#mette il minimo e il massimo dell'immagine
plt.colorbar() #nella visualizzazione esce la scala di associazione
plt.show() #lancia il comando di visualizzazione

xmin=np.min(x)
xmax=np.max(x)
print("minimo", xmin)
print("massimo", xmax)

y=x[60:100, 50:100]
plt.figure(2)
plt.imshow(y, cmap='gray', clim=[0,255])
plt.show()

z= np.fromfile('immagini/house.y', np.uint8) #per visualizzare un file in estensione. raw non serve io 
#ma una funzione di np. dobbiamo conoscere necessariamente la dimensione e la formattazione

z=np.reshape(z, (512,512)) #fromfile restituisce un vettore. con reshape lo trasformiamo 
#in matrice indicando la dimensione delle righe e colonne
plt.figure(3)
plt.imshow(z, cmap='gray', clim=[0,255])
plt.show()

#3 modi per salvare 
y.tofile('nuovoraw.y') #salviamo in formato raw. per riaprire dobbiamo usare la dim, e la formattazione
io.imsave('nuovo.png', np.uint8(y)) #formato standard
np.save('nuovo.npy', y) #salva un qualasiasi array

