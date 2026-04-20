import numpy as np
import skimage.io as io #io è una libreria pèer leggere e scrivere immagini in formato standar
import matplotlib.pyplot as plt #serve per visualizzare le immagini

plt.close("all")

x=io.imread('immagini/baba.jpg')
x=np.float64(x)/255 #se facciamo il cast a float i valori saranno visualizzati tra 0 e 1
#dobbiamo dividere per forza per 255 altrimenti tutti quelli fuori 0 e 1 saranno tagliati e l'immagine sarà bianca

plt.figure(1)
plt.imshow(x)
plt.show()

R= x[:,:, 0] #matrice del rosso, il primo indice della terza colonna
G= x[:,:, 1] #matrice del verde, il secondo indice della terza colonna
B= x[:,:, 2] #matrice del blu, il terzo indice della terza colonna

#R,G;B sono matrici bidimensionali. per visualizzarli va specificato il map
plt.figure(2)


plt.subplot(1, 3, 1) #subplot ci serve per dividere per vedere più immmagini insieme. 1 colonna, 3 righe. 
#La rossa la metto nella prima riga

plt.imshow(R, cmap='gray', clim=[0,1])
plt.title('rosso')


plt.subplot(1, 3, 2)#la verde sulla seconda riga
plt.imshow(G, cmap='gray', clim=[0,1])
plt.title('verde')


plt.subplot(1, 3, 3)
plt.imshow(B, cmap='gray', clim=[0,1]) #la blu sulla terza riga
plt.title('blu')

plt.show()

#vogliamo togliere la componente rossa
M,N= R.shape #shape da la dimensione. Shape è un attributo non un metodo
Rnew=np.zeros((M,N), dtype=R.dtype) #ho creato una matrice di tutti 0, con le dimensioni di R, e il tipo di R 

#ora dobbimao unire questa rnew con la verde e la blu di prima
y= np.stack([Rnew, G, B], 2)

plt.figure(4)
plt.imshow(y)
plt.title('Senza rosso')
plt.show()