import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.transform import warp
plt.close('all')

x = np.float64(io.imread('Immagini/mare.png'))

plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(x, clim=[0,255], cmap='gray')
plt.title('immagine originale')

X=np.fft.fft2(x)
R=np.real(np.fft.ifft2(np.abs(X)**2))
R=np.fft.fftshift(R)
Nr, Nc=R.shape #numero righe e colonne

#np.arange(Nc) crea un vettore di numeri interi che va da 0 a N_c - 1.Se la tua immagine ha 256 colonne, 
#avrai:[0, 1, 2, 3, ..., 255]. Sottraendo metà della dimensione, sposti l'intervallo in modo che lo zero si trovi esattamente al centro.
#Quando calcoli l'autocorrelazione, il picco massimo rappresenta uno "spostamento zero".
#Se usassi le coordinate standard (da 0 a 255), il centro dell'immagine sarebbe il punto (128, 128).
#Se il secondo picco (la copia) si trovasse 40 pixel più a destra, lo leggeresti alle coordinate $128 + 40 = 168$.
#Spostando l'origine a zero con quella sottrazione, il valore che leggi sulla mappa corrisponde direttamente allo spostamento fisico (il vettore di traslazione). 
#Se il picco è sul numero 40 della scala n, sai immediatamente che la scogliera è stata spostata di 40 pixel, 
#senza dover fare calcoli complicati dopo.
#non possiamo fare fftfreq perchè questa è nello spazio non in frequenza. 
n = np.arange(Nc) - Nc/2
m = np.arange(Nr) - Nr/2
n,m = np.meshgrid(n,m)

plt.figure(1)
plt.subplot(2,2,2)
plt.imshow(R, clim=None, cmap='jet', extent = (- Nc/2, Nc/2-1, Nr/2-1, - Nr/2))
plt.colorbar()
plt.title('autocorrelazione')

fig, ax = plt.subplots(num=2, subplot_kw={"projection": "3d"}) # 3d plot
ax.plot_surface(n,m,R, cmap='jet')

# punto 2
# Individuzione del secondo picco della funzione di autocorrelazione
#confronta la matrice L con R e mette true dove i pixel risultano uguali. 
'''
R=[25 12 15 35 
   60 13 9  56 
   1  45  66 79]

con blocco 3x3 
L=[60  60  56  56
   60  66  79  79 
   60  66  79  79]

dobbiamo marcare ogni singolo pixel =true se è il massimo del blocco di R

L=[F   F  F  F 
   T   F  F  F
   F   F   F  T]
 Ora dobbiamo trovare il secondo valore più alto. quindi prendiamo tutti i valori in cui ho true
 R[l]=[60, 79]
 p=[60,79] ordine crescente. ci serve il secondo più alto, per questo p[-2]
'''
L = ndi.generic_filter(R, np.max, (5,5)) == R
p = np.sort(R[L])
mappa = L & (R == p[-2]) #senza la and con L rischiamo di prendere valori che non erano massimi 
#nella finestra, ma che comunque sono picchi

#coordinate del secondo picco
tn = n[mappa]
tm = m[mappa]
print(tn[0], tm[0])

# punto 3
A = np.array([[1,0, tn[0]], [0, 1, tm[0]], [0, 0, 1]])
y = warp(x, A)
mask = x==y
plt.figure(1)
plt.subplot(2,2,3)
plt.imshow(y, clim=[0,255], cmap='gray')
plt.title('immagine traslata')
plt.subplot(2,2,4)
plt.imshow(mask, clim=[0,1], cmap='gray')
plt.title('mask')