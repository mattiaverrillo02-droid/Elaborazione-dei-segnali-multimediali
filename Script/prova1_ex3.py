import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from sklearn.cluster import k_means

plt.close('all')

#convertiamo subito a float perchè sarà oggetto di operazioni
x=np.float64(np.fromfile('Immagini/rice.y', np.uint8))/255
x=np.reshape(x, (256,256))


def T_opt(x, K=2): 
    
    N,M=x.shape
    d=np.reshape(x, (M*N,1))
    centroid, idx, sum_var=k_means(d,K)
    C1 = centroid[0]
    C2 = centroid[1]
    
    # Calcoliamo la semisomma
    t = (C1 + C2) / 2
    return t
    


def T_opt2(x,K=2):
    #L è la lista dei gruppi disgiunti di righe
    M,N=x.shape
    rice_ideal = np.float64(np.fromfile('Immagini/rice_bw.y', np.uint8)).reshape(256, 256) / 255>0
    L=[1,2,4,8,16,32,64,128,256]
    errors=[] #teniamo traccia dell'errore per ogni l
    best_error = float('inf') #salviamo il migliore errore
    best_segmentation = None #salviamo la migliore segmentazione locale
    for l in L: 
        
        x_locale=np.zeros((M,N))
        for i in range(0,256,l): 
            
            striscia=x[i:i+l, :] #prendiamo il blocco di l righe e tutte le colonne
            t_locale=T_opt(striscia) #chiamiamo la nostra funzione su questa sotto matrice
            x_locale[i:i+l,:]= striscia>t_locale
        
        #x_locale, con l blocchi di righe disgiunte, è pronta
        error = np.sum(np.abs(x_locale - rice_ideal))
        errors.append(error)
        
        # Verifichiamo se è la migliore finora
        if error < best_error:
            best_error = error
            best_l = l
            best_segmentation = x_locale
            
    return best_segmentation, best_l
            
plt.figure(3)

# 1. Originale
plt.subplot(1, 3, 1)
plt.imshow(x, cmap='gray', clim=[0,1])
plt.title('Originale')

# 2. Globale (L=256)
t_global = T_opt(x)
plt.subplot(1, 3, 2)
plt.imshow(x > t_global, cmap='gray')
plt.title('Globale (L=256)')

# 3. Migliore Locale
best_segmentation, best_l=T_opt2(x)
plt.subplot(1, 3, 3)
plt.imshow(best_segmentation, cmap='gray')
plt.title(f'Migliore Locale (L={best_l})')

plt.figure(2)
rice_ideal = np.float64(np.fromfile('Immagini/rice_bw.y', np.uint8)).reshape(256, 256) / 255
plt.imshow(rice_ideal)
plt.show()

