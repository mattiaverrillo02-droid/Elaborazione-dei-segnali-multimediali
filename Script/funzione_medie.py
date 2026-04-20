#funzione dal prototipo medie(x, k)

import numpy as np
import matplotlib.pyplot as plt

def medie(x,k): 
    N, M= x.shape#N righe, M colonne
    y= np.zeros((N+k,M+k))
        
    for i in range (N):
        for j in range (M):
            y[i][j]= np.mean(x[i:i+k,j:j+k])
    
    
    plt.figure(1)
    plt.imshow(y, cmap='gray', clim=[0,255])
    plt.show()
            
            
