import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.color as cl
from sklearn.cluster import k_means
from skimage.exposure import equalize_hist




def trasf(x, k):
    #x viene vettorizzato quindi lo riportiamo a matrice. 
    x=np.reshape(x, (k,k))
    
    #punto 1
    X=np.fft.fftshift(np.fft.fft2(x))
    
    #punto2
    X0=X[0,0]
    X1=X[0,1]
    X2=X[0,2]
    X3=X[1,2]
    X4=X[2,2]
    
    a = np.array([np.real(X1),np.imag(X1),np.real(X2),np.imag(X2),
                  np.real(X3),np.imag(X3),np.real(X4),np.imag(X4)])
    
    c = a > 0
    
    # 5) conversione del vettore in un numero decimale
    w = 2**np.arange(8)
    y = np.sum(w*c)
    
    return y
    
    
    
    

if __name__=='__main__': 
    plt.close('all')
    k=9
    x=np.float64(io.imread('Immagini/impronta.jpg'))

    plt.figure(1)
    plt.imshow(x, clim=[0,255], cmap='gray')
    plt.title('Originale')
    
    
    
    
    y=ndi.generic_filter(x, trasf, (k,k), extra_arguments=(k,))
    
    plt.figure(2)
    plt.hist(y)
    n,b= np.histogram(x.flatten(), bins=257)
    
    plt.plot(b,n)
    
    
    
    

    plt.show()