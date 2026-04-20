import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.exposure import equalize_hist 
from scipy.ndimage import correlate


plt.close('all')





def detect(x): 
    M,N = x.shape;
    
    h = np.array([[0,0,0],[-1, 2, -1],[0, 0, 0]])
    
    #punto1: derivata seconda
    d2_hori = correlate(x,h) 
    
    #punto2: pseduovarianza orizzontale e derivata prima
    v_hori = np.sum(np.abs(d2_hori),0) 
    
    mask_derivata_1= np.array([-1,1,0])
    d_hori=correlate(v_hori, mask_derivata_1)
    
    #punto3: modulo della trasformata su N-2 punti
    D_hori = np.abs(np.fft.fftshift(np.fft.fft(d_hori, N-2))) 
    
    #punto4: trovare il picco e il fattore di scala R
    f_hori = np.fft.fftshift(np.fft.fftfreq(N-2))
    
    #la frequenza dove ho il picco è la frequenza massima della trasformata tra 0 e 1/2. siccome il massimo 
    #è 1/2 allora serve solo dire >0
    ni_hori = f_hori[(D_hori==np.max(D_hori)) & (f_hori>0)] #frequenza v0: picco tra 0 e 1/2
    R_hori = 1/ni_hori #R=1/vo
    
    d2_vert = correlate(x,h.T)
    v_vert = np.sum(np.abs(d2_vert),1)
    d_vert = v_vert[1:] - v_vert[:-1] 
    D_vert = np.abs(np.fft.fftshift(np.fft.fft(d_vert,M-2)))
    f_vert = np.fft.fftshift(np.fft.fftfreq(M-2))
    ni_vert = f_vert[(D_vert==np.max(D_vert)) & (f_vert>0)]
    R_vert = 1/ni_vert
    
    return D_hori, D_vert, R_hori, R_vert

      

    
if __name__=='__main__': 
    
    x = np.fromfile('Immagini/zoom.y',np.float32)
    x = np.reshape(x, (128, 128))
    D_hori, D_vert, R_hori, R_vert = detect(x)
    f_hori = np.fft.fftshift(np.fft.fftfreq(len(D_hori)))
    f_vert = np.fft.fftshift(np.fft.fftfreq(len(D_vert)))
    
    plt.figure() 
    plt.subplot(2,1,1)
    plt.plot(f_hori, D_hori,'-or')
    plt.title(f'DFT derivata pseudo-varianza orizontale, fattori di scala: {R_hori}')
    plt.subplot(2,1,2)
    plt.plot(f_vert, D_vert,'-or')
    plt.title(f'DFT derivata pseudo-varianza verticale, fattori di scala: {R_vert}')
    