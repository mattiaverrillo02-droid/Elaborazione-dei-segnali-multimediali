
import numpy as np
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi
import skimage.io as io

def fhs(x): 
    xmin=  np.min(x)
    xmax=  np.max(x)
    a= 255/(xmax-xmin)
    b= -255*xmin/(xmax-xmin)
    g= a*x + b
    
    return g
             
    