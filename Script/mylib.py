import numpy as np

def FSHS (x,k=256): 
    xmax=np.max(x)
    xmin=np.min(x)
    k=256
    y=((k-1)/(xmax-xmin))*(x-xmin)
    return y
