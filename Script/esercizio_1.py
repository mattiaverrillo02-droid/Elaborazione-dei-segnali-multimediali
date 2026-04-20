# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:07:33 2026

@author: matti
"""

import numpy as np
import skimage.io as io #io è una libreria pèer leggere e scrivere immagini in formato standar
import matplotlib.pyplot as plt #serve per visualizzare le immagini
from funzione_medie import medie
plt.close("all")

x=io.imread('immagini/test.jpg')
plt.figure(2)
plt.imshow(x, cmap='gray', clim=[0,255])
plt.show()
medie(x, 10)