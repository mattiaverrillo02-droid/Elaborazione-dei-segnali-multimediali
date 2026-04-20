import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.restoration import inpaint_biharmonic
from skimage import img_as_float
from skimage.color import rgb2hsv, hsv2rgb
import scipy.ndimage as ndi

plt.close('all')

# 1. Caricamento immagine (Originale o precedente output)
x = img_as_float(io.imread('Immagini/coppia.jpeg'))

# --- PASSO A: RIMOZIONE DEL LENS FLARE (come prima) ---
maschera_flare = np.zeros(x.shape[:2], dtype=bool)
# Coordinate per il rettangolo del flare in alto a destra
r_f_inz, r_f_fine = 50, 600
c_f_inz, c_f_fine = 700, 1200
maschera_flare[r_f_inz:r_f_fine, c_f_inz:c_f_fine] = True
x_no_flare = inpaint_biharmonic(x, maschera_flare, channel_axis=-1)



# --- PASSO B: RIDUZIONE LOCALE DELL'ALONE (NUOVO) ---
# Passiamo allo spazio colore HSV
y = rgb2hsv(x_no_flare)
H, S, V = y[:,:,0], y[:,:,1], y[:,:,2]

# 1. Creiamo una maschera morbida centrata sull'alone (intorno alla coppia)
# NOTA: Queste coordinate e il sigma definiscono l'ampiezza dell'intervento.
centro_r, centro_c = 700, 650 # Coordinate approssimative del centro dell'alone
sigma_alone = 250 # Controllo della sfumatura del bordo

rr, cc = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), indexing='ij')
distanza_quadrata = (rr - centro_r)**2 + (cc - centro_c)**2
# Maschera gaussiana che sfuma l'effetto verso i bordi
maschera_morbida = np.exp(-distanza_quadrata / (2.0 * sigma_alone**2))

# 2. Applichiamo la correzione Tonale (Luminosità) e Cromatica (Saturazione)
# Definiamo i fattori di correzione
fattore_scurimento = 0.55 # Riduce la luminosità locale (più basso = più scuro)
fattore_desaturazione = 0.4 # Riduce il colore cast (più basso = meno verde/giallo)

# L'azione è pesata dalla maschera morbida: massima correzione al centro, minima ai bordi.
# Usiamo np.clip per assicurarci di non scendere sotto zero.
V_corretto = np.clip(V - (maschera_morbida * (1.0 - fattore_scurimento) * V), 0, 1)
S_corretto = np.clip(S - (maschera_morbida * (1.0 - fattore_desaturazione) * S), 0, 1)

# 3. Ricomponiamo e torniamo in RGB
y_finale = np.stack((H, S_corretto, V_corretto), axis=2)
x_finale = hsv2rgb(y_finale)

# --- VISUALIZZAZIONE A CONFRONTO ---
fig, assi = plt.subplots(1, 2, figsize=(16, 8))

# Originale con solo il flare rimosso
assi[0].imshow(x_no_flare)
assi[0].set_title("1. Solo Rimozione Lens Flare")
assi[0].axis('off')

# Finale con il flare rimosso E l'alone corretto
assi[1].imshow(x_finale)
assi[1].set_title("2. Rimozione Flare + Riduzione Alone Locale")
assi[1].axis('off')


y_sharp = rgb2hsv(x_finale)
H_s, S_s, V_s = y_sharp[:,:,0], y_sharp[:,:,1], y_sharp[:,:,2]
# 1. Definiamo il kernel del Laplaciano
kernel_laplaciano = np.array([[ -1, -1,  -1],
                              [-1,  8, -1],
                              [ -1, -1,  -1]])

# 2. Calcoliamo la derivata seconda (i bordi)
# Usiamo mode='reflect' per gestire i bordi dell'immagine senza creare artefatti
bordi = ndi.correlate(V_s, kernel_laplaciano, mode='reflect')

# 3. La Maschera di Soglia (IL TRUCCO INGEGNERISTICO)
# Definiamo una soglia sotto la quale consideriamo il segnale come "rumore" e non come "bordo"
soglia_rumore = 0.05 

# Azzeriamo tutti i bordi deboli (il rumore del cielo)
bordi_puliti = np.where(np.abs(bordi) > soglia_rumore, bordi, 0)

# 4. Aggiungiamo i bordi puliti all'immagine originale (Sharpening)
# Moltiplichiamo per un 'fattore_forza' per decidere quanto rendere nitida l'immagine
fattore_forza = 0.8
V_nitido = np.clip(V_s + (bordi_puliti * fattore_forza), 0, 1)

# 5. Ricomponiamo l'immagine finale
y_ultranitida = np.stack((H_s, S_s, V_nitido), axis=2)
x_ultranitida = hsv2rgb(y_ultranitida)

# --- VISUALIZZAZIONE ---
fig, assi = plt.subplots(1, 2, figsize=(16, 8))

assi[0].imshow(x_finale)
assi[0].set_title("Pre-Sharpening (Senza Flare e Alone)")
assi[0].axis('off')

assi[1].imshow(x_ultranitida)
assi[1].set_title("Post-Sharpening (Laplaciano con Soglia)")
assi[1].axis('off')

plt.tight_layout()
plt.show()

io.imsave('Nitida.jpeg', x_ultranitida)