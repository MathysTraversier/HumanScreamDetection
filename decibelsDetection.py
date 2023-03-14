import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import time

tinit = time.time()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

SEUIL = 50

moyenneDb = 0
echantillon = []

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Création du graphe
fig, ax = plt.subplots()
barplot = ax.bar(['dB'], [0], width=0.5)
ax.set_ylim(0,100)

# Boucle principale
while True:
    # Lecture des données audio
    data = stream.read(CHUNK)
    # Conversion des données audio en un tableau NumPy
    data_np = np.frombuffer(data, dtype=np.int16)
    # Calcul du niveau de décibels
    db = 20 * np.log10(np.abs(data_np).max() / 32767) + 70
    t=time.time()
    #Garder une fenêtre d'échantillon de 2 minutes
    if(t - tinit > 120):
        echantillon.remove(1)
    echantillon.append(db)
    moyenneDb = statistics.mean(echantillon)
    if(db > moyenneDb*(1+SEUIL/100)):
        print("Attention, le seuil a été dépassé !")
    # Mise à jour du barplot
    barplot[0].set_height(db)
    # Affichage du graphe
    plt.draw()
    plt.pause(0.001)