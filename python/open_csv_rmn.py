
import csv
import matplotlib.pyplot as plt
import numpy as np
import datetime
import paramiko
import os
import time
from scipy import signal
import datetime
import tkinter as tk
from tkinter import filedialog
import NMR_Library as nmr
SAMPLING_RATE = 125e+6



root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


FidNb = 10 #-1 Pour prendre toutes les FID


print("Ouverture de : "+file_path)
graph_name = "FID Antenne 0.8mm"##file_path[65:len(file_path)] ## On prend le nom de la figure


time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_csv(file_path, nombre_de_FID=FidNb)
print("Affichage de la FID...")

nmr.plot_acc_only(graph_name, time_array[200:], np.array(voltage_array_matrix)[:,200:], amountFID=FidNb)

""" 
if(len(voltage_array_matrix)>10):
    plot_acc_only(graph_name = graph_name, time_axis= time_array, voltage_matrix= voltage_array_matrix,amountFID=10)

 """
nmr.plot_single(graph_name, time_array, voltage_array_matrix, FID_nb=1)
print("Affichage de la Transformée de Fourrier...")
nmr.plot_fourier_transform(graph_name,time_array,voltageAcc_array[200:])


# Paramètres du signal
t = time_array
fs = 1/((time_array[10]-time_array[0])/10)           #SAMPLING_RATE/decimation  # Fréquence d'échantillonnage (Hz)


freq_basse = 500   # Fréquence basse (Hz)
freq_haute = 1500  # Fréquence haute (Hz)
ordre = 1          # Ordre du filtre

signal_filtre = nmr.butter_bandpass_filter(voltageAcc_array[200:], freq_basse, freq_haute, fs, ordre)
plt.figure(figsize=(10, 4))
plt.plot(time_array[200:], signal_filtre)
plt.title("Signal filtre")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

signal_filtre2 = nmr.butter_bandpass_filter(voltageAcc_array[1,:], freq_basse, freq_haute, fs, ordre)
plt.figure(figsize=(10, 4))
plt.plot(time_array[200:], signal_filtre2[200:])
plt.title("Signal filtre")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

graph_name_tf_filtré2 = graph_name + " - filtré2"
nmr.plot_fourier_transform(graph_name=graph_name_tf_filtré2, time=time_array, voltage=signal_filtre)

nmr.plot_fourier_transform(graph_name=graph_name_tf_filtré2, time=time_array[200:], voltage=signal_filtre2[200:])
plt.show()

