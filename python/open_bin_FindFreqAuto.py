
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
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import struct
from tqdm import tqdm
import NMR_Library as nmr
import plotly.graph_objects as go

SAMPLING_RATE = 125e+6


root = tk.Tk()
root.withdraw()

file_path_all = filedialog.askopenfilename()
FidNb = -1 #-1 Pour prendre toutes les FID

######### EXTRACTION OF PARAMETERS FROM FILENAME #############
Start_freq = 0 #float(file_path_all.split('_')[3]) - 5000
Step_freq = float(file_path_all.split('_')[2])
Number_of_files = int(file_path_all.split('_')[1])
graphstart = 0.00005 # in ms
graphstop = 0.1 # in ms

print("Ouverture de : "+file_path_all)
graph_name = "FindFreq_Auto"

file_path_all = file_path_all[:-1] # Getting the path without the last character (which is supposed to be a number) for looping over all files
progress_bar = tqdm(total=Number_of_files-1, desc="Processing files FID", unit="file") # Initialize the progress bar

fig1 = go.Figure()
fig2 = go.Figure()

for i in range(Number_of_files):
    progress_bar.update(1)
    file_path = file_path_all + str(i)

    # Opening the binary file
    time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=FidNb)

    #Filtrage :
    fs = 1/((time_array[10]-time_array[0])/10) 
    lowcut = 1000000.0
    highcut = 20000000
    voltageAcc_array_filtered = nmr.butter_bandpass_filter(voltageAcc_array, lowcut, highcut, fs, order=1)

    ## Removing the edges of the FID according to graphstart and graphstop
    dt = np.abs(time_array[0] - time_array[1])   
    voltageAcc_array = voltageAcc_array_filtered[int(graphstart/(1000*dt)):len(voltageAcc_array)-int(graphstop/(1000*dt))]
    time_array = time_array[int(graphstart/(1000*dt)):len(voltageAcc_array)+int(graphstart/(1000*dt))]
    
    # omputing the excitation frequency for this acquisition
    freq_ex = Start_freq + Step_freq*i 

    ## Calcul de la TF
    dt = np.abs(time_array[0] - time_array[1])   
    N = len(voltageAcc_array)
    freq = np.fft.fftfreq(N, dt)
    fft_values = np.fft.fft(voltageAcc_array)
    #freq = freq + freq_ex
    magnitude = np.abs(fft_values) * 2 / N  # Normalize amplitude

    # Summing all TFs together
    if i==0:
        freq_all = freq
        tf_sum = magnitude
    g0 = interp1d(freq_all, tf_sum,bounds_error=False,fill_value=0.0)
    freq_all = np.union1d(freq_all, freq)
    g1 = interp1d(freq, magnitude,bounds_error=False,fill_value=0.0)
    g1_values = g1(freq_all)
    g0_values = g0(freq_all)
    tf_sum = g1_values + g0_values 

    fig1.add_trace(go.Scattergl( # <--- Scattergl est essentiel pour la performance
        x=time_array, 
        y=voltageAcc_array, 
        mode='lines', 
        opacity=1,       # Transparence pour gérer la superposition
        line=dict(width=1), # Ligne fine
        showlegend=False    # Important : désactiver la légende si vous avez bcp de courbes
    ))


    fig2.add_trace(go.Scattergl(
        x=freq, 
        y=magnitude, 
        mode='lines', 
        opacity=1, 
        line=dict(width=1),
        showlegend=False
    ))

progress_bar.close()

max_tf = np.max(tf_sum)
max_freq = freq_all[np.argmax(tf_sum)]
print(f"\033[92m Larmor Frequency : {max_freq} Hz\033[0m")

#plotting the summed TF
# plt.figure(3)
# plt.legend(['Sum TF', f"Max: {max_tf:.2f} at {max_freq:.2f} Hz"], loc='center left', bbox_to_anchor=(1, 0.5))
# plt.plot(freq_all, tf_sum, label='Sum TF', marker='x', linestyle='-')
# plt.title("Sum Fourier Transform to find freq- " + graph_name)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")
# plt.legend(loc='center left',title="Larmor Freq ="+str(max_freq))
# plt.tight_layout()
# plt.grid(True, which='both')
# plt.minorticks_on()
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)
# plt.show()

fig1.update_layout(title="Figure 1")
fig2.update_layout(title="Figure 2")

fig1.show()
fig2.show()