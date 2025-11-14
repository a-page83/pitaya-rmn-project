
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

SAMPLING_RATE = 125e+6

root = tk.Tk()
root.withdraw()
file_path_all = filedialog.askopenfilename()

FidNb = -1 #-1 Pour prendre toutes les FID

######### EXTRACTION OF PARAMETERS FROM FILENAME #############
Number_of_files = int(file_path_all.split('_')[1])
excitation_initial = float(file_path_all.split('_')[2])
step_excitation = float(file_path_all.split('_')[3])

print(Number_of_files)
print("Ouverture de : "+file_path_all)
graph_name = "FindP90_Auto"

file_path_all = file_path_all[:-1]


progress_bar = tqdm(total=Number_of_files-1, desc="Processing files", unit="file")
for i in range(Number_of_files):
    # Progress bar using tqdm
    
    progress_bar.update(1)
    
    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=FidNb)
     
    plt.figure(1) #Figure 1 : plot time domain
    plt.plot(time_array, voltageAcc_array, marker='+', linestyle='-', label=str(i), linewidth=2)
    plt.title(f'{graph_name} - Accumulation de {FidNb}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
progress_bar.close()

# Initialisation of the loop to find the maximum
max_magnitude = 0
freq_at_max_magnitude = 0
progress_bar = tqdm(total=Number_of_files-1, desc="Processing files", unit="file")
for i in range(Number_of_files):
    progress_bar.update(1)

    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=FidNb)
    
    #Filtrage :
    fs = 1/((time_array[10]-time_array[0])/10) 
    lowcut = 100.0
    highcut = 2000.0
    voltageAcc_array_filtered = nmr.butter_bandpass_filter(voltageAcc_array, lowcut, highcut, fs, order=3)
    
    ## Calcul de la TF
    dt = np.abs(time_array[0] - time_array[1])
    N = len(voltageAcc_array)
    freq = np.fft.fftfreq(N, dt)
    fft_values = np.fft.fft(voltageAcc_array)
    magnitude = np.abs(fft_values) * 2 / N  # Normalize amplitude
    
    #### Find max magnitude and corresponding frequency
    max_magnitude_current = np.max(magnitude)
    if np.max(magnitude) > max_magnitude:
        index_max = i
        max_magnitude = np.max(magnitude)

    plt.figure(2) # Figure 2 : plot frequency domain
    plt.plot(freq, magnitude, label= str(i),marker='x', linestyle='-')
    plt.title("Fourier Transform - " + graph_name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
progress_bar.close()

excitation_duration_seconds_best = (index_max)*step_excitation + excitation_initial
print("Durée d'excitation optimale trouvée : "+str(excitation_duration_seconds_best)+" s pour un max de TF de "+str(max_magnitude)+" à l'index "+str(index_max))
plt.show()

