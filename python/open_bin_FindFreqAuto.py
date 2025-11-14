
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
Start_freq = float(file_path_all.split('_')[3])
Step_freq = float(file_path_all.split('_')[2])
Number_of_files = int(file_path_all.split('_')[1])

print("Ouverture de : "+file_path_all)
graph_name = "FindFreq_Auto"

file_path_all = file_path_all[:-1] # Getting the path without the last character (which is supposed to be a number) for looping over all files
progress_bar = tqdm(total=Number_of_files-1, desc="Processing files FID", unit="file") # Initialize the progress bar

for i in range(Number_of_files):
    progress_bar.update(1)
    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=FidNb)
    
    # Affichage du signal accumulé
    plt.figure(1)
    plt.plot(time_array, voltageAcc_array, marker='+', linestyle='-', label=str(i), linewidth=2)
    plt.title(f'{graph_name} - Accumulation de {FidNb}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

### plot FT
progress_bar.close()
progress_bar = tqdm(total=Number_of_files-1, desc="Processing files TF", unit="file")

for i in range(Number_of_files):
    progress_bar.update(1)
    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=FidNb)
    

    freq_ex = Start_freq + Step_freq*i
    

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
    freq = freq + freq_ex
    magnitude = np.abs(fft_values) * 2 / N  # Normalize amplitude

    if i==0:
        freq_all = freq
        tf_sum = magnitude

    g0 = interp1d(freq_all, tf_sum,bounds_error=False,fill_value=0.0)
    freq_all = np.union1d(freq_all, freq)
    g1 = interp1d(freq, magnitude,bounds_error=False,fill_value=0.0)
    g1_values = g1(freq_all)
    g0_values = g0(freq_all)

    tf_sum = g1_values + g0_values 

    plt.figure(2)
    plt.plot(freq, magnitude, label= str(i),marker='x', linestyle='-')
    plt.title("Fourier Transform - " + graph_name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

progress_bar.close()
plt.figure()
max_tf = np.max(tf_sum)
max_freq = freq_all[np.argmax(tf_sum)]
print(f"\033[92m Larmor Frequency : {max_freq} Hz\033[0m")

plt.legend(['Sum TF', f"Max: {max_tf:.2f} at {max_freq:.2f} Hz"], loc='center left', bbox_to_anchor=(1, 0.5))
plt.plot(freq_all, tf_sum, label='Sum TF', marker='x', linestyle='-')
plt.title("Sum Fourier Transform to find freq- " + graph_name)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.legend(loc='center left',title="Larmor Freq ="+str(max_freq))
plt.tight_layout()
plt.grid(True, which='both')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.show()