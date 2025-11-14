
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


SAMPLING_RATE = 125e+6


root = tk.Tk()
root.withdraw()

file_path_all = filedialog.askopenfilename()

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def accumulate(voltage_matrix,nb_accumulated):
    dsize = len(voltage_matrix[0])
    if ((nb_accumulated==-1) or (nb_accumulated>=len(voltage_matrix))):
        nb_accumulated = len(voltage_matrix)

    voltage_acc = np.zeros(dsize)

    for i in range(nb_accumulated-1):
        voltage_acc = voltage_acc + voltage_matrix[:][i]
    
    # Calcul de la moyenne et centrage du signal accumulé
    moyenne = np.mean(voltage_acc[int((dsize*0.5)):int((dsize*0.98))])
    voltage_acc = voltage_acc - moyenne
    
    return voltage_acc

def open_file(pathFile_csv, nombre_de_FID):
        voltage_acc = []


        with open(pathFile_csv, 'r', encoding='utf-8') as fichier_:
            lecteur = csv.reader(fichier_)

            # Lire et parser l'entête du fichier
            ligne_entete = next(lecteur)
            dsize = int(ligne_entete[0])
            decimation = int(ligne_entete[1])
            if nombre_de_FID <0 : nombre_de_FID = int(ligne_entete[2])
            gain = float(ligne_entete[3])
            offset = float(ligne_entete[4])
            nb_bits = int(ligne_entete[5])

            # Initialisation
            voltage = [[] for _ in range(nombre_de_FID)]
            mean = []
            voltage_acc = np.zeros(dsize)  # Initialiser au bon format et taille

            # Lecture et conversion des tensions
            for j in range(nombre_de_FID):
                ligne = next(lecteur)

                signal = []

                # for val_bin in ligne:
                #     #val_bin = int(val_bin)
                #     val = convert_to_volt(val_bin, nb_bits, gain, offset)
                #     signal.append(val)  # pas besoin de np.append ici


                for val in ligne:
                    signal.append(float(val))

                # Convertir la ligne en tableau numpy
                signal = np.array(signal)

                #### Calcul de la moyenne et centrage
                #moyenne = np.mean(signal)
                #signal_centre = signal - moyenne

                # Stocker les données
                voltage[j] = signal
                #mean.append(moyenne)                

        # Création du tableau temps
        duree_mesure = (dsize * decimation) / SAMPLING_RATE 
        time = np.linspace(0, duree_mesure, dsize, endpoint=False)

        voltage_acc = accumulate(voltage,nb_accumulated=nombre_de_FID)

        print(f"Fichier {pathFile_csv} lu. {len(voltage)} signaux FID chargés.")
        return time, voltage, voltage_acc

def open_file_bin(pathFile_csv,nombre_de_FID):
        
        "open the file at the given path and give back the numpy vectors time, voltage"
        voltage_acc = []

        with open(pathFile_csv, mode='rb') as file: # b is important -> binary
            # Lire tout le contenu binaire du fichier ouvert (retourne des bytes)
            fileContent = file.read()

            # Les 16 premiers octets correspondent à l'en-tête (4 entiers de 4 octets chacun)
            headerbin = fileContent[0:16]
            # Décomposer ces 16 octets en 4 entiers (iiii)
            header = struct.unpack("iiii", headerbin)

            # Afficher l'en-tête pour debug (ex: (dsize, decimation, nombre_de_FID, ...))
            #print(header)
            # Récupérer les valeurs utiles depuis le tuple d'entiers
            dsize           = header[0]  # nombre d'échantillons par FID
            decimation      = header[1]  # facteur de décimation utilisé lors de l'acquisition
            nombre_de_FID   = header[2]  # nombre de FID présents dans le fichier

            # Initialisation : créer une liste contenant une sous-liste vide par FID
            # (sera remplie plus tard avec les tableaux numpy de chaque signal)
            voltage = [[] for _ in range(nombre_de_FID)]
            mean = []
            voltage_acc = np.zeros(dsize)  # Initialiser au bon format et taille

            # Lecture et conversion des tensions
            for j in range(nombre_de_FID):
                # Extract bytes for this FID (each sample is int16 -> 2 bytes)
                start = 16 + j * dsize*2  # 16 bytes for header + offset for FID (*2 because int16=2 bytes)
                end = start + dsize*2
                if end > len(fileContent):
                    raise ValueError(f"Unexpected file size: need bytes {start}:{end}, file has {len(fileContent)} bytes")
                
                values = struct.unpack("<" + "h" * dsize, fileContent[start:end])

                signal = np.array(values,dtype=np.int16)
                signal = signal.astype(np.float32)/8190 # Conversion en volt PIN LOW
                # Stocker les données
                voltage[j] = signal

                # Accumulation du signal total
                voltage_acc += signal

        # Création du tableau temps
        duree_mesure = (dsize * decimation) / SAMPLING_RATE 
        time = np.linspace(0, duree_mesure, dsize, endpoint=False)

        # Calcul de la moyenn et centrage du signal accumulé
        moyenne = np.mean(voltage_acc)
        voltage_acc = voltage_acc - moyenne

        #print(f"Fichier {pathFile_csv} lu. {len(voltage)} signaux FID chargés.")
        return time, voltage, voltage_acc

def plot_acc(graph_name, time_axis, voltage_matrix):
    
    amountFID = len(voltage_matrix)

    plt.figure(figsize=(12, 7))

    # Affichage des courbes FID
    for w in range(amountFID):
        plt.plot(time_axis, voltage_matrix[w], marker='+', linestyle='-', label=f'FID {w+1}')

    # Affichage du signal accumulé
    ###############################
    ####----> plt.plot(time_axis, voltage_accumulated_axis, marker='+', linestyle='-', label='Total', linewidth=2, color='black')
    ##############################
    
    # Mise en forme du graphique
    plt.title(f'Superposition - {graph_name}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(False, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    #plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_single(graph_name, time_axis, voltage_matrix, FID_nb):
    
    amountFID = len(voltage_matrix)

    plt.figure(figsize=(12, 7))

    # Affichage de la premiere FID
    plt.plot(time_axis, voltage_matrix[FID_nb-1], marker='+', linestyle='-')

    # Affichage du signal accumulé
    ###############################
    ####----> plt.plot(time_axis, voltage_accumulated_axis, marker='+', linestyle='-', label='Total', linewidth=2, color='black')
    ##############################
    
    # Mise en forme du graphique
    plt.title(f'Single FID - {graph_name}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(False, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    #plt.legend(loc='upper right')

def plot_acc_only(graph_name, time_axis, voltage_matrix, amountFID):
    if amountFID <0 :
        amountFID = len(voltage_matrix)

    plt.figure(figsize=(12, 7))
    voltage_accumulated_axis = accumulate(voltage_matrix, nb_accumulated=amountFID)

    # Affichage du signal accumulé
    plt.plot(time_axis, voltage_accumulated_axis, marker='+', linestyle='-', label='Total', linewidth=2, color='black')

    # Mise en forme du graphique
    plt.title(f'{graph_name} - Accumulation de {amountFID}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    #plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.show(block=True)

def subpolts_acc(graph_name, time_axis, voltage_matrix, nb_of_accumulated):
    # Supposons que voltage_matrix et time_axis soient déjà définis
    amountFID = len(voltage_matrix)

    # Créer une figure et un ensemble de sous-graphes
    fig, axs = plt.subplots(amountFID, 1, figsize=(12, 7 * amountFID), sharex=True)

    # Affichage des courbes FID sur des sous-graphes séparés
    for w in range(amountFID):
        axs[w].plot(time_axis, voltage_matrix[w], marker='+', linestyle='-', label=f'FID {w+1}')
        axs[w].set_title(f'FID {w+1}')
        axs[w].set_ylabel('Tension (V)')
        axs[w].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[w].legend(loc='upper right')

    # Mise en forme globale du graphique
    fig.suptitle(f'Superposition - {graph_name}', y=1.02)  # y est légèrement ajusté pour éviter le chevauchement
    plt.xlabel('Temps (s)')
    plt.tight_layout()

def plot_fourier_transform(graph_name, time, voltage):
    
    # Ensure time and voltage are numpy arrays
    time = np.array(time)
    voltage = np.array(voltage)

    # Sampling interval and frequency
    dt = time[1] - time[0]
    fs = 1 / dt

    # Compute FFT
    N = len(voltage)
    fft_values = np.fft.fft(voltage)
    freq = np.fft.fftfreq(N, dt)

    # Keep only the positive frequencies
    #idx = np.where(freq >= 0)
    #freq = freq[idx]
    magnitude = np.abs(fft_values) * 2 / N  # Normalize amplitude

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitude)
    plt.title("Fourier Transform - " + graph_name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    #plt.legend()
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    #plt.tight_layout()
    #plt.show(block=True)

#############################################################

FidNb = -1 #-1 Pour prendre toutes les FID

Number_of_files = int(file_path_all.split('_')[1])
print(Number_of_files)

print("Ouverture de : "+file_path_all)
graph_name = "FindP90_Auto"

file_path_all = file_path_all[:-1]

plt.figure()
progress_bar = tqdm(total=Number_of_files-1, desc="Processing files", unit="file")
for i in range(Number_of_files):
    # Progress bar using tqdm
    
    progress_bar.update(1)
    
    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = open_file_bin(file_path, nombre_de_FID=FidNb)
     
    #print("Affichage de la FID...")

    # Affichage du signal accumulé
    plt.plot(time_array, voltageAcc_array, marker='+', linestyle='-', label=str(i), linewidth=2)
    
    # Mise en forme du graphique
    plt.title(f'{graph_name} - Accumulation de {FidNb}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()

### plot FT
plt.figure()
# Initialisation
max_magnitude = 0
freq_at_max_magnitude = 0

progress_bar.close()
progress_bar = tqdm(total=Number_of_files-1, desc="Processing files", unit="file")
for i in range(Number_of_files):
    progress_bar.update(1)

    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = open_file_bin(file_path, nombre_de_FID=FidNb)
        
    #voltageAcc_array = voltageAcc_array[300:] 

    #Filtrage :
    fs = 1/((time_array[10]-time_array[0])/10) 
    lowcut = 100.0
    highcut = 2000.0
    voltageAcc_array_filtered = butter_bandpass_filter(voltageAcc_array, lowcut, highcut, fs, order=3)
    
    ## Calcul de la TF
    
    dt = np.abs(time_array[0] - time_array[1])
    N = len(voltageAcc_array)
    freq = np.fft.fftfreq(N, dt)
    
    fft_values = np.fft.fft(voltageAcc_array)
    magnitude = np.abs(fft_values) * 2 / N  # Normalize amplitude
    
    max_magnitude_current = np.max(magnitude)
    if np.max(magnitude) > max_magnitude:
        index_max = i
        max_magnitude = np.max(magnitude)

    #print("Affichage de la TF...")
    plt.plot(freq, magnitude, label= str(i),marker='x', linestyle='-')
    plt.title("Fourier Transform - " + graph_name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

progress_bar.close()
step_excitation = 2e-6 #float(file_path_all.split('_')[3])
excitation_initial = float(file_path_all.split('_')[2])
excitation_duration_seconds_best = (index_max)*step_excitation + excitation_initial
print("Durée d'excitation optimale trouvée : "+str(excitation_duration_seconds_best)+" s pour un max de TF de "+str(max_magnitude)+" à l'index "+str(index_max))
plt.show()

