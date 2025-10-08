
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

SAMPLING_RATE = 125e+6



root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


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

#############################################################""

FidNb = -1 #-1 Pour prendre toutes les FID

print("Ouverture de : "+file_path)
graph_name = input("Graph name : ")


time_array, voltage_array_matrix, voltageAcc_array = open_file(file_path, nombre_de_FID=FidNb)
print("Affichage de la FID...")

plot_acc_only(graph_name, time_array, voltage_array_matrix, amountFID=FidNb)

if(len(voltage_array_matrix)>10):
    plot_acc_only(graph_name = graph_name, time_axis= time_array, voltage_matrix= voltage_array_matrix,amountFID=10)

plot_single(graph_name, time_array, voltage_array_matrix, FID_nb=1)
print("Affichage de la Transformée de Fourrier...")
plot_fourier_transform(graph_name,time_array ,voltageAcc_array)


# Paramètres du signal
t = time_array
fs = 1/((time_array[10]-time_array[0])/10)           #SAMPLING_RATE/decimation  # Fréquence d'échantillonnage (Hz)


freq_basse = 500   # Fréquence basse (Hz)
freq_haute = 10000  # Fréquence haute (Hz)
ordre = 3         # Ordre du filtre

butter = signal.butter(ordre,[freq_basse,freq_haute], 
                    btype='bandpass', fs=fs, output='sos')

signal_filtre = signal.sosfilt(butter, voltageAcc_array)


plt.figure(figsize=(10, 4))
plt.plot(time_array, signal_filtre)
plt.title("Signal filtre")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

graph_name_tf_filtré = graph_name + " - filtré"
plot_fourier_transform(graph_name=graph_name_tf_filtré, time=time_array, voltage=signal_filtre)



plt.show()

