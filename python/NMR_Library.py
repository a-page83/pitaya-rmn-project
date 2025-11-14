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

SAMPLING_RATE = 125e+6
PORT = 22
USERNAME = "root"
PASSWORD = "root"
REMOTE_PATH = "Pitaya-Tests/" 
REMOTE_FOLDER = "Pitaya-Tests"
SAMPLING_RATE = 125e+6

hostName = "169.254.215.235"
port = 22

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

        #print(f"Fichier {pathFile_csv} lu. {len(voltage)} signaux FID chargés.")
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

def create_file_wdate(nameFile):
    # Créer le dossier local avec timestamp
    now = datetime.datetime.now()
    name_local_file = f"python/mesures/{nameFile}_{now.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(name_local_file, exist_ok=True)
    
    return name_local_file

def run_acquisition_command(samplesNb, dec,FidNb, FileName, larmorFrequency, excitationDuration, delayRepeat): #voir si file path marche    ùù
    
    filePath = "mesures/" + FileName
    command = f"cd {REMOTE_FOLDER} && ./Acquisition_axi.exe {samplesNb} {dec} {FidNb} {filePath} {larmorFrequency} {excitationDuration} {delayRepeat}"
    stdin, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode()
    errors = stderr.read().decode()
    #print(f"[CMD-SSH] {command}")
    #if output:
        #print("[OUTPUT SSH]\n", output)
    
    if errors:
        print("[ERROR SHH]\n", errors)

def download_file_sftp(nameRemoteFile,nameRemoteFolder,nameLocalFolder):
    """Télécharge le fichier CSV via SFTP"""
    
    # Téléchargement du fichier
    remote_path = REMOTE_PATH + nameRemoteFolder+'/' + nameRemoteFile
    local_path = os.path.join(nameLocalFolder, nameRemoteFile)
    
    try:
        sftp.get(remote_path, local_path)
        #print(f"Téléchargé: {nameRemoteFile}")
    except FileNotFoundError:
        print(f"Fichier non trouvé: {remote_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement de {nameRemoteFile}: {e}")

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

