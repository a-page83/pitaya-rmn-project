import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import datetime
import paramiko
import os
import time
import datetime
from scipy import signal
import struct
import NMR_Library as nmr
from tqdm import tqdm
import sys


PORT = 22
USERNAME = "root"
PASSWORD = "root"
REMOTE_PATH = "Pitaya-Tests/" 
REMOTE_FOLDER = "Pitaya-Tests"
SAMPLING_RATE = 125e+6


### Connection SSH to Pitaya
hostName = "169.254.215.235"
port = 22

nmr.client = paramiko.SSHClient()
nmr.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
nmr.client.connect(hostName, username=USERNAME, password=PASSWORD,port=port)
print(f"[INFO] Connexion SSH établie à {hostName}")

transport = paramiko.Transport((hostName, PORT))
transport.connect(username=USERNAME, password=PASSWORD)
nmr.sftp = paramiko.SFTPClient.from_transport(transport)
print(f"[INFO] Connexion SFTP établie à {hostName}")

name_of_device = "Pure Device"
type_of_antenna = "0.8"
experiment_name = "test.bin"

########################################### PARAMETERS TO MODIFY #####################################################

sample_Amount = 424*10              #Number of points measured (MAX = 524288) must be a multiple of 2
decimation = 1216*2                   #Decimation
aquisiton_Amount = 1             #Nubmer of acquisitons
larmor_Frequency_Hertz = 24.3425e+6 - 100000    #Larmor frequency # 24.3417e+6   
excitation_duration_seconds = 50e-6     #Excitation time
delay_repeat_useconds = 0.1e+6
experiment_name_all_files = "Stepfreq"

Number_of_files = 10
step_excitation_duration_seconds = 5e-6

##"###################################### END OF PARAMETERS TO MODIFY ##################################################

total_time = (sample_Amount * decimation)/SAMPLING_RATE
nb_cycles = larmor_Frequency_Hertz*excitation_duration_seconds
#print(f"nb cycles burst : {nb_cycles}")
temps_secondes = (total_time+delay_repeat_useconds*1e-6)*aquisiton_Amount

print(f"Temps mesuré : {total_time}s")
print(f"Temps total :"+str(datetime.timedelta(seconds=temps_secondes*Number_of_files)))
print("Name of file : " + experiment_name_all_files)

### Display parameters and ask for confirmation
try:
    if not sys.stdin.isatty():
        print("No interactive terminal detected. Aborting.")
        sys.exit(1)
    resp = input("Continue and start acquisitions? [y/N]: ").strip().lower()
except (EOFError, KeyboardInterrupt):
    print("\nAborted.")
    sys.exit(1)

if resp not in ("y", "yes"):
    print("Operation cancelled by user.")
    sys.exit(0)
################ FOLDER NAME IS MADE TO WORK WITH OPEN_BIN_FIND_P90 ##############
nameLocalFolder = nmr.create_file_wdate("FindP90Auto_"+str(Number_of_files)+"_"+str(excitation_duration_seconds))
##################################################################################
progress_bar = tqdm(total=Number_of_files-1, desc="Processing Acquisitions to find Frequency", unit="file")
max_magnitude = 0

for i in range(Number_of_files):
    progress_bar.update(1)
    experiment_name = experiment_name_all_files+str(i)
    nmr.run_acquisition_command(sample_Amount, 
                                decimation, 
                                aquisiton_Amount, 
                                experiment_name, 
                                larmor_Frequency_Hertz, 
                                excitation_duration_seconds,
                                delay_repeat_useconds)

    nameRemoteFile = experiment_name
    nameRemoteFolder = "mesures"
    nmr.download_file_sftp(nameRemoteFile,nameRemoteFolder,nameLocalFolder)

    file_path_all = nameLocalFolder + "/" + experiment_name_all_files
    file_path = file_path_all + str(i)
    time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=Number_of_files)
     

    
    ###########################                 Filtrage
    fs = 1/((time_array[10]-time_array[0])/10) 
    lowcut = 100.0
    highcut = 2000.0
    voltageAcc_array_filtered = nmr.butter_bandpass_filter(voltageAcc_array, lowcut, highcut, fs, order=3)
    

    #######################################     Calcul de la TF
    dt = np.abs(time_array[0] - time_array[1])
    N = len(voltageAcc_array)
    freq = np.fft.fftfreq(N, dt)
    
    fft_values = np.fft.fft(voltageAcc_array)
    magnitude = np.abs(fft_values) * 2 / N
    

    ########################################    Recherche du maximum parmis les acquisitions
    max_magnitude_current = np.max(magnitude)
    if np.max(magnitude) > max_magnitude:
        index_max = i
        max_magnitude = np.max(magnitude)


    
    ######################################      Affichage du signal accumulé
    plt.figure(1)
    plt.plot(time_array, voltageAcc_array, marker='+', linestyle='-', label=str(i), linewidth=2)  
    plt.title(f'"FID Find Freq" - Accumulation de {Number_of_files}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)



    ####################################        Affichage de la TF
    plt.figure(2)
    plt.plot(freq, magnitude, label= str(i),marker='x', linestyle='-')
    plt.title("Fourier Transform - " + "FID Find Freq")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    excitation_duration_seconds += step_excitation_duration_seconds

print("Acquisitions et affichage terminés. Chemin du fichier :"+str(file_path_all))
plt.show()

