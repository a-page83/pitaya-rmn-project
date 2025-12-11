import csv                       # reading CSV measurement files
import matplotlib.pyplot as plt  # plotting helpers
import numpy as np               # numeric arrays and math
import datetime                  # timestamps for saved folders
import paramiko                  # SSH / SFTP (used elsewhere in project)
import os                        # filesystem operations
import time                      # time helpers (not heavily used here)
from scipy import signal         # signal processing helpers (some unused imports remain)
import datetime                  # duplicate import (harmless, but redundant)
import tkinter as tk             # simple file dialog GUI
from tkinter import filedialog
from scipy.signal import freqz   # not used in current code, left for future filtering work
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import struct                    # binary unpacking for .bin reader
import plotly.graph_objects as go

# Constants used across the module
SAMPLING_RATE = 125e+6          # device sampling rate in Hz (important for time axis)
PORT = 22
USERNAME = "root"               # credential present here; avoid committing secrets in real projects
PASSWORD = "root"
REMOTE_PATH = "Pitaya-Tests/" 
REMOTE_FOLDER = "Pitaya-Tests"
SAMPLING_RATE = 125e+6          # repeated definition (same value); redundant but harmless

# Default remote host settings (used by functions that create SSH/SFTP clients elsewhere)
hostName = "169.254.215.235"
port = 22

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create bandpass IIR filter coefficients using a Butterworth design.
    Returns (b, a) filter coefficients.
    Parameters:
      lowcut, highcut : passband edges in Hz
      fs              : sampling frequency in Hz
      order           : filter order
    """
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to `data`.
    Returns the filtered signal (same shape as input).
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def accumulate(voltage_matrix,nb_accumulated):
    """
    Sum (accumulate) a number of FID traces from voltage_matrix.
    Parameters:
      voltage_matrix   : list or array-like of 1D arrays, one per FID
      nb_accumulated   : number of FIDs to accumulate; if -1 or >= available, all are used
    Returns:
      voltage_acc : 1D numpy array containing the accumulated and baseline-centered signal
    Notes:
      - voltage_matrix is expected to be indexable like voltage_matrix[0], voltage_matrix[1], ...
      - There is a potential bug: the loop uses `voltage_matrix[:][i]`, which is equivalent to
        `voltage_matrix[i]` for lists but can be confusing for numpy arrays. It currently sums
        up to nb_accumulated-1 (exclusive of the last index) because range(nb_accumulated-1)
        is used; that may be unintended.
    """
    dsize = len(voltage_matrix[0])
    if ((nb_accumulated==-1) or (nb_accumulated>=len(voltage_matrix))):
        nb_accumulated = len(voltage_matrix)

    voltage_acc = np.zeros(dsize)

    # Accumulate the requested number of FIDs.
    # NOTE: range(nb_accumulated-1) excludes the last item — likely an off-by-one bug.
    for i in range(nb_accumulated-1):
        voltage_acc = voltage_acc + voltage_matrix[:][i]
    
    # Compute mean on a tail region and subtract to center the baseline.
    # Uses indices from 50% to 98% of the record for baseline estimate.
    moyenne = np.mean(voltage_acc[int((dsize*0.5)):int((dsize*0.98))])
    voltage_acc = voltage_acc - moyenne
    
    return voltage_acc

def open_file_csv(pathFile_csv, nombre_de_FID):
    """
    Read a CSV-format measurement file and return time axis, list of FID arrays, and accumulated signal.
    CSV header expected: [dsize, decimation, nombre_de_FID, gain, offset, nb_bits]
    Parameters:
      pathFile_csv   : path to CSV file
      nombre_de_FID  : number of FIDs to read; if < 0, use value from file header
    Returns:
      time           : 1D numpy array of time values (seconds)
      voltage        : list of numpy arrays, one per FID (float)
      voltage_acc    : accumulated and baseline-centered numpy array
    """
    voltage_acc = []

    with open(pathFile_csv, 'r', encoding='utf-8') as fichier_:
        lecteur = csv.reader(fichier_)

        # Read and parse the header
        ligne_entete = next(lecteur)
        dsize = int(ligne_entete[0])
        decimation = int(ligne_entete[1])
        if nombre_de_FID <0 : nombre_de_FID = int(ligne_entete[2])
        gain = float(ligne_entete[3])
        offset = float(ligne_entete[4])
        nb_bits = int(ligne_entete[5])

        # Initialize containers
        voltage = [[] for _ in range(nombre_de_FID)]
        mean = []
        voltage_acc = np.zeros(dsize)  # accumulator array

        # Read each FID line and convert to floats
        for j in range(nombre_de_FID):
            ligne = next(lecteur)

            signal = []

            # The file already contains float values (as strings) so convert directly.
            for val in ligne:
                signal.append(float(val))

            # Convert list to numpy array for later processing
            signal = np.array(signal)

            # Store the FID
            voltage[j] = signal

    # Build time axis using SAMPLING_RATE and decimation
    duree_mesure = (dsize * decimation) / SAMPLING_RATE 
    time = np.linspace(0, duree_mesure, dsize, endpoint=False)

    # Accumulate all FIDs (nombre_de_FID) and return results
    voltage_acc = accumulate(voltage,nb_accumulated=nombre_de_FID)

    return time, voltage, voltage_acc

def open_file_bin(pathFile_bin,nombre_de_FID):
    """
    Read a binary-format measurement file and return time axis, list of FID arrays, and accumulated signal.
    Binary header: first 16 bytes = 4 x 4-byte little-endian ints (dsize, decimation, nombre_de_FID, ...)
    Following data: signed 16-bit int samples for each FID sequentially.
    Parameters:
      pathFile_bin   : path to binary file
      nombre_de_FID  : ignored; the value in the file header is used
    Returns:
      time, voltage (list of arrays), voltage_acc (accumulated & baseline-centered)
    """
    voltage_acc = []

    with open(pathFile_bin, mode='rb') as file: # open in binary mode
        fileContent = file.read()

        # Parse 16-byte header (4 little-endian ints)
        headerbin = fileContent[0:16]
        header = struct.unpack("iiii", headerbin)

        # Extract fields from header
        dsize           = header[0]  # samples per FID
        decimation      = header[1]  # decimation factor
        nombre_de_FID   = header[2]  # number of FIDs in file

        # Prepare containers
        voltage = [[] for _ in range(nombre_de_FID)]
        mean = []
        voltage_acc = np.zeros(dsize)  # accumulator

        # For each FID, unpack int16 samples and scale to volts
        for j in range(nombre_de_FID):
            # Compute byte ranges for this FID (int16 -> 2 bytes per sample)
            start = 16 + j * dsize*2
            end = start + dsize*2
            if end > len(fileContent):
                raise ValueError(f"Unexpected file size: need bytes {start}:{end}, file has {len(fileContent)} bytes")
            
            # Unpack little-endian signed 16-bit integers
            values = struct.unpack("<" + "h" * dsize, fileContent[start:end])

            signal = np.array(values,dtype=np.int16)
            # Convert integer samples to voltages (scaling chosen for this device)
            signal = signal.astype(np.float32)/8190  # PIN LOW convention in this project

            # Store and accumulate
            voltage[j] = signal
            voltage_acc += signal

    # Time axis from dsize/decimation/SAMPLING_RATE
    duree_mesure = (dsize * decimation) / SAMPLING_RATE 
    time = np.linspace(0, duree_mesure, dsize, endpoint=False)

    # Subtract mean baseline from accumulated signal
    moyenne = np.mean(voltage_acc)
    voltage_acc = voltage_acc - moyenne

    return time, voltage, voltage_acc

def create_file_wdate(nameFile):
    """
    Create a local measurements folder with a timestamped name under pitaya-rmn-project/python/mesures/.
    Returns the created folder path.
    """
    now = datetime.datetime.now()
    projectDir = os.path.dirname(os.path.abspath(__file__))
    path_local_file = f"mesures/{nameFile}_{now.strftime('%Y%m%d_%H%M%S')}"
    abs_path_local_file = os.path.join(projectDir, path_local_file)
    os.makedirs(abs_path_local_file, exist_ok=True)
    return abs_path_local_file

def run_acquisition_echo_command(samplesNb, dec,FidNb, FileName, larmorFrequency, excitationDuration, delayRepeat, echoTime, verbose=False):
    """
    Compose and run a remote acquisition command via an existing SSH client.
    This function expects a global `client` paramiko.SSHClient to be already connected.
    Parameters match the remote Acquisition_axi.exe command-line arguments.
    larmorfrequency in Hz
    excitation duration in seconds
    delayRepat in micro_seconds
    echoTime in micro_seconds
    """
    filePath = "mesures/" + "mesure.bin"
    command = f"cd {REMOTE_FOLDER} && ./Acquisition_echo.exe {samplesNb} {dec} {FidNb} {filePath} {larmorFrequency} {excitationDuration} {delayRepeat} {echoTime}"
    print(command)
    stdin, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode()
    errors = stderr.read().decode()
    # Errors are printed; output currently not used elsewhere.
    if verbose == True : 
        print(output)
    
    if errors:
        print("[ERROR SHH]\n", errors)

def run_acquisition_fid_command(samplesNb, dec,FidNb, FileName, larmorFrequency, excitationDuration, delayRepeat, verbose=False):
    """
    Compose and run a remote acquisition command via an existing SSH client.
    This function expects a global `client` paramiko.SSHClient to be already connected.
    Parameters match the remote Acquisition_axi.exe command-line arguments.
    """
    filePath = "mesures/" + "mesure.bin"
    command = f"cd {REMOTE_FOLDER} && ./Acquisition_axi.exe {samplesNb} {dec} {FidNb} {filePath} {larmorFrequency} {excitationDuration} {delayRepeat}"
    stdin, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode()
    errors = stderr.read().decode()
    if verbose == True : 
        print(output)
    if errors:
        print("[ERROR SHH]\n", errors)

def download_file_sftp(nameLocalFile,nameRemoteFolder,nameLocalFolder):
    """
    Download a remote file using an existing global SFTP client `sftp`.
    Parameters:
        nameLocalFile    : filename on the local side
        nameRemoteFolder : remote subfolder under REMOTE_PATH
        nameLocalFolder  : local directory to save into
    To work the paramiko SFTP client `sftp` must be already connected.
    to do so, use the all the connexion lines with the nmr. before
    example : nmr.sftp = paramiko.SFTPClient.from_transport(transport)

    """

        
    remote_path = REMOTE_PATH + nameRemoteFolder+'/' + "mesure.bin"
    local_path = os.path.join(nameLocalFolder, nameLocalFile)
    
    try:
        sftp.get(remote_path, local_path)
    except FileNotFoundError:
        print(f"Fichier non trouvé: {remote_path}")
    except Exception as e:
        print(f"Erreur lors du téléchargement de {nameLocalFile}: {e}")

def plot_acc(graph_name, time_axis, voltage_matrix):
    """
    Plot all FIDs overlayed. The function currently does not compute or accept
    the accumulated trace as an argument; placeholder lines show where to add it.
    """
    amountFID = len(voltage_matrix)
    plt.figure(figsize=(12, 7))

    # Plot each FID
    for w in range(amountFID):
        plt.plot(time_axis, voltage_matrix[w], marker='+', linestyle='-', label=f'FID {w+1}')

    # Placeholder for plotting accumulated signal (not provided to this function)
    ###############################
    ####----> plt.plot(time_axis, voltage_accumulated_axis, marker='+', linestyle='-', label='Total', linewidth=2, color='black')
    ##############################
    
    plt.title(f'Superposition - {graph_name}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(False, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.show()

def plot_single(graph_name, time_axis, voltage_matrix, FID_nb):
    """
    Plot a single FID (1-based index).
    """
    amountFID = len(voltage_matrix)
    plt.figure(figsize=(12, 7))

    # Plot requested FID (convert 1-based FID_nb to 0-based index)
    plt.plot(time_axis, voltage_matrix[FID_nb-1], marker='+', linestyle='-')

    # Placeholder for plotting accumulated signal (not provided to this function)
    ###############################
    ####----> plt.plot(time_axis, voltage_accumulated_axis, marker='+', linestyle='-', label='Total', linewidth=2, color='black')
    ##############################
    
    plt.title(f'Single FID - {graph_name}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(False, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # plt.legend can be enabled if needed

def plot_acc_only(graph_name, time_axis, voltage_matrix, amountFID):
    """
    Compute and plot only the accumulated (summed) signal from the matrix of FIDs.
    If amountFID < 0, all FIDs are used.
    """
    if amountFID <0 :
        amountFID = len(voltage_matrix)

    plt.figure(figsize=(12, 7))
    voltage_accumulated_axis = accumulate(voltage_matrix, nb_accumulated=amountFID)

    plt.plot(time_axis, voltage_accumulated_axis, marker='+', linestyle='-', label='Total', linewidth=2, color='black')

    plt.title(f'{graph_name} - Accumulation de {amountFID}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Tension (V)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    # plt.show is intentionally commented in some contexts to allow notebook control

def subpolts_acc(graph_name, time_axis, voltage_matrix, nb_of_accumulated):
    """
    Plot each FID on its own subplot vertically stacked.
    nb_of_accumulated parameter is currently unused inside the function (kept for interface compatibility).
    """
    amountFID = len(voltage_matrix)

    fig, axs = plt.subplots(amountFID, 1, figsize=(12, 7 * amountFID), sharex=True)

    for w in range(amountFID):
        axs[w].plot(time_axis, voltage_matrix[w], marker='+', linestyle='-', label=f'FID {w+1}')
        axs[w].set_title(f'FID {w+1}')
        axs[w].set_ylabel('Tension (V)')
        axs[w].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[w].legend(loc='upper right')

    fig.suptitle(f'Superposition - {graph_name}', y=1.02)
    plt.xlabel('Temps (s)')
    plt.tight_layout()

def plot_fourier_transform(graph_name, time, voltage):
    """
    Compute and plot a simple (unshifted) FFT of a single voltage trace.
    - time : time axis (1D)
    - voltage : corresponding samples (1D)
    The plot shows both positive and negative frequencies unless further filtered.
    """
    time = np.array(time)
    voltage = np.array(voltage)

    # Sampling interval and frequency
    dt = time[1] - time[0]
    fs = 1 / dt

    # FFT
    N = len(voltage)
    fft_values = np.fft.fft(voltage)
    freq = np.fft.fftfreq(N, dt)

    # Magnitude (normalized)
    magnitude = np.abs(fft_values) * 2 / N

    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitude)
    plt.title("Fourier Transform - " + graph_name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    # Note: consider plotting only positive frequencies (freq >= 0) for clarity.

def open_file_dialog():
    """
    Open a file-selection dialog and return the selected path as a string.
    Useful for interactive selection of CSV or BIN files in a desktop environment.
    """
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename()
    return file_path

def plot_fourier_transform_plotly(graph_name, time, voltage):
    """
    Compute and plot a simple (unshifted) FFT of a single voltage trace.
    - time : time axis (1D)
    - voltage : corresponding samples (1D)
    The plot shows both positive and negative frequencies unless further filtered.
    """
    time = np.array(time)
    voltage = np.array(voltage)

    # Sampling interval and frequency
    dt = time[1] - time[0]
    fs = 1 / dt

    # FFT
    N = len(voltage)
    fft_values = np.fft.fft(voltage)
    freq = np.fft.fftfreq(N, dt)

    # Magnitude (normalized)
    magnitude = np.abs(fft_values) * 2 / N

    fig = go.Figure()
    fig.add_trace(go.Scattergl( #Scattergl to use opengl
        x=freq, 
        y=magnitude, 
        mode='lines', 
        opacity=1,       
        showlegend=False    # Legende désactivée car bcp de courbes
    ))

    fig.update_layout(title="TF")

    fig.show_dash(mode='external')
    # Note: consider plotting only positive frequencies (freq >= 0) for clarity.