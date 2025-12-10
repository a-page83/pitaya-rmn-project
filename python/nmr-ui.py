import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import paramiko
import datetime
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from scipy.interpolate import interp1d
from tkinter import filedialog
from tqdm import tqdm

# Importation de votre librairie
try:
    import NMR_Library as nmr
except ImportError:
    print("ATTENTION: NMR_Library non trouvé.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "settings.json")

class NMRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contrôle Pitaya NMR - Plotly & Save")
        self.root.geometry("800x1000") # Fenêtre plus compacte car les graphiques sont externes
        
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Données partagées pour les graphiques
        self.data_store = {
            "time": None, "voltage": None,
            "freq": None, "mag": None,
            "freq_sum": None, "mag_sum": None,
            "iter": 0
        }

        self.setup_ui()
        self.load_settings() # Chargement automatique au démarrage

        # Sauvegarder les réglages en quittant
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.configure("Bold.TLabel", font=("Segoe UI", 9, "bold"))


        # --- Panneau Principal ---
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Section Connexion ---
        conn_frame = ttk.LabelFrame(main_frame, text="1. Connexion SSH", padding="10")
        conn_frame.pack(fill=tk.X, pady=5)

        self.inputs = {} # Dictionnaire pour stocker les widgets Entry

        self.create_entry(conn_frame, "ip", "IP Pitaya:", "169.254.215.235", 0)
        self.create_entry(conn_frame, "user", "Utilisateur:", "root", 1)
        self.create_entry(conn_frame, "pass", "Mot de passe:", "root", 2)

        # --- Section Paramètres Généraux ---
        param_frame = ttk.LabelFrame(main_frame, text="2. Paramètres RMN", padding="10")
        param_frame.pack(fill=tk.X, pady=5)

        self.create_entry(param_frame, "sample_Amount", "Sample Amount:", "131072", 0, 0)
        self.create_entry(param_frame, "decimation", "Decimation:", "2", 1, 0)
        self.create_entry(param_frame, "acq_amt", "Nombre de FID mesurés (sur une accumulation):", "100", 2, 0)
        self.create_entry(param_frame, "larmor_Frequency_Hertz", "Fréquence larmor_Frequency_Hertz (Hz):", "13900000", 0, 2)
        self.create_entry(param_frame, "excitation_duration_seconds", "Durée Excitation (s):", "30e-6", 1, 2)
        self.create_entry(param_frame, "fid_time", "Temps FID (s):", "5e6", 2, 2)
        self.create_entry(param_frame, "echo_time", "Echo Time (s):", "1",3,0)

        # --- FRAME FILTRE ET BALAYAGE --- 
        sweep_filter_frame = ttk.Frame(main_frame, padding="10")
        sweep_filter_frame.pack(fill=tk.X,expand=True)
        # --- Section Balayage Subframe ---
        sweep_frame = ttk.LabelFrame(sweep_filter_frame, text="3. Balayage & Fichiers", padding="5")
        sweep_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,padx=2.5)
        self.create_entry(sweep_frame, "nb_files", "Nb Fichiers (Steps):", "1", 0)
        self.create_entry(sweep_frame, "step_freq", "Pas de Fréquence (Hz):", "3000", 1)
        self.create_entry(sweep_frame, "step_p90", "Pas de P90 (s):", "0", 2)
        self.create_entry(sweep_frame, "exp_name", "Nom Expérience:", "Stepfreq", 3)
        self.create_entry(sweep_frame, "graph_start", "Début Graphe (ms):", "0", 4)
        # --- Section filtre Subframe --- 
        filter_frame = ttk.LabelFrame(sweep_filter_frame, text="4. Réglages du filtre", padding="5")
        filter_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,padx=2.5) 
        self.create_entry(filter_frame, "high_freq", "Fréquence haute (Hz)", "1000", 0)
        self.create_entry(filter_frame, "low_freq", "Fréquence basse (Hz):", "3000", 1)
        self.create_entry(filter_frame, "order", "Ordre du fitre", "1", 2)
        
        # --- Push buttons ---      
        btn_frame = ttk.LabelFrame(main_frame, text="Configuration de l'ouverture", padding="10")
        btn_frame.pack(fill=tk.X, pady=10)
        self.var_chk_btn_offset_freq = tk.BooleanVar(value=False)
        # Widget 
        # We link it to the variable using 'variable='
        self.chk_btn_offset_freq = ttk.Checkbutton(
            btn_frame, 
            text="Ouvrir avec décalage", 
            variable=self.var_chk_btn_offset_freq
        )
        self.chk_btn_offset_freq.grid(column=1,row=1,sticky="ew", padx=5, pady=2)
        ## - Freq multiple files -
        self.var_chk_btn_files = tk.BooleanVar(value=False)
        self.chk_btn_files = ttk.Checkbutton(
            btn_frame, 
            text="Ouvrir plusieurs fichiers", 
            variable=self.var_chk_btn_files
        )
        self.chk_btn_files.grid(column=1,row=2,sticky="ew", padx=5, pady=2)
        ## - Freq filter -
        self.var_chk_btn_filter = tk.BooleanVar(value=False)
        self.chk_btn_filter = ttk.Checkbutton(
            btn_frame, 
            text="Ouvrir en filtrant le signal", 
            variable=self.var_chk_btn_filter
        )
        self.chk_btn_filter.grid(column=5,row=1,sticky="ew", padx=5, pady=2)
        ## - dash -
        self.var_chk_btn_dash = tk.BooleanVar(value=False)
        self.chk_btn_dash = ttk.Checkbutton(
            btn_frame, 
            text="Ouvrir en utilisant le moteur d'affichage", 
            variable=self.var_chk_btn_dash
        )
        self.chk_btn_dash.grid(column=5,row=2,sticky="ew", padx=5, pady=2)
        
        ## - dash -
        self.var_chk_btn_sumtf = tk.BooleanVar(value=False)
        self.chk_btn_sumtf = ttk.Checkbutton(
            btn_frame, 
            text="Ouvrir le graphe sommant les TF", 
            variable=self.var_chk_btn_sumtf
        )
        self.chk_btn_sumtf.grid(column=7,row=1,sticky="ew", padx=5, pady=2)
 
        # --- Boutons to launch ---
        btn_frame = ttk.LabelFrame(main_frame, text="Boutons de lancement", padding="10")
        btn_frame.pack(fill=tk.X, pady=10)

        # Calcul des paramètres
        self.btn_single = ttk.Button(btn_frame, text="ESTIMATION DU TEMPS", 
                                     command=lambda: self.print_parameters())
        self.btn_single.pack(fill=tk.X, pady=5)

        self.btn_plot = ttk.Button(btn_frame, text="📁 OUVRIR GRAPHIQUES", 
                                   command=lambda: self.browse_open_file())
        self.btn_plot.pack(side=tk.BOTTOM,fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(btn_frame, text="⏹ ARRÊTER", 
                                   command=self.stop_acquisition, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.BOTTOM,fill=tk.X, pady=5)

        btn_fid_frame = ttk.LabelFrame(btn_frame,text =" Boutons pour lancer avec fid")
        btn_fid_frame.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=5)
        
        # Mode 1 = Acquisition Simple (Fréquence Fixe)
        self.btn_single = ttk.Button(btn_fid_frame, text="▶ DÉMARRER FID SIMPLE", 
                                     command=lambda: self.start_thread_acq(mode=0)) #Single FID
        self.btn_single.pack(fill=tk.X, pady=5)
        # Mode 0 = Frequency Sweep
        self.btn_sweep = ttk.Button(btn_fid_frame, text="▶ DÉMARRER FID SWEEP FREQ", 
                                    command=lambda: self.start_thread_acq(mode=1)) #Sweep FID
        self.btn_sweep.pack(fill=tk.X, pady=5)
        
        self.btn_sweep = ttk.Button(btn_fid_frame, text="▶ DÉMARRER FID SWEEP P90", 
                                    command=lambda: self.start_thread_acq(mode=2)) #Sweep P90 FID
        self.btn_sweep.pack(fill=tk.X, pady=5)

        btn_echo_frame = ttk.LabelFrame(btn_frame,text =" Boutons pour lancer avec echo")
        btn_echo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,padx=5)
        # Mode 2 = Acquisition ECHO (Fréquence Fixe)
        self.btn_single = ttk.Button(btn_echo_frame, text="▶ DÉMARRER ECHO SIMPLE", 
                                     command=lambda: self.start_thread_acq(mode=3)) #Single Echo
        self.btn_single.pack(fill=tk.X, pady=5)

        self.btn_single = ttk.Button(btn_echo_frame, text="▶ DÉMARRER ECHO SWEEP FREQ", 
                                     command=lambda: self.start_thread_acq(mode=4)) #Sweep Freq Echo
        self.btn_single.pack(fill=tk.X, pady=5)
        self.btn_sweep = ttk.Button(btn_echo_frame, text="▶ DÉMARRER FID SWEEP P90", 
                                     command=lambda: self.start_thread_acq(mode=5)) #Sweep p90 echo
        self.btn_sweep.pack(fill=tk.X, pady=5)

        # --- Logs ---
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # -- msgType settings --- 
        self.log_text.tag_config("INFO", foreground="black")   # Texte normal
        self.log_text.tag_config("SUCCESS", foreground="green") # Vert pour succès
        self.log_text.tag_config("WARNING", foreground="orange") # Orange pour info importante
        self.log_text.tag_config("ERROR", foreground="red")    # Rouge pour erreurs
        self.log_text.tag_config("BLUE", foreground="blue")    # Bleu pour fréquence

    def create_entry(self, parent, key, label, default, row, col=1):
        """Helper pour créer label + entry et stocker la ref"""
        ttk.Label(parent, text=label).grid(row=row, column=col*2, sticky="w", padx=5, pady=2)
        entry = ttk.Entry(parent, width=15)
        entry.insert(0, default)
        entry.grid(row=row, column=col*2+1, sticky="ew", padx=5, pady=2)
        self.inputs[key] = entry

    def log(self, msg,msgType="INFO"):
        self.log_text.insert(tk.END, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n",msgType)
        #print(msg)
        self.log_text.see(tk.END)

    def save_settings(self): # --- Gestion des Paramètres (JSON) --- 

        settings = {key: entry.get() for key, entry in self.inputs.items()}
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
            self.log("--- --- --- --- --- ")
            self.log("Paramètres sauvegardés.")
        except Exception as e:
            self.log(f"Erreur sauvegarde: {e}")

    def load_settings(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, 'r') as f:
                settings = json.load(f)
            for key, val in settings.items():
                if key in self.inputs:
                    self.inputs[key].delete(0, tk.END)
                    self.inputs[key].insert(0, val)
            self.log("Paramètres chargés.")
        except Exception as e:
            self.log(f"Erreur chargement: {e}")

    def on_close(self):
        self.save_settings()
        self.root.destroy()

    def start_thread_acq(self, mode):
        if self.is_running: return
        
        self.save_settings() 
        self.is_running = True
        self.stop_event.clear()

        # Désactiver les boutons de start, activer le stop
        self.btn_sweep.config(state=tk.DISABLED)
        self.btn_single.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_plot.config(state=tk.NORMAL)

        # On passe 'mode' à la fonction cible via args=(mode,)
        # La virgule est importante pour créer un tuple
        t = threading.Thread(target=self.run_acquisition, args=(mode,))
        t.daemon = True
        t.start()

    def start_thread_dash(self, figure,port):
            
        self.is_running = True
        self.stop_event.clear()
        self.log(f"Affichage sur http://127.0.0.1:{port}/")
        # La virgule est importante pour créer un tuple
        t = threading.Thread(target=self.dash_launcher, args=(figure,port))
        t.daemon = True
        t.start()

    def dash_launcher(self,figure,port):
        config = {'scrollZoom': True}
        figure.show_dash(mode='external',port=port,config=config)
        return

    def stop_acquisition(self):
        if self.is_running:
            self.log("Arrêt demandé...")
            self.stop_event.set()

    def print_parameters(self):
        self.save_settings()

        p = {k: v.get() for k, v in self.inputs.items()}
        
        sample_Amount = int(float(p['sample_Amount']))
        decimation = int(p['decimation'])
        acq_Amt = int(p['acq_amt'])
        larmor_Frequency_Hertz = float(p['larmor_Frequency_Hertz'])
        excitation_duration_seconds = float(p['excitation_duration_seconds'])
        total_FID_Time_s = float(p['fid_time'])
        nb_files = int(p['nb_files'])

        step_freq = float(p['step_freq'])
        ##setp_p90 = 

        graph_start = float(p['graph_start'])

        # nb_cycles = larmor_Frequency_Hertz*excitation_duration_seconds
        # total_time = (sample_Amount * decimation) / 125e6
        # delay_rep = fid_time - total_time * 1e6
        # temps_secondes = total_time*nb_files*acq_Amt

        total_time_fid_mes = (sample_Amount * decimation)/125e6
        nb_cycles = larmor_Frequency_Hertz*excitation_duration_seconds
        temps_total_step_secondes = (total_FID_Time_s)*acq_Amt+1
        self.log(f"Sweep set from {larmor_Frequency_Hertz}Hz to {larmor_Frequency_Hertz + step_freq*nb_files}")
        self.log(f"total time for one step ({acq_Amt} accumulations): {str(datetime.timedelta(seconds=temps_total_step_secondes))}")
        self.log(f"total sweep time :{datetime.timedelta(seconds=temps_total_step_secondes*nb_files+3)} ")

        if nb_cycles < 0 or nb_cycles > 50000:
            self.log(f"Bad Number of cycles for the burst : {nb_cycles} check pulse time or frequency !!!","ERROR")  # Prints in red
        if  sample_Amount%2 != 0 :
            self.log(f"Sample Amount must be a multiple of 2","ERROR")
        if total_time_fid_mes > total_FID_Time_s :
            self.log(f"total measured time > total time of one accumulation","ERROR")
        if total_FID_Time_s > 1e6 :
            self.log(f"total_time_fid in seconds ?","ERROR")

    def run_acquisition(self,mode):
        try:
            # Récupération des valeurs
            p = {k: v.get() for k, v in self.inputs.items()}
            
            HOST, USER, PASS = p['ip'], p['user'], p['pass']
            sample_Amount = int(float(p['sample_Amount']))
            decimation = int(p['decimation'])
            acq_Amt = int(p['acq_amt'])
            larmor_Frequency_Hertz = float(p['larmor_Frequency_Hertz'])
            excitation_duration_seconds = float(p['excitation_duration_seconds'])
            total_time_s = float(p['fid_time'])
            nb_files = int(p['nb_files'])
            echo_time_us = float(p['echo_time'])*1e6 # Conversion de secondes à us
            step_freq = 0
            setp_p90 = 0 #On met step freq et P90 à 0 par défaut, seront modifiés en fonction du mode

            echo = False                            # Variable indiquant si l'echo est activé ou non

            match mode :
                case 0 :
                    # MODE SINGLE : On force 1 seul fichier (ou accumulation sans changer freq)
                    nb_files = 1      # On force à 1 cycle pour une acquisition simple
                    exp_prefix = "Single_"
                    self.log(">>> Mode: Mesure d'une accumulation de FID (Freq Fixe)")
                case 1 :
                    # MODE SWEEP : On utilise les champs "Nb Fichiers" et "Step"
                    nb_files = int(p['nb_files'])
                    step_freq = float(p['step_freq'])
                    exp_prefix = "SweepFreq_"
                    self.log(">>> Mode: Frequency Sweep FID")
                case 2 :
                    # MODE SWEEP P90 : On utilise les champs "Nb Fichiers" et "Stepp90"
                    nb_files = int(p['nb_files'])
                    step_p90 = float(p['step_freq'])
                    exp_prefix = "SweepP90_"
                    self.log(">>> Mode: P90 Sweep FID")
                case 3 :
                    # MODE ECHO : Les paramètres sont les mêmes que case 1 : MODE SINGLE avec la variable echo à 1
                    nb_files = 1      # On force à 1 cycle pour une accumulation simple
                    exp_prefix = "SingleEcho_"
                    self.log(">>> Mode: Mesure d'une accumulation d'Echo (Freq Fixe)")
                    echo = True
                    if total_time_s <= echo_time_us*1e-6 :
                        self.log("ERREUR : Echo time trop long","ERROR")
                        return
                case 4 :
                    # MODE ECHO SWEEP : Les paramètres sont les mêmes que MODE SWEEP avec la variable echo à 1
                    # On utilise les champs "Nb Fichiers" et "Step"
                    nb_files = int(p['nb_files'])
                    step_freq = float(p['step_freq'])
                    exp_prefix = "SweepFreqEcho_"
                    self.log(">>> Mode: Frequency Sweep with echo")
                    echo = True
                    if total_time_s <= echo_time_us*1e-6 :
                        self.log("ERREUR : Echo time trop long","ERROR")
                        return
                case 5 :
                    # MODE SWEEP P90 ECHO : On utilise les champs "Nb Fichiers" et "Stepp90"
                    nb_files = int(p['nb_files'])
                    step_p90 = float(p['step_freq'])
                    exp_prefix = "SweepP90_"
                    echo = True
                    self.log(">>> Mode: Frequency Sweep with echo")
            if echo == True :
                meas_time = (sample_Amount * decimation) / 125e6 + echo_time_us*3e-6 + excitation_duration_seconds*3
            else : 
                meas_time = (sample_Amount * decimation) / 125e6 + excitation_duration_seconds

            delay_rep = (total_time_s - meas_time) * 1e6
            print(delay_rep)

            # Connexion
            
            self.log(f"Connexion à {HOST}...")
            nmr.client = paramiko.SSHClient()
            nmr.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            nmr.client.connect(HOST, username=USER, password=PASS, port=22,timeout=10)
            
            if not nmr.client.get_transport(): 
                return
            
            transport = paramiko.Transport((HOST, 22))
            transport.connect(username=USER, password=PASS)
            nmr.sftp = paramiko.SFTPClient.from_transport(transport)
            
            nameLocalFolder = nmr.create_file_wdate(exp_prefix+str(nb_files)+"_"+str(step_freq)+"_"+str(larmor_Frequency_Hertz))
            
            #progress_bar = tqdm(total=nb_files, desc="Processing Acquisitions to find Frequency", unit="file")
            for i in range(nb_files):
                #progress_bar.update(1)
                if self.stop_event.is_set(): break
                
                self.log(f"--- Step {i}/{nb_files} : {larmor_Frequency_Hertz/1e6:.3f} MHz {excitation_duration_seconds/1e-6:.3f}µs ---")
                
                # Acquisition
                if echo == True :
                    nmr.run_acquisition_echo_command(sample_Amount, decimation, acq_Amt, "mesures.bin", larmor_Frequency_Hertz, excitation_duration_seconds, delay_rep,echoTime=echo_time_us)
                else :    
                    nmr.run_acquisition_fid_command(sample_Amount, decimation, acq_Amt, "mesures.bin", larmor_Frequency_Hertz, excitation_duration_seconds, delay_rep) 
                
                # Téléchargement
                experiment_name = f"{p['exp_name']}{i}"
                nameRemoteFolder = "mesures" 
                nmr.download_file_sftp(experiment_name,nameRemoteFolder,nameLocalFolder)
                
                # Traitement
                file_path = os.path.join(nameLocalFolder, experiment_name)
             
                larmor_Frequency_Hertz += step_freq
                excitation_duration_seconds += step_p90
            
            #progress_bar.close()

            self.log("Acquisition terminée.")
            self.data_store = {"file_path" : file_path}
            print("Chemin renvoyé par l'acquisition"+str(file_path))
            self.open_file(filepath_all=file_path[:-1]) # Remove the index
            
        except Exception as e:
            self.log(f"ERREUR: {e}")
            print(e) # Pour debug console
        finally:
            try:
                nmr.client.close()
                transport.close()
                self.btn_sweep.config(state=tk.NORMAL)
                self.btn_single.config(state=tk.NORMAL)
            except: pass
            self.is_running = False
            self.btn_stop.config(state=tk.DISABLED)

    def browse_open_file(self):
        # Ouvre l'explorateur
        self.log(f"Ouverture et affichage...")
        filepath = filedialog.askopenfilename(
            title="Sélectionner un fichier",
        )
        
        # Si l'utilisateur n'a pas annulé
        if filepath:
            filepath = filepath[:-1]
            self.open_file(filepath_all=filepath)
            return

    def open_file(self,filepath_all):
        
        p = {k: v.get() for k, v in self.inputs.items()}

        # Initialisation variables accumulation
        freq_all = None
        tf_sum = None

        graph_start = float(p['graph_start'])
        if not(self.var_chk_btn_files.get()) and self.var_chk_btn_sumtf.get():
            self.log("SUM TF option requires multiple files to be enabled","ERROR")
            return
        
        # --- multiple file enabled ---
        if self.var_chk_btn_files.get(): ## IF TICKBOX MULTIPLE FILES IS ON
            Start_freq = 0 #float(filepath.split('_')[3]) - 5000
            Step_freq = float(filepath_all.split('_')[2])
            Number_of_files = int(filepath_all.split('_')[1])
            if (Number_of_files >= 100) and (not self.var_chk_btn_dash.get()):
                self.log("ENABLE DASH","ERROR")
                return
        else :
            Number_of_files = 1

        print(f"Opening {Number_of_files}")
        # --- if dash is enabled ---
        if self.var_chk_btn_dash.get():
            fig1 = FigureResampler(go.Figure(), default_n_shown_samples=1000)
            fig2 = FigureResampler(go.Figure(), default_n_shown_samples=1000)
            self.log("No html file will be stored")
        else :
            fig1 = go.Figure()
            fig2 = go.Figure()

        fig3 = go.Figure() # For sum TF

        progress_bar = tqdm(total=Number_of_files, desc="Processing Acquisitions to find Frequency", unit="file")
        for i in range(Number_of_files):
            progress_bar.update(1)
            self.log(f"Chargement du fichier {i}/{Number_of_files}")

            filepath = filepath_all+str(i)
            time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(filepath, nombre_de_FID=-1)
            
            dt = np.abs(time_array[0] - time_array[1])
            
            # --- if filter is enabled ---
            if self.var_chk_btn_filter.get():                
                lowcut=float(p['low_freq'])
                highcut=float(p['high_freq'])
                if lowcut >= highcut :
                    self.log("Filtre : FREQUENCE BASSE > FREQ HAUTE","ERROR")
                    return
                
                voltageAcc_array = nmr.butter_bandpass_filter(data=voltageAcc_array, lowcut=lowcut, highcut=highcut, fs=1/dt, order=int(p['order']))

            # Coupe
            
            idx = int(graph_start/(1000*dt))
            volt_cut = voltageAcc_array[idx:]
            time_cut = time_array[idx:]

            # FFT
            N = len(volt_cut)
            freq = np.fft.fftfreq(N, dt)
            mag = np.abs(np.fft.fft(volt_cut)) * 2 / N        

            # --- if Multiple files is enabled ==> SUM TF ---    
            if self.var_chk_btn_files.get() and self.var_chk_btn_sumtf.get():
                # Accumulation TF
                if freq_all is None:
                    freq_all = freq
                    tf_sum = mag
                else:
                    g0 = interp1d(freq_all, tf_sum, bounds_error=False, fill_value=0.0)
                    freq_all = np.union1d(freq_all, freq)
                    g1 = interp1d(freq, mag, bounds_error=False, fill_value=0.0)
                    tf_sum = g1(freq_all) + g0(freq_all)

                if self.var_chk_btn_offset_freq.get():
                    freq = freq + Start_freq + i*Step_freq  

            ## --- check if dash is enabled ---
            if self.var_chk_btn_dash.get():
                self.log(f"sending file {i} on the plot with dash...")
                fig1.add_trace(go.Scattergl( #Scattergl to use opengl
                    mode='lines', 
                    opacity=1,       
                    showlegend=False    # Legende désactivée car bcp de courbes
                ),hf_x=time_cut, hf_y = volt_cut)

                fig2.add_trace(go.Scattergl(
                    mode='lines', 
                    opacity=1, 
                    showlegend=False
                ),hf_x = freq[:len(freq)//2], hf_y = mag[:len(mag)//2])
                
            else :
                self.log(f"sending file {i} on the plot...")
                fig1.add_trace(go.Scattergl( #Scattergl to use opengl
                    x=time_cut, 
                    y=volt_cut, 
                    mode='lines', 
                    opacity=1,       
                    showlegend=False    # Legende désactivée car bcp de courbes
                ))

                fig2.add_trace(go.Scattergl(
                    x=freq, 
                    y=mag, 
                    mode='lines', 
                    opacity=1, 
                    showlegend=False
                ))
        
        progress_bar.close()

      ## -- plot of sum TF if multiple files is enabled --
        if self.var_chk_btn_sumtf.get() and self.var_chk_btn_files.get(): 
            fig3.add_trace(go.Scattergl( #Scattergl to use opengl
                x=freq_all, 
                y=tf_sum, 
                mode='lines', 
                name='TF_SUM',       
                line=dict(color='blue', width=2)
            ))
            fig3.show(config=config)


        if self.var_chk_btn_dash.get():
            self.start_thread_dash(fig1,port = 8080)
            self.start_thread_dash(fig2,port = 8081)

            self.log("check the console to open the plot","WARNING")
        else :
            config = {'scrollZoom': True}
            fig1.show(renderer="browser", config = config)
            fig2.show(renderer="browser", config = config)

            folder = os.path.dirname(filepath_all)
            fig1.write_html(os.path.join(folder,"Figure FID.html"))
            fig2.write_html(os.path.join(folder,"Figure TF.html"))
                        
        print(filepath)
        self.is_running = False

    def show_plotly(self):
        A = NULL
        #self.browse_open_file()

        # if d["time"] is None:
        #     self.log("Pas de données à afficher.")
        #     return

        # self.log("Génération du graphique Plotly...")
        
        # # Création de la figure avec sous-graphes
        # fig1 = go.Figure()
        # fig2 = go.Figure()
        # fig3 = go.Figure()
        # if self.var_chk_btn_offset_freq.get():
        #     # 1. Temporel
        #     fig1.add_trace(go.Scatter(x=d['time'], y=d['voltage'], name="FID", mode='lines', line=dict(color='blue', width=1)))

        # # 2. FFT Instant
        # fig2.add_trace(go.Scatter(x=d['freq'], y=d['mag'], name="FFT Inst.", mode='lines', line=dict(color='orange')))

        # # 3. FFT Somme
        # fig3.add_trace(go.Scatter(x=d['freq_sum'], y=d['mag_sum'], name="FFT Somme", mode='lines', line=dict(color='green')))

        # # Mise en forme
        # fig1.update_xaxes(title_text="Temps (s)")
        # fig2.update_xaxes(title_text="Fréquence (Hz)")
        # fig3.update_xaxes(title_text="Fréquence (Hz)")
        
        # # Affichage (Ouvre le navigateur)
        # fig1.show()
        # fig2.show()
        # fig3.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = NMRApp(root)
    root.mainloop()