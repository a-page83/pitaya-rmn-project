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
from scipy.interpolate import interp1d

# Importation de votre librairie
try:
    import NMR_Library as nmr
except ImportError:
    print("ATTENTION: NMR_Library non trouvé.")

CONFIG_FILE = "settings.json"

class NMRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contrôle Pitaya NMR - Plotly & Save")
        self.root.geometry("600x750") # Fenêtre plus compacte car les graphiques sont externes
        
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

        # --- Section Paramètres ---
        param_frame = ttk.LabelFrame(main_frame, text="2. Paramètres RMN", padding="10")
        param_frame.pack(fill=tk.X, pady=5)

        # Grille de paramètres
        self.create_entry(param_frame, "sample_Amount", "Sample Amount:", "131072", 0, 0)
        self.create_entry(param_frame, "decimation", "Decimation:", "2", 1, 0)
        self.create_entry(param_frame, "acq_amt", "Acquisitions:", "100", 2, 0)
        self.create_entry(param_frame, "larmor_Frequency_Hertz", "Fréquence larmor_Frequency_Hertz (Hz):", "13900000", 0, 2)
        self.create_entry(param_frame, "excitation_duration_seconds", "Durée Excitation (s):", "30e-6", 1, 2)
        self.create_entry(param_frame, "fid_time", "Temps FID (us):", "5e6", 2, 2)

        # --- Section Balayage ---
        sweep_frame = ttk.LabelFrame(main_frame, text="3. Balayage & Fichiers", padding="10")
        sweep_frame.pack(fill=tk.X, pady=5)
        
        self.create_entry(sweep_frame, "nb_files", "Nb Fichiers (Steps):", "1", 0)
        self.create_entry(sweep_frame, "step_freq", "Pas de Fréquence (Hz):", "3000", 1)
        self.create_entry(sweep_frame, "exp_name", "Nom Expérience:", "Stepfreq", 2)
        self.create_entry(sweep_frame, "graph_start", "Début Graphe (ms):", "0", 3)

        # --- Boutons ---
        btn_frame = ttk.Frame(main_frame, padding="10")
        btn_frame.pack(fill=tk.X, pady=10)


        #Mode 2 = Calcul des paramètres
        self.btn_single = ttk.Button(btn_frame, text="ESTIMATION DU TEMPS", 
                                     command=lambda: self.start_thread(mode=2))
        self.btn_single.pack(fill=tk.X, pady=5)

        # Mode 0 = Frequency Sweep
        self.btn_sweep = ttk.Button(btn_frame, text="▶ DÉMARRER FREQ SWEEP", 
                                    command=lambda: self.start_thread(mode=0))
        self.btn_sweep.pack(fill=tk.X, pady=5)

        # Mode 1 = Acquisition Simple (Fréquence Fixe)
        self.btn_single = ttk.Button(btn_frame, text="▶ DÉMARRER ACQUISITION (FIXE)", 
                                     command=lambda: self.start_thread(mode=1))
        self.btn_single.pack(fill=tk.X, pady=5)


        self.btn_plot = ttk.Button(btn_frame, text="📊 OUVRIR GRAPHIQUES (PLOTLY)", 
                                   command=self.show_plotly, state=tk.DISABLED)
        self.btn_plot.pack(fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(btn_frame, text="⏹ ARRÊTER", 
                                   command=self.stop_acquisition, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=5)
        # --- Logs ---
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_entry(self, parent, key, label, default, row, col=1):
        """Helper pour créer label + entry et stocker la ref"""
        ttk.Label(parent, text=label).grid(row=row, column=col*2, sticky="w", padx=5, pady=2)
        entry = ttk.Entry(parent, width=15)
        entry.insert(0, default)
        entry.grid(row=row, column=col*2+1, sticky="ew", padx=5, pady=2)
        self.inputs[key] = entry

    def log(self, msg):
        self.log_text.insert(tk.END, f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END)

    # --- Gestion des Paramètres (JSON) ---
    def save_settings(self):
        settings = {key: entry.get() for key, entry in self.inputs.items()}
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
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

    def start_thread(self, mode):
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

    def stop_acquisition(self):
        if self.is_running:
            self.log("Arrêt demandé...")
            self.stop_event.set()

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
            fid_time = float(p['fid_time'])
            nb_files = int(p['nb_files'])

            step_freq = float(p['step_freq'])
            ##setp_p90 = 

            graph_start = float(p['graph_start'])

            nb_cycles = larmor_Frequency_Hertz*excitation_duration_seconds
            total_time = (sample_Amount * decimation) / 125e6
            delay_rep = fid_time - total_time * 1e6
            temps_secondes = total_time*nb_files*acq_Amt

            match mode :
                case 0 :
                    # MODE SWEEP : On utilise les champs "Nb Fichiers" et "Step"
                    nb_files = int(p['nb_files'])
                    step_freq = float(p['step_freq'])
                    exp_prefix = "Sweep_"
                    self.log(">>> Mode: Frequency Sweep")
                case 1 :
                    # MODE SINGLE : On force 1 seul fichier (ou accumulation sans changer freq)
                    nb_files = 1      # On force à 1 cycle pour une acquisition simple
                    step_freq = 0     # Pas de changement de fréquence
                    exp_prefix = "Single_"
                    self.log(">>> Mode: Acquisition Simple (Freq Fixe)")
                case 2 : 
                    self.log(f"total time : {str(datetime.timedelta(seconds=temps_secondes))}")
                    if nb_cycles < 0 or nb_cycles > 50000:
                        self.log(f"\033[91m Bad Number of cycle : {nb_cycles} check pulse time or frequency !!!\033[0m")  # Prints in red
                    return

            # Connexion
            self.log(f"Connexion à {HOST}...")
            nmr.client = paramiko.SSHClient()
            nmr.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            nmr.client.connect(HOST, username=USER, password=PASS, port=22)
            
            transport = paramiko.Transport((HOST, 22))
            transport.connect(username=USER, password=PASS)
            nmr.sftp = paramiko.SFTPClient.from_transport(transport)
            
            nameLocalFolder = nmr.create_file_wdate("FindFreqAuto_"+str(nb_files)+"_"+str(step_freq)+"_"+str(larmor_Frequency_Hertz))
            
            # Initialisation variables accumulation
            freq_all = None
            tf_sum = None

            for i in range(nb_files):
                if self.stop_event.is_set(): break
                
                self.log(f"--- Step {i+1}/{nb_files} : {larmor_Frequency_Hertz/1e6:.3f} MHz ---")
                
                # Acquisition
                nmr.run_acquisition_command(sample_Amount, decimation, acq_Amt, "mesures.bin", larmor_Frequency_Hertz, excitation_duration_seconds, delay_rep)
                
                # Téléchargement
                experiment_name = f"{p['exp_name']}{i}"
                nameRemoteFolder = "mesures" 
                nmr.download_file_sftp(experiment_name,nameRemoteFolder,nameLocalFolder)
                
                # Traitement
                file_path = os.path.join(nameLocalFolder, experiment_name)


                # Note: adaptation légère si open_file_bin attend un chemin sans index
                time_array, voltage_array_matrix, voltageAcc_array = nmr.open_file_bin(file_path, nombre_de_FID=-1)
                
                # Filtrage
                fs = 1/((time_array[10]-time_array[0])/10)
                voltageAcc_array = nmr.butter_bandpass_filter(voltageAcc_array, 1e6, 20e6, fs, order=3)
                
                # Coupe
                dt = np.abs(time_array[0] - time_array[1])
                idx = int(graph_start/(1000*dt))
                volt_cut = voltageAcc_array[idx:]
                time_cut = time_array[idx:]
                
                # FFT
                N = len(volt_cut)
                freq = np.fft.fftfreq(N, dt)
                mag = np.abs(np.fft.fft(volt_cut)) * 2 / N

                # Accumulation TF
                if freq_all is None:
                    freq_all = freq
                    tf_sum = mag
                else:
                    g0 = interp1d(freq_all, tf_sum, bounds_error=False, fill_value=0.0)
                    freq_all = np.union1d(freq_all, freq)
                    g1 = interp1d(freq, mag, bounds_error=False, fill_value=0.0)
                    tf_sum = g1(freq_all) + g0(freq_all)

                # Mise à jour données partagées
                self.data_store = {
                    "time": time_cut, "voltage": volt_cut,
                    "freq": freq, "mag": mag,
                    "freq_sum": freq_all, "mag_sum": tf_sum,
                    "iter": i+1,
                    "max_freq_curr": freq[np.argmax(mag)],
                    "max_freq_sum": freq_all[np.argmax(tf_sum)]
                }
                
                larmor_Frequency_Hertz += step_freq

            self.log("Acquisition terminée.")
            self.show_plotly() # Ouvre automatiquement à la fin

        except Exception as e:
            self.log(f"ERREUR: {e}")
            print(e) # Pour debug console
        finally:
            try:
                nmr.client.close()
                transport.close()
            except: pass
            self.is_running = False
            self.btn_stop.config(state=tk.DISABLED)

    # --- Plotly ---
    def show_plotly(self):
        d = self.data_store
        if d["time"] is None:
            self.log("Pas de données à afficher.")
            return

        self.log("Génération du graphique Plotly...")
        
        # Création de la figure avec sous-graphes
        fig1 = go.Figure()
        fig2 = go.Figure()
        fig3 = go.Figure()

        # 1. Temporel
        fig1.add_trace(go.Scatter(x=d['time'], y=d['voltage'], name="FID", mode='lines', line=dict(color='blue', width=1)))

        # 2. FFT Instant
        fig2.add_trace(go.Scatter(x=d['freq'], y=d['mag'], name="FFT Inst.", mode='lines', line=dict(color='orange')))

        # 3. FFT Somme
        fig3.add_trace(go.Scatter(x=d['freq_sum'], y=d['mag_sum'], name="FFT Somme", mode='lines', line=dict(color='green')))

        # Mise en forme
        fig1.update_xaxes(title_text="Temps (s)")
        fig2.update_xaxes(title_text="Fréquence (Hz)")
        fig3.update_xaxes(title_text="Fréquence (Hz)")
        
        # Affichage (Ouvre le navigateur)
        fig1.show()
        fig2.show()
        fig3.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = NMRApp(root)
    root.mainloop()