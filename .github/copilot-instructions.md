Repository: pitaya-rmn-project — Copilot / AI agent instructions

Purpose
- Help an AI code assistant be immediately productive: where to look, what patterns matter, and the concrete workflows to run, debug and extend acquisition and processing code.

Quick high-level architecture
- Two main concerns live side-by-side:
  - python/: analysis, notebooks and helpers that orchestrate SSH/SFTP to the Red Pitaya, parse measurement files, plot and filter signals. Key file: `python/Link Red Pitaya RMN.ipynb`.
  - src-c/: low-level acquisition code and build artifacts (Acquisition_axi.c, Makefile). The compiled program (used remotely) is `Acquisition_axi.exe`.

Primary data flow (typical run)
1. Notebook composes a shell command and triggers the remote executable via SSH (paramiko). See `run_acquisition_command(...)` in `Link Red Pitaya RMN.ipynb`. Example command pattern:
   cd Pitaya-Tests && ./Acquisition_axi.exe {samplesNb} {dec} {FidNb} {filePath} {larmorFreq} {excitationDuration} {delayRepeat}
2. Notebook uses SFTP to download results into a local `mesures/` subfolder (create_file_wdate creates timestamped local folders).
3. Files are read and parsed by either `read_file(...)` (CSV) or `read_file_bin(...)` (binary) functions in the same notebook.
4. Data is processed (accumulation, filtering using scipy.signal, FFT) and plotted with provided helpers (`plot_acc`, `plot_acc_only`, `plot_fourier_transform`).

Important constants & conventions
- Sampling rate: `SAMPLING_RATE = 125e+6` (Hz). Time axis calculation uses: duree_mesure = (dsize * decimation) / SAMPLING_RATE.
- Remote folder constants are defined at top of notebook: `REMOTE_FOLDER = "Pitaya-Tests"`, `REMOTE_PATH = "Pitaya-Tests/"`.
- Filenames/local structure: measurements are stored under `python/mesures/` with names like `{experiment_name}-{YYYYmmdd_HHMMSS}`.

Discoverable file formats (useful for parsing / adding readers)
- CSV header (first line read by `read_file`): [dsize, decimation, nombre_de_FID, gain, offset, nb_bits]
  - `dsize` = number of samples per FID
  - `decimation` = decimation factor used on acquisition
  - `nombre_de_FID` = number of FID traces in the file
- Binary format (read by `read_file_bin`): first 16 bytes = 4 little-endian 4-byte ints (dsize, decimation, nombre_de_FID, ...). Following data are signed 16-bit samples (little-endian). The notebook unpacks samples with format like `struct.unpack('<'+'h'*dsize, fileContent[16:])` and scales by 32768.

Project-specific patterns & quirks
- The notebooks mix French/English variable names (e.g., `nombre_de_FID`, `duree_mesure`). Keep literal names when modifying functions to avoid breaking uses.
- `read_file_bin(fileName, folderName)` currently constructs `pathOfFile` but opens `fileName` directly in the provided notebook; double-check the path variable when modifying/rewriting this function.
- Many values are hard-coded in the notebook for quick experimentation (host IP, root/root credentials, REMOTE_FOLDER). Treat these as configuration to extract if you add automation or CI.

Developer workflows (how a developer actually runs things)
- To run acquisition from the notebook:
  1. Ensure SSH credentials and `hostName` are correct in `Link Red Pitaya RMN.ipynb`.
  2. Execute `run_acquisition_command(...)` cell to launch remote `Acquisition_axi.exe`.
  3. Use `download_file_sftp(...)` to fetch results, then `read_file(...)` or `read_file_bin(...)` and plotting helpers.
- To (re)build the acquisition program:
  - On a Linux-compatible build host (or the device), go to `src-c/` and run `make`. The repository contains `src-c/Makefile`. Cross-compilation details are not embedded in the repo — build on a host compatible with the runtime target.

Actionable tasks for AI contributions
- When modifying data readers, validate against both CSV and binary readers and add an automated small test (a tiny fixture file under `python/mesures/` is appropriate).
- When changing plotting or filtering parameters, update or add a notebook cell demonstrating the new behavior using an existing `mesures/...` folder so maintainers can run it interactively.
- When adding new configuration, centralize remote constants (host, REMOTE_FOLDER, SAMPLING_RATE) into a small `python/config.py` module and update the notebooks to import it.

Files to check first
- `python/Link Red Pitaya RMN.ipynb` — primary orchestration: SSH, SFTP, parsing and plotting.
- `python/README.md` — environment setup: create venv and `pip install -r requirements.txt`.
- `src-c/Acquisition_axi.c` and `src-c/Makefile` — native acquisition program used by the device.

Notes / safety
- Notebook contains credentials and an IP address in clear text. Do not exfiltrate secrets. Use environment variables or a developer-only config file when automating.

If anything is unclear or you'd like the instructions to include more examples (e.g., a minimal test fixture, example output shapes, or a short `config.py` proposal), tell me which section to expand and I'll iterate.
