import os
import mne
import numpy as np
import re
from pathlib import Path

# --- CONFIGURATION ---
# Ensure your folders (chb01, chb02...) are inside this DATA_ROOT
DATA_ROOT = Path("data/raw") 
OUTPUT_DIR = Path("data/processed_signals")

# Parameters
SAMPLING_RATE = 256 
WINDOW_SECONDS = 30
OVERLAP_SECONDS = 15
PREICTAL_MINUTES = 15 

# We use the 5 patients you have ready
TARGET_PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_summary_info(summary_path):
    """Parses the summary text file for seizure times."""
    if not summary_path.exists():
        return {}
    
    with open(summary_path, 'r') as f:
        content = f.read()

    seizure_info = {}
    current_file = None
    lines = content.split('\n')
    
    for line in lines:
        if "File Name" in line:
            current_file = line.split(':')[-1].strip()
            if current_file not in seizure_info:
                seizure_info[current_file] = []
        
        if "Seizure Start Time" in line:
            try:
                # Regex to grab the number
                start_sec = int(re.search(r'\d+', line).group())
                seizure_info[current_file].append({'start': start_sec, 'end': None})
            except: pass

    return seizure_info

def process_patient(patient_id):
    patient_dir = DATA_ROOT / patient_id
    if not patient_dir.exists():
        print(f"⚠️  Skipping {patient_id} (Folder not found in {DATA_ROOT})")
        return

    print(f"\n--- Processing Patient: {patient_id} ---")
    summary_file = patient_dir / f"{patient_id}-summary.txt"
    seizure_map = parse_summary_info(summary_file)
    
    edf_files = list(patient_dir.glob("*.edf"))
    
    for edf_path in edf_files:
        filename = edf_path.name
        
        try:
            # Read EDF (suppress warnings)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
        except:
            print(f"  [Skip] Error reading {filename}")
            continue

        # Filter 1-50Hz
        raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose='error')
        
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Handle Sampling Rate
        if sfreq != SAMPLING_RATE:
            data = mne.filter.resample(data, down=sfreq/SAMPLING_RATE)
            sfreq = SAMPLING_RATE

        file_seizures = seizure_map.get(filename, [])
        
        # Windowing
        win_samp = int(WINDOW_SECONDS * sfreq)
        step_samp = int(OVERLAP_SECONDS * sfreq)
        n_samples = data.shape[1]
        
        windows = []
        labels = []
        
        for start_idx in range(0, n_samples - win_samp, step_samp):
            end_idx = start_idx + win_samp
            window_end_sec = end_idx / sfreq
            
            label = 0 
            for seizure in file_seizures:
                sz_start = seizure['start']
                if sz_start is None: continue
                
                # Pre-ictal Labeling (15 mins before)
                pre_ictal_start = max(0, sz_start - (PREICTAL_MINUTES * 60))
                if pre_ictal_start <= window_end_sec < sz_start:
                    label = 1
                    break
            
            chunk = data[:, start_idx:end_idx]
            windows.append(chunk.astype(np.float32))
            labels.append(label)
            
        if len(windows) > 0:
            X = np.stack(windows)
            y = np.array(labels)
            
            # Save Raw Signal
            save_name = OUTPUT_DIR / f"{filename.replace('.edf', '')}_processed.npz"
            np.savez_compressed(save_name, X=X, y=y)
            
            if 1 in y:
                print(f"  -> {filename}: Saved {len(X)} windows (Contains Seizure!)")

if __name__ == "__main__":
    for patient in TARGET_PATIENTS:
        process_patient(patient)
    print("\n✅ PHASE 1 COMPLETE. Raw signals ready.")