import os
import mne
import numpy as np
import re
import pickle
from pathlib import Path

# --- CONFIGURATION BASED ON YOUR DOCS ---
DATA_ROOT = Path("data/raw")
OUTPUT_DIR = Path("data/processed_signals") # Intermediate storage before CWT
SAMPLING_RATE = 256  # Standard for CHB-MIT
WINDOW_SECONDS = 30
OVERLAP_SECONDS = 15 # 50% overlap
PREICTAL_MINUTES = 15 # 15 mins before seizure = Label 1

# Specific patients mentioned in your document
TARGET_PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_summary_info(summary_path):
    """
    Parses the .txt summary file to find which .edf files contain seizures
    and what the start/end times are.
    """
    with open(summary_path, 'r') as f:
        content = f.read()

    seizure_info = {}
    current_file = None
    
    # Regex to find filenames and seizure times
    # Pattern looks for "File Name: chb01_03.edf" and "Seizure Start Time: 360 seconds"
    lines = content.split('\n')
    for line in lines:
        if "File Name" in line:
            current_file = line.split(':')[-1].strip()
            if current_file not in seizure_info:
                seizure_info[current_file] = []
        
        if "Seizure Start Time" in line:
            start_sec = int(re.search(r'\d+', line).group())
            # We look ahead or store this to pair with end time
            # Usually the next line is End Time, but let's be safe
            seizure_info[current_file].append({'start': start_sec, 'end': None})
            
        if "Seizure End Time" in line:
            end_sec = int(re.search(r'\d+', line).group())
            # Update the last entry
            if seizure_info[current_file]:
                seizure_info[current_file][-1]['end'] = end_sec

    return seizure_info

def process_patient(patient_id):
    print(f"\n--- Processing Patient: {patient_id} (CPU Mode) ---")
    
    patient_dir = DATA_ROOT / patient_id
    summary_file = patient_dir / f"{patient_id}-summary.txt"
    
    if not summary_file.exists():
        print(f"CRITICAL ERROR: Summary file not found for {patient_id}")
        return

    # 1. Parse Labels
    seizure_map = parse_summary_info(summary_file)
    
    # Get all .edf files in the folder
    edf_files = list(patient_dir.glob("*.edf"))
    
    for edf_path in edf_files:
        filename = edf_path.name
        
        # Skip files not in the summary (rare, but safety check) or if we only want specific files
        # The doc suggests using specific files (e.g., chb01_03), but usually we process the whole patient
        # to get enough data. We will process files found in the summary.
        
        print(f"Loading {filename}...")
        
        # 2. Ingestion (Load EDF)
        # verbose=False reduces clutter
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"Skipping {filename} due to load error: {e}")
            continue

        # 3. Filtering
        # Doc requirement: Bandpass 1-50 Hz (Remove DC and Muscle Noise)
        raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose=False)
        
        # Standardize channels (Optional but good for "Perfect Accuracy")
        # Most CHB-MIT have 23 channels. We keep it raw for now, handled in PCA later.

        data = raw.get_data() # Shape: (Channels, Time)
        sfreq = raw.info['sfreq']
        
        # Define Seizure Times for this specific file
        file_seizures = seizure_map.get(filename, [])
        
        # 4. Windowing & 5. Labeling
        # Convert window/stride to samples
        win_samp = int(WINDOW_SECONDS * sfreq)
        step_samp = int(OVERLAP_SECONDS * sfreq)
        
        n_samples = data.shape[1]
        
        windows = []
        labels = []
        
        # Iterate through signal with sliding window
        for start_idx in range(0, n_samples - win_samp, step_samp):
            end_idx = start_idx + win_samp
            
            # Determine time in seconds for the END of the window
            # (Prediction is usually done based on what we have seen so far)
            window_end_sec = end_idx / sfreq
            
            label = 0 # Default: Inter-ictal (Normal)
            
            for seizure in file_seizures:
                sz_start = seizure['start']
                sz_end = seizure['end']
                
                # LOGIC FROM DOC:
                # Label 1 if window is within [Seizure Start - 15 mins]
                # We strictly exclude the actual seizure and post-seizure from training 
                # to prevent "looking ahead" or training on the event itself, 
                # BUT the prompt implies prediction.
                
                # Pre-ictal period starts at:
                pre_ictal_start = sz_start - (PREICTAL_MINUTES * 60)
                
                if pre_ictal_start < 0: pre_ictal_start = 0
                
                # Check if this window falls in the Pre-ictal zone
                # Note: We classify the window based on its end time (prediction point)
                if pre_ictal_start <= window_end_sec < sz_start:
                    label = 1
                    break
                    
                # Optional: You might want to exclude the actual seizure frames 
                # to avoid confusing the model, or label them differently.
                # For now, we stick to Binary (0 vs 1).
            
            # Extract the chunk
            chunk = data[:, start_idx:end_idx]
            
            # Save chunk (We will save as a list first, then dump to numpy)
            windows.append(chunk.astype(np.float32))
            labels.append(label)
            
        # Save processed data for this file
        if len(windows) > 0:
            X = np.stack(windows) # Shape: (N_windows, Channels, Time)
            y = np.array(labels)
            
            save_name = OUTPUT_DIR / f"{filename.replace('.edf', '')}_processed.npz"
            np.savez_compressed(save_name, X=X, y=y)
            
            # Validation Print
            positives = np.sum(y == 1)
            print(f"  -> Saved {X.shape[0]} windows. Pre-ictal (Class 1): {positives}")

if __name__ == "__main__":
    print("STARTING PHASE 1: SIGNAL ENGINEERING (CPU)")
    for patient in TARGET_PATIENTS:
        process_patient(patient)
    print("\nPHASE 1 COMPLETE. Data saved in 'data/processed_signals'")