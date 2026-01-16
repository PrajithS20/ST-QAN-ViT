import mne
import numpy as np
import pywt
import os
import matplotlib.pyplot as plt
import cv2

# --- Configuration & Paths ---
RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/scalograms"
PLOT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Dataset Configuration from Math_Sem4.docx
PATIENTS = {
    "chb01": [("chb01_03.edf", 2996, 3036), ("chb01_04.edf", 1467, 1494), ("chb01_01.edf", None, None)],
    "chb02": [("chb02_16.edf", 130, 212), ("chb02_01.edf", None, None)],
    "chb03": [("chb03_01.edf", 362, 414), ("chb03_05.edf", None, None)],
    "chb05": [("chb05_06.edf", 417, 532), ("chb05_01.edf", None, None)],
    "chb08": [("chb08_02.edf", 2670, 2841), ("chb08_03.edf", None, None)] # Adjusted for existing files
}

def generate_scalogram(data):
    """Generates 32x32 CWT Scalogram using Complex Morlet"""
    scales = np.arange(1, 33)
    # Using 'cmor1.5-1.0' for mandated time-frequency localization
    coef, _ = pywt.cwt(data, scales, 'cmor1.5-1.0')
    magnitude = np.abs(coef)
    
    # Resize to 32x32 for AI input
    resized = cv2.resize(magnitude, (32, 32))
    # Normalize 0-1 for training stability
    normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-8)
    return normalized

def process_dataset():
    print("🚀 Starting Phase 1: Signal Engineering...")
    
    for patient, files in PATIENTS.items():
        for edf_name, start_s, end_s in files:
            file_path = os.path.join(RAW_DATA_DIR, patient, edf_name)
            if not os.path.exists(file_path):
                print(f"⚠️ Skipping missing file: {file_path}")
                continue
                
            print(f"📂 Processing {edf_name}...")
            # 1. Ingestion: Load EDF files
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # 2. Mandatory Clinical Filtering
            raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
            raw.notch_filter(60.0, verbose=False)
            
            # Process first channel for feature extraction
            data = raw.get_data(picks='eeg')[0] 
            fs = int(raw.info['sfreq'])
            window_size = 30 * fs # 30-second windows
            step_size = 15 * fs   # 50% overlap
            
            for start in range(0, len(data) - window_size, step_size):
                segment = data[start : start + window_size]
                current_time = start / fs
                
                # 3. Labeling: 1 if within 15 mins (900s) of onset (Pre-ictal)
                label = 0
                if start_s is not None:
                    if (start_s - 900) <= current_time <= end_s:
                        label = 1
                
                # 4. Transformation: Create 32x32 Image
                scalogram = generate_scalogram(segment)
                
                # 5. Result #1 Visualization (Save first successful instance of each label)
                if start == 0:
                    plt.figure(figsize=(5, 5))
                    plt.imshow(scalogram, cmap='jet', aspect='auto')
                    plt.title(f"Result #1: {'Seizure' if label==1 else 'Normal'}")
                    plt.axis('off')
                    plt.savefig(f"{PLOT_DIR}/result_1_{patient}_lab{label}.png")
                    plt.close()

                # Save locally for Phase 2: Quantum Extraction
                save_name = f"{patient}_{edf_name.split('.')[0]}_t{int(current_time)}_lab{label}.npy"
                np.save(os.path.join(OUTPUT_DIR, save_name), scalogram)

    print(f"✨ Phase 1 Complete. Data ready in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()