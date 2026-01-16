import os
import numpy as np
import pywt
import cv2
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_DIR = Path("data/processed_signals")
OUTPUT_DIR = Path("data/scalograms")
IMG_SIZE = (224, 224) # Standard input for Vision Transformers
SCALES = np.arange(1, 65) # Scale range for CWT (approx 1-64 Hz coverage)
WAVELET = 'cmor1.5-1.0'   # Complex Morlet wavelet (great for EEG spectral power)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def apply_pca_to_window(window_data, n_components=3):
    """
    Reduces (23, Time) -> (3, Time) using PCA.
    Essentially compresses all electrodes into 3 'Super Channels'.
    """
    # Transpose to (Time, Channels) for PCA
    X = window_data.T
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Transpose back to (3, Time)
    return X_pca.T

def compute_cwt(channel_data):
    """
    Applies Continuous Wavelet Transform to a single channel.
    Returns magnitude scalogram.
    """
    coef, freqs = pywt.cwt(channel_data, SCALES, WAVELET, sampling_period=1/256)
    # Return magnitude (ignore phase)
    return np.abs(coef)

def process_and_save_images():
    files = list(INPUT_DIR.glob("*.npz"))
    
    total_samples = 0
    
    print(f"Found {len(files)} processed files. Starting CWT conversion...")
    
    all_X = []
    all_y = []
    
    for fpath in files:
        print(f"Processing {fpath.name}...")
        data = np.load(fpath)
        X_raw = data['X'] # Shape: (N_windows, 23, 7680)
        y_raw = data['y']
        
        # Skip if empty
        if len(X_raw) == 0: continue
        
        file_images = []
        
        for i in range(len(X_raw)):
            window = X_raw[i] # (23, Time)
            
            # 1. PCA: Reduce 23 channels -> 3 channels
            # This makes the data look like an RGB image to the model
            window_pca = apply_pca_to_window(window, n_components=3)
            
            rgb_image = []
            
            # 2. CWT: Apply to each of the 3 components
            for ch in range(3):
                scalogram = compute_cwt(window_pca[ch]) # Shape: (Scales, Time)
                
                # 3. Resize to 224x224 (Model Input Size)
                # cv2.resize expects (Width, Height), so we pass (224, 224)
                resized = cv2.resize(scalogram, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
                
                # Flip vertically so low freq is at bottom (standard plot style)
                resized = np.flipud(resized)
                
                rgb_image.append(resized)
            
            # Stack to create (3, 224, 224)
            img_tensor = np.stack(rgb_image, axis=0) 
            
            # 4. Normalization (Min-Max scaling to 0-1 range for Neural Network)
            img_min, img_max = img_tensor.min(), img_tensor.max()
            if img_max - img_min > 0:
                img_tensor = (img_tensor - img_min) / (img_max - img_min)
            else:
                img_tensor = np.zeros_like(img_tensor)
                
            file_images.append(img_tensor.astype(np.float32))
        
        # Save processed images for this file
        X_imgs = np.stack(file_images)
        
        # Save to disk immediately to save RAM, or keep in list?
        # Let's save individually per file to be safe.
        save_name = OUTPUT_DIR / f"{fpath.stem}_scalograms.npz"
        np.savez_compressed(save_name, X=X_imgs, y=y_raw)
        
        total_samples += len(X_imgs)
        print(f"  -> Converted {len(X_imgs)} scalograms. Saved to {save_name.name}")

    print(f"\nPHASE 2 COMPLETE. Total Scalograms Generated: {total_samples}")
    print("Ready for GPU Training.")

if __name__ == "__main__":
    process_and_save_images()