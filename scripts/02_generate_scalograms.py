import os
import numpy as np
import pywt
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

# --- CONFIG ---
INPUT_DIR = Path("data/processed_signals")
OUTPUT_DIR = Path("data/scalograms")
IMG_SIZE = (224, 224) 
SCALES = np.arange(1, 65) 
WAVELET = 'cmor1.5-1.0' 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_cwt_magnitude(channel_data):
    # CWT is CPU heavy. We keep it simple here.
    coef, _ = pywt.cwt(channel_data, SCALES, WAVELET, sampling_period=1/256)
    return np.abs(coef)

def process_single_file(fpath):
    """
    Worker function to process ONE file.
    Returns: Number of windows generated (or 0 if error)
    """
    try:
        data = np.load(fpath)
        X_raw = data['X'] 
        y_raw = data['y']
        y_raw = np.atleast_1d(y_raw)
        
        count = 0
        
        for i in range(len(X_raw)):
            window = X_raw[i]
            label = y_raw[i]
            
            # --- MAX ENERGY ENCODING ---
            cwt_maps = []
            
            # This loop is the bottleneck, but now it runs in parallel across files
            for ch in range(window.shape[0]):
                cwt = compute_cwt_magnitude(window[ch])
                cwt_maps.append(cwt)
            
            cwt_stack = np.stack(cwt_maps, axis=0) # (23, Scales, Time)
            
            # Extract Features
            max_feat = np.max(cwt_stack, axis=0)
            mean_feat = np.mean(cwt_stack, axis=0)
            std_feat = np.std(cwt_stack, axis=0)
            
            # Create RGB Image
            rgb_img = []
            for feat in [max_feat, mean_feat, std_feat]:
                # Resize (linear is faster than cubic, almost same result for ML)
                resized = cv2.resize(feat, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
                resized = np.flipud(resized)
                rgb_img.append(resized)
                
            img_tensor = np.stack(rgb_img, axis=0)
            
            # Normalize
            img_min, img_max = img_tensor.min(), img_tensor.max()
            if img_max - img_min > 0:
                img_tensor = (img_tensor - img_min) / (img_max - img_min)
            else:
                img_tensor = np.zeros_like(img_tensor)
            
            # Save
            fname = fpath.stem.replace('_processed', '')
            save_name = OUTPUT_DIR / f"{fname}_w{i}_L{int(label)}_scalogram.npz"
            np.savez_compressed(save_name, X=img_tensor.astype(np.float32), y=label)
            count += 1
            
        return count
        
    except Exception as e:
        return f"Error {fpath.name}: {e}"

def process_and_save_images():
    files = list(INPUT_DIR.glob("*.npz"))
    print(f"Found {len(files)} signal files.")
    print(f"Starting Multiprocessing on {os.cpu_count()} CPU cores...")
    
    total_generated = 0
    
    # Use ProcessPoolExecutor to utilize all cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = list(tqdm(executor.map(process_single_file, files), total=len(files), desc="Generating Scalograms"))
        
        for result in futures:
            if isinstance(result, int):
                total_generated += result
            else:
                print(result) # Print errors

    print(f"\n✅ PHASE 2 COMPLETE. {total_generated} 'Max-Energy' images generated.")

if __name__ == "__main__":
    # Windows requires this protection for multiprocessing
    process_and_save_images()