import sys
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_vit import HybridViT

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_map():
    print("--- GENERATING ATTENTION MAP ---")
    
    # 1. Load ONE Seizure Sample directly
    # We search for a file that contains a seizure (Class 1)
    all_files = list(DATA_DIR.glob("*.npz"))
    target_image = None
    
    print("Searching for a valid Seizure sample...")
    for f in all_files:
        loaded = np.load(f)
        ys = loaded['y']
        if np.sum(ys) > 0: # If this file has a seizure
            Xs = loaded['X']
            # Find the index of the seizure
            idx = np.where(ys == 1)[0][0]
            target_image = Xs[idx]
            print(f"Found Seizure sample in {f.name} at index {idx}")
            break
    
    if target_image is None:
        print("Error: No seizure samples found in dataset.")
        return

    # 2. Prepare Image for Model
    # Convert to Tensor (3, 224, 224) -> (1, 3, 224, 224)
    img_tensor = torch.tensor(target_image, dtype=torch.float32).to(DEVICE).unsqueeze(0)
    
    # CRITICAL FIX: Enable Gradients on the Input Image
    img_tensor.requires_grad_()
    
    # 3. Load Model
    model = HybridViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Eval mode for layers like Dropout
    
    # 4. Forward Pass
    output = model(img_tensor)
    
    # 5. Backward Pass (Calculate Saliency)
    # We want to maximize the "Seizure" class score (Index 0 if 1 output, or simple scalar)
    score = output[0] 
    score.backward()
    
    # 6. Process Gradients
    # Get maximum gradient across channels
    gradients = torch.max(torch.abs(img_tensor.grad[0]), dim=0)[0].detach().cpu().numpy()
    
    # Normalize to 0-255
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
    gradients = np.uint8(255 * gradients)
    
    # Apply Heatmap Color
    heatmap = cv2.applyColorMap(gradients, cv2.COLORMAP_JET)
    
    # Prepare Original Image (Grayscale/RGB for overlay)
    original = target_image.transpose(1, 2, 0) # (H, W, 3)
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    original = np.uint8(255 * original)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    
    # Superimpose
    superimposed = cv2.addWeighted(heatmap, 0.5, original_bgr, 0.5, 0)
    
    # Save
    save_path = RESULTS_DIR / "explainability_map.png"
    cv2.imwrite(str(save_path), superimposed)
    print(f"✅ Success! Attention Map saved to: {save_path}")

if __name__ == "__main__":
    generate_map()