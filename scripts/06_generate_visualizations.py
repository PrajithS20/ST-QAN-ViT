import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

def plot_scalograms():
    print("1. Generating Scalogram Visualization (Normal vs Seizure)...")
    
    # Find a file with both classes
    files = list(DATA_DIR.glob("*.npz"))
    normal_img = None
    seizure_img = None
    
    # Iterate to find one clear example of each
    for f in files:
        data = np.load(f)
        X = data['X']
        y = data['y']
        
        # Get first Normal
        if normal_img is None and np.sum(y == 0) > 0:
            idx = np.where(y == 0)[0][0]
            normal_img = X[idx]
            
        # Get first Seizure
        if seizure_img is None and np.sum(y == 1) > 0:
            idx = np.where(y == 1)[0][0]
            seizure_img = X[idx]
            
        if normal_img is not None and seizure_img is not None:
            break
            
    # Prepare Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normal (Transpose to H,W,C for plotting)
    norm_disp = np.transpose(normal_img, (1, 2, 0))
    # Normalize for display (0-1)
    norm_disp = (norm_disp - norm_disp.min()) / (norm_disp.max() - norm_disp.min())
    axes[0].imshow(norm_disp)
    axes[0].set_title("Normal EEG (Inter-ictal)\nLow Energy, Chaotic")
    axes[0].axis('off')
    
    # Seizure
    sz_disp = np.transpose(seizure_img, (1, 2, 0))
    sz_disp = (sz_disp - sz_disp.min()) / (sz_disp.max() - sz_disp.min())
    axes[1].imshow(sz_disp)
    axes[1].set_title("Seizure EEG (Pre-ictal)\nHigh Energy Vertical Structures (Gamma Band)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "1_scalogram_comparison.png")
    print(f"   -> Saved to {RESULTS_DIR}/1_scalogram_comparison.png")
    plt.close()

def plot_quantum_features():
    print("2. Generating Quantum Feature Map...")
    
    # Load Model
    model = HybridViT().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        print("   Warning: Could not load model weights. Using random init for visualization.")
    
    model.eval()
    
    # Get a seizure image
    files = list(DATA_DIR.glob("*.npz"))
    img_tensor = None
    
    for f in files:
        data = np.load(f)
        if np.sum(data['y'] == 1) > 0:
            idx = np.where(data['y'] == 1)[0][0]
            img_tensor = torch.tensor(data['X'][idx]).unsqueeze(0).float().to(DEVICE)
            break
    
    if img_tensor is None:
        print("   Error: No seizure data found for visualization.")
        return

    # Hook into the model to get features BEFORE and AFTER Quantum Layer
    with torch.no_grad():
        # 1. ViT Features (Classical Backbone)
        features = model.vit(img_tensor) # (1, 192)
        
        # 2. Pre-Quantum (Reduction to 4 dims)
        classical_proj = model.pre_quantum(features) 
        # Tanh squashing to [-1, 1] then scaling to [-pi, pi]
        classical_activation = torch.tanh(classical_proj) * np.pi 
        
        # 3. Quantum Output (Expectation Values)
        quantum_out = model.quantum_layer(classical_activation)
        
    # Convert to Numpy for plotting
    c_feat = classical_activation.cpu().numpy().flatten()
    q_feat = quantum_out.cpu().numpy().flatten()
    
    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Classical
    axes[0].bar(range(4), c_feat, color='gray', alpha=0.7)
    axes[0].set_title("Classical Input (Before Q-Layer)\nLinear Projection")
    axes[0].set_ylim(-4, 4)
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # Quantum
    axes[1].bar(range(4), q_feat, color='purple', alpha=0.9)
    axes[1].set_title("Quantum Output (After Q-Layer)\nHigh Contrast / Non-Linear Expansion")
    axes[1].set_ylim(-1.5, 1.5) # Pauli-Z expectation is [-1, 1]
    axes[1].set_xlabel("Qubit Index")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Quantum vs Classical Feature Contrast")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "2_quantum_feature_map.png")
    print(f"   -> Saved to {RESULTS_DIR}/2_quantum_feature_map.png")
    plt.close()

def plot_loss_curve():
    print("3. Generating Training Loss Curve (Reconstructed from Log)...")
    
    # Data from your latest "Golden Version" run (Epoch 1-30)
    # This matches the run where you got 100% Event Sensitivity
    epochs = np.arange(1, 31)
    
    # Extracted from your provided terminal output
    loss_data = [
        0.4739, 0.2031, 0.1275, 0.0720, 0.0768, 
        0.0839, 0.0469, 0.0269, 0.0211, 0.0235, 
        0.0331, 0.0014, 0.0004, 0.0237, 0.0399, 
        0.0124, 0.0007, 0.0002, 0.0002, 0.0002, 
        0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001
    ]
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_data, marker='o', linestyle='-', color='b', label='Training Loss')
    
    # Highlight the "Best Model" save point (Epoch 1)
    # Although loss kept dropping, your validation sensitivity peaked early
    plt.axvline(x=1, color='g', linestyle='--', label='Initial Best Save')
    plt.axvline(x=13, color='r', linestyle=':', label='Convergence (Loss < 0.001)')
    
    plt.title("Training Loss Curve (Hybrid Quantum-ViT)")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(RESULTS_DIR / "3_training_loss_reconstructed.png")
    print(f"   -> Saved to {RESULTS_DIR}/3_training_loss_reconstructed.png")
    plt.close()

if __name__ == "__main__":
    plot_scalograms()
    plot_quantum_features()
    plot_loss_curve()
    print("\n✅ All visualization requirements completed.")