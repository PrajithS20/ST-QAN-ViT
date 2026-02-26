import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import timm
import re

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cpu") 

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. MODEL ARCHITECTURE (Required for Loading)
# ==========================================
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_features, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.fc_compress = nn.Linear(n_features, n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        q_in = torch.tanh(self.fc_compress(x)) * np.pi 
        q_out = self.q_layer(q_in) 
        return q_out

class HybridViT(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.n_features = self.vit.head.in_features
        self.vit.head = nn.Identity() 
        self.quantum = QuantumLayer(self.n_features)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.n_features + 4),
            nn.Linear(self.n_features + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.vit(x) 
        q_out = self.quantum(features) 
        combined = torch.cat([features, q_out], dim=1)
        return self.classifier(combined)

# ==========================================
# 2. QUANTUM WEIGHTS HEATMAP
# ==========================================
def plot_quantum_weights():
    print("--- GENERATING QUANTUM WEIGHT HEATMAP ---")
    
    # Load Model
    model = HybridViT().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    
    # Extract Weights: Shape (Layers, Qubits, 3 Parameters)
    # The PennyLane TorchLayer stores weights in 'weights' parameter
    weights = model.quantum.q_layer.weights.detach().cpu().numpy()
    
    # Reshape for visualization: (Layers * Qubits, 3) or (Layers, Qubits * 3)
    # Let's visualize Layer 0 vs Layer 1
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Layer 1 Weights
    sns.heatmap(weights[0], ax=axes[0], cmap="viridis", annot=True, fmt=".2f", cbar=False)
    axes[0].set_title("Quantum Layer 1 Weights\n(Rotation Angles)", fontsize=12)
    axes[0].set_ylabel("Qubit Index")
    axes[0].set_xlabel("Rotation Params (Rx, Ry, Rz)")
    
    # Layer 2 Weights
    sns.heatmap(weights[1], ax=axes[1], cmap="magma", annot=True, fmt=".2f")
    axes[1].set_title("Quantum Layer 2 Weights\n(Entanglement Strength)", fontsize=12)
    axes[1].set_ylabel("Qubit Index")
    axes[1].set_xlabel("Rotation Params")
    
    plt.suptitle("Learned Quantum Parameters (The 'Black Box' Opened)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "quantum_weights_heatmap.png", dpi=300)
    print("✅ Saved: quantum_weights_heatmap.png")

# ==========================================
# 3. SEIZURE RADAR (TEMPORAL TRACK)
# ==========================================
# ==========================================
# 3. SEIZURE RADAR (FIXED SCALING)
# ==========================================
def plot_seizure_radar():
    print("\n--- GENERATING SEIZURE RADAR (TEMPORAL TRACK) ---")
    
    model = HybridViT().to(DEVICE)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    all_files = list(DATA_DIR.glob("*.npz"))
    
    # Group files by Recording ID
    recordings = {}
    for f in all_files:
        parts = f.name.split('_')
        rec_id = "_".join(parts[:2]) 
        try:
            w_str = [p for p in parts if p.startswith('w')][0]
            w_idx = int(w_str[1:])
            if rec_id not in recordings: recordings[rec_id] = []
            recordings[rec_id].append((w_idx, f))
        except: continue

    # Find transition
    target_rec = None
    target_sequence = []
    
    for rec_id, files in recordings.items():
        files.sort(key=lambda x: x[0])
        labels = [1 if "_L1_" in f[1].name else 0 for f in files]
        
        # Look for onset
        for i in range(len(labels) - 5):
            if labels[i] == 0 and labels[i+1] == 0 and labels[i+2] == 1:
                # Grab MORE context (30 windows)
                start_idx = max(0, i - 10)
                end_idx = min(len(files), i + 20)
                target_sequence = files[start_idx:end_idx]
                target_rec = rec_id
                break
        if target_rec: break
    
    if not target_sequence:
        print("Could not find a clean seizure onset sequence.")
        return

    print(f"Plotting Radar for: {target_rec} ({len(target_sequence)} windows)")
    
    probs = []
    true_labels = []
    stitched_img = []
    
    for w_idx, fpath in target_sequence:
        with np.load(fpath) as d: 
            img = d['X']
            if img.ndim == 4: img = img.squeeze(0)
            
            t_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(model(t_img)).item()
                
            probs.append(prob)
            true_labels.append(1 if "_L1_" in fpath.name else 0)
            stitched_img.append(img[0, :, :]) 

    full_spectrogram = np.concatenate(stitched_img, axis=1)
    
    # --- PLOTTING FIX ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. Spectrogram
    # extent=[left, right, bottom, top] ensures it aligns with the 0..N axis of the plot below
    num_windows = len(probs)
    ax1.imshow(full_spectrogram, cmap='jet', aspect='auto', origin='lower', 
               extent=[0, num_windows, 0, full_spectrogram.shape[0]])
    
    ax1.set_title(f"EEG Energy Spectrum (Time Progression) - {target_rec}", fontsize=12)
    ax1.set_ylabel("Frequency Scale")
    
    # 2. Probability Track
    x_axis = np.arange(num_windows) + 0.5 # Shift slightly to center on window
    
    # Draw logic
    ax2.plot(x_axis, probs, color='blue', linewidth=3, label="Model Probability")
    ax2.fill_between(x_axis, probs, alpha=0.3, color='blue')
    
    # Ground Truth (Step plot)
    ax2.step(np.arange(num_windows), true_labels, where='post', color='red', linestyle='--', linewidth=2, label="Actual Seizure", alpha=0.8)
    
    ax2.set_ylabel("Seizure Probability")
    ax2.set_xlabel("Time (30s Windows)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(0, num_windows) # Lock axis to window count
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    ax2.axhline(0.5, color='gray', linestyle=':', label='Threshold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "seizure_radar_track.png", dpi=300)
    print("✅ Saved Corrected Radar: seizure_radar_track.png")

if __name__ == "__main__":
    plot_quantum_weights()
    plot_seizure_radar()