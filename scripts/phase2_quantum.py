import torch
import numpy as np
import pennylane as qml
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "data/scalograms"
OUTPUT_DIR = "data/quantum_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. CPU Device Setup 
# Faster for sliding-window tasks as it avoids PCIe data movement
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Map 4 pixels to 4 qubits
    qml.AngleEmbedding(inputs, wires=range(4))
    # Entangle qubits for feature extraction
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Fixed random weights (MUST be the same for all files)
weights = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4))

def process_scalogram_cpu(image_np):
    """Processes 32x32 image into 4x16x16 quantum features using CPU"""
    output = np.zeros((4, 16, 16))
    
    # Quanv2D: Sliding 2x2 window with stride 2
    for i in range(0, 32, 2):
        for j in range(0, 32, 2):
            # Extract and normalize patch
            patch = image_np[i:i+2, j:j+2].flatten()
            patch = patch * np.pi 
            
            # Execute on CPU directly
            q_out = quantum_circuit(patch, weights)
            output[:, i//2, j//2] = q_out
            
    return torch.tensor(output).float()

if __name__ == "__main__":
    print("🚀 Running Phase 2 on CPU (Removing PCIe Bottleneck)...")
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.npy')]
    
    # Filter out files already processed to save time
    existing = [f.replace('.pt', '.npy') for f in os.listdir(OUTPUT_DIR)]
    files_to_process = [f for f in files if f not in existing]

    for filename in tqdm(files_to_process):
        img = np.load(os.path.join(INPUT_DIR, filename))
        
        # CPU-based extraction
        features = process_scalogram_cpu(img)
        
        # Save as .pt for Phase 3 training
        save_name = filename.replace('.npy', '.pt')
        torch.save(features, os.path.join(OUTPUT_DIR, save_name))

    print(f"✨ Phase 2 Complete. Features ready in: {OUTPUT_DIR}")