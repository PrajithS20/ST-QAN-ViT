import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import timm
import random

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. MODEL ARCHITECTURE (Matches your High-Acc Run)
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
# 2. GENERATE GALLERY
# ==========================================
def generate_gallery(num_examples=5):
    print(f"--- GENERATING {num_examples} EXPLAINABILITY MAPS ---")
    
    # 1. Load Model
    model = HybridViT().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    # 2. Find Seizure Files
    all_files = list(DATA_DIR.glob("*.npz"))
    seizure_files = [f for f in all_files if "_L1_" in f.name]
    
    if len(seizure_files) < num_examples:
        print(f"Warning: Only found {len(seizure_files)} seizure files.")
        num_examples = len(seizure_files)

    # Pick Random Examples
    selected_files = random.sample(seizure_files, num_examples)

    for i, target_path in enumerate(selected_files):
        print(f"[{i+1}/{num_examples}] Analyzing: {target_path.name}")
        
        # 3. Load Data
        with np.load(target_path) as d: 
            img = d['X']
            # Handle dimensions (Batch, C, H, W) -> (C, H, W)
            if img.ndim == 4: img = img.squeeze(0)
        
        # Prepare Tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_()
        
        # 4. Forward & Backward
        output = model(img_tensor)
        score = output[0]
        model.zero_grad()
        score.backward()
        
        # 5. Extract Saliency (Max gradient across RGB channels)
        gradients = torch.max(torch.abs(img_tensor.grad[0]), dim=0)[0].detach().cpu().numpy()
        
        # Normalize
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        gradients = np.uint8(255 * gradients)
        
        # Create Heatmap
        heatmap = cv2.applyColorMap(gradients, cv2.COLORMAP_JET)
        
        # Prepare Original (Grayscale for background)
        original_gray = img[0, :, :] # Channel 0 = Max Energy
        original_gray = (original_gray - original_gray.min()) / (original_gray.max() - original_gray.min() + 1e-8)
        original_uint8 = np.uint8(255 * original_gray)
        original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2BGR)
        
        # Superimpose (Heatmap + Original)
        overlay = cv2.addWeighted(heatmap, 0.6, original_bgr, 0.4, 0)
        
        # 6. Plotting Side-by-Side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original
        axes[0].imshow(original_gray, cmap='gray')
        axes[0].set_title("Original Scalogram (Max Energy)", fontsize=12)
        axes[0].axis('off')
        
        # Saliency
        # Convert BGR to RGB for matplotlib
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Saliency Map (Model Focus)", fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save
        save_name = RESULTS_DIR / f"explainability_map_{i+1}.png"
        plt.savefig(save_name, dpi=300)
        plt.close()
        print(f"  -> Saved: {save_name.name}")

    print("\n✅ All maps generated in results/plots/")

if __name__ == "__main__":
    generate_gallery(5)