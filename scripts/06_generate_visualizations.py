import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timm
import random
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
TRAIN_LOG_PATH = Path("results/tables/training_metrics.csv")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cpu") 

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. MODEL ARCHITECTURE (Required to load weights)
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
        return features, q_out, self.classifier(torch.cat([features, q_out], dim=1))

# ==========================================
# 2. SCALOGRAM GALLERY GENERATOR
# ==========================================
def plot_scalogram_gallery(num_examples=5):
    print(f"1. Generating {num_examples} Scalogram Comparisons...")
    
    all_files = list(DATA_DIR.glob("*.npz"))
    seizure_files = [f for f in all_files if "_L1_" in f.name]
    normal_files = [f for f in all_files if "_L0_" in f.name]
    
    if not seizure_files:
        print("Error: No seizure files found.")
        return

    for i in range(1, num_examples + 1):
        s_path = random.choice(seizure_files)
        n_path = random.choice(normal_files)
        
        try:
            with np.load(s_path) as d: s_img = d['X']
            with np.load(n_path) as d: n_img = d['X']
            
            if s_img.ndim == 4: s_img = s_img.squeeze(0)
            if n_img.ndim == 4: n_img = n_img.squeeze(0)
            
            s_disp = s_img[0, :, :]
            n_disp = n_img[0, :, :]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            im1 = axes[0].imshow(n_disp, cmap='jet', aspect='auto', origin='lower')
            axes[0].set_title(f"Normal EEG\n{n_path.name[:20]}...", fontsize=11)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            im2 = axes[1].imshow(s_disp, cmap='jet', aspect='auto', origin='lower')
            axes[1].set_title(f"Seizure EEG (Ictal)\n{s_path.name[:20]}...", fontsize=11, fontweight='bold', color='red')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"Example {i}: Feature Input Visualization", fontsize=15)
            plt.tight_layout()
            
            save_path = RESULTS_DIR / f"1_scalogram_comparison_{i}.png"
            plt.savefig(save_path, dpi=150)
            print(f"  -> Saved: {save_path.name}")
            plt.close()
            
        except Exception as e:
            print(f"  Skipping file due to load error: {e}")

# ==========================================
# 3. QUANTUM FEATURE MAP
# ==========================================
def plot_quantum_analysis():
    print("\n2. Generating Quantum Feature Analysis...")
    
    model = HybridViT().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    all_files = list(DATA_DIR.glob("*.npz"))
    seizure_files = [f for f in all_files if "_L1_" in f.name]
    if not seizure_files: return
    
    with np.load(seizure_files[0]) as d: 
        img = torch.tensor(d['X'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        if img.ndim == 5: img = img.squeeze(1)

    with torch.no_grad():
        c_feats, q_feats, _ = model(img)
    
    c_feats = c_feats.numpy().flatten()
    q_feats = q_feats.numpy().flatten()
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, :])
    c_grid = c_feats.reshape(12, 16)
    sns.heatmap(c_grid, ax=ax1, cmap="viridis", cbar=True)
    ax1.set_title("Classical ViT Features (192 Dimensions)\nExtracted from Temporal Backbone", fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[1, :])
    colors = ['#FF5733' if x > 0 else '#33C1FF' for x in q_feats]
    ax2.bar(range(4), q_feats, color=colors, width=0.6, edgecolor='black')
    ax2.set_title("Quantum Entanglement Features (4 Qubits)\nNon-Linear Decision Boundaries", fontsize=12, fontweight='bold')
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel("Qubit Index")
    ax2.set_ylabel("Expectation Value <Z>")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "2_quantum_feature_map.png"
    plt.savefig(save_path, dpi=300)
    print(f"  -> Saved: {save_path.name}")
    plt.close()

# ==========================================
# 4. TRAINING LOSS CURVE (NO SENSITIVITY)
# ==========================================
def plot_training_curve():
    print("\n3. Generating Training Loss Curve...")
    
    if not TRAIN_LOG_PATH.exists():
        json_path = Path("results/history.json")
        if json_path.exists():
            import json
            with open(json_path) as f: history = json.load(f)
            df = pd.DataFrame({
                "Epoch": range(1, len(history['loss']) + 1),
                "Loss": history['loss'],
                "Acc": history['accuracy']
            })
        else:
            print("Warning: No logs found.")
            return
    else:
        df = pd.read_csv(TRAIN_LOG_PATH)
        if "Accuracy" in df.columns: df.rename(columns={"Accuracy": "Acc"}, inplace=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Loss (Red)
    sns.lineplot(data=df, x="Epoch", y="Loss", ax=ax1, color="#E74C3C", label="Training Loss", linewidth=3, marker='o')
    ax1.set_ylabel("Cross Entropy Loss", color="#E74C3C", fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor="#E74C3C")
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy Only (Blue)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Epoch", y="Acc", ax=ax2, color="#2E86C1", label="Val Accuracy", linewidth=2.5, linestyle="--")
    
    # REMOVED SENSITIVITY PLOT HERE
    
    ax2.set_ylabel("Validation Accuracy (0-1)", color="#2E86C1", fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 1.05)
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", shadow=True)
    
    plt.title("ST-QAN-ViT Training Dynamics", fontsize=14, fontweight='bold')
    plt.savefig(RESULTS_DIR / "3_training_loss_curve.png", dpi=300)
    print(f"  -> Saved: {RESULTS_DIR / '3_training_loss_curve.png'}")

if __name__ == "__main__":
    plot_scalogram_gallery(5)
    plot_quantum_analysis()
    plot_training_curve()
    print("\n✅ All visualizations generated successfully.")