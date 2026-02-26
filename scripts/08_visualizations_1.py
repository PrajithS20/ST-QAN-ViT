import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
import timm
import random
import seaborn as sns

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cpu") # CPU is safer for t-SNE memory

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. MODEL ARCHITECTURE (Required for t-SNE)
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
    def forward_features(self, x):
        # Returns the Feature Vector BEFORE classification
        feat = self.vit(x)
        q_out = self.quantum(feat)
        return torch.cat([feat, q_out], dim=1) # (Batch, 196)

# ==========================================
# 2. 3D SCALOGRAM SURFACE PLOT
# ==========================================
def plot_3d_scalogram():
    print("--- GENERATING 3D SURFACE PLOT ---")
    
    # Find a Seizure File
    all_files = list(DATA_DIR.glob("*.npz"))
    seizure_files = [f for f in all_files if "_L1_" in f.name]
    
    if not seizure_files: return
    target_path = random.choice(seizure_files)
    
    with np.load(target_path) as d: 
        img = d['X']
        if img.ndim == 4: img = img.squeeze(0)
    
    # Use Max-Energy Channel (0)
    data = img[0, ::2, ::2] # Downsample slightly for smoother 3D mesh
    
    # Grid
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface
    surf = ax.plot_surface(X, Y, data, cmap='plasma', edgecolor='none', alpha=0.9)
    
    ax.set_title("3D Energy Landscape of Seizure (Gamma Spikes)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency Scale')
    ax.set_zlabel('Energy Intensity')
    ax.view_init(elev=30, azim=225) # Rotate for best view
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "3d_scalogram_surface.png", dpi=300)
    print("✅ Saved: 3d_scalogram_surface.png")

# ==========================================
# 3. t-SNE FEATURE CLUSTERING
# ==========================================
def plot_tsne_clustering(num_samples=1000):
    print(f"\n--- GENERATING t-SNE CLUSTER MAP ({num_samples} samples) ---")
    
    # Load Model
    model = HybridViT().to(DEVICE)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except: model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    # Prepare Data Subset (Balanced)
    all_files = list(DATA_DIR.glob("*.npz"))
    pos_files = [f for f in all_files if "_L1_" in f.name]
    neg_files = [f for f in all_files if "_L0_" in f.name]
    
    # Take 500 Seizure, 500 Normal
    n_per_class = num_samples // 2
    subset = random.sample(pos_files, min(len(pos_files), n_per_class)) + \
             random.sample(neg_files, min(len(neg_files), n_per_class))
    
    features = []
    labels = []
    
    print("Extracting Features...")
    with torch.no_grad():
        for f in tqdm(subset):
            try:
                with np.load(f) as d: 
                    img = torch.tensor(d['X'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    y = d['y']
                    if img.ndim == 5: img = img.squeeze(1)
                
                # Get the 196-dim feature vector (192 classical + 4 quantum)
                feat_vec = model.forward_features(img)
                features.append(feat_vec.numpy().flatten())
                labels.append(int(np.atleast_1d(y)[0]))
            except: pass
            
    features = np.array(features)
    labels = np.array(labels)
    
    print("Running t-SNE (Dimensionality Reduction)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels, 
                    palette={0: '#3498DB', 1: '#E74C3C'}, style=labels, 
                    s=60, alpha=0.8)
    
    plt.title("t-SNE Feature Space: Separation of Seizure vs. Normal", fontsize=14, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title='Class', labels=['Normal', 'Seizure'])
    
    plt.savefig(RESULTS_DIR / "tsne_feature_clusters.png", dpi=300)
    print("✅ Saved: tsne_feature_clusters.png")

if __name__ == "__main__":
    plot_3d_scalogram()
    plot_tsne_clustering()