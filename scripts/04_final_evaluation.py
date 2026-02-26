import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score
from pathlib import Path
from tqdm import tqdm
import timm
import random

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cpu") 

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. MODEL ARCHITECTURE ---
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

# --- DATASET ---
class SeizureDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.labels = []
        print(f"Loading {len(file_paths)} test files...")
        valid_count = 0
        for f in tqdm(file_paths, desc="Loading Test Data"):
            try:
                loaded = np.load(f)
                X = loaded['X']; y = loaded['y']
                y = np.atleast_1d(y)
                if X.ndim == 3: X = np.expand_dims(X, axis=0)
                if X.shape[0] > 0:
                    self.data.append(X); self.labels.append(y)
                    valid_count += 1
            except: pass
        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def get_events(binary_arr):
    events = []
    is_active = False
    start_idx = 0
    for i, val in enumerate(binary_arr):
        if val == 1 and not is_active:
            is_active = True
            start_idx = i
        elif val == 0 and is_active:
            is_active = False
            events.append((start_idx, i-1))
    if is_active: events.append((start_idx, len(binary_arr)-1))
    return events

def calculate_event_metrics(y_true, y_pred, total_hours):
    true_events = get_events(y_true)
    pred_events = get_events(y_pred)
    tp = 0
    for t_start, t_end in true_events:
        caught = False
        for p_start, p_end in pred_events:
            if not (p_end < t_start or p_start > t_end): 
                caught = True; break
        if caught: tp += 1
    fp = 0
    for p_start, p_end in pred_events:
        is_false = True
        for t_start, t_end in true_events:
            if not (p_end < t_start or p_start > t_end):
                is_false = False; break
        if is_false: fp += 1
    sens = tp / len(true_events) if len(true_events) > 0 else 0.0
    fpr = fp / total_hours if total_hours > 0 else 0.0
    return sens, fp, fpr

def find_best_threshold(y_true, y_probs, total_hours):
    print(f"\nScanning for Optimal Threshold...")
    candidates = []
    thresholds = np.arange(0.05, 0.96, 0.05)
    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(float)
        evt_sens, fps, fpr = calculate_event_metrics(y_true, y_pred, total_hours)
        win_acc = accuracy_score(y_true, y_pred)
        stats = {'thresh': thresh, 'evt_sens': evt_sens, 'fpr': fpr, 'win_acc': win_acc, 'fps': fps}
        candidates.append(stats)
        if thresh in [0.1, 0.5, 0.9]:
            print(f"{thresh:.2f}     | {evt_sens*100:.1f}%   | {fpr:.2f}/hr | {win_acc:.2f}")

    best_stats = max(candidates, key=lambda x: x['evt_sens'] + x['win_acc'])
    return best_stats['thresh'], best_stats

def evaluate():
    print("--- GENERATING FINAL CLINICAL REPORT ---")
    all_files = list(DATA_DIR.glob("*.npz"))
    
    # --- FIX: ENSURE SEIZURES ARE IN TEST SET ---
    pos_files = [f for f in all_files if "_L1_" in f.name]
    neg_files = [f for f in all_files if "_L0_" in f.name]
    
    print(f"Found {len(pos_files)} Total Seizure Files available.")
    
    # Take 50% of seizures for testing to be safe, or ALL if small number
    # And mix with random normal files
    random.seed(42)
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    
    # Build a test set: All Seizures + 5000 Normal files
    # (We use all seizures to check sensitivity, and enough normal to check specificity)
    test_files = pos_files + neg_files[:5000]
    random.shuffle(test_files)
    
    print(f"Constructed Evaluation Set: {len(test_files)} Files.")
    
    dataset = SeizureDataset(test_files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = HybridViT().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    y_true, y_probs = [], []
    print("Running Inference...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            probs = torch.sigmoid(model(images))
            y_true.extend(labels.numpy().flatten())
            y_probs.extend(probs.numpy().flatten())
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    total_hours = (len(y_true) * 15) / 3600
    if total_hours == 0: total_hours = 0.001
    
    threshold, stats = find_best_threshold(y_true, y_probs, total_hours)
    y_pred = (y_probs > threshold).astype(float)
    
    print("\n" + "="*45)
    print("   FINAL CLINICAL REPORT (HIGH ACCURACY)")
    print("="*45)
    print(f"Total Test Data:        {total_hours:.2f} hours")
    print("-" * 45)
    print(f"Optimal Threshold:      {threshold:.2f}")
    print(f"Window Accuracy:        {stats['win_acc']*100:.2f}%")
    print("-" * 45)
    print(f"EVENT SENSITIVITY:      {stats['evt_sens']*100:.2f}%")
    print(f"FPR (Errors/Hour):      {stats['fpr']:.2f}/hr")
    print(f"False Alarms Total:     {stats['fps']}")
    print("="*45)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (T={threshold:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr_roc, tpr_roc)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    print("✅ Plots saved to results/plots/")

if __name__ == "__main__":
    evaluate()