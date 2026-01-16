import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_vit import HybridViT

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_SAVE_DIR = Path("results/models")
PLOT_SAVE_DIR = Path("results/plots")
BATCH_SIZE = 16 
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

class SeizureDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.labels = []
        print("Loading dataset into RAM...")
        for f in tqdm(file_paths):
            loaded = np.load(f)
            X = loaded['X'] 
            y = loaded['y'] 
            if X.shape[0] > 0:
                self.data.append(X)
                self.labels.append(y)
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def get_metrics(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds, zero_division=0)
    return acc, sens

def plot_training_results(history):
    epochs = range(1, len(history['loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Loss (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, history['loss'], color=color, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot Sensitivity (Right Axis)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Validation Sensitivity (Recall)', color=color) 
    ax2.plot(epochs, history['sensitivity'], color=color, linewidth=2, linestyle='--', label='Val Sensitivity')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Dynamics: Loss vs Sensitivity')
    fig.tight_layout()
    
    save_path = PLOT_SAVE_DIR / "3_training_loss_curve.png"
    plt.savefig(save_path)
    print(f"✅ Training Graph saved to {save_path}")

def train():
    print(f"--- Starting Retraining (GOLDEN VERSION) on {DEVICE} ---")
    
    # Data Setup
    all_files = list(DATA_DIR.glob("*.npz"))
    full_dataset = SeizureDataset(all_files)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Weighted Sampler
    train_indices = train_dataset.indices
    all_labels = full_dataset.labels.squeeze()
    train_labels = all_labels[train_indices]
    n_pos = torch.sum(train_labels == 1).item()
    n_neg = torch.sum(train_labels == 0).item()
    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    sample_weights = torch.zeros(len(train_dataset))
    sample_weights[train_labels == 1] = weight_pos
    sample_weights[train_labels == 0] = weight_neg
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    model = HybridViT().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # --- RESTORED SCHEDULER (Crucial for high accuracy) ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Metric History
    history = {'loss': [], 'accuracy': [], 'sensitivity': []}
    best_sensitivity = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        val_acc, val_sens = get_metrics(val_loader, model, DEVICE)
        
        # Scheduler Step
        scheduler.step(avg_train_loss) # Monitor Loss to drop LR
        
        # Store metrics
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(val_acc)
        history['sensitivity'].append(val_sens)
        
        print(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | SENSITIVITY: {val_sens:.4f}")

        # Save Best Model (Logic: Prioritize Sensitivity)
        if val_sens > best_sensitivity:
            best_sensitivity = val_sens
            torch.save(model.state_dict(), MODEL_SAVE_DIR / "best_model.pth")
            print("  --> Best Sensitivity Model Saved!")
        elif val_sens == best_sensitivity and val_acc > 0.90:
             # Tie-breaker: Accuracy
             torch.save(model.state_dict(), MODEL_SAVE_DIR / "best_model.pth")
             print("  --> Best Model Updated (Higher Acc)")

    # Final Plot
    plot_training_results(history)
    print("\nTraining Complete.")

if __name__ == "__main__":
    train()