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
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_vit import HybridViT

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_SAVE_DIR = Path("results/models")
BATCH_SIZE = 16 
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

class SeizureDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.labels = []
        print(f"Scanning {len(file_paths)} files for labels...")
        
        # Fast Scan
        for f in tqdm(file_paths):
            # We trust the filename contains the label to speed up loading
            # Filename format: "..._L1_..." or "..._L0_..."
            if "_L1_" in f.name:
                self.labels.append(1.0)
            else:
                self.labels.append(0.0)
            
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with np.load(path) as data:
                X = data['X']
                y = data['y']
            if X.ndim == 4: X = X.squeeze(0)
            y = np.atleast_1d(y)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        except:
            return torch.zeros((3, 224, 224)), torch.tensor([0.0])

def get_stratified_split(all_files):
    """
    Splits data 80/20 but GUARANTEES that 20% of seizures go to Val.
    """
    # 1. Separate by Class
    pos_files = [f for f in all_files if "_L1_" in f.name]
    neg_files = [f for f in all_files if "_L0_" in f.name]
    
    print(f"Found {len(pos_files)} Seizure files and {len(neg_files)} Normal files.")
    
    # 2. Shuffle independently
    random.seed(42)
    random.shuffle(pos_files)
    random.shuffle(neg_files)
    
    # 3. Split Positives (80/20)
    split_pos = int(0.8 * len(pos_files))
    train_pos = pos_files[:split_pos]
    val_pos = pos_files[split_pos:]
    
    # 4. Split Negatives (80/20)
    split_neg = int(0.8 * len(neg_files))
    train_neg = neg_files[:split_neg]
    val_neg = neg_files[split_neg:]
    
    # 5. Combine
    train_files = train_pos + train_neg
    val_files = val_pos + val_neg
    
    # Shuffle the combined lists so batches are mixed
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    return train_files, val_files

def train():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"--- Starting High-Acc Training (Stratified Split) on {DEVICE} ---")
    
    all_files = list(DATA_DIR.glob("*.npz"))
    
    # --- USE STRATIFIED SPLIT ---
    train_files, val_files = get_stratified_split(all_files)
    
    print(f"Training Set: {len(train_files)} files")
    print(f"Validation Set: {len(val_files)} files (Guaranteed to contain seizures)")
    
    # Save Manifest for Evaluation Script
    with open("data/test_files_manifest.txt", "w") as f:
        for path in val_files:
            f.write(str(path) + "\n")
            
    train_dataset = SeizureDataset(train_files)
    val_dataset = SeizureDataset(val_files)
    
    # Weighted Sampler
    n_pos = torch.sum(train_dataset.labels == 1).item()
    n_neg = torch.sum(train_dataset.labels == 0).item()
    
    weight_pos = 1.0 / n_pos if n_pos > 0 else 1.0
    weight_neg = 1.0 / n_neg
    
    sample_weights = torch.zeros(len(train_dataset))
    for i, label in enumerate(train_dataset.labels):
        sample_weights[i] = weight_pos if label == 1 else weight_neg
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = HybridViT().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                p = (torch.sigmoid(model(images)) > 0.5).float()
                preds.extend(p.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        acc = accuracy_score(targets, preds)
        sens = recall_score(targets, preds, zero_division=0)
        
        print(f"  Val Acc: {acc:.4f} | Sens: {sens:.4f}")
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_DIR / "best_model.pth")
            print("  --> Saved Best Model!")

if __name__ == "__main__":
    train()