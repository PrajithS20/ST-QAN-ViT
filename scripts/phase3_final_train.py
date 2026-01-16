import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import glob
import numpy as np
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIR = "data/quantum_features"
MODEL_SAVE_PATH = "results/models/st_qan_vit_final.pth"

# Ensure the results directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 1. Custom Dataset Class
class SeizureDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        # Load the 4x16x16 quantum tensor generated in Phase 2
        data = torch.load(path, weights_only=True) 
        # Label 1 for seizure files, Label 0 for normal
        label = 1.0 if "_lab1" in path else 0.0
        return data, torch.tensor(label, dtype=torch.float32)

# 2. Advanced Loss: Combined Focal + Weighted BCE
# This forces the model to focus on the rare seizure events.
class SeizureFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(SeizureFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Weighting seizures 8 times more than normal samples
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]).to(DEVICE))

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss

# 3. Optimized Dataset with Class Balancing
# Every batch will contain roughly 50% Normal and 50% Seizure samples.
def get_balanced_loader(files, batch_size=32):
    dataset = SeizureDataset(files)
    labels = [1 if "_lab1" in f else 0 for f in files]
    
    class_count = np.array([labels.count(0), labels.count(1)])
    weight = 1. / class_count
    sample_weights = np.array([weight[t] for t in labels])
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# 4. Training Execution logic
def train_to_understand():
    print(f"🧠 Training for Sensitivity using ViT on: {DEVICE}")
    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    
    if not files:
        print(f"❌ Error: No .pt files found in {FEATURE_DIR}. Run Phase 2 first.")
        return

    train_loader = get_balanced_loader(files)

    # FIXED: Added num_heads=8 to make it compatible with embed_dim=256
    model = VisionTransformer(
        img_size=16, 
        patch_size=2, 
        in_chans=4, 
        num_classes=1, 
        embed_dim=256, 
        depth=6,
        num_heads=8  # 256 is divisible by 8
    ).to(DEVICE)
    
    criterion = SeizureFocalLoss()
    # AdamW optimizer as specified in the Architecture Showdown
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for epoch in range(25):
        model.train()
        epoch_loss = 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            
            # Forward pass through the Vision Transformer
            output = model(imgs)
            loss = criterion(output, labs)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1:02d} | Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Save the final model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Final Model Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_to_understand()