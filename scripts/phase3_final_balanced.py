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
MODEL_SAVE_PATH = "results/models/st_qan_vit_balanced.pth"

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 1. Custom Dataset Class
class SeizureDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        # Load the 4x16x16 quantum tensor
        data = torch.load(path, weights_only=True) 
        # Label 1 for seizure files, 0 for normal
        label = 1.0 if "_lab1" in path else 0.0
        return data, torch.tensor(label, dtype=torch.float32)

# 2. Balanced Focal Loss
# Adjusting pos_weight to 3.0 reduces the massive False Positive count.
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

# 3. Balanced Data Loader
# Ensures the model sees Seizures and Normal signals equally in every batch.
def get_balanced_loader(files, batch_size=32):
    dataset = SeizureDataset(files)
    labels = [1 if "_lab1" in f else 0 for f in files]
    
    class_count = np.array([labels.count(0), labels.count(1)])
    weight = 1. / class_count
    sample_weights = np.array([weight[t] for t in labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# 4. Main Training Logic
def train_balanced():
    print(f"🧠 Training Balanced ST-QAN-ViT on: {DEVICE}")
    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    
    if not files:
        print(f"❌ Error: No files found in {FEATURE_DIR}. Check your Phase 2 output.")
        return

    train_loader = get_balanced_loader(files)

    # ViT Config: patch_size=2 for high resolution of quantum spikes
    model = VisionTransformer(
        img_size=16, patch_size=2, in_chans=4, 
        num_classes=1, embed_dim=256, depth=6, num_heads=8
    ).to(DEVICE)
    
    criterion = BalancedFocalLoss()
    # AdamW with weight decay handles hybrid model weights better
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    for epoch in range(30): 
        model.train()
        epoch_loss = 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labs)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1:02d} | Avg Loss: {epoch_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Balanced Model Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_balanced()