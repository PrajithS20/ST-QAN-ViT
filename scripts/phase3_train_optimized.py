import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import glob
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_SAVE_PATH = "results/models/st_qan_vit_optimized.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset Class (Must be included in the new file)
class SeizureDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = torch.load(path, weights_only=True) 
        label = 1.0 if "_lab1" in path else 0.0
        return data, torch.tensor(label, dtype=torch.float32)

# 2. Focal Loss for Hard-Example Mining
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

# 3. Enhanced ViT Architecture
class ST_QAN_ViT_Large(nn.Module):
    def __init__(self):
        super(ST_QAN_ViT_Large, self).__init__()
        self.vit = VisionTransformer(
            img_size=16, patch_size=4, in_chans=4, 
            num_classes=1, embed_dim=256, depth=6, num_heads=8
        )

    def forward(self, x):
        return self.vit(x)

def train_optimized():
    print(f"🚀 Starting Optimized Phase 3 on: {DEVICE}")
    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    dataset = SeizureDataset(files)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = ST_QAN_ViT_Large().to(DEVICE)
    criterion = FocalLoss(alpha=0.8, gamma=2) 
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_val = float('inf')
    for epoch in range(25):
        model.train()
        t_loss = 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
                v_loss += criterion(model(imgs), labs).item()
        
        avg_v = v_loss/len(val_loader)
        scheduler.step(avg_v)
        print(f"Epoch {epoch+1:02d} | Val Loss: {avg_v:.4f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"✨ Optimized Model Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_optimized()