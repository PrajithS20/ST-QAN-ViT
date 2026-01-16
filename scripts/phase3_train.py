import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import glob
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_SAVE_DIR = "results/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class SeizureDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        # Use weights_only=True for security
        data = torch.load(path, weights_only=True) 
        label = 1.0 if "_lab1" in path else 0.0
        return data, torch.tensor(label, dtype=torch.float32)

class ST_QAN_ViT(nn.Module):
    def __init__(self):
        super(ST_QAN_ViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=16, patch_size=4, in_chans=4, 
            num_classes=1, embed_dim=128, depth=4, num_heads=4
        )
        # REMOVED Sigmoid: BCEWithLogitsLoss handles it for better stability

    def forward(self, x):
        return self.vit(x)

def train_model():
    print(f"🚀 Starting Weighted Training on: {DEVICE}")
    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    dataset = SeizureDataset(files)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = ST_QAN_ViT().to(DEVICE)
    
    # CALCULATE WEIGHT: (Normal Samples / Seizure Samples)
    # Based on your confusion matrix: 2167 / 276 = ~7.85
    # We use 8.0 to strongly penalize missed seizures
    class_weight = torch.tensor([8.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')

    for epoch in range(20): # Increased epochs for better convergence with weighting
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/st_qan_vit_best.pth")

    print(f"✨ Model re-trained and saved.")

if __name__ == "__main__":
    train_model()