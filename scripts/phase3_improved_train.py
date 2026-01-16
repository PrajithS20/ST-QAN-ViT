import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import glob
import numpy as np
from timm.models.vision_transformer import VisionTransformer
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIR = "data/quantum_features"
MODEL_SAVE_PATH = "results/models/st_qan_vit_improved.pth"
BEST_MODEL_PATH = "results/models/st_qan_vit_improved_best.pth"
PLOT_DIR = "results/plots"

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. Custom Dataset Class with Augmentation
class SeizureDataset(Dataset):
    def __init__(self, file_list, augment=False):
        self.file_list = file_list
        self.augment = augment
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = torch.load(path, weights_only=True)
        
        # Data augmentation: random noise + scaling for robustness
        if self.augment:
            noise = torch.randn_like(data) * 0.05
            scale = torch.tensor(np.random.uniform(0.9, 1.1))
            data = data * scale + noise
            data = torch.clamp(data, -1.0, 1.0)
        
        label = 1.0 if "_lab1" in path else 0.0
        return data, torch.tensor(label, dtype=torch.float32)

# 2. Advanced Loss: Weighted Focal Loss for Class Imbalance
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2.0, pos_weight=10.0):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss

# 3. Balanced Data Loader
def get_balanced_loader(files, batch_size=32, augment=False, shuffle=True):
    dataset = SeizureDataset(files, augment=augment)
    labels = [1 if "_lab1" in f else 0 for f in files]
    
    class_count = np.array([labels.count(0), labels.count(1)])
    print(f"📊 Class Distribution: Normal={class_count[0]}, Seizure={class_count[1]}")
    
    # Weighted sampling ensures balanced batches
    weight = 1. / class_count
    sample_weights = np.array([weight[t] for t in labels])
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)

# 4. Improved ViT Architecture (smaller, more efficient)
class ImprovedViT(nn.Module):
    def __init__(self, img_size=16, patch_size=2, in_chans=4, 
                 embed_dim=256, depth=6, num_heads=8):
        super(ImprovedViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )
    
    def forward(self, x):
        return self.vit(x)

# 5. Training with Early Stopping
def train_improved():
    print(f"🚀 Starting Improved Training on: {DEVICE}")
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
    
    if not files:
        print(f"❌ Error: No .pt files found in {FEATURE_DIR}. Run Phase 2 first.")
        return

    # Split data: 85% train, 15% validation
    val_count = max(1, len(files) // 7)
    train_files = files[:-val_count]
    val_files = files[-val_count:]
    
    train_loader = get_balanced_loader(train_files, batch_size=16, augment=True)
    val_loader = get_balanced_loader(val_files, batch_size=16, augment=False, shuffle=False)

    # Initialize model
    model = ImprovedViT(
        img_size=16, 
        patch_size=2, 
        in_chans=4, 
        embed_dim=256, 
        depth=6,
        num_heads=8
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = ImprovedFocalLoss(alpha=0.3, gamma=2.0, pos_weight=10.0)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_balanced_acc = 0.0
    patience = 8
    patience_counter = 0
    
    print(f"📊 Training Samples: {len(train_files)}, Validation Samples: {len(val_files)}")
    
    for epoch in range(40):
        # Training phase
        model.train()
        train_loss = 0.0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labs)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_labels.extend(labs.cpu().numpy().flatten())
        
        # Calculate metrics
        val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds]
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds_binary)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/40 | TrLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | BalAcc: {val_balanced_acc:.4f}")
        
        # Early stopping with best model saving
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   ✅ New best model saved! Balanced Accuracy: {best_balanced_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✨ Training Complete!")
    print(f"💾 Final model saved to: {MODEL_SAVE_PATH}")
    print(f"🏆 Best model saved to: {BEST_MODEL_PATH}")
    
    # Evaluate on all data
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION ON ALL DATA")
    print("="*60)
    evaluate_model(model, files)

def evaluate_model(model, files):
    """Comprehensive evaluation"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            output = model(data)
            prob = torch.sigmoid(output).item()
            y_true.append(1 if "_lab1" in path else 0)
            y_prob.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)
    
    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / len(files)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"   ✅ Raw Accuracy:      {accuracy:.2%}")
    print(f"   ⚖️  Balanced Accuracy: {balanced_acc:.2%}")
    print(f"   🔴 Sensitivity (TPR): {sensitivity:.2%}")
    print(f"   🔵 Specificity (TNR): {specificity:.2%}")
    print(f"   📍 Precision:         {precision:.2%}")
    print(f"   💯 F1-Score:         {f1:.4f}")
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   TP (Seizure Detected):  {tp}")
    print(f"   TN (Normal Detected):   {tn}")
    print(f"   FP (False Alarm):       {fp}")
    print(f"   FN (Missed Seizure):    {fn}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Normal', 'Seizure'], 
                yticklabels=['Normal', 'Seizure'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title('Result #4: Improved ST-QAN-ViT Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/improved_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Confusion matrix saved to: {PLOT_DIR}/improved_confusion_matrix.png")

if __name__ == "__main__":
    train_improved()
