import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import os
import glob
import numpy as np
from timm.models.vision_transformer import VisionTransformer
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIR = "data/quantum_features"
MODEL_SAVE_PATH = "results/models/st_qan_vit_best_optimized.pth"
BEST_MODEL_PATH = "results/models/st_qan_vit_best_optimized_checkpoint.pth"
PLOT_DIR = "results/plots"

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. Enhanced Dataset with stronger augmentation
class SeizureDataset(Dataset):
    def __init__(self, file_list, augment=False):
        self.file_list = file_list
        self.augment = augment
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = torch.load(path, weights_only=True)
        
        # Enhanced augmentation for better generalization
        if self.augment:
            # Random scaling
            scale = torch.tensor(np.random.uniform(0.85, 1.15))
            # Random noise
            noise = torch.randn_like(data) * np.random.uniform(0.02, 0.08)
            # Mixup-like augmentation for seizure/normal
            alpha = np.random.uniform(0.0, 0.1)
            data = data * scale + noise
            data = torch.clamp(data, -1.0, 1.0)
        
        label = 1.0 if "_lab1" in path else 0.0
        return data, torch.tensor(label, dtype=torch.float32)

# 2. Improved Focal Loss with hard negative mining
class HardNegativeFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.5, pos_weight=8.0):
        super(HardNegativeFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # Compute raw BCE loss
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, min=1e-7, max=1-1e-7)
        
        # BCE with logits (more numerically stable)
        ce_loss = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        
        # Apply class weighting
        ce_loss = ce_loss * (targets * self.pos_weight + (1 - targets))
        
        # Focal weight: focus on hard examples
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        focal_loss = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()

# 3. Balanced Data Loader with careful sampling
def get_balanced_loader(files, batch_size=24, augment=False):
    dataset = SeizureDataset(files, augment=augment)
    labels = [1 if "_lab1" in f else 0 for f in files]
    
    class_count = np.array([labels.count(0), labels.count(1)])
    print(f"📊 Class Distribution - Normal: {class_count[0]}, Seizure: {class_count[1]}")
    
    # More aggressive weighting for minority class
    weight = np.array([1.0 / class_count[0], 2.0 / class_count[1]])
    sample_weights = np.array([weight[t] for t in labels])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)

# 4. Improved ViT with regularization
class RegularizedViT(nn.Module):
    def __init__(self, img_size=16, patch_size=2, in_chans=4, 
                 embed_dim=256, depth=6, num_heads=8):
        super(RegularizedViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=0.15,
            attn_drop_rate=0.15,
            drop_path_rate=0.15
        )
    
    def forward(self, x):
        return self.vit(x)

# 5. Training with improved learning strategy
def train_best_optimized():
    print(f"🔥 Starting Best Optimized Training on: {DEVICE}\n")
    
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
    
    if not files:
        print(f"❌ No .pt files found in {FEATURE_DIR}")
        return

    # 80/20 split
    train_count = int(0.80 * len(files))
    train_files = files[:train_count]
    val_files = files[train_count:]
    
    train_loader = get_balanced_loader(train_files, batch_size=24, augment=True)
    val_loader = get_balanced_loader(val_files, batch_size=24, augment=False)

    model = RegularizedViT(embed_dim=256, depth=6, num_heads=8).to(DEVICE)
    
    # Loss with better hyperparameters
    criterion = HardNegativeFocalLoss(alpha=0.35, gamma=2.5, pos_weight=8.0)
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.02, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)
    
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"📊 Training: {len(train_files)} | Validation: {len(val_files)}\n")
    
    for epoch in range(50):
        # Training
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
        
        # Validation
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labs)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.extend(probs.flatten())
                val_preds.extend((probs >= 0.5).astype(int).flatten())
                val_labels.extend(labs.cpu().numpy().flatten())
        
        # Metrics
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"E{epoch+1:2d} | TrL: {avg_train_loss:.4f} | VaL: {avg_val_loss:.4f} | F1: {val_f1:.4f} | BAcc: {val_balanced_acc:.4f}", end="")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(" ✅ NEW BEST")
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✨ Training Complete!")
    print(f"💾 Final model: {MODEL_SAVE_PATH}")
    print(f"🏆 Best model:  {BEST_MODEL_PATH}\n")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    print_comprehensive_evaluation(model, files)

def print_comprehensive_evaluation(model, files):
    """Print detailed evaluation metrics"""
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
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / len(files)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("=" * 70)
    print("📊 FINAL COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"\n✅ ACCURACY METRICS:")
    print(f"   Raw Accuracy:      {accuracy:7.2%}")
    print(f"   Balanced Acc:      {balanced_acc:7.2%}")
    print(f"   F1-Score:          {f1:7.4f}")
    print(f"\n🎯 CLINICAL METRICS:")
    print(f"   Sensitivity (TPR): {sensitivity:7.2%}  ← Must detect seizures!")
    print(f"   Specificity (TNR): {specificity:7.2%}  ← Minimize false alarms")
    print(f"   Precision:         {precision:7.2%}")
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   True Positives:    {tp:4d}  ✅ Correctly detected seizures")
    print(f"   True Negatives:    {tn:4d}  ✅ Correctly identified normal")
    print(f"   False Positives:   {fp:4d}  ❌ False alarms")
    print(f"   False Negatives:   {fn:4d}  ❌ MISSED SEIZURES (BAD!)")
    print(f"   Total:             {len(files):4d}")
    print("=" * 70)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0],
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'],
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontweight='bold')
    axes[0].set_ylabel('Actual', fontweight='bold')
    axes[0].set_title('Confusion Matrix\n(Optimized Model)', fontweight='bold', fontsize=12)
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
    values = [accuracy, sensitivity, specificity, precision, f1]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylim([0, 1.1])
    axes[1].set_ylabel('Score', fontweight='bold')
    axes[1].set_title('Performance Metrics', fontweight='bold', fontsize=12)
    axes[1].grid(alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2%}' if val <= 1 else f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/final_optimized_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Results saved to: {PLOT_DIR}/final_optimized_matrix.png")

if __name__ == "__main__":
    train_best_optimized()
