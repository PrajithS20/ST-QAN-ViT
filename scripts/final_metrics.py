import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Final Architecture (Must match scripts/phase3_final_train.py)
class ST_QAN_ViT_V3(nn.Module):
    def __init__(self):
        super(ST_QAN_ViT_V3, self).__init__()
        self.vit = VisionTransformer(
            img_size=16, patch_size=2, in_chans=4, 
            num_classes=1, embed_dim=384, depth=8, num_heads=12
        )
    def forward(self, x): return self.vit(x)

def run_final_evaluation():
    print(f"📊 Analyzing Final Model Performance on: {DEVICE}")
    model = ST_QAN_ViT_V3().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    y_true, y_pred = [], []

    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            y_true.append(1 if "_lab1" in path else 0)
            # Use 0.5 threshold for standard accuracy reporting
            prob = torch.sigmoid(model(data)).item()
            y_pred.append(1 if prob >= 0.5 else 0)

    # 2. Metrics Calculation
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / len(files)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_hour = fp / ((len(files) * 15) / 3600)

    print("\n" + "🏁" * 10 + " FINAL PROJECT METRICS " + "🏁" * 10)
    print(f"✅ Raw Accuracy:       {accuracy:.2%}")
    print(f"⚖️  Balanced Accuracy:  {b_acc:.2%}") 
    print(f"🔥 Final Sensitivity:  {sensitivity:.2%}")
    print(f"📉 Final FPR / Hour:   {fpr_hour:.4f}")
    print("🏁" * 31)

    # 3. Save Final Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', 
                xticklabels=['Normal', 'Seizure'], 
                yticklabels=['Normal', 'Seizure'])
    plt.title(f"Final Result #4: Advanced ST-QAN-ViT Matrix")
    plt.savefig("results/plots/final_advanced_matrix.png")
    plt.show()

if __name__ == "__main__":
    run_final_evaluation()