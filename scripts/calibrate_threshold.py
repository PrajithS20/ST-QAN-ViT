import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_optimized.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ST_QAN_ViT_Large(nn.Module):
    def __init__(self):
        super(ST_QAN_ViT_Large, self).__init__()
        self.vit = VisionTransformer(
            img_size=16, patch_size=4, in_chans=4, 
            num_classes=1, embed_dim=256, depth=6, num_heads=8
        )
    def forward(self, x): return self.vit(x)

def calibrate():
    model = ST_QAN_ViT_Large().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    y_true, y_probs = [], []

    print("📊 Gathering model probabilities...")
    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            label = 1 if "_lab1" in path else 0
            prob = torch.sigmoid(model(data)).item()
            y_true.append(label)
            y_probs.append(prob)

    # 1. Find Optimal Threshold for Sensitivity >= 92%
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Filter thresholds where sensitivity (tpr) is at least 92%
    idx = np.where(tpr >= 0.92)[0][0] 
    optimal_threshold = thresholds[idx]

    # 2. Final Evaluation
    preds = [1 if p >= optimal_threshold else 0 for p in y_probs]
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    sensitivity = tp / (tp + fn)
    fpr_hour = fp / ((len(files) * 15) / 3600)

    print("\n" + "🏆" * 10 + " FINAL CALIBRATED REPORT " + "🏆" * 10)
    print(f"📍 Calibrated Threshold: {optimal_threshold:.4f}")
    print(f"🔥 Target Sensitivity:   {sensitivity:.2%}") 
    print(f"📉 Resulting FPR / Hour: {fpr_hour:.4f}")
    print("🏆" * 32)

    # 3. Result #4: Final Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Result #4: Final Clinical Matrix (T={optimal_threshold:.3f})")
    plt.savefig("results/plots/final_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    calibrate()