import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMAL_T = 0.86  # Based on your last run

def run_smoothed_evaluation():
    model = VisionTransformer(
        img_size=16, patch_size=2, in_chans=4, 
        num_classes=1, embed_dim=256, depth=6, num_heads=8
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Sort files to maintain time-sequence for smoothing
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
    y_true, y_probs = [], []

    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            y_true.append(1 if "_lab1" in path else 0)
            y_probs.append(torch.sigmoid(model(data)).item())

    # 1. Apply the 0.86 Threshold
    raw_preds = [1 if p >= OPTIMAL_T else 0 for p in y_probs]

    # 2. Apply Temporal Smoothing (5-window consensus)
    # A prediction is only 1 if at least 4 of the last 5 windows were 1
    smoothed_preds = []
    for i in range(len(raw_preds)):
        if i < 4:
            smoothed_preds.append(raw_preds[i])
        else:
            window = raw_preds[i-4 : i+1]
            # Requires 4/5 majority to trigger an alarm
            smoothed_preds.append(1 if sum(window) >= 4 else 0)

    # 3. Final Metrics
    cm = confusion_matrix(y_true, smoothed_preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / len(files)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_hour = fp / ((len(files) * 15) / 3600)

    print("\n" + "🔥" * 10 + " TEMPORAL SMOOTHED RESULTS " + "🔥" * 10)
    print(f"✅ Accuracy:    {accuracy:.2%}")
    print(f"🔥 Sensitivity: {sensitivity:.2%}") 
    print(f"🛡️ Specificity: {specificity:.2%}")
    print(f"📉 FPR / Hour:  {fpr_hour:.4f}")
    print("🔥" * 38)

    # Plot Result #4
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=['Normal', 'Seizure'], 
                yticklabels=['Normal', 'Seizure'])
    plt.title(f"Result #4: Final Smoothed Matrix (T={OPTIMAL_T})")
    plt.savefig("results/plots/final_smoothed_matrix.png")
    plt.show()

if __name__ == "__main__":
    run_smoothed_evaluation()