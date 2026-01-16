import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
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

def optimize_for_accuracy():
    model = ST_QAN_ViT_Large().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    y_true, y_probs = [], []

    print("📊 Extracting probabilities for Accuracy Optimization...")
    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            y_true.append(1 if "_lab1" in path else 0)
            y_probs.append(torch.sigmoid(model(data)).item())

    # 1. Grid Search for Maximum Accuracy
    thresholds = np.linspace(0.1, 0.9, 100)
    accuracies = [accuracy_score(y_true, [1 if p >= t else 0 for p in y_probs]) for t in thresholds]
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    max_accuracy = accuracies[best_idx]

    # 2. Final Metrics at Best Accuracy Threshold
    final_preds = [1 if p >= best_threshold else 0 for p in y_probs]
    tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()
    
    sensitivity = tp / (tp + fn)
    fpr_hour = fp / ((len(files) * 15) / 3600)

    print("\n" + "🎯" * 10 + " ACCURACY OPTIMIZED REPORT " + "🎯" * 10)
    print(f"📍 Optimal Accuracy Threshold: {best_threshold:.4f}")
    print(f"✅ Maximum Accuracy:           {max_accuracy:.2%}") 
    print(f"🔥 Resulting Sensitivity:      {sensitivity:.2%}")
    print(f"📉 Resulting FPR / Hour:       {fpr_hour:.4f}")
    print("🎯" * 32)

    # 3. Plot Result #4: Accuracy-First Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, final_preds), annot=True, fmt='d', cmap='YlGn')
    plt.title(f"Result #4: Accuracy-Optimized Matrix (T={best_threshold:.2f})")
    plt.savefig("results/plots/accuracy_optimized_matrix.png")
    plt.show()

if __name__ == "__main__":
    optimize_for_accuracy()