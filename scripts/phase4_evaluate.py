import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_best.pth"
PLOT_DIR = "results/plots"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Architecture Setup (Must match the new weighted training script)
class ST_QAN_ViT(nn.Module):
    def __init__(self):
        super(ST_QAN_ViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=16, patch_size=4, in_chans=4, 
            num_classes=1, embed_dim=128, depth=4, num_heads=4
        )

    def forward(self, x):
        return self.vit(x) # Returns raw logits

# 2. Evaluation Logic
def evaluate():
    print(f"📊 Running Evaluation on Best Model (Epoch 12 State)...")
    
    # Load Model with weights_only for security
    model = ST_QAN_ViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = glob.glob(os.path.join(FEATURE_DIR, "*.pt"))
    y_true, y_pred = [], []

    with torch.no_grad():
        for path in files:
            # Load quantum tensor and move to GPU
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            label = 1 if "_lab1" in path else 0
            
            # Apply Sigmoid to the raw logit output
            logits = model(data)
            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability > 0.5 else 0
            
            y_true.append(label)
            y_pred.append(prediction)

    # 3. Clinical Metrics Calculation
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / len(files)
    
    # FPR per hour (15s window steps)
    total_hours = (len(files) * 15) / 3600
    fpr_hour = fp / total_hours

    print("\n" + "="*30)
    print(f"✅ Accuracy:    {accuracy:.2%}")
    print(f"🔥 Sensitivity: {sensitivity:.2%}") 
    print(f"🛡️ Specificity: {specificity:.2%}")
    print(f"📉 FPR / Hour:  {fpr_hour:.4f}")
    print("="*30)

    # 4. Result #4: Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Seizure'], 
                yticklabels=['Normal', 'Seizure'])
    plt.title("Result #4: Clinical Confusion Matrix (Weighted)")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{PLOT_DIR}/result_4_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()