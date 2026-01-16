import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_improved_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "results/plots"

# 1. Model Architecture (Must match training script)
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

def run_improved_evaluation():
    print(f"📊 Running Improved Evaluation on: {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print(f"   Please run: python scripts/phase3_improved_train.py")
        return
    
    model = ImprovedViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
    y_true, y_pred, y_prob = [], [], []

    print(f"\n🔍 Evaluating {len(files)} samples...")
    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            output = model(data)
            prob = torch.sigmoid(output).item()
            y_true.append(1 if "_lab1" in path else 0)
            y_prob.append(prob)
            y_pred.append(1 if prob >= 0.5 else 0)

    # Metrics Calculation
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / len(files)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    fpr_hour = fp / ((len(files) * 15) / 3600)

    # ROC Curve for threshold optimization
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print("\n" + "🏁" * 12 + " FINAL PERFORMANCE REPORT " + "🏁" * 12)
    print(f"✅ Raw Accuracy:       {accuracy:.2%}")
    print(f"⚖️  Balanced Accuracy:  {balanced_acc:.2%}")
    print(f"🔴 Sensitivity (TPR):  {sensitivity:.2%}  [Detect seizures correctly]")
    print(f"🔵 Specificity (TNR):  {specificity:.2%}  [Avoid false alarms]")
    print(f"📍 Precision:          {precision:.2%}")
    print(f"💯 F1-Score:          {f1:.4f}")
    print(f"📈 ROC-AUC:           {roc_auc:.4f}")
    print(f"📉 False Positives/Hr: {fpr_hour:.4f}")
    print("🏁" * 50)

    print(f"\n📊 CONFUSION MATRIX BREAKDOWN:")
    print(f"   ✅ True Positives (TP):  {tp:4d}  [Correctly identified seizures]")
    print(f"   ✅ True Negatives (TN):  {tn:4d}  [Correctly identified normal]")
    print(f"   ❌ False Positives (FP): {fp:4d}  [False alarms]")
    print(f"   ❌ False Negatives (FN): {fn:4d}  [Missed seizures]")
    print(f"   📊 Total Samples:        {len(files):4d}")

    # Visualization 1: Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes[0],
                xticklabels=['Normal', 'Seizure'], 
                yticklabels=['Normal', 'Seizure'],
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=11, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    
    # Normalize by actual class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Normal', 'Seizure'], 
                yticklabels=['Normal', 'Seizure'],
                cbar_kws={'label': 'Percentage'})
    axes[1].set_xlabel('Predicted', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=11, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/improved_confusion_matrix_final.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 2: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    plt.title('ROC Curve - Improved ST-QAN-ViT', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/improved_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 3: Probability Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    normal_probs = [y_prob[i] for i in range(len(y_true)) if y_true[i] == 0]
    seizure_probs = [y_prob[i] for i in range(len(y_true)) if y_true[i] == 1]
    
    axes[0].hist(normal_probs, bins=30, alpha=0.7, label='Normal', color='blue', edgecolor='black')
    axes[0].hist(seizure_probs, bins=30, alpha=0.7, label='Seizure', color='red', edgecolor='black')
    axes[0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Probability Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Metrics visualization
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    values = [accuracy, sensitivity, specificity, precision, f1]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
    bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Performance Metrics', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/improved_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizations saved:")
    print(f"   📊 {PLOT_DIR}/improved_confusion_matrix_final.png")
    print(f"   📈 {PLOT_DIR}/improved_roc_curve.png")
    print(f"   📉 {PLOT_DIR}/improved_analysis.png")

if __name__ == "__main__":
    run_improved_evaluation()
