import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer
from scipy.ndimage import gaussian_filter1d

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_improved_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "results/plots"

# Model Architecture
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

def apply_temporal_smoothing(probabilities, files, method='voting', window=5, threshold=0.5):
    """
    Apply temporal smoothing to reduce false positives
    
    Methods:
    1. 'voting': Majority voting within window
    2. 'gaussian': Gaussian smoothing
    3. 'median': Median filtering
    """
    probs_array = np.array(probabilities)
    
    if method == 'voting':
        smoothed = np.zeros_like(probs_array)
        half_window = window // 2
        
        for i in range(len(probs_array)):
            start = max(0, i - half_window)
            end = min(len(probs_array), i + half_window + 1)
            window_probs = probs_array[start:end]
            # Average probabilities in window
            smoothed[i] = np.mean(window_probs)
    
    elif method == 'gaussian':
        smoothed = gaussian_filter1d(probs_array, sigma=2.0)
    
    elif method == 'median':
        from scipy.ndimage import median_filter
        smoothed = median_filter(probs_array, size=window)
    
    else:
        smoothed = probs_array
    
    return smoothed

def find_optimal_operating_point(y_true, y_prob):
    """Find optimal threshold balancing sensitivity and false positive rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Compute F1-like metric: balance TPR and (1-FPR)
    f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    return optimal_threshold, optimal_tpr, optimal_fpr

def evaluate_with_optimization():
    print(f"🚀 Optimizing ST-QAN-ViT on: {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    
    model = ImprovedViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
    y_true, y_prob = [], []

    print(f"\n🔍 Evaluating {len(files)} samples...")
    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            output = model(data)
            prob = torch.sigmoid(output).item()
            y_true.append(1 if "_lab1" in path else 0)
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Find optimal threshold
    optimal_threshold, optimal_tpr, optimal_fpr = find_optimal_operating_point(y_true, y_prob)
    
    print(f"\n🎯 THRESHOLD OPTIMIZATION:")
    print(f"   Default (0.5):  TPR={np.mean((y_prob >= 0.5) == y_true):.2%}")
    print(f"   Optimal:        TPR={optimal_tpr:.2%}, FPR={optimal_fpr:.2%}, Threshold={optimal_threshold:.4f}")
    
    # Test different strategies
    results = {}
    
    # Strategy 1: Baseline (threshold 0.5)
    y_pred_base = (y_prob >= 0.5).astype(int)
    cm_base = confusion_matrix(y_true, y_pred_base)
    tn, fp, fn, tp = cm_base.ravel()
    results['Baseline (0.5)'] = {
        'cm': cm_base,
        'threshold': 0.5,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'y_pred': y_pred_base
    }
    
    # Strategy 2: Optimized threshold
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm_opt.ravel()
    results['Optimized Threshold'] = {
        'cm': cm_opt,
        'threshold': optimal_threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'y_pred': y_pred_opt
    }
    
    # Strategy 3: Temporal voting (smoothing)
    y_prob_smooth = apply_temporal_smoothing(y_prob, files, method='voting', window=5)
    y_pred_smooth = (y_prob_smooth >= optimal_threshold).astype(int)
    cm_smooth = confusion_matrix(y_true, y_pred_smooth)
    tn, fp, fn, tp = cm_smooth.ravel()
    results['Temporal Voting'] = {
        'cm': cm_smooth,
        'threshold': optimal_threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'y_pred': y_pred_smooth
    }
    
    # Strategy 4: Higher threshold for conservative prediction
    high_threshold = optimal_threshold + 0.15
    y_pred_cons = (y_prob >= high_threshold).astype(int)
    cm_cons = confusion_matrix(y_true, y_pred_cons)
    tn, fp, fn, tp = cm_cons.ravel()
    results['Conservative (Higher Threshold)'] = {
        'cm': cm_cons,
        'threshold': high_threshold,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'y_pred': y_pred_cons
    }
    
    # Print comparison
    print("\n" + "="*90)
    print("STRATEGY COMPARISON")
    print("="*90)
    
    for strategy, data in results.items():
        cm = data['cm']
        accuracy = (data['tp'] + data['tn']) / len(y_true)
        balanced_acc = (data['sensitivity'] + data['specificity']) / 2
        precision = data['tp'] / (data['tp'] + data['fp']) if (data['tp'] + data['fp']) > 0 else 0
        f1 = 2 * (precision * data['sensitivity']) / (precision + data['sensitivity']) if (precision + data['sensitivity']) > 0 else 0
        
        print(f"\n🎯 {strategy}")
        print(f"   Threshold:    {data['threshold']:.4f}")
        print(f"   Sensitivity:  {data['sensitivity']:.2%}  (Detect seizures)")
        print(f"   Specificity:  {data['specificity']:.2%}  (Avoid false alarms)")
        print(f"   Accuracy:     {accuracy:.2%}")
        print(f"   Bal. Acc:     {balanced_acc:.2%}")
        print(f"   Precision:    {precision:.2%}")
        print(f"   F1-Score:     {f1:.4f}")
        print(f"   TP/TN/FP/FN:  {data['tp']}/{data['tn']}/{data['fp']}/{data['fn']}")
    
    # Visualization: Compare strategies
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ST-QAN-ViT: Strategy Comparison', fontsize=16, fontweight='bold')
    
    strategy_names = list(results.keys())
    for idx, (ax, strategy) in enumerate(zip(axes.flat, strategy_names)):
        cm = results[strategy]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
                    xticklabels=['Normal', 'Seizure'],
                    yticklabels=['Normal', 'Seizure'],
                    cbar_kws={'label': 'Count'})
        
        sensitivity = results[strategy]['sensitivity']
        specificity = results[strategy]['specificity']
        ax.set_title(f"{strategy}\nSens={sensitivity:.1%}, Spec={specificity:.1%}", 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Strategy comparison saved: {PLOT_DIR}/strategy_comparison.png")
    
    # Find best strategy
    best_strategy = max(results.items(), 
                       key=lambda x: (x[1]['sensitivity'] + x[1]['specificity']) / 2)
    
    print("\n" + "🏆" * 20)
    print(f"🏆 BEST STRATEGY: {best_strategy[0]}")
    print("🏆" * 20)
    
    # Save recommendation
    with open(f"{PLOT_DIR}/optimization_report.txt", 'w') as f:
        f.write("ST-QAN-ViT OPTIMIZATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Recommended Strategy: {best_strategy[0]}\n")
        f.write(f"Threshold: {best_strategy[1]['threshold']:.4f}\n")
        f.write(f"Sensitivity: {best_strategy[1]['sensitivity']:.2%}\n")
        f.write(f"Specificity: {best_strategy[1]['specificity']:.2%}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"TP: {best_strategy[1]['tp']}, TN: {best_strategy[1]['tn']}\n")
        f.write(f"FP: {best_strategy[1]['fp']}, FN: {best_strategy[1]['fn']}\n")
    
    print(f"✅ Report saved: {PLOT_DIR}/optimization_report.txt")

if __name__ == "__main__":
    evaluate_with_optimization()
