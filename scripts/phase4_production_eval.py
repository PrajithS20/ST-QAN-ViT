import torch
import torch.nn as nn
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models.vision_transformer import VisionTransformer

# --- Configuration ---
FEATURE_DIR = "data/quantum_features"
MODEL_PATH = "results/models/st_qan_vit_best_optimized_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "results/plots"

# Model Architecture
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

def evaluate_with_threshold_optimization():
    print(f"🚀 Final Production Evaluation on: {DEVICE}\n")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    model = RegularizedViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "*.pt")))
    y_true, y_prob = [], []

    print(f"🔍 Evaluating {len(files)} samples...")
    with torch.no_grad():
        for path in files:
            data = torch.load(path, weights_only=True).unsqueeze(0).to(DEVICE)
            output = model(data)
            prob = torch.sigmoid(output).item()
            y_true.append(1 if "_lab1" in path else 0)
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Test multiple threshold strategies
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Strategy 1: Standard 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results['0.50 - Standard'] = {
        'threshold': 0.5,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'y_pred': y_pred,
        'cm': cm
    }
    
    # Strategy 2: Find threshold for 95% sensitivity
    idx_95 = np.argmax(tpr >= 0.95)
    t_95 = thresholds[idx_95]
    y_pred = (y_prob >= t_95).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results[f'{t_95:.2f} - 95% Sensitivity'] = {
        'threshold': t_95,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'y_pred': y_pred,
        'cm': cm
    }
    
    # Strategy 3: Find threshold for 90% sensitivity
    idx_90 = np.argmax(tpr >= 0.90)
    t_90 = thresholds[idx_90]
    y_pred = (y_prob >= t_90).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results[f'{t_90:.2f} - 90% Sensitivity'] = {
        'threshold': t_90,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'y_pred': y_pred,
        'cm': cm
    }
    
    # Strategy 4: Balanced (max F1)
    f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-8)
    idx_best = np.argmax(f1_scores)
    t_best = thresholds[idx_best]
    y_pred = (y_prob >= t_best).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results[f'{t_best:.2f} - Best F1 (Balanced)'] = {
        'threshold': t_best,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'y_pred': y_pred,
        'cm': cm
    }
    
    # Strategy 5: Strict (high threshold to minimize FP)
    idx_strict = np.argmax(tpr >= 0.80)
    t_strict = thresholds[idx_strict]
    y_pred = (y_prob >= t_strict).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results[f'{t_strict:.2f} - Conservative (80% Sens)'] = {
        'threshold': t_strict,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'y_pred': y_pred,
        'cm': cm
    }
    
    # Print comparison
    for strategy, data in results.items():
        cm = data['cm']
        accuracy = (data['tp'] + data['tn']) / len(y_true)
        precision = data['tp'] / (data['tp'] + data['fp']) if (data['tp'] + data['fp']) > 0 else 0
        f1 = f1_score(y_true, data['y_pred'], zero_division=0)
        fpr_hour = data['fp'] / ((len(files) * 15) / 3600)
        
        print(f"\n{strategy}")
        print(f"   Sensitivity: {data['sensitivity']:6.2%} (Detect seizures - CRITICAL!)")
        print(f"   Specificity: {data['specificity']:6.2%} (Avoid false alarms)")
        print(f"   Accuracy:    {accuracy:6.2%}")
        print(f"   Precision:   {precision:6.2%}")
        print(f"   F1-Score:    {f1:6.4f}")
        print(f"   FP/Hour:     {fpr_hour:7.2f} (False alarms per hour)")
        print(f"   Matrix: TP={data['tp']:3d} | TN={data['tn']:4d} | FP={data['fp']:4d} | FN={data['fn']:2d}")
    
    # Find best strategy (prioritize sensitivity, then specificity)
    best_strategy = None
    best_score = -1
    for strategy, data in results.items():
        score = data['sensitivity'] * 0.7 + data['specificity'] * 0.3  # Weigh sensitivity more
        if score > best_score:
            best_score = score
            best_strategy = strategy
    
    print("\n" + "🏆" * 30)
    print(f"🏆 RECOMMENDED STRATEGY: {best_strategy}")
    print("🏆" * 30)
    
    best_data = results[best_strategy]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC (AUC={roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    
    # Mark strategies on ROC
    for strategy, data in results.items():
        t = data['threshold']
        idx = np.argmin(np.abs(thresholds - t))
        ax1.plot(fpr[idx], tpr[idx], 'o', markersize=8)
    
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve - All Thresholds', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Probability Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    normal_probs = [y_prob[i] for i in range(len(y_true)) if y_true[i] == 0]
    seizure_probs = [y_prob[i] for i in range(len(y_true)) if y_true[i] == 1]
    ax2.hist(normal_probs, bins=40, alpha=0.6, label='Normal', color='blue', edgecolor='black')
    ax2.hist(seizure_probs, bins=40, alpha=0.6, label='Seizure', color='red', edgecolor='black')
    ax2.axvline(best_data['threshold'], color='green', linestyle='--', linewidth=3, label=f'Optimal: {best_data["threshold"]:.3f}')
    ax2.set_xlabel('Predicted Probability', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Probability Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Sensitivity vs Specificity tradeoff
    ax3 = fig.add_subplot(gs[1, :])
    strategies_list = list(results.keys())
    sensitivities = [results[s]['sensitivity'] for s in strategies_list]
    specificities = [results[s]['specificity'] for s in strategies_list]
    
    x = np.arange(len(strategies_list))
    width = 0.35
    bars1 = ax3.bar(x - width/2, sensitivities, width, label='Sensitivity (TPR)', color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, specificities, width, label='Specificity (TNR)', color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Sensitivity vs Specificity Comparison', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies_list, rotation=15, ha='right')
    ax3.set_ylim([0, 1.1])
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Best strategy confusion matrix
    ax4 = fig.add_subplot(gs[2, 0])
    sns.heatmap(best_data['cm'], annot=True, fmt='d', cmap='RdYlGn', ax=ax4,
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'],
                cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted', fontweight='bold')
    ax4.set_ylabel('Actual', fontweight='bold')
    ax4.set_title(f'Best Strategy Confusion Matrix\n{best_strategy}', fontweight='bold')
    
    # 5. Metrics for best strategy
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    metrics_text = f"""
    ★ RECOMMENDED CONFIGURATION ★
    
    Threshold: {best_data['threshold']:.4f}
    
    ✅ PERFORMANCE:
       • Sensitivity: {best_data['sensitivity']:.2%}
       • Specificity: {best_data['specificity']:.2%}
    
    📊 CONFUSION MATRIX:
       • True Positives:  {best_data['tp']}
       • True Negatives:  {best_data['tn']}
       • False Positives: {best_data['fp']}
       • False Negatives: {best_data['fn']}
    
    Use this threshold in production!
    """
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('ST-QAN-ViT: Complete Threshold Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f"{PLOT_DIR}/final_production_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save report
    with open(f"{PLOT_DIR}/production_recommendation.txt", 'w', encoding='utf-8') as f:
        f.write("ST-QAN-ViT PRODUCTION RECOMMENDATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Recommended Threshold: {best_data['threshold']:.6f}\n")
        f.write(f"Strategy: {best_strategy}\n\n")
        f.write("EXPECTED PERFORMANCE:\n")
        f.write(f"  Sensitivity (Detect Seizures): {best_data['sensitivity']:.2%}\n")
        f.write(f"  Specificity (Avoid False Alarms): {best_data['specificity']:.2%}\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Positives:  {best_data['tp']}\n")
        f.write(f"  True Negatives:  {best_data['tn']}\n")
        f.write(f"  False Positives: {best_data['fp']}\n")
        f.write(f"  False Negatives: {best_data['fn']}\n\n")
        f.write("IMPLEMENTATION:\n")
        f.write(f"  1. Load model: {MODEL_PATH}\n")
        f.write(f"  2. Apply sigmoid to output\n")
        f.write(f"  3. Compare to threshold: {best_data['threshold']:.6f}\n")
        f.write(f"  4. Classification: prob >= {best_data['threshold']:.6f} yields Seizure, else yields Normal\n")
    
    print(f"\n✅ Analysis saved:")
    print(f"   📊 {PLOT_DIR}/final_production_analysis.png")
    print(f"   📄 {PLOT_DIR}/production_recommendation.txt")

if __name__ == "__main__":
    evaluate_with_threshold_optimization()
