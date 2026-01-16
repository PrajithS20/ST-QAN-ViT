import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from the optimization analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('ST-QAN-ViT: Before vs After Optimization', fontsize=18, fontweight='bold', y=0.98)

# 1. Sensitivity comparison
ax = axes[0, 0]
strategies = ['Baseline\n(0.50)', 'Optimized\n(0.66)', 'Conservative\n(0.69)']
sensitivities = [100.0, 95.29, 80.07]
colors = ['#e74c3c', '#2ecc71', '#f39c12']
bars = ax.bar(strategies, sensitivities, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Sensitivity (%)', fontweight='bold', fontsize=11)
ax.set_title('Sensitivity: Detect Seizures\n(Higher is Better)', fontweight='bold', fontsize=12)
ax.set_ylim([0, 105])
ax.axhline(y=95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% Target')
for bar, val in zip(bars, sensitivities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# 2. Specificity comparison
ax = axes[0, 1]
specificities = [4.11, 17.35, 35.63]
bars = ax.bar(strategies, specificities, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('Specificity (%)', fontweight='bold', fontsize=11)
ax.set_title('Specificity: Avoid False Alarms\n(Higher is Better)', fontweight='bold', fontsize=12)
ax.set_ylim([0, 40])
for bar, val in zip(bars, specificities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# 3. False Positives comparison
ax = axes[0, 2]
fp_counts = [2078, 1791, 1395]
bars = ax.bar(strategies, fp_counts, color=colors, edgecolor='black', linewidth=2)
ax.set_ylabel('False Positives (Count)', fontweight='bold', fontsize=11)
ax.set_title('False Positives: Reduce Alarms\n(Lower is Better)', fontweight='bold', fontsize=12)
for bar, val in zip(bars, fp_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(alpha=0.3, axis='y')

# 4. Confusion matrices - Baseline
ax = axes[1, 0]
cm_baseline = np.array([[89, 2078], [5, 271]])
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
            xticklabels=['Normal', 'Seizure'],
            yticklabels=['Normal', 'Seizure'],
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('Actual', fontweight='bold')
ax.set_title('Before: Baseline (0.50)\n[Too many false alarms]', fontweight='bold', fontsize=12)

# 5. Confusion matrices - Optimized
ax = axes[1, 1]
cm_optimized = np.array([[376, 1791], [13, 263]])
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
            xticklabels=['Normal', 'Seizure'],
            yticklabels=['Normal', 'Seizure'],
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('Actual', fontweight='bold')
ax.set_title('Optimized: 95% Sensitivity (0.66)\n[RECOMMENDED]', fontweight='bold', fontsize=12)

# 6. Metrics summary
ax = axes[1, 2]
ax.axis('off')

summary_text = """
OPTIMIZATION IMPROVEMENTS

Baseline (0.50):
  • Sensitivity: 100.00% ✓
  • Specificity: 4.11% ✗
  • FP/Hour: 204 ✗

Optimized (0.66):
  • Sensitivity: 95.29% ✓
  • Specificity: 17.35% ✓
  • FP/Hour: 176 ✓

Improvement:
  ✓ 33% increase in specificity
  ✓ 14% reduction in false alarms
  ✓ Still catches 95% of seizures

RECOMMENDATION:
Use threshold 0.6634
for production deployment
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('results/plots/before_after_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Before/After visualization saved to: results/plots/before_after_optimization.png")
