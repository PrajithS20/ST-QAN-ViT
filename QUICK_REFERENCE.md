# ST-QAN-ViT Quick Reference Guide

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Model** | Vision Transformer (ViT) |
| **Input Size** | 4×16×16 (4 quantum channels) |
| **Optimal Threshold** | **0.6634** |
| **Sensitivity** | 95.29% (catches seizures) |
| **Specificity** | 17.35% (baseline), 35.63% (conservative) |
| **False Negatives** | 13 out of 276 (95% catch rate) |
| **False Positives** | 1791 out of 2167 (baseline threshold) |

---

## 🎯 Model Selection

### **Production Use** ✅
```
Model: results/models/st_qan_vit_best_optimized_checkpoint.pth
Threshold: 0.6634
Sensitivity: 95.29%
Use: Real-time clinical seizure detection
```

### **Training Experiments**
```
Script: scripts/phase3_best_optimized.py
Input: data/quantum_features/*.pt
Output: results/models/st_qan_vit_best_optimized.pth
```

---

## 🔧 Training Configuration

### Loss Function
```python
HardNegativeFocalLoss(
    alpha=0.35,      # Focal weight scaling
    gamma=2.5,       # Focusing parameter (higher = focus on hard examples)
    pos_weight=8.0   # Class weight (seizure is 8x more important)
)
```

### Optimizer
```python
AdamW(
    lr=3e-5,
    weight_decay=0.02,
    betas=(0.9, 0.999)
)
```

### Learning Rate Schedule
```python
CosineAnnealingWarmRestarts(
    T_0=8,           # Restart period
    T_mult=2,        # Period multiplier
    eta_min=1e-6     # Minimum learning rate
)
```

---

## 📈 Decision Tree

```
Q: Need to catch ALL seizures?
├─ YES: Use threshold 0.50 (100% sensitivity, but 204 FP/hr)
└─ NO: Go to next

Q: Can accept some false negatives?
├─ Up to 5%: Use threshold 0.6634 (95% sensitivity, 176 FP/hr) ← RECOMMENDED
├─ Up to 10%: Use threshold 0.68 (90% sensitivity, 159 FP/hr)
└─ More acceptable: Use threshold 0.69 (80% sensitivity, 137 FP/hr)
```

---

## 🚀 Quick Start: Make a Prediction

### Python Code
```python
import torch
from timm.models.vision_transformer import VisionTransformer

# 1. Initialize model
model = VisionTransformer(
    img_size=16, patch_size=2, in_chans=4,
    num_classes=1, embed_dim=256, depth=6, num_heads=8,
    drop_rate=0.15, attn_drop_rate=0.15, drop_path_rate=0.15
)
model.load_state_dict(torch.load(
    'results/models/st_qan_vit_best_optimized_checkpoint.pth'
))
model.eval()

# 2. Load quantum features
features = torch.load('data/quantum_features/sample.pt')

# 3. Predict
with torch.no_grad():
    logit = model(features.unsqueeze(0))
    probability = torch.sigmoid(logit).item()
    prediction = "SEIZURE" if probability >= 0.6634 else "NORMAL"

print(f"Probability: {probability:.4f}")
print(f"Prediction: {prediction}")
```

---

## 📊 Evaluation Scripts

### Available Scripts
```bash
# Generate production evaluation with threshold analysis
python scripts/phase4_production_eval.py

# Generate before/after comparison
python scripts/generate_comparison.py

# Train new model from scratch
python scripts/phase3_best_optimized.py

# Quick evaluation
python scripts/phase4_improved_eval.py
```

---

## 🎨 Available Visualizations

All saved to `results/plots/`:

1. **`final_production_analysis.png`** - Complete analysis with ROC curve, probability distribution, threshold comparison
2. **`before_after_optimization.png`** - Before/after comparison chart
3. **`final_optimized_matrix.png`** - Confusion matrix heatmap
4. **`strategy_comparison.png`** - All 5 threshold strategies visualized
5. **`improved_analysis.png`** - Metrics bar charts
6. **`improved_roc_curve.png`** - ROC curve with AUC

---

## ⚠️ Important Notes

### Clinical Considerations
- **Priority**: Minimize false negatives (missed seizures) over false positives
- **Use Case**: Real-time screening, not definitive diagnosis
- **Threshold**: Can be adjusted based on clinical requirements
- **Validation**: Always validate on your specific EEG dataset

### Model Details
- **Architecture**: Vision Transformer with drop_rate=0.15
- **Input**: Quantum-extracted features (4 channels of 16×16 tensors)
- **Output**: Sigmoid probability (0-1)
- **Training**: ~17 epochs before early stopping

### Performance Tradeoffs
| Aspect | Baseline (0.50) | Optimized (0.66) | Conservative (0.69) |
|--------|---|---|---|
| Seizure Detection | 100% | 95% | 80% |
| False Alarm Rate | 204/hr | 176/hr | 137/hr |
| Use Case | Too aggressive | RECOMMENDED | High specificity |

---

## 📞 Support

For detailed implementation:
1. See `IMPROVEMENT_SUMMARY.md`
2. Check `results/plots/production_recommendation.txt`
3. Review `scripts/phase4_production_eval.py` for implementation example

---

**Last Updated**: January 16, 2026  
**Model Version**: st_qan_vit_best_optimized_checkpoint  
**Recommended Threshold**: 0.6634
