import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_balanced_model(model, test_loader, device, class_names, save_path="results/plots"):
    """
    Evaluates the model, prints metrics, and saves the confusion matrix plot.
    """
    model.eval()
    all_preds = []
    all_labels = []

    print(f"🔍 Starting Evaluation on {device}...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Generate Metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print("\n📊 Classification Report:")
    print(report)

    # 2. Save the Report to a Text File
    os.makedirs("results/eval", exist_ok=True)
    with open("results/eval/metrics_report.txt", "w") as f:
        f.write(report)

    # 3. Create and Save the Confusion Matrix Plot
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Result #4: Confusion Matrix - Balanced ST-QAN-ViT')
    
    plot_file = os.path.join(save_path, "confusion_matrix_balanced.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close() # Close to free up memory
    
    print(f"✅ Confusion Matrix saved to: {plot_file}")
    print(f"✅ Metrics report saved to: results/eval/metrics_report.txt")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # This part allows you to run 'python scripts/evaluate.py' directly
    # You will need to initialize your Model and DataLoader here
    pass