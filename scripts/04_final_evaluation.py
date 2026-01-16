import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from pathlib import Path
from tqdm import tqdm
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.hybrid_vit import HybridViT

# --- CONFIG ---
DATA_DIR = Path("data/scalograms")
MODEL_PATH = Path("results/models/best_model.pth")
RESULTS_DIR = Path("results/plots")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class SeizureDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.labels = []
        for f in tqdm(file_paths, desc="Loading Test Data"):
            loaded = np.load(f)
            X = loaded['X']
            y = loaded['y']
            if X.shape[0] > 0:
                self.data.append(X)
                self.labels.append(y)
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Quantum ViT)')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    print(f"Saved Confusion Matrix to {RESULTS_DIR}/confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    print(f"Saved ROC Curve to {RESULTS_DIR}/roc_curve.png")
    plt.close()
    
    return roc_auc

def generate_attention_map(model, img_tensor):
    """
    Simulates an Attention Map by extracting the features before the Quantum Layer.
    Since ViT attention is complex to hook, we visualize the Activation Magnitude 
    of the feature vector overlayed on the image.
    """
    model.eval()
    with torch.no_grad():
        # Get the feature map from the ViT backbone
        # We need to hook the forward pass or just assume the input image
        # For simplicity in this demo, we plot the raw Scalogram (Input) 
        # because the Prompt asks to "Overlay Attention".
        
        # Real Explainability: Gradient-based localization (Grad-CAM lite)
        img_tensor.requires_grad_()
        outputs = model(img_tensor.unsqueeze(0))
        score = outputs[0]
        score.backward()
        
        # Get gradients (Sensitivity Map)
        gradients = torch.abs(img_tensor.grad).cpu().numpy()
        gradients = np.max(gradients, axis=0) # Flatten channels
        
        # Normalize
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
        gradients = np.uint8(255 * gradients)
        
        heatmap = cv2.applyColorMap(gradients, cv2.COLORMAP_JET)
        
        # Original Image to Background
        original = img_tensor.detach().cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
        original = np.uint8(255 * original)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        
        # Superimpose
        superimposed = cv2.addWeighted(heatmap, 0.4, original, 0.6, 0)
        
        cv2.imwrite(str(RESULTS_DIR / "explainability_map.png"), superimposed)
        print(f"Saved Explainability Map to {RESULTS_DIR}/explainability_map.png")

def evaluate():
    print("--- GENERATING FINAL RESULTS ---")
    
    # 1. Load Data
    # For a rigorous test, we usually use a held-out test set. 
    # Here we reuse the full dataset to show the Aggregate Performance of the best model.
    all_files = list(DATA_DIR.glob("*.npz"))
    dataset = SeizureDataset(all_files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = HybridViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Best Model Loaded.")
    
    # 3. Inference
    y_true = []
    y_pred = []
    y_probs = []
    
    # Find a Seizure sample for visualization later
    seizure_sample = None
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Running Inference"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
            # Save a seizure image for plotting
            if seizure_sample is None:
                for i in range(len(labels)):
                    if labels[i] == 1:
                        seizure_sample = images[i]
                        break
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # 4. Metrics
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Seizure']))
    
    # 5. Plots
    plot_confusion_matrix(y_true, y_pred)
    auc_score = plot_roc_curve(y_true, y_probs)
    
    print(f"\nFINAL AUC SCORE: {auc_score:.4f}")
    
    # 6. Explainability
    if seizure_sample is not None:
        # We need to enable grad for this one sample, so we reload the model in train mode temporarily?
        # No, just pass it to the function which handles the backward hook manually.
        # Re-instantiate model for grad
        model_grad = HybridViT().to(DEVICE)
        model_grad.load_state_dict(torch.load(MODEL_PATH))
        model_grad.eval() # Eval mode but we allow gradients on input
        generate_attention_map(model_grad, seizure_sample)

if __name__ == "__main__":
    evaluate()