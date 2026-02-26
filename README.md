# Maths-ST_QAN_ViT: Spatio-Temporal Quantum Attention Network

A hybrid classical-quantum deep learning pipeline for Epileptic Seizure Prediction using Electroencephalogram (EEG) data.

This project implements a novel architecture that fuses a classical Vision Transformer (ViT) with a 4-qubit Quantum Neural Network (QNN) to accurately detect pre-ictal states (the period up to 15 minutes before a seizure) from EEG signals. 

## 🧠 Architecture Overview

The pipeline leverages time-frequency representations (scalograms) of EEG signals to extract both macro-level spatial features and micro-level quantum states.

1. **Signal Preprocessing**: Raw `.edf` files (e.g., CHB-MIT dataset) are bandpass filtered (1-50Hz) and segment into 30-second windows.
2. **Feature Extraction**: Continuous Wavelet Transform (CWT) generates Maximum Energy Scalograms (224x224 RGB images) representing the frequency spectrum across all EEG channels.
3. **Classical Backbone (ViT)**: A pre-trained `vit_tiny_patch16_224` (via `timm`) extracts high-level 192-dimensional features from the scalograms.
4. **Quantum Layer (QNN)**: The classical features are compressed and embedded into a 4-qubit parameterized quantum circuit (via `PennyLane`), utilizing `AngleEmbedding` and `StronglyEntanglingLayers` to capture complex multidimensional correlations.
5. **Fusion & Classification**: Classical and Quantum features are concatenated and passed through a Multi-Layer Perceptron (MLP) for binary classification (Seizure vs. Normal).

## 🚀 Key Features

*   **Hybrid ViT + Quantum Layer**: Integrates PyTorch and PennyLane for a state-of-the-art hybrid architecture.
*   **Multiprocessing Scalogram Generation**: Fast, parallelized CWT feature extraction across multiple CPU cores.
*   **Stratified Training Split**: Guarantees actual seizure events are properly represented in the validation and test sets.
*   **Weighted Random Sampling**: Effectively handles severe class imbalance between normal EEG and pre-ictal periods.
*   **Comprehensive Evaluation metrics**: Reports Clinical standards including Event Sensitivity, False Positive Rate per Hour (FPR/hr), and Window Accuracy.
*   **Rich Visualizations**: Provides automatically generated Saliency Maps, Confusion Matrices, ROC Curves, and more.

## 📂 Project Structure

```text
Maths-ST_QAN_ViT/
├── data/
│   ├── raw/                    # Place raw .edf files here (e.g., chb01, chb02)
│   ├── processed_signals/      # .npz files containing 30s EEG windows
│   └── scalograms/             # 224x224 Scalogram images (.npz format)
├── results/
│   ├── models/                 # Saved model checkpoints (best_model.pth)
│   └── plots/                  # Generated evaluation and explainability images
├── scripts/
│   ├── 01_signal_preprocessing.py   # EDF to 30s windows
│   ├── 02_generate_scalograms.py    # Windows to CWT Scalograms
│   ├── 03_train_model.py            # Train HybridViT
│   ├── 04_final_evaluation.py       # Clinical Metrics & Reporting
│   ├── 05_explainability.py         # Generate Saliency/Heatmaps
│   └── 06_generate_visualizations.py # Additional Visualizations
├── src/
│   └── models/
│       └── hybrid_vit.py            # Core PyTorch/PennyLane Model Definition
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages include: `torch`, `torchvision`, `timm`, `pennylane`, `mne`, `pywt`, `cv2`, `scikit-learn`, `matplotlib`, `seaborn`.*

## 🏃 Usage Pipeline

Run the scripts in sequential order:

1. **Preprocess Data**:
   Place your patient folders (e.g., `chb01`) containing the `.edf` and summary text files inside `data/raw/`.
   ```bash
   python scripts/01_signal_preprocessing.py
   ```

2. **Generate Scalograms**:
   *Note: This process is computationally heavy and will utilize all available CPU cores.*
   ```bash
   python scripts/02_generate_scalograms.py
   ```

3. **Train the Model**:
   ```bash
   python scripts/03_train_model.py
   ```

4. **Evaluate Performance**:
   Generates clinical metrics, ROC curves, and confusion matrices in `results/plots/`.
   ```bash
   python scripts/04_final_evaluation.py
   ```

5. **Generate Explainability Maps**:
   Outputs visual saliency maps showing which time-frequencies the model focuses on.
   ```bash
   python scripts/05_explainability.py
   ```

## 📊 Results & Visualizations

The scripts automatically output detailed figures to the `results/plots/` directory, including:
- `confusion_matrix.png` & `roc_curve.png`
- `explainability_map.png` (Gradient-based focus maps over scalograms)
- `vit_attention_map.png`
- `3_training_loss_curve.png`
- Various comparative analyses (PSD, latency profiling).
