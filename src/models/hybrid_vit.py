import torch
import torch.nn as nn
import pennylane as qml
import timm
import numpy as np

# --- 1. QUANTUM LAYER (Standard Dense) ---
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_features, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.fc_compress = nn.Linear(n_features, n_qubits)
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        # Compress (192 -> 4)
        q_in = torch.tanh(self.fc_compress(x)) * np.pi 
        # Quantum Calculation
        q_out = self.q_layer(q_in) 
        return q_out

class HybridViT(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # 1. ViT Backbone (Standard)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.n_features = self.vit.head.in_features
        self.vit.head = nn.Identity() # Remove head
        
        # 2. Quantum Branch
        self.quantum = QuantumLayer(self.n_features)
        
        # 3. Classifier (Residual Fusion)
        # We concatenate Classical (192) + Quantum (4)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.n_features + 4),
            nn.Linear(self.n_features + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 1. Classical Features
        features = self.vit(x) # (Batch, 192)
        
        # 2. Quantum Features
        q_out = self.quantum(features) # (Batch, 4)
        
        # 3. Fusion
        combined = torch.cat([features, q_out], dim=1)
        
        # 4. Classification
        return self.classifier(combined)