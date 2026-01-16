import torch
import torch.nn as nn
import pennylane as qml
import timm
import numpy as np

class QuantumLayer(nn.Module):
    """
    Hybrid Quantum Layer with 4 Qubits.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            # Angle Embedding (Encodes data into rotation angles)
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            # Entangling Layers (The "Quantum Neural Network" part)
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measure Expectation Value (Returns 4 values)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.qnode = circuit
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

class HybridViT(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # 1. Load Pretrained ViT
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        
        # Get feature size (usually 192 for tiny, 384 for small)
        n_features = self.vit.head.in_features
        self.vit.head = nn.Identity() # Remove original head
        
        # 2. Quantum Branch
        self.n_qubits = 4
        self.pre_quantum = nn.Linear(n_features, self.n_qubits)
        self.quantum_layer = QuantumLayer(n_qubits=self.n_qubits)
        
        # 3. Classical Branch (Residual Connection)
        # We keep the original features to ensure gradient flow
        
        # 4. Final Classifier
        # Input = Original Features + Quantum Features (Concatenation)
        self.classifier = nn.Sequential(
            nn.Linear(n_features + self.n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 1. Extract Features from ViT
        features = self.vit(x) # Shape: (Batch, 192)
        
        # 2. Quantum Path
        q_in = torch.tanh(self.pre_quantum(features)) * np.pi # Squash to [-pi, pi]
        q_out = self.quantum_layer(q_in) # Shape: (Batch, 4)
        
        # 3. Concatenate (Residual Connection)
        # This is the secret sauce: Mix Classical and Quantum info
        combined = torch.cat([features, q_out], dim=1)
        
        # 4. Classify
        logits = self.classifier(combined)
        
        return logits