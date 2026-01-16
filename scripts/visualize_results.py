import torch
import matplotlib.pyplot as plt
import os

# Load one seizure feature map
feature_path = "data/quantum_features"
sample_file = [f for f in os.listdir(feature_path) if "lab1" in f][0]
features = torch.load(os.path.join(feature_path, sample_file)) # Shape: (4, 16, 16)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle(f"Result #2: Quantum Feature Maps (Seizure - {sample_file})")

for i in range(4):
    axes[i].imshow(features[i].numpy(), cmap='magma')
    axes[i].set_title(f"Quantum Channel {i+1}")
    axes[i].axis('off')

plt.savefig("results/plots/result_2_quantum_features.png")
plt.show()