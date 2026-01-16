import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# --- CONFIG ---
RESULTS_DIR = Path("results/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data from your final "Balanced" run
metrics = {
    'Window Sensitivity': 89.2,  # Technical (157/176)
    'Event Sensitivity': 100.0   # Clinical (Caught every event)
}

def plot_dual_sensitivity():
    print("Generating Dual Sensitivity Comparison...")
    
    # Setup
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#5C6BC0', '#4CAF50'] # Blue (Tech) vs Green (Clinical)
    
    # Plot Bars
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    
    # Styling
    ax.set_ylim(0, 115) # Give room for labels
    ax.set_ylabel('Sensitivity (%)', fontsize=12)
    ax.set_title('Metric Analysis: Technical vs. Clinical Sensitivity', fontsize=14, fontweight='bold')
    
    # Target Line
    ax.axhline(y=92, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Project Target (92%)')
    ax.legend(loc='upper left')
    
    # Add Explanatory Text inside the plot
    ax.text(0, 50, "Strict Definition:\nRequires every 30s\nchunk to be caught", 
            ha='center', color='white', fontweight='bold', fontsize=10)
    
    ax.text(1, 50, "Safety Definition:\nRequires alarm to ring\nat least once per seizure", 
            ha='center', color='white', fontweight='bold', fontsize=10)

    # Add Value Labels on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')

    plt.tight_layout()
    save_path = RESULTS_DIR / "Sensitivity_Comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved graph to {save_path}")

if __name__ == "__main__":
    plot_dual_sensitivity()