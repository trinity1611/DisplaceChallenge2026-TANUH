import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Create directory for plots if it doesn't exist
os.makedirs("./outputs/plots", exist_ok=True)

# 1. Raw Counts Data
counts = {'True Positives (TP)': 57, 'False Positives (FP)': 37, 'False Negatives (FN)': 49}
names = list(counts.keys())
values = list(counts.values())

# Plot 1: Classification Counts
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c', '#3498db']
bars = plt.bar(names, values, color=colors, width=0.5, edgecolor='black', linewidth=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{int(yval)}', ha='center', va='bottom', weight='bold')

plt.title("Track 3: Medical Topic Identification - Prediction Counts", pad=15)
plt.ylabel("Number of Topics")
plt.ylim(0, max(values) + 10)
plt.tight_layout()
plt.savefig("./outputs/plots/prediction_counts.png", dpi=300)
plt.close()

# 2. Performance Scores Data
metrics = {
    'Precision': 0.606,
    'Recall': 0.538,
    'F1-Score': 0.570,
    'Accuracy': 0.629,
    'BERTScore': 0.620,
    'ROUGE-1': 0.243
}

m_names = list(metrics.keys())
m_values = list(metrics.values())

# Plot 2: Performance Summary
plt.figure(figsize=(10, 5))
palette = sns.color_palette("viridis", len(m_names))
bars_m = plt.bar(m_names, m_values, color=palette, width=0.6, edgecolor='black', linewidth=0.7)

# Add value labels on top of metrics bars
for bar in bars_m:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', weight='bold')

plt.title("Track 3: Final Baseline Performance Metrics", pad=15)
plt.ylabel("Score Range (0.0 - 1.0)")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("./outputs/plots/final_metrics_summary.png", dpi=300)
plt.close()

print("Visualizations successfully generated and saved to ./outputs/plots/")