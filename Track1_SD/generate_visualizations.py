import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

# 1. Setup paths
ref_dir = r"./data_directory/Track_1_SD_DevData_1/Hindi/data/rttm"
sys_dir = r"./gen_rttm"

ref_files = glob.glob(os.path.join(ref_dir, "*.rttm"))
sys_files = glob.glob(os.path.join(sys_dir, "*.rttm"))
sys_map = {os.path.basename(f).replace(".rttm", "").strip(): f for f in sys_files}

metric = DiarizationErrorRate()
file_scores = []

# 2. Collect scores
for ref_path in ref_files:
    ref_base = os.path.basename(ref_path).replace(".rttm", "").replace("_SPEAKER", "").strip()
    if ref_base in sys_map:
        try:
            ref_anno = next(iter(load_rttm(ref_path).values()))
            sys_anno = next(iter(load_rttm(sys_map[ref_base]).values()))
            der_val = metric(ref_anno, sys_anno) * 100
            file_scores.append((ref_base, der_val))
        except Exception:
            pass

# Sort files by performance
file_scores.sort(key=lambda x: x[1])
scores_only = [x[1] for x in file_scores]
avg_der = sum(scores_only) / len(scores_only) if scores_only else 0

# 3. Create a split-screen dashboard
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left Plot: Distribution of DER across all files
sns.histplot(scores_only, bins=12, kde=True, ax=axes[0], color="royalblue", edgecolor="black")
axes[0].axvline(avg_der, color="crimson", linestyle="--", linewidth=2.5, label=f"Average DER ({avg_der:.2f}%)")
axes[0].set_title("Diarization Error Rate (DER) Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("DER (%)", fontsize=12)
axes[0].set_ylabel("Number of Audio Files", fontsize=12)
axes[0].legend(fontsize=11)

# Right Plot: Best 5 vs Worst 5 files to show variance
extreme_files = file_scores[:5] + file_scores[-5:]
extreme_names = [x[0] for x in extreme_files]
extreme_values = [x[1] for x in extreme_files]
colors = ["g"] * 5 + ["r"] * 5

sns.barplot(x=extreme_values, y=extreme_names, palette=colors, ax=axes[1], hue=extreme_names, legend=False)
axes[1].axvline(avg_der, color="black", linestyle=":", label="Global Avg")
axes[1].set_title("Extreme Performers: Top 5 (Green) vs Bottom 5 (Red)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("DER (%)", fontsize=12)

plt.suptitle(f"DISPLACE 2026 Baseline Performance Report (Overall Average DER: {avg_der:.2f}%)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()

# Save image file
output_image = "der_baseline_report.png"
plt.savefig(output_image, dpi=300)
print(f"Success! Visualization generated and saved as: {output_image}")
