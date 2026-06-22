import json
import os
import matplotlib.pyplot as plt

# Path to your generated metrics file
json_path = r"outputs\metrics\asr_cer_wer_results.json"

if not os.path.exists(json_path):
    print(f"Error: Could not find metrics file at {json_path}")
    exit()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract file-level metrics
files_data = data.get("files", [])
if not files_data:
    print("No individual file data found in the JSON.")
    exit()

rec_ids = [item["rec_id"] for item in files_data]
wers = [item["wer"] * 100 for item in files_data]  # Convert decimal to %
cers = [item["cer"] * 100 for item in files_data]

# Plotting Distribution/Sorted Error Rates
plt.figure(figsize=(14, 6))

# Sort by WER to make the graph highly readable
sorted_data = sorted(zip(rec_ids, wers, cers), key=lambda x: x[1])
s_ids, s_wers, s_cers = zip(*sorted_data)

plt.plot(s_wers, label="Word Error Rate (WER %)", color="crimson", marker="o", linewidth=2)
plt.plot(s_cers, label="Character Error Rate (CER %)", color="darkblue", linestyle="--", marker="s", alpha=0.7)

plt.title(f"DISPLACE 2026 ASR Baseline Performance Across {len(rec_ids)} Files\nAvg WER: {data['average_wer']*100:.1f}% | Avg CER: {data['average_cer']*100:.1f}%", fontsize=14, fontweight='bold')
plt.xlabel("Audio Files (Sorted by Difficulty/WER)", fontsize=12)
plt.ylabel("Error Rate (%)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the visualization plot
output_img = r"outputs\metrics\asr_baseline_chart.png"
plt.savefig(output_img, dpi=300)
plt.show()

print(f"Success! Chart generated and saved to: {output_img}")