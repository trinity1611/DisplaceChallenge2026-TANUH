import os
import glob
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

ref_dir = r"./data_directory/Track_1_SD_DevData_1/Hindi/data/rttm"
sys_dir = r"./gen_rttm"

ref_files = glob.glob(os.path.join(ref_dir, "*.rttm"))
sys_files = glob.glob(os.path.join(sys_dir, "*.rttm"))
sys_map = {os.path.basename(f).replace(".rttm", "").strip(): f for f in sys_files}

metric = DiarizationErrorRate()
total_der, count = 0, 0

print("# DISPLACE 2026 - Track 1 Speaker Diarization Baseline Results\n")
print(f"| File Name | Baseline DER (%) | Status |")
print(f"| :--- | :--- | :--- |")

for ref_path in ref_files:
    ref_base = os.path.basename(ref_path).replace(".rttm", "").replace("_SPEAKER", "").strip()
    if ref_base in sys_map:
        try:
            ref_anno = next(iter(load_rttm(ref_path).values()))
            sys_anno = next(iter(load_rttm(sys_map[ref_base]).values()))
            der_val = metric(ref_anno, sys_anno) * 100
            print(f"| {ref_base}.rttm | {der_val:.2f}% | Success |")
            total_der += der_val
            count += 1
        except Exception:
            print(f"| {ref_base}.rttm | Error | Failed to parse |")

print(f"\n**OVERALL AVERAGE BENCHMARK DER:** {total_der/count:.2f}%")
