import os
import glob
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

ref_dir = r"C:\Users\Hp\DISPLACE-2026-Baselines\Track1_SD\data_directory\Track_1_SD_DevData_1\Hindi\data\rttm"
sys_dir = r"C:\Users\Hp\DISPLACE-2026-Baselines\Track1_SD\gen_rttm"

ref_files = glob.glob(os.path.join(ref_dir, "*.rttm"))
sys_files = glob.glob(os.path.join(sys_dir, "*.rttm"))

metric = DiarizationErrorRate()

total_der = 0
count = 0

sys_map = {os.path.basename(f).replace(".rttm", "").strip(): f for f in sys_files}

print("File Name | DER (%)")
print("-" * 30)

for ref_path in ref_files:
    ref_base = os.path.basename(ref_path).replace(".rttm", "").replace("_SPEAKER", "").strip()
    
    if ref_base in sys_map:
        try:
            ref_data = load_rttm(ref_path)
            sys_data = load_rttm(sys_map[ref_base])
            
            ref_anno = next(iter(ref_data.values()))
            sys_anno = next(iter(sys_data.values()))
            
            der_val = metric(ref_anno, sys_anno)
            print(f"{ref_base}.rttm | {der_val*100:.2f}%")
            total_der += der_val
            count += 1
        except Exception as e:
            pass

print("-" * 30)
if count > 0:
    print(f"OVERALL AVERAGE DER: {(total_der/count)*100:.2f}%")
else:
    print("Still no match. Verify names manually.")
