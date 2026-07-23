# export_pkl_info.py
import pickle
from pathlib import Path
import numpy as np

pkl_path = Path(r"C:\Users\CMME260629\projects_cmme\SuperResolutionAI\super_resolution\final_training_data\sc1.0\scans\scan_benign_10_HR.pkl")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("type:", type(data))
print("keys:", list(data.keys()))
print()

for key, value in data.items():
    print(f"[{key}]")
    print("  type:", type(value))

    if isinstance(value, dict):
        print("  dict keys:", list(value.keys()))
        for sub_key, sub_value in value.items():
            print(f"    - {sub_key}: {sub_value}")
    elif isinstance(value, np.ndarray):
        print("  shape:", value.shape)
        print("  dtype:", value.dtype)
        print("  min:", float(np.min(value)))
        print("  max:", float(np.max(value)))
    else:
        print("  value:", value)

    print()