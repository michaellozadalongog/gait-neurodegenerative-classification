"""
Generate a small fake dataset so we can verify the full pipeline works
without needing PhysioNet access. Run this once if you want to test
your environment before downloading real data.

Usage:
    python src/00_smoke_test.py
    python src/03_train_models.py  # then run training on fake data
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent.parent / "data" / "features.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Mimic the real dataset: 16 control, 15 park, 20 hunt, 13 als (64 total)
groups = (
    [("control", i + 1) for i in range(16)]
    + [("park", i + 1) for i in range(15)]
    + [("hunt", i + 1) for i in range(20)]
    + [("als", i + 1) for i in range(13)]
)

rows = []
for grp, i in groups:
    # Disease subjects have: higher stride variability, more double support,
    # more asymmetry. Controls have tighter, more regular gait.
    is_disease = grp != "control"
    noise = 0.15 if is_disease else 0.05
    base_stride = np.random.normal(1.1, noise)
    rows.append({
        "subject_id": f"{grp}{i}",
        "group": grp,
        "is_disease": int(is_disease),
        "stride_mean": base_stride,
        "stride_std": noise * np.random.uniform(0.8, 1.2),
        "stride_cv": noise / base_stride * np.random.uniform(0.9, 1.1),
        "stride_asymmetry": np.random.uniform(0.005, 0.05) * (3 if is_disease else 1),
        "swing_pct_mean": np.random.normal(38 - (3 if is_disease else 0), 1.5),
        "swing_pct_std": np.random.uniform(0.5, 2.0) * (1.5 if is_disease else 1),
        "stance_pct_mean": np.random.normal(62 + (3 if is_disease else 0), 1.5),
        "stance_pct_std": np.random.uniform(0.5, 2.0) * (1.5 if is_disease else 1),
        "double_support_mean": np.random.normal(22 + (5 if is_disease else 0), 2),
        "double_support_std": np.random.uniform(1, 4) * (1.4 if is_disease else 1),
        "double_support_cv": np.random.uniform(0.05, 0.2) * (1.5 if is_disease else 1),
        "cadence": 120.0 / base_stride,
        "stride_range": np.random.uniform(0.1, 0.4) * (1.8 if is_disease else 1),
        "n_strides": np.random.randint(80, 200),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"Wrote synthetic dataset: {len(df)} subjects")
print(df["group"].value_counts())
print(f"\nFile: {OUT_PATH}")
print("Now run: python src/03_train_models.py")
