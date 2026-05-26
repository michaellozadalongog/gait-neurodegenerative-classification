"""
Extract gait features from PhysioNet gaitndd .ts files.

Each .ts file has 13 columns:
   1  Elapsed Time (sec)
   2  Left Stride Interval (sec)
   3  Right Stride Interval (sec)
   4  Left Swing Interval (sec)
   5  Right Swing Interval (sec)
   6  Left Swing Interval (% of stride)
   7  Right Swing Interval (% of stride)
   8  Left Stance Interval (sec)
   9  Right Stance Interval (sec)
  10  Left Stance Interval (% of stride)
  11  Right Stance Interval (% of stride)
  12  Double Support Interval (sec)
  13  Double Support Interval (% of stride)

We compute summary statistics over each subject's walking record. These are the
classic gait features used in Hausdorff et al. and follow-on papers.

Usage:
    python src/02_extract_features.py
    -> writes data/features.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "gaitndd"
OUT_PATH = Path(__file__).parent.parent / "data" / "features.csv"

COLUMNS = [
    "time",
    "stride_L", "stride_R",
    "swing_L_sec", "swing_R_sec",
    "swing_L_pct", "swing_R_pct",
    "stance_L_sec", "stance_R_sec",
    "stance_L_pct", "stance_R_pct",
    "double_support_sec", "double_support_pct",
]

GROUP_PREFIXES = {"control": 16, "park": 15, "hunt": 20, "als": 13}


def load_ts(path: Path) -> pd.DataFrame:
    """Load a .ts file as a DataFrame. Drop the first 20 seconds (startup transients)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df = df[df["time"] >= 20.0].reset_index(drop=True)
    return df


def extract_features(df: pd.DataFrame) -> dict:
    """Compute summary statistics for one subject's gait record."""
    feats = {}

    # Symmetric stride: mean of left+right stride times
    stride_all = pd.concat([df["stride_L"], df["stride_R"]])
    feats["stride_mean"] = stride_all.mean()
    feats["stride_std"] = stride_all.std()
    feats["stride_cv"] = stride_all.std() / stride_all.mean()  # coefficient of variation

    # Asymmetry: difference between L and R
    feats["stride_asymmetry"] = abs(df["stride_L"].mean() - df["stride_R"].mean())

    # Swing %
    swing_pct_all = pd.concat([df["swing_L_pct"], df["swing_R_pct"]])
    feats["swing_pct_mean"] = swing_pct_all.mean()
    feats["swing_pct_std"] = swing_pct_all.std()

    # Stance %
    stance_pct_all = pd.concat([df["stance_L_pct"], df["stance_R_pct"]])
    feats["stance_pct_mean"] = stance_pct_all.mean()
    feats["stance_pct_std"] = stance_pct_all.std()

    # Double support — important fall-risk indicator
    feats["double_support_mean"] = df["double_support_pct"].mean()
    feats["double_support_std"] = df["double_support_pct"].std()
    feats["double_support_cv"] = (
        df["double_support_sec"].std() / df["double_support_sec"].mean()
    )

    # Cadence proxy: steps per minute (2 steps per stride)
    feats["cadence"] = 120.0 / stride_all.mean()  # steps/min

    # Variability metric: range of stride times
    feats["stride_range"] = stride_all.max() - stride_all.min()

    # Number of strides recorded (rough record length proxy)
    feats["n_strides"] = len(df)

    return feats


def main() -> None:
    rows = []
    for prefix, count in GROUP_PREFIXES.items():
        for i in range(1, count + 1):
            subject_id = f"{prefix}{i}"
            ts_path = DATA_DIR / f"{subject_id}.ts"
            if not ts_path.exists():
                print(f"  missing: {ts_path}")
                continue
            df = load_ts(ts_path)
            feats = extract_features(df)
            feats["subject_id"] = subject_id
            feats["group"] = prefix
            feats["is_disease"] = int(prefix != "control")  # binary label
            rows.append(feats)

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(features_df)} rows to {OUT_PATH}")
    print("\nClass balance:")
    print(features_df["group"].value_counts())
    print("\nBinary balance (disease vs control):")
    print(features_df["is_disease"].value_counts())


if __name__ == "__main__":
    main()
