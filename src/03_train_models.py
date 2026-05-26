"""
Train and evaluate three classifiers on the gait features.

Task: binary classification of disease (Parkinson's/Huntington's/ALS) vs control.

Why 5-fold stratified CV: dataset is small (64 subjects). A single train/test
split would have huge variance. CV gives a real estimate.

Usage:
    python src/03_train_models.py
    -> writes figures/confusion_matrix_<model>.png
    -> writes figures/roc_curves.png
    -> prints results table
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    raise ImportError("skip xgboost")
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features.csv"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def get_models() -> dict:
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
        )
    return models


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in {"subject_id", "group", "is_disease"}]
    X = df[feature_cols].values
    y = df["is_disease"].values

    print(f"Loaded {len(df)} subjects, {len(feature_cols)} features")
    print(f"Class balance: {y.sum()} disease / {len(y) - y.sum()} control\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    results = []

    plt.figure(figsize=(7, 6))

    for name, model in models.items():
        # CV predictions for confusion matrix and metrics
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)

        results.append({
            "model": name,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        })

        # Confusion matrix figure
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Control", "Disease"],
                    yticklabels=["Control", "Disease"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close()

        # Add ROC to combined plot
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Finalize ROC curve figure
    plt.figure(1)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — 5-fold CV (Disease vs Control)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # Print and save results
    results_df = pd.DataFrame(results)
    print("=" * 70)
    print("RESULTS (5-fold stratified CV)")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format="%.3f"))
    results_df.to_csv(FIGURES_DIR.parent / "data" / "results.csv", index=False)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Results CSV: {FIGURES_DIR.parent / 'data' / 'results.csv'}")


if __name__ == "__main__":
    main()
