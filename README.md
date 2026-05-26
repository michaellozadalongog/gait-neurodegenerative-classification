# Gait-Based Classification of Neurodegenerative Disease

Binary classification of neurodegenerative disease (Parkinson's, Huntington's, ALS) vs. healthy controls from gait timing features, using the PhysioNet Gait in Neurodegenerative Disease Database.

This is **Project A** of a three-project biomedical ML portfolio (Projects B and C cover ECG arrhythmia detection and diabetic retinopathy imaging).

## Why this matters

Falls are the leading cause of injury death in adults over 65. Early gait abnormalities are one of the strongest predictors of fall risk and disease progression in conditions like Parkinson's. Models that can flag at-risk individuals from short walking samples are a building block for at-home monitoring and clinical decision support — the same problem space targeted by wearables like the Apple Watch and clinical devices from Medtronic and Stryker.

## Dataset

**Hausdorff et al., PhysioNet Gait in Neurodegenerative Disease Database** (n = 64):

| Group | n | Condition |
|---|---|---|
| Control | 16 | Healthy |
| Parkinson's | 15 | Parkinson's disease |
| Huntington's | 20 | Huntington's disease |
| ALS | 13 | Amyotrophic lateral sclerosis |

Each subject provides a 5-minute walking trial sampled by force-sensitive resistors under each foot, with 13 stride-by-stride timing measures (stride interval, swing/stance time, double support).

> Hausdorff JM et al. *Dynamic markers of altered gait rhythm in amyotrophic lateral sclerosis.* J Applied Physiology 88:2045-2053, 2000.

## Approach

1. **Preprocess.** Drop the first 20 seconds of each record (startup transients). Pool left/right limb measurements for symmetric features and compute asymmetry separately.
2. **Feature engineering.** 14 features capturing central tendency, variability, asymmetry, and rhythm:
   - Stride: mean, std, coefficient of variation, asymmetry, range
   - Swing %: mean, std
   - Stance %: mean, std
   - Double support: mean, std, CV
   - Cadence (steps/min)
   - Record length (n strides)
3. **Models.** Logistic regression (with feature scaling), random forest, XGBoost.
4. **Evaluation.** 5-fold stratified cross-validation. Reported: accuracy, precision, recall, F1, ROC-AUC.

## Results

_To be filled in after running on real data. Expected AUC range from prior literature on this dataset: 0.80–0.92._

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

See `figures/roc_curves.png` and `figures/confusion_matrix_*.png`.

## Repo structure

```
project-a/
├── data/                      # downloaded data + processed features
│   └── gaitndd/
├── src/
│   ├── 00_smoke_test.py       # synthetic data for pipeline testing
│   ├── 01_download_data.py    # PhysioNet download
│   ├── 02_extract_features.py # .ts files → features.csv
│   └── 03_train_models.py     # train + evaluate
├── figures/                   # generated plots
├── notebooks/                 # exploratory work
└── README.md
```

## Reproduce

```bash
# 1. Set up env
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) verify pipeline with synthetic data
python src/00_smoke_test.py
python src/03_train_models.py

# 3. Real run
python src/01_download_data.py
python src/02_extract_features.py
python src/03_train_models.py
```

Runtime: < 2 minutes on a laptop, no GPU needed.

## What I learned

_To be filled in after running. Plan to write about:_

- Which gait features are most predictive (double support variability and stride CV are the literature's top picks — does that match what feature importance tells us?)
- Why a small dataset still gives useful signal (clean labels, large effect sizes)
- Where this would fail in the real world (lab conditions vs. free-living gait)
- What I'd do next (1D CNN on raw stride sequences — Project B will use the same pipeline shape)

## Limitations

- Small n (64). All performance estimates are wide confidence intervals.
- Subjects with disease are also older on average → age is a confounder. A more rigorous analysis would control for it.
- Disease vs. control is a coarser task than disease subtyping. Subtyping is harder and worth attempting once the binary task is solid.
- Force-sensor data may not transfer to consumer wearables (IMU) without domain adaptation.

## Citation

If you use this code:

```
Lozada-Longog, M. (2026). Gait-Based Classification of Neurodegenerative Disease.
GitHub: https://github.com/<your-handle>/gait-neurodegenerative-classification
```

Built as part of an independent undergraduate research portfolio.
Biomedical Engineering, Hawaii Pacific University.
