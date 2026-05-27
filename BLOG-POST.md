# Can a Computer Spot Neurodegenerative Disease from a 5-Minute Walk?

*First project in my biomedical ML research portfolio. Code on [GitHub](https://github.com/michaellozadalongog/gait-neurodegenerative-classification).*

---

Walking looks simple. We do it without thinking. But for a clinician evaluating a patient with Parkinson's, Huntington's, or ALS, the way someone walks carries an enormous amount of information: stride length, timing variability, asymmetry between left and right legs. These tiny details are often the earliest warning signs that something neurological is changing.

I wanted to see if a machine learning model could pick up on those signals from raw gait timing data alone. No cameras, no doctors interpreting video, just numbers off a sensor.

This is the first of three research projects I'm building as an undergraduate Biomedical Engineering student at Hawaii Pacific University, with the goal of going deep on the intersection of ML and biomedical signals. The next two will tackle ECG arrhythmia detection and diabetic retinopathy from retinal images.

## The question

**Can a classical machine learning model distinguish people with neurodegenerative disease from healthy controls, using only stride timing features from a 5-minute walking trial?**

## The data

I used the [PhysioNet Gait in Neurodegenerative Disease Database](https://physionet.org/content/gaitndd/1.0.0/), originally collected by Hausdorff and colleagues at Beth Israel Hospital and MIT. 64 subjects total:

- 16 healthy controls
- 15 with Parkinson's disease
- 20 with Huntington's disease
- 13 with ALS

Each subject walked for about 5 minutes while force-sensitive resistors under their feet recorded every footfall. The data comes as 13 columns per subject, capturing left and right stride times, swing and stance phases, and double-support intervals.

64 subjects is tiny by ML standards. That mattered for my methodology, but as you'll see, the signal in the data turned out to be strong enough to work with.

## What I did

1. **Dropped the first 20 seconds** of each recording. People speed up and stabilize in their first few steps, and those transients aren't representative of steady-state gait.
2. **Engineered 14 features** capturing what gait researchers have studied for decades: stride mean, stride variability (coefficient of variation), left-right asymmetry, swing and stance percentages, double support, cadence.
3. **Framed it as binary classification**: disease vs. healthy control. Not predicting which specific disease, just whether neurodegenerative pathology is present.
4. **Trained two models**: logistic regression as a linear baseline, and random forest, which captures non-linear feature interactions.
5. **Used 5-fold stratified cross-validation.** This part matters more than it sounds. With only 16 controls against 48 disease subjects, a non-stratified split could land all the controls in a single fold, making cross-validation meaningless. Stratification forces each fold to keep the original class ratio, so every fold is actually testing the same problem.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.828 | 0.849 | 0.938 | 0.891 | 0.879 |
| **Random Forest** | **0.859** | **0.898** | 0.917 | **0.907** | **0.928** |

The random forest hit **AUC 0.928**, meaning if you handed it a random sick person and a random healthy person, it would correctly rank them by risk about 93% of the time.

For context, published papers on this dataset report AUCs that are roughly comparable, though the comparison isn't quite apples-to-apples. Many use leave-one-out cross-validation, different feature sets, or per-disease AUCs rather than binary disease-vs-control. My result sits in the same ballpark as the literature, which is what I was looking for as a sanity check.

The other thing worth flagging: the class imbalance (48 disease vs 16 controls) inflates how good the headline numbers look. A trivial baseline that just predicts "diseased" for everyone would get 100% recall and 75% accuracy. So 92% recall sounds great in isolation, but the relevant comparison is precision and AUC, where the random forest clears that baseline meaningfully.

![ROC curves comparing logistic regression and random forest across 5-fold stratified cross-validation](https://raw.githubusercontent.com/michaellozadalongog/gait-neurodegenerative-classification/main/figures/roc_curves.png)

Both ROC curves hug the top-left corner, which is what good classifiers look like. The diagonal dashed line is what random guessing would produce.

## What I learned

**1. 64 subjects was enough, and that's the surprising part.** I went in expecting the dataset to be too small for reliable results. It wasn't. The biological difference between Parkinsonian gait and healthy gait is huge, with stride variability several times higher in the disease group, and the models pick that up easily. The lesson: when the effect size is large, you don't always need millions of samples to find real signal. This reframed how I think about biomedical datasets going forward.

**2. The relationship isn't purely linear.** Random forest beat logistic regression by a meaningful AUC margin (0.928 vs 0.879). That gap is the model telling you there are interactions between features. High stride variability probably matters more when double support is also elevated than either does alone. A linear model can't capture that. A tree-based model can.

**3. Recall was high, but that's partly the imbalance talking.** Both models caught over 91% of the actually-sick subjects. For a screening application, that's the metric you want high, since missing someone with a real condition is worse than a false alarm. But the trivial all-diseased baseline already hits 100% recall, so the more honest read is that the model is high-precision *and* high-recall, not just that recall is great.

## What I'd do next

- **Feature importance analysis.** Which of the 14 features actually drove the random forest's predictions? Gait researchers point at double-support variability and stride coefficient of variation as the strongest markers. Does the model agree?
- **Subtyping, not just detection.** Can a model distinguish Parkinson's from Huntington's from ALS, instead of lumping them all as "disease"? Much harder problem, more clinically useful.
- **Age confounding.** Disease subjects in this dataset are older on average. A more rigorous version would control for age explicitly so the model doesn't just learn "old = sick."
- **From hand-engineered features to deep learning.** This is exactly what Project B will tackle, using a 1D CNN on raw signal data (ECG arrhythmia detection from the MIT-BIH dataset), with the same pipeline shape. Then I'll compare what deep learning actually buys you over classical ML.

## Tech stack

- Python 3.14
- scikit-learn for the models
- pandas and NumPy for data wrangling
- matplotlib and seaborn for figures
- The PhysioNet team for keeping this data free and accessible

Full code, reproducible from scratch: [github.com/michaellozadalongog/gait-neurodegenerative-classification](https://github.com/michaellozadalongog/gait-neurodegenerative-classification)

Project B (ECG arrhythmia detection) is in progress. I'll post it when the results are in.

---

*Michael Lozada-Longog is a Biomedical Engineering undergraduate at Hawaii Pacific University. KapoleiBioForge is where I show my work.*
