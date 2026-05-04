# Chronic Kidney Disease Prediction: Heterogeneous Ensemble ML Pipeline in R

[![R](https://img.shields.io/badge/R-4.3%2B-276DC3?logo=r)](https://www.r-project.org/)
[![caret](https://img.shields.io/badge/caret-6.0%2B-orange)](https://topepo.github.io/caret/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Clinical Problem

Chronic Kidney Disease (CKD) is a progressive condition affecting over 850 million people worldwide. Early detection is critical — by the time symptoms appear, significant and often irreversible kidney damage has already occurred. Routine clinical and laboratory measurements (serum creatinine, hemoglobin, blood pressure, etc.) can flag high-risk patients long before end-stage disease, but manual interpretation across large patient populations is resource-intensive.

This project builds and evaluates a **heterogeneous soft-voting ensemble classifier** that combines logistic regression, random forest, and SVM predictions to identify CKD from tabular clinical data — with the goal of supporting early triage in a clinical decision support context.

> **Why ensembles for clinical AI?** Individual models encode different inductive biases — logistic regression captures linear effects, random forests learn feature interactions, and SVMs find non-linear boundaries. Combining diverse learners produces more stable predictions and reduces the risk of model-specific failures, which matters when the cost of a wrong prediction is patient harm.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Chronic Kidney Disease](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)

**Reference:** Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, School of Information and Computer Science.

400 patients | 24 clinical features | Binary target: CKD vs. Not-CKD

### Feature Overview

| Category | Features |
|----------|----------|
| Vitals | Age, Blood Pressure |
| Urinalysis | Specific Gravity, Albumin, Sugar, Red Blood Cells, Pus Cells, Pus Cell Clumps, Bacteria |
| Blood work | Blood Glucose, Blood Urea, Serum Creatinine, Sodium, Potassium, Hemoglobin, Packed Cell Volume, WBC Count, RBC Count |
| Comorbidities | Hypertension, Diabetes Mellitus, Coronary Artery Disease |
| Symptoms | Appetite, Pedal Edema, Anemia |
| **Target** | **Classification (CKD / Not-CKD)** |

---

## Methods Overview

```
Raw Data → EDA → Missing Value Imputation → Feature Engineering
       → Base Models (LR, RF, SVM) → Bagged Models → Heterogeneous Ensemble
       → Evaluation (Accuracy, Sensitivity, Specificity, AUC)
```

### Models Built

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Base learner | L2 (ridge) regularized via glmnet |
| Random Forest | Base learner | Tuned via caret (optimal mtry = 2) |
| SVM (RBF Kernel) | Base learner | Radial kernel, tuned C + σ via 10-fold CV |
| Bagged Logistic Regression | Ensemble | Bootstrap aggregation over LR |
| Bagged SVM | Ensemble | Bootstrap aggregation over SVM |
| **Heterogeneous Soft-Voting Ensemble** | **Final model** | **Averaged probabilities: LR + RF + SVM** |

### Key Design Decisions

- **KNN imputation** for missing values — preserves relationships between clinical variables better than mean/median imputation
- **One-hot encoding** applied before SVM — SVMs require numeric features; encoding done post-split to prevent leakage
- **Feature scaling** — applied for LR and SVM sensitivity to magnitude
- **Soft-voting** chosen over stacking — stacking was attempted but produced unstable out-of-fold predictions due to dataset size and model heterogeneity (documented in Section 5 of the report)
- **Bootstrap resampling** (bagging) used for LR and SVM to improve stability on smaller clinical datasets
- **ROC as the tuning metric** throughout — not accuracy, reflecting clinical priority on discriminative ability

---

## Results

| Model | Accuracy | Sensitivity | Specificity | AUC |
|-------|----------|-------------|-------------|-----|
| Logistic Regression | 0.9875 | 0.9667 | 1.0000 | 1.0000 |
| Random Forest | 0.9875 | 0.9667 | 1.0000 | 1.0000 |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Bagged Logistic | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Bagged SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Ensemble (LR + RF + SVM)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Evaluated on a held-out test set (25% stratified split). n = 80 test patients.*

### Note on High Performance Metrics

All models achieve near-perfect or perfect classification. This is expected and well-documented for this dataset — once missing values are properly imputed and features standardized, CKD vs. Not-CKD becomes highly separable. The separation is driven by a small number of strong clinical signals: serum creatinine, blood urea, hemoglobin, and packed cell volume. This phenomenon is consistent across published literature on the UCI CKD dataset.

This does **not** indicate overfitting — it reflects genuine clinical separability after appropriate preprocessing. Section 7 of the html report discusses this in full detail, including why results should be interpreted in context and why external validation on a different patient population would be the appropriate next step before any deployment consideration.

---

## Repository Structure

```
ckd-ensemble-ml/
├── README.md
├── MullerC_DA5030_Project.Rmd     ← full analysis: EDA, modeling, evaluation
└── MullerC_DA5030_Project.html
```

> The dataset is loaded directly from a public GitHub mirror in the Rmd script — no manual download required.

---

## How to Run

### Prerequisites

```r
install.packages(c(
  "tidyverse", "caret", "randomForest", "kernlab",
  "glmnet", "pROC", "DMwR2", "VIM", "knitr", "rmarkdown"
))
```

### Knit the report

```r
rmarkdown::render("MullerC_DA5030_Project.Rmd", output_format = "html_document")
```

This will run the full pipeline end-to-end and produce an HTML report with all figures and model comparison tables embedded.

---

## Academic Integrity

This project was completed as a **final course assignment for DA5030 (Practicum in Machine Learning, Northeastern University, Fall 2025)**. It is shared publicly as part of a professional portfolio with full transparency about its academic origin.

The work — including all code, analysis, clinical framing, and written interpretation — is entirely my own. It is posted here to demonstrate applied ML skills to prospective employers and collaborators, not to facilitate academic dishonesty. Any reuse of this code for academic submissions at any institution would constitute a violation of academic integrity policies.

---

## Author

**Cynric Joshua Muller**
MS Bioinformatics, Northeastern University
[LinkedIn] | [GitHub]
