# Heart Disease Prediction – Paper Reproduction

**DS 301 – Mid-Term Project | CICCC**

**Team:** Felipe Kuerten Neves, Karina Adorno, Carlos Alvarez

---

## About the project

This project reproduces the research paper:

> **Aldila, A. S., & Supriyono, L. A. (2026).** *Evaluating Logistic Regression, SVM, KNN, and Ensemble Models for Accurate Heart Disease Risk Prediction.* JITK, 11(3), 949–957. DOI: [10.33480/jitk.v11i3.6738](https://doi.org/10.33480/jitk.v11i3.6738)

The goal is to predict **whether a patient has heart disease or not** (binary classification) using clinical data. We reproduced the paper's steps for the models covered in class: **Logistic Regression, KNN, SVM and Decision Tree**, compared our results with the paper, and suggested improvements.

## Dataset

**Cleveland Heart Disease dataset** – UCI Machine Learning Repository
- 303 patients, 13 features + 1 target (297 patients after removing missing values)
- Target: 0 = No Disease, 1 = Heart Disease

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0)

> We used `processed.cleveland.data` (14 attributes). The raw `cleveland.data` file is reported as corrupted in the dataset's WARNING note (David Aha, UCI), so it was not used.

## Repository structure

| File | Description |
|---|---|
| `Felipe_heart_disease.ipynb` | Main notebook (data preparation, models, evaluation) |
| `processed.cleveland.data` | Original dataset file from UCI (no header, `?` for missing values) |
| `heart_cleveland.csv` | Same data saved as a CSV with header |
| `heart-disease.names` | Dataset documentation from UCI |
| `Heart Disease Paper.pdf` | The research paper |

## How to run

1. Open `Felipe_heart_disease.ipynb` in **Google Colab** (or download it and open it in Jupyter).
2. Download `processed.cleveland.data` from this repository.
3. Run the notebook cells **in order** (*Runtime → Run all* in Colab).
4. When **Cell 1** asks for a file, upload `processed.cleveland.data`.

**Requirements:** Python 3, pandas, numpy, scikit-learn, matplotlib (all already installed in Google Colab).

## Methodology

1. **Data cleaning:** removed 6 rows with missing values in `ca` and `thal`; converted the target (0–4) to binary (0/1)
2. **Label encoding** for `cp` and `thal`; outliers checked with boxplots and kept (clinical relevance)
3. **80/20 stratified train/test split** (`random_state = 42`)
4. **Min-Max scaling** to the (0, 10) range, fitted on training data only
5. **Paper reproduction** with the hyperparameters reported in the paper
6. **Hyperparameter tuning** with GridSearchCV (stratified 5-fold cross-validation, optimizing **recall**)
7. **Evaluation:** accuracy, recall, precision, specificity, F1-score, AUC, confusion matrices and ROC curves

## Results (tuned models, test set)

| Model | Accuracy | Recall | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 83.3% | 0.79 | 0.88 | 0.95 |
| **KNN (K = 7)** | **86.7%** | **0.82** | **0.91** | 0.94 |
| SVM (RBF) | 78.3% | 0.75 | 0.81 | 0.90 |
| Decision Tree | 80.0% | 0.79 | 0.81 | 0.82 |

**KNN (K = 7) was the best model**, which agrees with the paper. The paper's SVM hyperparameters were not compatible with its (0, 10) scaling; after tuning, SVM accuracy improved from 53% to 78%.
