# Heart Disease Prediction – Paper Reproduction

**DS 301 – Mid-Term Project | CICCC**

**Team:** Carlos Alvarez, Felipe Kuerten Neves, Karina Adorno

---

## About the project

This project reproduces the research paper:

> **Aldila, A. S., & Supriyono, L. A. (2026).** *Evaluating Logistic Regression, SVM, KNN, and Ensemble Models for Accurate Heart Disease Risk Prediction.* JITK, 11(3), 949–957. DOI: [10.33480/jitk.v11i3.6738](https://doi.org/10.33480/jitk.v11i3.6738)

The goal is to predict **whether a patient has heart disease or not** (binary classification) using clinical data. We reproduced the paper's steps for the four models covered in class: **Logistic Regression, KNN, SVM and Decision Tree**. Then we tuned the models, compared our results with the paper, and suggested improvements.

## Dataset

**Cleveland Heart Disease dataset** – UCI Machine Learning Repository
- 303 patients, 13 features + 1 target (297 patients after removing missing values)
- Target: 0 = No Disease, 1 = Heart Disease (original values 1–4 grouped as "disease")

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0)

> We used `processed.cleveland.data` (14 attributes). The raw `cleveland.data` file is reported as corrupted in the dataset's WARNING note (David Aha, UCI), so it was not used.

## Repository structure

| File | Description |
|---|---|
| `Heart_Disease_Combinedvfinal.ipynb` | Final notebook (data preparation, models, evaluation, comparison with the paper) |
| `Final- Heart_Disease_Presentatio.pptx` | Final presentation |
| `Heart Disease Paper.pdf` | The research paper |
| `processed.cleveland.data` | Original dataset file from UCI (no header, `?` for missing values) |
| `heart_cleveland.csv` | Same data saved as a CSV with header |
| `heart-disease.names` | Dataset documentation from UCI |

Each team member also has an individual branch (`Carlos`, `Felipe`, `Karina`) with their own notebook.

## How to run

1. Open `Heart_Disease_Combinedvfinal.ipynb` in **Google Colab** (or Jupyter).
2. Run all cells in order (*Runtime → Run all* in Colab).

The notebook downloads the data automatically with `ucimlrepo`. If that fails, it loads `processed.cleveland.data` from this repository. No manual upload is needed.

**Requirements:** Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn, ucimlrepo (installed by the notebook).

## Methodology

1. **Data cleaning:** removed 6 patients with missing values in `ca` and `thal`; converted the target (0–4) to binary (0/1).
2. **Train/test split:** 80/20, stratified, `random_state = 42` (237 train, 60 test patients).
3. **Pipeline (no data leakage):** ordinal encoding of `cp`, `restecg`, `slope` and `thal`, then Min-Max scaling to the (0, 10) range, as in the paper. The pipeline is fitted on training data only.
4. **Paper reproduction:** each model trained with the hyperparameters reported in the paper.
5. **Hyperparameter tuning:** GridSearchCV with stratified **6-fold** cross-validation (as in the paper), choosing the best settings by **recall**.
6. **Evaluation on the test set:** accuracy, recall, precision, specificity, F1-score, ROC-AUC, confusion matrices and ROC curves.

**Main metric: recall**, because in medicine the worst error is to predict a sick patient as healthy.

## Results

### 1. Reproduction with the paper's settings (test set)

| Model | Paper accuracy | Our accuracy | Sick patients found |
|---|---|---|---|
| Logistic Regression (C = 0.1) | 86.9% | 83.3% | 22 of 28 |
| KNN (K = 7) | 88.5% | 86.7% | 23 of 28 |
| Decision Tree | 85.3% | 73.3% | 19 of 28 |
| SVM (RBF, C = 3, gamma = 0.65) | 86.9% | **53.3%** | **1 of 28** |

Logistic Regression and KNN were close to the paper. The Decision Tree overfits (100% on training data). The SVM fails: with the (0, 10) scaling, the paper's gamma = 0.65 is too large, so every patient looks different from all the others and the model predicts "healthy" for almost everyone.

### 2. Tuned models (test set, 60 patients, 28 sick)

| Model | Best settings | Accuracy | Recall | Precision | Specificity | F1 | AUC | Missed |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | C = 0.01 | 83.3% | 0.79 | 0.85 | 0.88 | 0.81 | 0.95 | 6 |
| KNN | K = 9, distance, euclidean | 81.7% | 0.75 | 0.84 | 0.88 | 0.79 | 0.94 | 7 |
| **SVM** | **RBF, C = 10, gamma = 0.001** | **85.0%** | 0.79 | **0.88** | **0.91** | **0.83** | **0.96** | 6 |
| Decision Tree | gini, depth 3, min leaf 5 | 83.3% | **0.82** | 0.82 | 0.84 | 0.82 | 0.87 | **5** |

*Missed = sick patients predicted as healthy (false negatives).*

## Conclusions

- **Recommended model: SVM (RBF, C = 10, gamma = 0.001)**, with the best overall balance: highest accuracy, precision, specificity, F1 and AUC. The Decision Tree had the highest recall, but only 1 patient more.
- The differences between the models are small (1 to 3 patients), so the choice should be confirmed with more data.
- We reproduced the paper approximately, with the same level of performance.
- **Problems found in the paper:** the KNN and SVM hyperparameters are swapped; the SVM gamma does not fit the (0, 10) scaling; some numbers differ between the text and the tables; the train/test split (random_state, stratification) is not described.
- **Limitations:** small and old dataset, mostly male patients; the model is not validated for clinical use.

## Suggested improvements

One-hot encoding for categorical features, standard (0, 1) scaling, repeated cross-validation, threshold tuning to increase recall, more data from the other UCI hospitals, and full reporting of the methodology.
