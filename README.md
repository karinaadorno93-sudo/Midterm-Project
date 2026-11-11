# Heart Disease Prediction — Machine Learning Model Comparison

## About the Project

This DS 301 Machine Learning midterm project reproduces and evaluates selected methods from Aldila and Supriyono (2026), *Evaluating Logistic Regression, SVM, KNN, and Ensemble Models for Accurate Heart Disease Risk Prediction*. We compare four supervised classification algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree—using the Cleveland Heart Disease dataset.

The primary hyperparameter-tuning metric is **Recall**, chosen to prioritize detection of patients with heart disease and reduce false negatives. Accuracy, Precision, F1-Score, Specificity, and ROC-AUC are also reported. This is an approximate reproduction, not an exact replication of the paper.

## Dataset

- Source: UCI Cleveland Heart Disease dataset.
- Original observations: **303**; observations after removing rows with missing values: **297**.
- Predictors: **13 clinical features**.
- Target: original `num` converted to binary (`0` = no heart disease; `1` = heart disease, original values 1–4).
- Split: **80% training / 20% testing**, stratified, `random_state=42` (**237** training and **60** test observations).
- Test set: **32 negative** and **28 positive** observations.
- Some categorical variables retain their original numeric codes; this is a methodological limitation.

## Repository Structure

```text
heart-disease-prediction/
├── README.md
├── data/
│   └── heart_cleveland.csv
├── models/
│   └── [project-notebook].ipynb
├── [research-paper].pdf
└── [presentation-file-or-link]
```

*Replace bracketed placeholders with the actual filenames or links in your repository. Keep the required `data/` and `models/` folders.*

## How to Run

1. Clone or download this repository.
2. Install Python and the required packages:

   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

3. Open the notebook in the `models/` folder using Jupyter Notebook or VS Code.
4. Confirm that the notebook's dataset path points to `data/heart_cleveland.csv` (adjust the relative path if needed).
5. Run the notebook cells in order to reproduce the preprocessing, model tuning, and evaluation.

## Methodology

1. Inspect the dataset, missing values, and feature distributions.
2. Remove six observations with missing values and binarize the target.
3. Retain the original numerical encodings; keep outliers for this experiment.
4. Create an 80/20 stratified train/test split.
5. Apply MinMax scaling to the range **0–10** inside the relevant model pipelines to avoid data leakage; Decision Tree uses unscaled features.
6. Train Logistic Regression, KNN, SVM (RBF), and Decision Tree.
7. Tune hyperparameters with **GridSearchCV**, **six-fold StratifiedKFold**, and **`scoring="recall"`**.
8. Evaluate tuned models using six-fold cross-validation summaries and the held-out test set.
9. Compare confusion matrices, ROC curves, and ROC-AUC with the published paper.

**Note:** The cross-validation summaries use the same folds as hyperparameter selection and are descriptive, not independent performance estimates.

## Best Hyperparameters (Recall Optimization)

| Model | Best Parameters | Mean CV Recall | Training Accuracy |
|---|---|---:|---:|
| Logistic Regression | `C=0.01` | 0.7788 | 0.8481 |
| KNN | `n_neighbors=9`, `weights='distance'` | **0.7875** | 1.0000 |
| SVM | `C=3`, `gamma=0.001`, `kernel='rbf'` | 0.7788 | 0.8481 |
| Decision Tree | `max_depth=3`, `min_samples_leaf=4`, `min_samples_split=2` | 0.7700 | 0.8354 |

KNN had the highest mean cross-validation Recall, but its 100% training accuracy with distance weighting does not establish generalization.

## Test Set Results

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8333 | 0.8462 | 0.7857 | 0.8148 | 0.8750 | 0.9554 |
| KNN | 0.8167 | 0.8400 | 0.7500 | 0.7925 | 0.8750 | 0.9330 |
| **SVM** | **0.8500** | **0.8800** | **0.7857** | **0.8302** | **0.9062** | **0.9609** |
| Decision Tree | 0.8167 | 0.8148 | 0.7857 | 0.8000 | 0.8438 | 0.8633 |

### Confusion Matrix Summary

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 28 | 4 | 6 | 22 |
| KNN | 28 | 4 | 7 | 21 |
| SVM | 29 | 3 | 6 | 22 |
| Decision Tree | 27 | 5 | 6 | 22 |

**SVM** had the strongest overall test-set results. It tied Logistic Regression and Decision Tree for the highest test Recall (**22 of 28 positive cases identified**) while producing the fewest false positives (**3**). These differences are based on a small test set and do not establish clinical reliability.

## Six-Fold Cross-Validation

| Model | Mean Recall | Standard Deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7788 | 0.1406 | 0.5556 | 0.8947 |
| KNN | 0.7875 | 0.1344 | 0.5556 | 0.9474 |
| SVM | 0.7788 | 0.1611 | 0.5556 | 0.9444 |
| Decision Tree | 0.7700 | 0.1226 | 0.6111 | 0.9444 |

## Comparison with the Research Paper

Although Recall was used to select hyperparameters, ROC-AUC is retained as a secondary metric to compare with the published results.

| Model | Paper ROC-AUC | Our ROC-AUC | Difference |
|---|---:|---:|---:|
| Logistic Regression | 0.9200 | 0.9554 | +0.0354 |
| KNN | 0.9200 | 0.9330 | +0.0130 |
| SVM | 0.9300 | 0.9609 | +0.0309 |
| Decision Tree | 0.8600 | 0.8633 | +0.0033 |

Our scores are numerically higher, but different data splits, preprocessing, model selection, and evaluation details mean that the comparison **does not prove superiority** over the original study. The paper does not clearly specify the GridSearchCV scoring argument.

## Suggested Improvements

- Compare One-Hot Encoding for nominal categorical variables with the original numeric encoding.
- Compare StandardScaler with MinMaxScaler.
- Explore classification thresholds to increase Recall while monitoring false positives.
- Use nested cross-validation for less biased model-selection assessment.
- Investigate feature selection and additional regularization.
- Validate models on larger, external datasets before considering clinical applications.

## Conclusion

Using Recall-based GridSearchCV with six-fold stratified cross-validation, **SVM achieved the best overall performance on the held-out test set**: **85.00% Accuracy, 78.57% Recall, 83.02% F1-Score, and 0.9609 ROC-AUC**. Logistic Regression and Decision Tree matched its test Recall, but SVM achieved stronger secondary metrics. KNN had the highest mean cross-validation Recall but not the highest test Recall.

This experiment demonstrates the importance of selecting evaluation metrics based on project goals, assessing performance on unseen data, and interpreting small-sample results cautiously.

## Reference

Aldila & Supriyono (2026). *Evaluating Logistic Regression, SVM, KNN, and Ensemble Models for Accurate Heart Disease Risk Prediction*. https://doi.org/10.33480/jitk.v11i3.6738

## Project Team

Felipe Kuerten Neves · Karina Adorno · Carlos Alvarez
