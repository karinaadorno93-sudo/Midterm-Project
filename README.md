# Heart Disease Prediction Using Machine Learning

**DS 301 – Mid-Term Project | CICCC**

**Team:** Felipe Kuerten Neves, Karina Adorno, Carlos Alvarez

---

## About the Project

This project reproduces and evaluates the methodology presented in the research paper:

> **Aldila, A. S., & Supriyono, L. A. (2026).** *Evaluating Logistic Regression, SVM, KNN, and Ensemble Models for Accurate Heart Disease Risk Prediction.* JITK, 11(3), 949–957. DOI: [10.33480/jitk.v11i3.6738](https://doi.org/10.33480/jitk.v11i3.6738)

The objective is to predict **whether a patient has heart disease or not** using clinical data and supervised machine learning.

Four classification algorithms were implemented:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

The project includes exploratory data analysis, data preprocessing, model training, hyperparameter optimization, cross-validation, performance evaluation, and comparison with the original research paper.

Additional improvements are proposed to enhance model generalization and classification performance.

## Dataset

**Cleveland Heart Disease Dataset — UCI Machine Learning Repository**

- Original dataset: 303 patients
- Cleaned dataset: 297 patients
- Input features: 13 clinical variables
- Target variable: `num`
- Classification type: Binary Classification

**Target encoding:**

- `0` = No Heart Disease
- `1` = Heart Disease

The original target values (0–4) were transformed into binary categories.

Six observations containing missing values in `ca` and `thal` were removed.

**Dataset reference:**

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository.

https://doi.org/10.24432/C52P4X

## Repository Structure

| File | Description |
|---|---|
| `heart_cleveland.csv` | Cleveland Heart Disease dataset |
| Main Jupyter Notebook (`.ipynb`) | EDA, preprocessing, training, tuning, evaluation, and visualizations |
| Research Paper (`.pdf`) | Original research article |

## How to Run

1. Open the project notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Ensure that `heart_cleveland.csv` is available in the working directory.
3. Install the required Python libraries if necessary.
4. Run the notebook cells sequentially.
5. Review the generated tables, charts, and model evaluation results.

**Requirements:** Python 3, pandas, numpy, scikit-learn, matplotlib, and seaborn.

## Methodology

**1. Exploratory Data Analysis (EDA)**

- Examined dataset structure, data types, and descriptive statistics.
- Identified missing values and duplicate observations.
- Detected outliers using the Interquartile Range (IQR) method.
- Analyzed numerical and categorical features.
- Visualized feature distributions, target distribution, and correlations.

**2. Data Preprocessing**

- Converted the original target values (0–4) into binary classification labels.
- Removed six observations containing missing values.
- Retained the original numerical encoding of categorical features.
- Preserved outliers due to their potential clinical relevance.

**3. Train/Test Split**

- Applied an 80/20 stratified train/test split.
- Used `random_state=42`.
- Training set: 237 patients.
- Testing set: 60 patients.

**4. Feature Scaling**

- Applied Min-Max Scaling to the (0, 10) range, following the research paper.
- Fitted scaling transformations on training data only.
- Used preprocessing pipelines during cross-validation to prevent data leakage.
- Decision Tree was trained without scaling because it uses threshold-based splits.

**5. Model Training**

Four classification models were trained using the configurations reported in the original research.

**6. Hyperparameter Optimization**

- Applied GridSearchCV to all four classification models.
- Used stratified six-fold cross-validation.
- Optimized ROC-AUC.
- Selected the best hyperparameter configurations using training data only.

**7. Model Evaluation**

Evaluated the tuned models using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC
- Confusion Matrices
- ROC Curves

The project also includes Decision Tree visualization, Feature Importance, and model performance comparison charts.

## Best Hyperparameters

| Model | Best Hyperparameters |
|---|---|
| Logistic Regression | C = 0.01, L2 regularization |
| KNN | K = 15, weights = uniform |
| SVM (RBF) | C = 0.1, gamma = 0.001 |
| Decision Tree | max_depth = 3, min_samples_leaf = 4, min_samples_split = 2 |

## Results — Tuned Models (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **83.33%** | 0.8462 | **0.7857** | **0.8148** | 0.8750 | **0.9554** |
| KNN (K = 15) | 80.00% | 0.8333 | 0.7143 | 0.7692 | 0.8750 | 0.9386 |
| SVM (RBF) | 81.67% | **0.8696** | 0.7143 | 0.7843 | **0.9062** | 0.9464 |
| Decision Tree | 81.67% | 0.8148 | **0.7857** | 0.8000 | 0.8438 | 0.8633 |

**Logistic Regression achieved the best overall performance**, with the highest Accuracy (83.33%), F1-Score (0.8148), and ROC-AUC (0.9554).

SVM achieved the highest Precision and Specificity, while Logistic Regression and Decision Tree achieved the highest Recall.

Logistic Regression correctly identified 22 of the 28 patients with heart disease in the independent test set.

## Six-Fold Cross-Validation Results

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| **Logistic Regression** | **0.8859** | 0.0710 |
| KNN | 0.8656 | **0.0689** |
| SVM | 0.8831 | 0.0749 |
| Decision Tree | 0.8348 | 0.0819 |

Logistic Regression achieved the highest mean ROC-AUC during hyperparameter optimization, closely followed by SVM.

KNN demonstrated the lowest variability across validation folds.

These cross-validation results were used during hyperparameter selection and do not represent an independent evaluation.

## Comparison with the Research Paper

| Model | Paper ROC-AUC | Our Test ROC-AUC | Difference |
|---|---:|---:|---:|
| Logistic Regression | 0.9200 | **0.9554** | +0.0354 |
| KNN | 0.9200 | 0.9386 | +0.0186 |
| SVM | 0.9300 | 0.9464 | +0.0164 |
| Decision Tree | 0.8600 | 0.8633 | +0.0033 |

All four models achieved slightly higher ROC-AUC scores than the values reported in the original research.

However, differences in data splitting, cross-validation implementation, and hyperparameter optimization mean that the experiment represents an **approximate reproduction rather than an exact replication**.

The higher scores do not establish that our models are statistically or clinically superior.

## Suggested Improvements

**1. One-Hot Encoding**

Evaluate One-Hot Encoding for nominal categorical features instead of retaining their original numerical codes.

**2. Alternative Feature Scaling**

Compare StandardScaler with MinMaxScaler (0–10) to determine whether standardization improves model performance.

**3. Overfitting Prevention**

Investigate additional regularization and model complexity controls, especially for SVM and Decision Tree, which initially achieved 100% training accuracy.

**4. Classification Threshold Optimization**

Evaluate alternative classification thresholds to improve Recall and reduce False Negatives, selecting thresholds using cross-validation rather than the independent test set.

## Conclusion

This project successfully implemented and evaluated four machine learning classification algorithms for heart disease prediction using the Cleveland Heart Disease Dataset.

**Logistic Regression demonstrated the strongest overall performance**, achieving 83.33% test accuracy and a ROC-AUC of 0.9554.

SVM also demonstrated strong discrimination, while KNN and Decision Tree achieved competitive results.

The experiment highlighted the importance of preprocessing, hyperparameter tuning, cross-validation, and evaluating multiple performance metrics.

Although the results are promising, the small dataset and limited test sample indicate that additional validation is necessary before considering these models for clinical decision-support applications.
