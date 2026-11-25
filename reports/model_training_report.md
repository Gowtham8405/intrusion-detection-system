# STEP 5 — Machine Learning Model Training Report

## 1. Overview
This report details the binary classification model training and validation for the AI-Based Intrusion Detection System.

- **Training Set Size**: 47,554 records
- **Validation Set Size**: 8,391 records
- **Preprocessed Input Features**: 188 dimensions
- **Random State**: Fixed `random_state=42` across all stochastic algorithms for full reproducibility.
- **Test Set Protection**: Test dataset (`X_test.npy`, `y_test.npy` with 107,740 records) was kept completely untouched and unseen.

---

## 2. Model Configuration & Parameters
1. **Logistic Regression**: `max_iter=1000`, `random_state=42`, default L2 penalty.
2. **Decision Tree Classifier**: `random_state=42`, Gini criterion, unconstrained depth.
3. **Random Forest Classifier**: `n_estimators=100`, `random_state=42`, `n_jobs=-1`.
4. **Extra Trees Classifier**: `n_estimators=100`, `random_state=42`, `n_jobs=-1`.

---

## 3. Model Comparison Table (Validation Set)

| Model Name | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Prediction Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8942 | 0.8837 | 0.8401 | **0.8614** | 2.330 | 0.0057 |
| **Decision Tree** | 0.9476 | 0.9341 | 0.9318 | **0.9329** | 1.064 | 0.0058 |
| **Random Forest** | 0.9622 | 0.9634 | 0.9391 | **0.9511** | 1.709 | 0.0335 |
| **Extra Trees** | 0.9554 | 0.9657 | 0.9187 | **0.9416** | 3.067 | 0.0580 |

---

## 4. Final Model Selection
- **Selected Model**: `Random Forest`
- **Selection Criteria**: Prioritized **F1-Score** and **Recall** on the validation set (`X_val`) because in intrusion detection, minimizing False Negatives (uncaught attacks) while maintaining high precision is critical.
- **Validation Performance**:
  - Accuracy: `0.9622`
  - Precision: `0.9634`
  - Recall: `0.9391`
  - **F1-Score**: `0.9511`

---

## 5. Artifact Locations
- Candidate Models:
  - `ml/artifacts/models/logistic_regression.joblib`
  - `ml/artifacts/models/decision_tree.joblib`
  - `ml/artifacts/models/random_forest.joblib`
  - `ml/artifacts/models/extra_trees.joblib`
- **Final Model Artifact**: `ml/artifacts/final_model.joblib`
- Comparison CSV: `ml/artifacts/model_comparison.csv`
- Model Metadata: `ml/artifacts/model_metadata.json`

---

## 6. Reproducibility & Limitations
- All models were initialized with explicit seeds (`random_state=42`).
- Hyperparameter tuning was restricted to validation split comparisons without leakage from `X_test`.
- **Limitation**: Classical tree ensembles are efficient for tabular intrusion data, but feature scaling and encodings remain tied to `preprocessor.joblib`.
