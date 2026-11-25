"""
ml/train_models.py

Machine Learning model training and validation script for UNSW-NB15 binary classification.

Models trained:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Extra Trees Classifier

Evaluation is strictly performed on X_val / y_val.
Test data (X_test / y_test) is untouched.
"""

import os
import time
import json
import pathlib
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Directory Configuration
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data():
    X_train = np.load(ARTIFACTS_DIR / "X_train.npy")
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
    X_val = np.load(ARTIFACTS_DIR / "X_val.npy")
    y_val = np.load(ARTIFACTS_DIR / "y_val.npy")
    return X_train, y_train, X_val, y_val

def train_and_evaluate():
    print("Loading preprocessed training and validation data...")
    X_train, y_train, X_val, y_val = load_data()
    
    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Candidate Models Configuration
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    filename_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "Extra Trees": "extra_trees.joblib"
    }
    
    results = []
    trained_model_objects = {}
    
    for name, model in candidate_models.items():
        print(f"\n--- Training {name} ---")
        
        # 1. Train Model
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # 2. Predict on Validation Split
        start_pred = time.time()
        y_val_pred = model.predict(X_val)
        pred_time = time.time() - start_pred
        
        # 3. Calculate Validation Metrics
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        print(f"Validation Accuracy:  {acc:.4f}")
        print(f"Validation Precision: {prec:.4f}")
        print(f"Validation Recall:    {rec:.4f}")
        print(f"Validation F1-Score:  {f1:.4f}")
        print(f"Training Time:        {train_time:.2f}s")
        print(f"Prediction Time:      {pred_time:.4f}s")
        
        # 4. Save Trained Model
        save_path = MODELS_DIR / filename_map[name]
        joblib.dump(model, save_path)
        print(f"Saved model artifact: {save_path}")
        
        trained_model_objects[name] = model
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Train_Time_Sec": round(train_time, 3),
            "Pred_Time_Sec": round(pred_time, 4),
            "Saved_Filename": filename_map[name]
        })
        
    # Create DataFrame of Results
    df_results = pd.DataFrame(results)
    comparison_csv_path = ARTIFACTS_DIR / "model_comparison.csv"
    df_results.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved model comparison CSV to: {comparison_csv_path}")
    
    # Select Best Model based on Validation F1-Score
    best_row = df_results.sort_values(by="F1_Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_model_objects[best_name]
    
    print(f"\n==========================================")
    print(f"SELECTED FINAL MODEL: {best_name}")
    print(f"Validation F1-Score:  {best_row['F1_Score']:.4f}")
    print(f"Validation Accuracy:  {best_row['Accuracy']:.4f}")
    print(f"==========================================")
    
    # Save Final Model
    final_model_path = ARTIFACTS_DIR / "final_model.joblib"
    joblib.dump(best_model, final_model_path)
    print(f"Saved final model artifact to: {final_model_path}")
    
    # Save Metadata
    metadata = {
        "selected_model_name": best_name,
        "selected_model_filename": best_row["Saved_Filename"],
        "num_features": X_train.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "random_state": 42,
        "validation_metrics": {
            "accuracy": float(best_row["Accuracy"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
            "f1_score": float(best_row["F1_Score"]),
            "train_time_sec": float(best_row["Train_Time_Sec"]),
            "pred_time_sec": float(best_row["Pred_Time_Sec"])
        }
    }
    metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {metadata_path}")
    
    # Generate Markdown Report
    generate_report(df_results, best_name, best_row, X_train.shape[1], len(X_train), len(X_val))

def generate_report(df_results, best_name, best_row, num_features, train_samples, val_samples):
    report_content = f"""# STEP 5 — Machine Learning Model Training Report

## 1. Overview
This report details the binary classification model training and validation for the AI-Based Intrusion Detection System.

- **Training Set Size**: {train_samples:,} records
- **Validation Set Size**: {val_samples:,} records
- **Preprocessed Input Features**: {num_features} dimensions
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
"""
    for idx, row in df_results.iterrows():
        report_content += f"| **{row['Model']}** | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | **{row['F1_Score']:.4f}** | {row['Train_Time_Sec']:.3f} | {row['Pred_Time_Sec']:.4f} |\n"

    report_content += f"""
---

## 4. Final Model Selection
- **Selected Model**: `{best_name}`
- **Selection Criteria**: Prioritized **F1-Score** and **Recall** on the validation set (`X_val`) because in intrusion detection, minimizing False Negatives (uncaught attacks) while maintaining high precision is critical.
- **Validation Performance**:
  - Accuracy: `{best_row['Accuracy']:.4f}`
  - Precision: `{best_row['Precision']:.4f}`
  - Recall: `{best_row['Recall']:.4f}`
  - **F1-Score**: `{best_row['F1_Score']:.4f}`

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
"""
    report_file = REPORTS_DIR / "model_training_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report saved to: {report_file}")

def verify_models():
    print("\n--- Running Verification Checks ---")
    
    # 1. Verify files exist
    final_model_path = ARTIFACTS_DIR / "final_model.joblib"
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
    
    assert final_model_path.exists(), "Final model artifact missing!"
    assert preprocessor_path.exists(), "Preprocessor artifact missing!"
    
    # 2. Test loading final model
    loaded_model = joblib.load(final_model_path)
    print(f"[OK] Successfully loaded final model: {type(loaded_model).__name__}")
    
    # 3. Test prediction on sample validation data
    X_val = np.load(ARTIFACTS_DIR / "X_val.npy")[:10]
    preds = loaded_model.predict(X_val)
    print(f"[OK] Sample predictions on 10 validation samples: {preds}")
    
    # 4. Check for NaN/Inf in predictions
    assert not np.isnan(preds).any(), "NaN values found in prediction output!"
    assert not np.isinf(preds).any(), "Inf values found in prediction output!"
    print("[OK] Prediction outputs are clean (no NaN/Inf).")

if __name__ == "__main__":
    train_and_evaluate()
    verify_models()
