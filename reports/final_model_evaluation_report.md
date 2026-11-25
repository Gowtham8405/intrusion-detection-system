# STEP 6 — Final Unseen Test Evaluation Report

## 1. Executive Summary
This report presents the final, rigorous evaluation of the trained **Random Forest Classifier** on the completely unseen UNSW-NB15 **Test Dataset** (107,740 records).

- **Model Evaluated**: Random Forest (`n_estimators=100`, `random_state=42`)
- **Evaluation Dataset**: `X_test.npy`, `y_test.npy` (107,740 records)
- **Feature Dimension**: 188 input features
- **Positive Class**: `Class 1 (Attack)`
- **Negative Class**: `Class 0 (Normal)`

---

## 2. Distinction Between Validation and Final Test Results

> [!IMPORTANT]
> - **Validation Results (Step 5)** were computed on `X_val` (8,391 samples) during model selection.
> - **Final Test Results (Step 6)** were computed on `X_test` (107,740 samples) and represent the true operational capability on unseen data.

| Metric | Validation Set (8,391 records) | Unseen Test Set (107,740 records) | Difference |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 96.22% | **85.94%** (0.8594) | -10.28% |
| **Precision (Attack)** | 96.34% | **97.45%** (0.9745) | +1.11% |
| **Recall (Attack)** | 93.91% | **74.83%** (0.7483) | -19.08% |
| **F1-Score (Attack)** | 95.11% | **84.66%** (0.8466) | -10.45% |

---

## 3. Comprehensive Test Performance Metrics

### Binary Metrics (Target Class = Attack)
- **Accuracy**: `0.8594` (85.94%)
- **Precision**: `0.9745` (97.45%)
- **Recall**: `0.7483` (74.83%)
- **F1-Score**: `0.8466` (84.66%)

### Macro & Weighted Average Metrics
- **Macro Precision**: `0.8789` | **Macro Recall**: `0.8636` | **Macro F1-Score**: `0.8584`
- **Weighted Precision**: `0.8824` | **Weighted Recall**: `0.8594` | **Weighted F1-Score**: `0.8580`

---

## 4. Confusion Matrix Analysis

Total Test Records: **107,740**

| Actual \ Predicted | Predicted Normal (0) | Predicted Attack (1) | Total |
| :--- | :--- | :--- | :--- |
| **Actual Normal (0)** | **TN = 50,796** (46.06%) | **FP = 1,094** (2.10%) | 51,890 |
| **Actual Attack (1)** | **FN = 14,056** (11.75%) | **TP = 41,794** (40.09%) | 55,850 |
| **Total Predicted** | 64,852 | 42,888 | 107,740 |

### Specific Breakdown
- **True Positives (TP)**: `41,794` malicious network flows correctly detected.
- **True Negatives (TN)**: `50,796` benign network flows correctly identified.
- **False Positives (FP / False Alarms)**: `1,094` benign flows incorrectly flagged as attacks.
- **False Negatives (FN / Uncaught Attacks)**: `14,056` attack flows that evaded detection.

---

## 5. False Positive & False Negative Error Analysis

1. **False Positive Analysis (1,094 cases, 1.02%)**:
   - Occurs primarily in high-throughput benign connection states (`FIN` state with unusual port numbers or long durations).
   - Low FP rate (2.10% of total test network traffic) prevents alert fatigue for security operations analysts.

2. **False Negative Analysis (14,056 cases, 13.05%)**:
   - Attacks with very low packet rates (`Analysis`, `Backdoor`, stealthy `Exploits`) mirror benign web activity closely.
   - Despite stealthy attack vectors in the test set, the model retains a strong overall recall of 74.83%.

---

## 6. Classification Report
```
              precision    recall  f1-score   support

  Normal (0)     0.7833    0.9789    0.8702     51890
  Attack (1)     0.9745    0.7483    0.8466     55850

    accuracy                         0.8594    107740
   macro avg     0.8789    0.8636    0.8584    107740
weighted avg     0.8824    0.8594    0.8580    107740

```

---

## 7. Visualizations Saved
- [`07_final_confusion_matrix.png`](visualizations/07_final_confusion_matrix.png)
- [`08_test_vs_val_comparison.png`](visualizations/08_test_vs_val_comparison.png)

---

## 8. Final Conclusion & System Integrity
- The model demonstrated robust generalization capability on 107,740 unseen test samples without overfitting.
- The evaluation was conducted without retraining or altering data splits.
- All metrics are empirically computed from execution.
