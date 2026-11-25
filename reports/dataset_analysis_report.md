# UNSW-NB15 Dataset Analysis & Visualization Report

## 1. Dataset Overview
- **Total Dataset Records**: 163,685
- **Total Columns**: 45 (including target `label` and `attack_cat`)
- **Train Split**: 47,554 records (29.05%)
- **Validation Split**: 8,391 records (5.13%)
- **Test Split**: 107,740 records (65.82%)

## 2. Target Class Distribution
- **Normal Traffic (0)**: 86,096 records (52.60%)
- **Attack Traffic (1)**: 77,589 records (47.40%)
- **Class Imbalance Result**: Slight imbalance (52.60% Normal vs 47.40% Attack, Difference: 5.20%). The dataset is well-balanced and does not require synthetic oversampling (SMOTE).

## 3. Attack Category Counts (Total Dataset)
| Category | Record Count | Percentage |
| :--- | :--- | :--- |
| Normal | 86,096 | 52.60% |
| Exploits | 27,453 | 16.77% |
| Fuzzers | 20,988 | 12.82% |
| Reconnaissance | 10,225 | 6.25% |
| Generic | 7,838 | 4.79% |
| DoS | 5,524 | 3.37% |
| Analysis | 2,040 | 1.25% |
| Backdoor | 1,881 | 1.15% |
| Shellcode | 1,469 | 0.90% |
| Worms | 171 | 0.10% |

## 4. Train / Validation / Test Comparison
- **Train Split**: Normal = 29,099 (61.19%), Attack = 18,455 (38.81%)
- **Validation Split**: Normal = 5,107 (60.86%), Attack = 3,284 (39.14%)
- **Test Split**: Normal = 51,890 (48.16%), Attack = 55,850 (51.84%)

## 5. Key Statistical Observations
1. **TTL Features (`sttl`, `dttl`)**: `sttl` (source time-to-live) exhibits strong correlation with attack traffic (correlation ~0.62 with `label`). Certain attack types utilize distinct initial TTL values.
2. **Rate & Packet Load**: Attack traffic shows high packet rates (`rate`) and high payload volumes (`sbytes`, `sload`) during Denial-of-Service and Generic flood attempts.
3. **Categorical Protocols & States**: `INT` and `FIN` connection states are heavily present in attack traffic (`Generic` category primarily uses `INT` state and `udp` protocol).
4. **Data Leakage Field**: `attack_cat` directly reveals attack sub-categories and must be excluded from binary classification model features to prevent target leakage.

## 6. Visualizations Saved
- `reports/visualizations/01_target_distribution.png`
- `reports/visualizations/02_attack_categories.png`
- `reports/visualizations/03_split_comparison.png`
- `reports/visualizations/04_numerical_distributions.png`
- `reports/visualizations/05_categorical_features.png`
- `reports/visualizations/06_correlation_matrix.png`
