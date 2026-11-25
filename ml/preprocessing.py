"""
ml/preprocessing.py

Reusable preprocessing pipeline for the UNSW-NB15 dataset.

- Loads train/validation/test CSV files from `datasets/`.
- Identifies numeric and categorical features automatically.
- Drops identifier columns (`id`) and leakage‑related columns (`label`, `attack_cat`).
- Handles missing values (median for numeric, most_frequent for categorical).
- Scales numeric features with StandardScaler.
- Encodes categorical features with OneHotEncoder (ignore unknown categories).
- Fits all transformers **only on the training split** to avoid data leakage.
- Saves the fitted ColumnTransformer as `ml/artifacts/preprocessor.joblib`.
- Saves transformed feature matrices and target vectors as NumPy ``.npy`` files.

The script can be imported as a module (e.g., by the Django prediction view) or executed directly.
"""

import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # Intrusion_Detection root
DATASET_DIR = PROJECT_ROOT / "datasets"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

TRAIN_PATH = DATASET_DIR / "train" / "train.csv"
VALID_PATH = DATASET_DIR / "validation" / "validation.csv"
TEST_PATH = DATASET_DIR / "test" / "test.csv"

# Ensure artifact directory exists
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_split(path: pathlib.Path) -> pd.DataFrame:
    """Load a CSV split and return a DataFrame."""
    return pd.read_csv(path)

def identify_columns(df: pd.DataFrame):
    """Identify feature categories.

    Returns a tuple of (numeric_features, categorical_features, target, identifier, leakage).
    """
    target_col = "label"
    identifier_cols = ["id"]
    # `attack_cat` is derived from the label and therefore considered leakage for a generic model.
    leakage_cols = ["attack_cat"]

    # Feature columns are everything except target, identifier, and leakage.
    feature_cols = [c for c in df.columns if c not in (identifier_cols + [target_col] + leakage_cols)]

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if df[c].dtype == "object"]

    return numeric_features, categorical_features, target_col, identifier_cols, leakage_cols

def build_preprocessor(numeric_features, categorical_features):
    """Create a ColumnTransformer that preprocesses numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,  # force dense output for easy saving
    )
    return preprocessor

def process_and_save():
    """Run the full preprocessing pipeline and persist artifacts.

    The function performs the following steps:
    1. Load train/validation/test CSV files.
    2. Identify column groups.
    3. Build and fit the preprocessing pipeline on the training data **only**.
    4. Transform validation and test data using the fitted pipeline.
    5. Save the fitted preprocessor (joblib) and the transformed NumPy arrays.
    """
    # Load data
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VALID_PATH)
    test_df = load_split(TEST_PATH)

    # Identify columns based on the training split (consistent across splits)
    num_feats, cat_feats, target_col, id_cols, leak_cols = identify_columns(train_df)

    # Separate target and features
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    X_train = train_df.drop(columns=id_cols + [target_col] + leak_cols)
    X_val = val_df.drop(columns=id_cols + [target_col] + leak_cols)
    X_test = test_df.drop(columns=id_cols + [target_col] + leak_cols)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(num_feats, cat_feats)

    # Fit on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Persist artifacts
    preprocessor_path = ARTIFACT_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    np.save(ARTIFACT_DIR / "X_train.npy", X_train_processed)
    np.save(ARTIFACT_DIR / "y_train.npy", y_train)
    np.save(ARTIFACT_DIR / "X_val.npy", X_val_processed)
    np.save(ARTIFACT_DIR / "y_val.npy", y_val)
    np.save(ARTIFACT_DIR / "X_test.npy", X_test_processed)
    np.save(ARTIFACT_DIR / "y_test.npy", y_test)

    # Report summary for user verification
    print("=== Preprocessing Summary ===")
    print(f"Numeric features ({len(num_feats)}): {num_feats}")
    print(f"Categorical features ({len(cat_feats)}): {cat_feats}")
    print(f"Identifier columns dropped: {id_cols}")
    print(f"Leakage columns dropped: {leak_cols}")
    print(f"Training shape after preprocessing: {X_train_processed.shape}")
    print(f"Validation shape after preprocessing: {X_val_processed.shape}")
    print(f"Test shape after preprocessing: {X_test_processed.shape}")
    print(f"Preprocessor saved to: {preprocessor_path}")

if __name__ == "__main__":
    process_and_save()
