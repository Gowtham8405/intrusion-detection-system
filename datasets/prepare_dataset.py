"""
Dataset Preparation Script for the AI-Based Intrusion Detection System.
========================================================================

Dataset: UNSW-NB15
Source:   University of New South Wales (UNSW) Canberra
URL:      https://research.unsw.edu.au/projects/unsw-nb15-dataset

This script:
    1. Attempts to download the UNSW-NB15 training & testing CSV files.
       If automatic download fails, provides manual placement instructions.
    2. Loads and inspects the raw CSV files.
    3. Reports column names, data types, missing values, duplicates, class distribution.
    4. Cleans the data (handle missing values, strip whitespace, fix types).
    5. Saves cleaned data to datasets/processed/.
    6. Creates train/validation/test splits and saves them.

Usage:
    python datasets/prepare_dataset.py

IMPORTANT: This script does NOT train any ML model.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Resolve project root (parent of datasets/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'datasets' else SCRIPT_DIR

DATASETS_DIR = PROJECT_ROOT / 'datasets'
RAW_DIR = DATASETS_DIR / 'raw'
PROCESSED_DIR = DATASETS_DIR / 'processed'
TRAIN_DIR = DATASETS_DIR / 'train'
VALIDATION_DIR = DATASETS_DIR / 'validation'
TEST_DIR = DATASETS_DIR / 'test'

# Expected filenames
TRAIN_FILE = 'UNSW_NB15_training-set.csv'
TEST_FILE = 'UNSW_NB15_testing-set.csv'

# Validation split ratio (from training set)
VALIDATION_RATIO = 0.15
RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def create_directories():
    """Create the required directory structure."""
    for d in [RAW_DIR, PROCESSED_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR] {d.relative_to(PROJECT_ROOT)}")


def download_dataset():
    """
    Attempt to download the UNSW-NB15 training and testing CSV files.
    
    Strategy:
      1. Check if files already exist in datasets/raw/
      2. Try kagglehub (no auth needed for public datasets)
      3. Fall back to manual download instructions
    
    Returns True if files are available (downloaded or already exist).
    """
    train_path = RAW_DIR / TRAIN_FILE
    test_path = RAW_DIR / TEST_FILE

    # Check if files already exist
    if train_path.exists() and test_path.exists():
        if train_path.stat().st_size > 1_000_000 and test_path.stat().st_size > 500_000:
            print(f"\n  >> Dataset files already exist in {RAW_DIR.relative_to(PROJECT_ROOT)}/")
            print(f"     - {TRAIN_FILE} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"     - {TEST_FILE} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return True

    # ── Method 1: kagglehub (no authentication required for public datasets) ──
    print("\n  Attempting download via kagglehub (Kaggle public dataset)...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download('mrwellsdavid/unsw-nb15')
        download_path = Path(download_path)
        print(f"    Dataset cached at: {download_path}")

        # Find and copy the training & testing CSV files
        found_train = False
        found_test = False

        for f in download_path.rglob('*'):
            if f.is_file():
                fname = f.name
                if fname == TRAIN_FILE:
                    shutil.copy2(str(f), str(train_path))
                    print(f"    >> Copied {TRAIN_FILE} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
                    found_train = True
                elif fname == TEST_FILE:
                    shutil.copy2(str(f), str(test_path))
                    print(f"    >> Copied {TEST_FILE} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")
                    found_test = True

        if found_train and found_test:
            print("    >> Dataset downloaded and copied successfully via kagglehub!")
            return True
        else:
            # List what we did find
            print("    >> Expected CSV files not found in download. Files found:")
            for f in download_path.rglob('*'):
                if f.is_file():
                    print(f"       - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")

    except ImportError:
        print("    >> kagglehub not installed. Install with: pip install kagglehub")
    except Exception as e:
        print(f"    >> kagglehub download failed: {e}")

    # ── Method 2: Manual instructions ──
    print("\n" + "=" * 70)
    print("  AUTOMATIC DOWNLOAD FAILED -- Manual Setup Required")
    print("=" * 70)
    print("""
  The UNSW-NB15 dataset requires manual download from one of these sources:

  OPTION 1 -- Official UNSW SharePoint (Recommended):
    1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    2. Click the "HERE" link to access the SharePoint folder
    3. Download: UNSW_NB15_training-set.csv
    4. Download: UNSW_NB15_testing-set.csv

  OPTION 2 -- Kaggle:
    1. Visit: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
    2. Download the dataset (requires free Kaggle account)
    3. Extract UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv

  After downloading, place the files in:
    {raw_dir}

  Then re-run this script:
    python datasets/prepare_dataset.py
""".format(raw_dir=RAW_DIR))
    return False


def load_and_inspect(filepath, name):
    """Load a CSV file and print inspection details."""
    print(f"\n{'-' * 60}")
    print(f"  Loading: {name}")
    print(f"  File:    {filepath.name}")
    print(f"{'-' * 60}")

    df = pd.read_csv(filepath)

    print(f"\n  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\n  Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        null_str = f" [{nulls} missing]" if nulls > 0 else ""
        print(f"    {i:3d}. {col:<25s} {str(dtype):<12s}{null_str}")

    return df


def report_missing_values(df, name):
    """Report missing values in detail."""
    print(f"\n  Missing Values -- {name}:")
    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("    [OK] No missing values found.")
    else:
        print(f"    Total missing values: {total_missing:,}")
        cols_with_missing = missing[missing > 0]
        for col, count in cols_with_missing.items():
            pct = count / len(df) * 100
            print(f"    - {col}: {count:,} ({pct:.2f}%)")

    return total_missing


def report_duplicates(df, name):
    """Report duplicate records."""
    # Exclude 'id' column if present when checking duplicates
    check_cols = [c for c in df.columns if c.lower() != 'id']
    n_dupes = df.duplicated(subset=check_cols, keep='first').sum()

    print(f"\n  Duplicate Records -- {name}:")
    if n_dupes == 0:
        print("    [OK] No duplicate records found.")
    else:
        print(f"    Found {n_dupes:,} duplicate records ({n_dupes / len(df) * 100:.2f}%)")

    return n_dupes


def report_class_distribution(df, label_col, attack_col, name):
    """Report target class distribution."""
    print(f"\n  Class Distribution ('{label_col}') -- {name}:")
    dist = df[label_col].value_counts().sort_index()
    for cls, count in dist.items():
        pct = count / len(df) * 100
        label_name = "Normal" if cls == 0 else "Attack"
        print(f"    {cls} ({label_name}): {count:>8,}  ({pct:5.2f}%)")

    if attack_col in df.columns:
        print(f"\n  Attack Category Distribution ('{attack_col}') -- {name}:")
        attack_dist = df[attack_col].value_counts()
        for cat, count in attack_dist.items():
            pct = count / len(df) * 100
            cat_display = cat.strip() if isinstance(cat, str) else str(cat)
            print(f"    {cat_display:<20s}: {count:>8,}  ({pct:5.2f}%)")


def clean_dataset(df, name):
    """
    Clean the dataset:
    - Strip whitespace from string columns
    - Clean attack_cat column
    - Handle missing values
    - Fix data types
    """
    print(f"\n  Cleaning: {name}...")
    original_shape = df.shape

    # Strip whitespace from object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Clean attack_cat: normalize values
    if 'attack_cat' in df.columns:
        # Replace empty strings and whitespace-only values with 'Normal'
        df['attack_cat'] = df['attack_cat'].replace({'': 'Normal', ' ': 'Normal', 'nan': 'Normal'})
        # Standardize 'Backdoor' vs 'Backdoors'
        df['attack_cat'] = df['attack_cat'].replace({'Backdoors': 'Backdoor'})

    # Handle missing values
    # For numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"    Filled {col} missing values with median ({median_val:.4f})")

    # For categorical columns: fill with mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"    Filled {col} missing values with mode ('{mode_val}')")

    # Remove duplicate records (keeping first occurrence)
    check_cols = [c for c in df.columns if c.lower() != 'id']
    before = len(df)
    df = df.drop_duplicates(subset=check_cols, keep='first')
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed:,} duplicate records")

    print(f"    Shape: {original_shape} -> {df.shape}")
    print(f"    [OK] Cleaning complete.")

    return df


def create_splits(train_df, test_df):
    """
    Create train/validation/test splits.
    - Test set: use the official UNSW-NB15 testing set as-is.
    - Validation set: split from the training set.
    - Train set: remaining training data.
    """
    print(f"\n{'-' * 60}")
    print(f"  Creating Train / Validation / Test Splits")
    print(f"{'-' * 60}")

    # Shuffle the training data
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split training -> train + validation
    val_size = int(len(train_df) * VALIDATION_RATIO)
    val_df = train_df.iloc[:val_size].reset_index(drop=True)
    final_train_df = train_df.iloc[val_size:].reset_index(drop=True)

    print(f"\n  Split Results:")
    print(f"    Training set:   {len(final_train_df):>8,} records ({len(final_train_df) / (len(final_train_df) + len(val_df) + len(test_df)) * 100:.1f}%)")
    print(f"    Validation set: {len(val_df):>8,} records ({len(val_df) / (len(final_train_df) + len(val_df) + len(test_df)) * 100:.1f}%)")
    print(f"    Test set:       {len(test_df):>8,} records ({len(test_df) / (len(final_train_df) + len(val_df) + len(test_df)) * 100:.1f}%)")

    return final_train_df, val_df, test_df


def save_data(df, directory, filename):
    """Save a DataFrame to CSV."""
    filepath = directory / filename
    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"    Saved: {filepath.relative_to(PROJECT_ROOT)} ({len(df):,} rows, {size_mb:.2f} MB)")


# ──────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  UNSW-NB15 Dataset Preparation")
    print("  AI-Based Intrusion Detection System")
    print("=" * 70)

    # Step 1: Create directories
    print("\n>> Step 1: Creating directory structure...")
    create_directories()

    # Step 2: Download / verify dataset
    print("\n>> Step 2: Checking dataset availability...")
    if not download_dataset():
        print("\n  [FAIL] Dataset files not available. Please download manually and re-run.")
        sys.exit(1)

    # Step 3: Load and inspect raw data
    print("\n>> Step 3: Loading and inspecting raw data...")
    train_raw = load_and_inspect(RAW_DIR / TRAIN_FILE, "Training Set")
    test_raw = load_and_inspect(RAW_DIR / TEST_FILE, "Testing Set")

    # Step 4: Report missing values
    print("\n>> Step 4: Checking for missing values...")
    train_missing = report_missing_values(train_raw, "Training Set")
    test_missing = report_missing_values(test_raw, "Testing Set")

    # Step 5: Report duplicates
    print("\n>> Step 5: Checking for duplicate records...")
    train_dupes = report_duplicates(train_raw, "Training Set")
    test_dupes = report_duplicates(test_raw, "Testing Set")

    # Step 6: Report class distribution
    print("\n>> Step 6: Analyzing class distribution...")
    label_col = 'label'
    attack_col = 'attack_cat'

    # Verify the target column exists
    if label_col not in train_raw.columns:
        print(f"\n  [FAIL] Target column '{label_col}' not found!")
        print(f"  Available columns: {list(train_raw.columns)}")
        sys.exit(1)

    report_class_distribution(train_raw, label_col, attack_col, "Training Set")
    report_class_distribution(test_raw, label_col, attack_col, "Testing Set")

    # Step 7: Clean data
    print("\n>> Step 7: Cleaning datasets...")
    train_clean = clean_dataset(train_raw.copy(), "Training Set")
    test_clean = clean_dataset(test_raw.copy(), "Testing Set")

    # Step 8: Save processed data
    print("\n>> Step 8: Saving processed data...")
    save_data(train_clean, PROCESSED_DIR, 'train_processed.csv')
    save_data(test_clean, PROCESSED_DIR, 'test_processed.csv')

    # Step 9: Create splits
    final_train, val, test = create_splits(train_clean, test_clean)

    print("\n>> Step 9: Saving split data...")
    save_data(final_train, TRAIN_DIR, 'train.csv')
    save_data(val, VALIDATION_DIR, 'validation.csv')
    save_data(test, TEST_DIR, 'test.csv')

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  DATASET PREPARATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"""
  Dataset:           UNSW-NB15
  Source:             UNSW Canberra (Australian Centre for Cyber Security)
  URL:               https://research.unsw.edu.au/projects/unsw-nb15-dataset

  Total Records:     {len(train_clean) + len(test_clean):,}
    Training (raw):  {len(train_clean):,}
    Testing (raw):   {len(test_clean):,}

  Features:          {len(train_clean.columns)} columns
  Target Column:     '{label_col}' (0=Normal, 1=Attack)
  Attack Categories: {attack_col}

  Missing Values:    Training={train_missing:,}, Testing={test_missing:,}
  Duplicates:        Training={train_dupes:,}, Testing={test_dupes:,}

  Split Summary:
    Train:           {len(final_train):,} records
    Validation:      {len(val):,} records
    Test:            {len(test):,} records

  Output Files:
    datasets/processed/train_processed.csv
    datasets/processed/test_processed.csv
    datasets/train/train.csv
    datasets/validation/validation.csv
    datasets/test/test.csv
""")
    print("  [OK] Ready for Step 3 (Model Training).")
    print("=" * 70)


if __name__ == '__main__':
    main()
