import pathlib
import pandas as pd
import numpy as np

# Resolve project root (Intrusion_Detection) from this script location
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / 'datasets'

def load_csv(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)

def report_split(df: pd.DataFrame, name: str):
    total = len(df)
    duplicate_rows = df.duplicated().sum()
    missing_values = df.isnull().sum().sum()
    infinite_vals = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    target_vals = set(df['label'].unique())
    unexpected_targets = target_vals - {0, 1}
    class_counts = df['label'].value_counts().to_dict()
    print(f"--- {name} Split ---")
    print(f"Total records: {total}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Missing values (any column): {missing_values}")
    print(f"Infinite numeric values: {infinite_vals}")
    print(f"Unexpected target values: {unexpected_targets if unexpected_targets else 'None'}")
    print(f"Class distribution: {class_counts}\n")

def main():
    train = load_csv(DATASET_DIR / 'train' / 'train.csv')
    val = load_csv(DATASET_DIR / 'validation' / 'validation.csv')
    test = load_csv(DATASET_DIR / 'test' / 'test.csv')

    for df, name in [(train, 'Train'), (val, 'Validation'), (test, 'Test')]:
        report_split(df, name)

    # Feature consistency across splits (excluding target and identifier columns)
    exclude = {'label', 'id', 'attack_cat'}
    train_features = set(train.columns) - exclude
    val_features = set(val.columns) - exclude
    test_features = set(test.columns) - exclude
    common_features = train_features & val_features & test_features
    print('Common feature columns across all splits (excluding target/id):')
    print(sorted(common_features))

    # Overlap checks by record id to verify no leakage across splits
    train_ids = set(train['id'])
    val_ids = set(val['id'])
    test_ids = set(test['id'])
    print('\nRecord ID overlap checks:')
    print(f"Train and Validation overlap: {len(train_ids & val_ids)}")
    print(f"Train and Test overlap: {len(train_ids & test_ids)}")
    print(f"Validation and Test overlap: {len(val_ids & test_ids)}")

if __name__ == '__main__':
    main()
