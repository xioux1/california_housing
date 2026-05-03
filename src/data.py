from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataset")
    return df.drop(columns=[target_column]), df[target_column]


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def summarize_dataset(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_name": y.name,
        "target_distribution": y.value_counts().to_dict(),
        "missing_values": int(X.isna().sum().sum() + y.isna().sum()),
    }
