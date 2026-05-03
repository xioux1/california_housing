from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
    ])
