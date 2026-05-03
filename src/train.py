from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


def train_model(model_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict[str, Any]]:
    start = time.perf_counter()
    model_pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    model = model_pipeline.named_steps["model"]
    summary = {
        "train_seconds": round(elapsed, 4),
        "n_iter_": int(getattr(model, "n_iter_", 0)),
        "loss_": float(getattr(model, "loss_", 0.0)),
        "converged": int(getattr(model, "n_iter_", 0)) < int(getattr(model, "max_iter", 1)),
    }
    return model_pipeline, summary
