from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline


def evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, pd.DataFrame]:
    y_pred = model_pipeline.predict(X_test)
    proba = model_pipeline.predict_proba(X_test)

    labels = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=labels)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, proba)),
        "roc_auc_ovr": float(roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    predictions = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
    })
    for i, label in enumerate(labels):
        predictions[f"proba_{label}"] = proba[:, i]
    return metrics, predictions
