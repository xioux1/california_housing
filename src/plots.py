from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


def save_plots(model_pipeline: Pipeline, X_test, y_test, metrics: dict[str, Any], plot_dir: Path) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []

    model = model_pipeline.named_steps["model"]

    loss_path = plot_dir / "loss_curve.png"
    if hasattr(model, "loss_curve_") and model.loss_curve_:
        plt.figure(figsize=(7, 4))
        plt.plot(model.loss_curve_)
        plt.title("MLP Training Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=120)
        plt.close()
        output.append(loss_path)

    cm_path = plot_dir / "confusion_matrix.png"
    y_pred = model_pipeline.predict(X_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(metrics["confusion_matrix"]))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close()
    output.append(cm_path)

    roc_path = plot_dir / "roc_curve.png"
    proba = model_pipeline.predict_proba(X_test)
    labels = np.unique(y_test)
    y_bin = label_binarize(y_test, classes=labels)
    plt.figure(figsize=(7, 4))
    for idx, label in enumerate(labels):
        RocCurveDisplay.from_predictions(y_bin[:, idx], proba[:, idx], name=f"Class {label}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC Curves (One-vs-Rest)")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=120)
    plt.close()
    output.append(roc_path)

    return output
