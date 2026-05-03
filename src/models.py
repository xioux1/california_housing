from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def build_model(config: dict) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=tuple(config.get("hidden_layer_sizes", [64, 32])),
        activation=config.get("activation", "relu"),
        alpha=float(config.get("alpha", 0.0001)),
        learning_rate_init=float(config.get("learning_rate_init", 0.001)),
        max_iter=int(config.get("max_iter", 300)),
        random_state=int(config.get("random_state", 42)),
    )


def build_training_pipeline(preprocess: Pipeline, model: MLPClassifier) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])
