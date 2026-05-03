"""Pipeline de referencia para tuning con MLP (clasificación y regresión).

Este script aplica la estructura del notebook de clase sobre el caso del repositorio.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_classification_pipeline():
    """Clasificación con Iris para replicar la referencia de clase."""
    iris = load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X = X[["sepal length (cm)", "petal length (cm)"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline_base_clf = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(50,),
                activation="relu",
                alpha=0.0001,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    pipeline_base_clf.fit(X_train, y_train)
    y_pred_base = pipeline_base_clf.predict(X_test)

    scores_cv_base = cross_val_score(
        pipeline_base_clf,
        X,
        y,
        cv=10,
        scoring="accuracy",
    )

    param_grid_clf = {
        "mlp__hidden_layer_sizes": [(20,), (50,), (100,), (50, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
    }

    combinaciones = list(ParameterGrid(param_grid_clf))

    grid_search_clf = GridSearchCV(
        estimator=pipeline_base_clf,
        param_grid=param_grid_clf,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    grid_search_clf.fit(X, y)

    best_model_grid_clf = grid_search_clf.best_estimator_
    y_pred_grid = best_model_grid_clf.predict(X_test)

    param_dist_clf = {
        "mlp__hidden_layer_sizes": [(20,), (50,), (100,), (150,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": np.logspace(-5, -1, 20),
        "mlp__learning_rate_init": np.logspace(-4, -1, 20),
    }

    random_search_clf = RandomizedSearchCV(
        estimator=pipeline_base_clf,
        param_distributions=param_dist_clf,
        n_iter=15,
        scoring="accuracy",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    random_search_clf.fit(X_train, y_train)

    best_model_random_clf = random_search_clf.best_estimator_
    y_pred_random = best_model_random_clf.predict(X_test)

    return {
        "base_test_accuracy": accuracy_score(y_test, y_pred_base),
        "base_cv_mean_accuracy": float(scores_cv_base.mean()),
        "base_cv_std_accuracy": float(scores_cv_base.std()),
        "grid_best_params": grid_search_clf.best_params_,
        "grid_best_cv_accuracy": float(grid_search_clf.best_score_),
        "grid_test_accuracy": accuracy_score(y_test, y_pred_grid),
        "grid_confusion_matrix": confusion_matrix(y_test, y_pred_grid).tolist(),
        "grid_classification_report": classification_report(
            y_test, y_pred_grid, target_names=iris.target_names
        ),
        "random_best_params": random_search_clf.best_params_,
        "random_best_cv_accuracy": float(random_search_clf.best_score_),
        "random_test_accuracy": accuracy_score(y_test, y_pred_random),
        "total_grid_combinations": len(combinaciones),
        "total_grid_models_cv5": len(combinaciones) * 5,
    }


def run_regression_pipeline(data_path: str = "california_housing.csv"):
    """Regresión con California Housing usando el CSV del repositorio."""
    df = pd.read_csv(data_path)

    target_col = "MedHouseVal" if "MedHouseVal" in df.columns else df.columns[-1]
    X_reg = df.drop(columns=[target_col])
    y_reg = df[target_col]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    pipeline_base_reg = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=(50,),
                activation="relu",
                alpha=0.0001,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    pipeline_base_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg_base = pipeline_base_reg.predict(X_test_reg)

    scores_cv_reg = cross_val_score(
        pipeline_base_reg,
        X_reg,
        y_reg,
        cv=5,
        scoring="r2",
    )

    param_grid_reg = {
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001],
    }

    grid_search_reg = GridSearchCV(
        estimator=pipeline_base_reg,
        param_grid=param_grid_reg,
        scoring="r2",
        cv=3,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    grid_search_reg.fit(X_reg, y_reg)

    best_model_grid_reg = grid_search_reg.best_estimator_
    y_pred_grid_reg = best_model_grid_reg.predict(X_test_reg)

    param_dist_reg = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (150,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": np.logspace(-5, -1, 20),
        "mlp__learning_rate_init": np.logspace(-4, -2, 20),
    }

    random_search_reg = RandomizedSearchCV(
        estimator=pipeline_base_reg,
        param_distributions=param_dist_reg,
        n_iter=8,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    random_search_reg.fit(X_train_reg, y_train_reg)

    best_model_random_reg = random_search_reg.best_estimator_
    y_pred_random_reg = best_model_random_reg.predict(X_test_reg)

    return {
        "base_mae": mean_absolute_error(y_test_reg, y_pred_reg_base),
        "base_rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_base))),
        "base_r2": r2_score(y_test_reg, y_pred_reg_base),
        "base_cv_r2_mean": float(scores_cv_reg.mean()),
        "base_cv_r2_std": float(scores_cv_reg.std()),
        "grid_best_params": grid_search_reg.best_params_,
        "grid_best_cv_r2": float(grid_search_reg.best_score_),
        "grid_mae": mean_absolute_error(y_test_reg, y_pred_grid_reg),
        "grid_rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred_grid_reg))),
        "grid_r2": r2_score(y_test_reg, y_pred_grid_reg),
        "random_best_params": random_search_reg.best_params_,
        "random_best_cv_r2": float(random_search_reg.best_score_),
        "random_mae": mean_absolute_error(y_test_reg, y_pred_random_reg),
        "random_rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred_random_reg))),
        "random_r2": r2_score(y_test_reg, y_pred_random_reg),
    }


def main():
    clf_results = run_classification_pipeline()
    reg_results = run_regression_pipeline()

    print("=== Clasificación (Iris) ===")
    for k, v in clf_results.items():
        print(f"{k}: {v}")

    print("\n=== Regresión (California Housing local) ===")
    for k, v in reg_results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
