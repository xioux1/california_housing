"""Ejercicio MLPRegressor para California Housing.

Objetivos:
- baseline reproducible
- variantes manuales
- comparación con R2/MAE/MSE/RMSE
- GridSearchCV + análisis de cv_results_
"""

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 31415
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)


def metricas_regresion(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }


def construir_modelos():
    """Devuelve las 10 configuraciones pedidas."""
    return [
        {"modelo": "Baseline_STD", "scaler": "standard", "hidden_layers": (64, 32), "activation": "relu", "alpha": 1e-4},
        {"modelo": "Variante_1_STD", "scaler": "standard", "hidden_layers": (32,), "activation": "relu", "alpha": 1e-4},
        {"modelo": "Variante_2_STD", "scaler": "standard", "hidden_layers": (128, 64), "activation": "relu", "alpha": 1e-4},
        {"modelo": "Variante_3_STD", "scaler": "standard", "hidden_layers": (64, 32), "activation": "tanh", "alpha": 1e-4},
        {"modelo": "Variante_4_STD", "scaler": "standard", "hidden_layers": (64, 32), "activation": "relu", "alpha": 1e-2},
        {"modelo": "Baseline_MM", "scaler": "minmax", "hidden_layers": (64, 32), "activation": "relu", "alpha": 1e-4},
        {"modelo": "Variante_1_MM", "scaler": "minmax", "hidden_layers": (32,), "activation": "relu", "alpha": 1e-4},
        {"modelo": "Variante_2_MM", "scaler": "minmax", "hidden_layers": (128, 64), "activation": "relu", "alpha": 1e-4},
        {"modelo": "Variante_3_MM", "scaler": "minmax", "hidden_layers": (64, 32), "activation": "tanh", "alpha": 1e-4},
        {"modelo": "Variante_4_MM", "scaler": "minmax", "hidden_layers": (64, 32), "activation": "relu", "alpha": 1e-2},
    ]


def crear_pipeline(cfg):
    scaler = StandardScaler() if cfg["scaler"] == "standard" else MinMaxScaler()
    mlp = MLPRegressor(
        hidden_layer_sizes=cfg["hidden_layers"],
        activation=cfg["activation"],
        alpha=cfg["alpha"],
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("scaler", scaler),
        ("mlp", mlp),
    ])


def ejecutar_variantes(X_train, X_test, y_train, y_test):
    rows = []
    for cfg in construir_modelos():
        pipe = crear_pipeline(cfg)
        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        m_train = metricas_regresion(y_train, pred_train)
        m_test = metricas_regresion(y_test, pred_test)

        rows.append({
            "Modelo": cfg["modelo"],
            "Scaler": cfg["scaler"],
            "Hidden Layers": cfg["hidden_layers"],
            "Activacion": cfg["activation"],
            "Alpha": cfg["alpha"],
            "R2_train": m_train["R2"],
            "MAE_train": m_train["MAE"],
            "MSE_train": m_train["MSE"],
            "RMSE_train": m_train["RMSE"],
            "R2_test": m_test["R2"],
            "MAE_test": m_test["MAE"],
            "MSE_test": m_test["MSE"],
            "RMSE_test": m_test["RMSE"],
        })

    return pd.DataFrame(rows).sort_values("R2_test", ascending=False).reset_index(drop=True)


def ejecutar_grid_search(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(max_iter=300, random_state=RANDOM_STATE)),
    ])

    param_grid = [
        {
            "scaler": [StandardScaler()],
            "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-2],
        },
        {
            "scaler": [MinMaxScaler()],
            "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-2],
        },
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    cv_results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    columnas = [
        "rank_test_score", "mean_test_score", "std_test_score", "mean_train_score",
        "param_scaler", "param_mlp__hidden_layer_sizes", "param_mlp__activation", "param_mlp__alpha",
    ]
    return gs, cv_results[columnas]


def main():
    print(f"RANDOM_STATE = {RANDOM_STATE}")

    df = pd.read_csv("california_housing.csv")
    target_col = "MedHouseVal"

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    tabla_modelos = ejecutar_variantes(X_train, X_test, y_train, y_test)
    print("\n=== Resultados de las 10 redes (ordenadas por R2_test) ===")
    print(tabla_modelos.to_string(index=False))

    gs, tabla_cv = ejecutar_grid_search(X_train, y_train)
    best_model = gs.best_estimator_
    pred_test_best = best_model.predict(X_test)
    metricas_best = metricas_regresion(y_test, pred_test_best)

    print("\n=== GridSearchCV ===")
    print("Mejores hiperparámetros:", gs.best_params_)
    print("Mejor score CV (R2):", gs.best_score_)
    print("Métricas en test del mejor modelo:", metricas_best)

    print("\n=== Top 10 de cv_results_ (por rank_test_score) ===")
    print(tabla_cv.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
