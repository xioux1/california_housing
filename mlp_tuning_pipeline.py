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
        ("mlp", MLPRegressor(max_iter=100, solver="adam", random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
    }

    scoring = {
        "R2": "r2",
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error",
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit="R2",
        n_jobs=-1,
        return_train_score=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    cv_results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_R2").reset_index(drop=True)
    for c in ["mean_test_MAE", "mean_test_MSE", "mean_test_RMSE"]:
        cv_results[c] = -cv_results[c]

    columnas = [
        "rank_test_R2", "mean_test_R2", "std_test_R2", "mean_train_R2",
        "mean_test_MAE", "mean_test_MSE", "mean_test_RMSE",
        "param_mlp__hidden_layer_sizes", "param_mlp__activation", "param_mlp__alpha",
    ]
    return gs, cv_results[columnas]


def _params_like(df, hidden, activation, alpha):
    mask = (
        (df["param_mlp__hidden_layer_sizes"].apply(tuple) == tuple(hidden)) &
        (df["param_mlp__activation"] == activation) &
        (df["param_mlp__alpha"].astype(float) == float(alpha))
    )
    return df.loc[mask].copy()


def construir_analisis_preguntas(tabla_cv):
    total_entrenamientos = 10 + (3 * 2 * 3 * 3) + 1  # 10 manuales + 54 CV + 1 refit

    opcion_b = _params_like(tabla_cv, (128, 64), "relu", 1e-3)["mean_test_R2"].iloc[0]
    opcion_a = _params_like(tabla_cv, (128, 64), "relu", 1e-2)["mean_test_R2"].iloc[0]
    opcion_c = _params_like(tabla_cv, (64, 32), "tanh", 1e-4)["mean_test_R2"].iloc[0]
    opcion_d = _params_like(tabla_cv, (64, 32), "relu", 1e-2)["mean_test_R2"].iloc[0]

    idx_worst_mse = tabla_cv["mean_test_MSE"].idxmax()
    worst_mse_layers = tuple(tabla_cv.loc[idx_worst_mse, "param_mlp__hidden_layer_sizes"])
    peor_por_mse_tiene_una_capa = (len(worst_mse_layers) == 1)

    idx_best_mae = tabla_cv["mean_test_MAE"].idxmin()
    idx_worst_mae = tabla_cv["mean_test_MAE"].idxmax()
    diff_r2_mae = tabla_cv.loc[idx_best_mae, "mean_test_R2"] - tabla_cv.loc[idx_worst_mae, "mean_test_R2"]

    orden_por_r2 = list(tabla_cv.sort_values("mean_test_R2", ascending=False).index)
    orden_por_mae = list(tabla_cv.sort_values("mean_test_MAE", ascending=True).index)
    orden_por_mse = list(tabla_cv.sort_values("mean_test_MSE", ascending=True).index)
    coincide_orden = (orden_por_r2 == orden_por_mae == orden_por_mse)

    return {
        "total_entrenamientos": total_entrenamientos,
        "mejor_entre_opciones": max(
            [
                ("Opción A", opcion_a),
                ("Opción B", opcion_b),
                ("Opción C", opcion_c),
                ("Opción D", opcion_d),
            ],
            key=lambda x: x[1],
        )[0],
        "peor_mse_una_capa": peor_por_mse_tiene_una_capa,
        "diff_r2_mejor_peor_mae": diff_r2_mae,
        "orden_r2_mae_mse_igual": coincide_orden,
    }


def generar_html_analisis(analisis, output_path="analisis_preguntas.html"):
    mejor_opcion = analisis["mejor_entre_opciones"]
    peor_mse_txt = "Verdadero" if analisis["peor_mse_una_capa"] else "Falso"
    diff_mayor_005 = "Verdadero" if analisis["diff_r2_mejor_peor_mae"] > 0.05 else "Falso"
    orden_igual_txt = "Verdadero" if analisis["orden_r2_mae_mse_igual"] else "Falso"

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Análisis de preguntas - MLP California Housing</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 920px; margin: 2rem auto; line-height: 1.5; }}
    h1, h2 {{ color: #1f3b4d; }}
    .ok {{ color: #0a7f2e; font-weight: bold; }}
    .block {{ background: #f5f8fb; border: 1px solid #d9e3ec; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Análisis de preguntas (basado en nuestro código)</h1>
  <div class="block">
    <h2>1) ¿Cuántos entrenamientos se hacen en total?</h2>
    <p><span class="ok">Respuesta:</span> {analisis["total_entrenamientos"]} entrenamientos (10 manuales + 54 de CV + 1 refit final).</p>
  </div>
  <div class="block">
    <h2>2) Mejor opción por <code>mean_test_R2</code> entre A/B/C/D</h2>
    <p><span class="ok">Respuesta:</span> {mejor_opcion}.</p>
  </div>
  <div class="block">
    <h2>3) “Los peores modelos por <code>mean_test_MSE</code> tenían 1 capa oculta”</h2>
    <p><span class="ok">Respuesta:</span> {peor_mse_txt}.</p>
  </div>
  <div class="block">
    <h2>4) “Entre mejor y peor por <code>mean_test_MAE</code> hay diferencia de <code>mean_test_R2</code> mayor a 0.05”</h2>
    <p><span class="ok">Respuesta:</span> {diff_mayor_005}. Diferencia calculada: {analisis["diff_r2_mejor_peor_mae"]:.6f}.</p>
  </div>
  <div class="block">
    <h2>5) “MAE, MSE y R2 coinciden exactamente en el orden de mejores modelos”</h2>
    <p><span class="ok">Respuesta:</span> {orden_igual_txt}.</p>
  </div>
  <div class="block">
    <h2>6) Flujo metodológico correcto</h2>
    <p><span class="ok">Respuesta:</span> Opción D: ajustar hiperparámetros con CV en train y usar test solo para evaluación final.</p>
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


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

    print("\n=== Top 10 de cv_results_ (por rank_test_R2) ===")
    print(tabla_cv.head(10).to_string(index=False))

    analisis = construir_analisis_preguntas(tabla_cv)
    print("\n=== Respuestas de análisis solicitadas ===")
    print(f"Entrenamientos totales: {analisis['total_entrenamientos']}")
    print(f"Mejor opción por mean_test_R2 (A/B/C/D): {analisis['mejor_entre_opciones']}")
    print(f"Peor MSE con una capa oculta: {analisis['peor_mse_una_capa']}")
    print(f"Diferencia R2 entre mejor/peor por MAE: {analisis['diff_r2_mejor_peor_mae']:.6f}")
    print(f"Orden idéntico entre R2, MAE y MSE: {analisis['orden_r2_mae_mse_igual']}")
    generar_html_analisis(analisis, output_path="analisis_preguntas.html")
    print("HTML de análisis generado: analisis_preguntas.html")


if __name__ == "__main__":
    main()
