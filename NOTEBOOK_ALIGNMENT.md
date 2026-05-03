# Alineación con notebook de clase

Este repositorio no tenía un pipeline previo implementado, por lo que no hubo componentes para marcar como **(ya estaba implementado, coincide con notebook)**.

Se agregó un pipeline completo en `mlp_tuning_pipeline.py` aplicando la referencia de clase y etiquetando conceptualmente cada bloque como **(extraido del notebook de clase)**:

- Imports, estructura general y uso de `Pipeline` con `StandardScaler` + MLP **(extraido del notebook de clase)**.
- Flujo de clasificación con Iris: split, baseline, `cross_val_score`, `ParameterGrid`, `GridSearchCV`, `RandomizedSearchCV` y comparación de métricas **(extraido del notebook de clase)**.
- Flujo de regresión con California Housing: baseline, CV con `r2`, `GridSearchCV`, `RandomizedSearchCV` y comparación final de métricas **(extraido del notebook de clase)**.
- Adaptación al caso presente del repositorio: para regresión se toma `california_housing.csv` local y se detecta `MedHouseVal` como target **(extraido del notebook de clase, adaptado al repo)**.

Criterio aplicado ante conflictos:
- Se priorizó la implementación enviada por clase como referencia principal, manteniendo hiperparámetros, tipos de búsqueda y estructura de evaluación.
