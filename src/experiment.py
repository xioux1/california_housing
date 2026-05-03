from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import save_config, save_json
from src.data import load_dataset, split_dataset, summarize_dataset
from src.evaluate import evaluate_model
from src.features import build_preprocessing_pipeline
from src.models import build_model, build_training_pipeline
from src.plots import save_plots
from src.reporting import generate_html_report
from src.train import train_model


def run_experiment(config: dict[str, Any]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = Path(config.get("results_dir", "results")) / run_id
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    train_cfg = config["train"]

    X, y = load_dataset(Path(data_cfg["path"]), data_cfg["target_column"])
    dataset_summary = summarize_dataset(X, y)
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, float(data_cfg.get("test_size", 0.2)), int(train_cfg.get("random_state", 42))
    )

    preprocess = build_preprocessing_pipeline()
    model = build_model(train_cfg)
    pipeline = build_training_pipeline(preprocess, model)

    trained_pipeline, training_summary = train_model(pipeline, X_train, y_train)
    metrics, predictions = evaluate_model(trained_pipeline, X_test, y_test)

    predictions_path = run_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics_path = run_dir / "metrics.json"
    save_json(metrics, metrics_path)

    plot_paths = save_plots(trained_pipeline, X_test, y_test, metrics, plots_dir)
    config_path = run_dir / "config.yaml"
    save_config(config, config_path)

    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "training_summary": training_summary,
    }
    artifacts = [config_path, metrics_path, predictions_path, *plot_paths]
    generate_html_report(
        run_dir=run_dir,
        run_metadata=run_metadata,
        dataset_summary=dataset_summary,
        config=config,
        metrics=metrics,
        plot_paths=plot_paths,
        artifact_paths=artifacts,
        warnings=[],
    )
    return run_dir
