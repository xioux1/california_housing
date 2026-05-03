from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full ML experiment and generate an HTML report.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    run_dir = run_experiment(config)
    print(f"Experiment completed. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
