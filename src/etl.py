"""High-level ETL orchestration utilities.

This module composes the lower-level functions in data_pipeline.py into an
end-to-end run() function suitable for CI/demo and for unit testing.
"""
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from src.data_pipeline import load_dataset, clean_data, build_model


def run(data_path: str | Path, model_out: str | Path) -> Tuple[str, dict]:
    """Run the full ETL -> train -> save pipeline.

    Returns a tuple (model_path, metrics)
    """
    df = load_dataset(data_path)
    cleaned = clean_data(df)
    model, metrics = build_model(cleaned)

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    return str(model_out), metrics


if __name__ == "__main__":
    # simple CLI convenience when invoked directly
    import argparse

    parser = argparse.ArgumentParser(description="Run ETL and train model")
    parser.add_argument("data", help="Path to CSV dataset")
    parser.add_argument("--out", default="artifacts/sales_model.joblib", help="Model output path")
    args = parser.parse_args()

    path, metrics = run(args.data, args.out)
    print(f"Saved model to: {path}")
    print(f"Metrics: {metrics}")
