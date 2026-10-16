from pathlib import Path

from src.data_pipeline import build_model, clean_data, load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "example_dataset" / "sales_data.csv"


def test_end_to_end_runs():
    df = load_dataset(DATA_PATH)
    cleaned = clean_data(df)
    model, metrics = build_model(cleaned)

    assert isinstance(metrics, dict)
    assert "mse" in metrics


def test_model_saved_to_artifacts(tmp_path):
    df = load_dataset(DATA_PATH)
    cleaned = clean_data(df)
    model, metrics = build_model(cleaned)

    out = tmp_path / "model.joblib"
    # write model artifact using joblib from the pipeline directly
    import joblib

    joblib.dump(model, out)

    assert out.exists()
    assert out.stat().st_size > 0
