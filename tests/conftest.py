# tests/conftest.py
import os
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def tiny_classification_df(rng):
    """Dataset sintético con la columna objetivo EXACTA que usa tu proyecto: 'shares'."""
    n = 120
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(4, 1.5, n)
    # Regla simple: popular si x1 + x2 > 4.3
    shares = (x1 + x2 > 4.3).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "shares": shares, "url": [f"u{i}" for i in range(n)], "timedelta": rng.integers(1, 10, n)})
    return df

@pytest.fixture
def isolated_mlflow_tmp(tmp_path, monkeypatch):
    # crea directorio temporal
    tracking_dir = tmp_path / "mlruns_isolated"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # convierte a URI de archivo
    uri = tracking_dir.resolve().as_uri() 
    # Aísla tanto tracking como registry al mismo directorio local
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)

    return tracking_dir