import re
import pytest
from pathlib import Path
from mlops_equipo_63.mlflow_utils import setup_mlflow_experiment

@pytest.mark.unit
def test_setup_mlflow_experiment_returns_uri_and_callback(tmp_path, monkeypatch):
    # usar dir temporal para evitar tocar "mlruns" real
    tracking_dir = tmp_path / "mlruns_tmp"
    uri, cb = setup_mlflow_experiment(
        experiment_name="Test_Exp",
        tracking_dir=str(tracking_dir),
        metric_name="auc"
    )
    # URI de tipo file://
    assert isinstance(uri, str) and uri.startswith("file://")
    # callback de Optuna-MLflow
    from optuna.integration.mlflow import MLflowCallback
    assert isinstance(cb, MLflowCallback)