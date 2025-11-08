from pathlib import Path
from mlops_equipo_63.mlflow_utils import setup_mlflow_experiment


def test_setup_mlflow_experiment(tmp_path):
    td = tmp_path / "mlruns"
    td.mkdir()
    tracking_uri, mlflow_cb = setup_mlflow_experiment(experiment_name="test_exp", tracking_dir=str(td), metric_name="auc")
    assert tracking_uri.startswith("file:")
    assert mlflow_cb is not None
