import sys
import importlib
import pandas as pd
from types import SimpleNamespace


def test_best_run_id_monkeypatch(monkeypatch, capsys):
    # Create a fake mlflow module to avoid real MLflow interactions on import
    fake_mlflow = SimpleNamespace()
    fake_experiment = SimpleNamespace(experiment_id="1")
    fake_mlflow.get_experiment_by_name = lambda name: fake_experiment

    # create a fake runs dataframe
    df = pd.DataFrame({
        "metrics.final_auc": [0.1, 0.9],
        "metrics.final_accuracy": [0.2, 0.8],
        "run_id": ["r1", "r2"],
    })
    # .loc[...] used in module; ensure idxmax works
    fake_mlflow.search_runs = lambda experiment_ids=None: df

    # inject fake mlflow
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    # import the module under test (it runs top-level code)
    import mlops_equipo_63.best_run_id as br
    # it should have printed best run lines
    captured = capsys.readouterr()
    assert "Best run ID" in captured.out
