import os
import pandas as pd
import numpy as np
from types import SimpleNamespace

from mlops_equipo_63.Retrain_and_Evaluate import retrain_and_evaluate_best


class DummyTrial:
    def __init__(self, params=None):
        self.params = params or {}
        self.system_attrs = {}


class DummyStudy:
    def __init__(self, best_params=None):
        self.best_params = best_params or {}
        self.best_trial = DummyTrial(params=best_params or {})


def test_retrain_and_evaluate_best_minimal(tmp_path):
    # create tiny dataset
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame(rng.randn(40, 4), columns=[f"f{i}" for i in range(4)])
    y_train = rng.randint(0, 2, size=40)
    X_test = pd.DataFrame(rng.randn(10, 4), columns=[f"f{i}" for i in range(4)])
    y_test = rng.randint(0, 2, size=10)

    best_params = {"classifier": "RandomForest", "rf_n_estimators": 5, "rf_max_depth": 3}
    study = DummyStudy(best_params=best_params)

    # run inside tmp_path to avoid creating/writing repo-level artifacts (models/final_model.pkl)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        model, metrics, importance_df = retrain_and_evaluate_best(
            study, X_train, y_train, X_test, y_test,
            feature_names=list(X_train.columns),
            experiment_name="test",
            tracking_uri=None,
            parent_from_best_trial=False,
        )
        assert 'final_accuracy' in metrics
        assert model is not None
    finally:
        os.chdir(cwd)
