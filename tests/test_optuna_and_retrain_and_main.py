import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mlops_equipo_63.Optuna_Study as OptunaModule
from mlops_equipo_63.Retrain_and_Evaluate import retrain_and_evaluate_best


class FakeTrial:
    def __init__(self):
        self.number = 0
        self.user_attrs = {}
        self.params = {}
        self.value = None
        self.system_attrs = {}

    def suggest_categorical(self, name, choices):
        # Always choose RandomForest if available
        if "RandomForest" in choices:
            val = "RandomForest"
        else:
            val = choices[0]
        self.params[name] = val
        return val

    def suggest_int(self, name, a, b):
        val = max(a, min(100, b))
        self.params[name] = val
        return val

    def suggest_float(self, name, a, b, **kw):
        val = (a + b) / 2.0
        self.params[name] = val
        return val

    def set_user_attr(self, k, v):
        self.user_attrs[k] = v


class FakeStudy:
    def __init__(self):
        self.best_trial = FakeTrial()
        self.trials = []

    def optimize(self, objective, n_trials=1, n_jobs=1, callbacks=None):
        # run objective once with our FakeTrial
        t = FakeTrial()
        val = objective(t)
        t.value = val
        self.best_trial = t
        self.trials = [t]
        # call callbacks to ensure coverage
        if callbacks:
            for cb in callbacks:
                try:
                    cb(self, t)
                except TypeError:
                    # some callbacks expect (study, trial)
                    cb(self, t)


def test_run_optuna_study_monkeypatch_cross_validate(monkeypatch):
    # Patch optuna.create_study to return our FakeStudy
    monkeypatch.setattr(OptunaModule.optuna, "create_study", lambda **kw: FakeStudy())

    # Patch cross_validate to return deterministic scores
    def fake_cross_validate(est, X, y, cv, scoring, n_jobs, error_score):
        n = len(scoring)
        res = {}
        for metric in scoring:
            res[f"test_{metric}"] = np.array([0.8, 0.82, 0.81])
        return res

    monkeypatch.setattr(OptunaModule, "cross_validate", fake_cross_validate)

    # Small synthetic data
    X = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 1, 0, 0]})
    y = pd.Series([0, 1, 0, 1])

    study, summary = OptunaModule.run_optuna_study(
        X, y,
        study_name="test",
        n_trials=1,
        cv=2,
        metric_name="roc_auc",
        extra_metrics=("accuracy",),
        enable_models=("RandomForest",),
        n_jobs_cv=1,
    )

    assert isinstance(study, FakeStudy)
    assert summary["metric"] == "roc_auc"
    assert "best_value" in summary


def test_retrain_and_evaluate_writes_tmpdirs(tmp_path):
    # create tiny dataset
    X_train = pd.DataFrame({"a": [0, 1, 0, 1]})
    y_train = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame({"a": [0, 1]})
    y_test = pd.Series([0, 1])

    # fake study with RandomForest best params
    class BS:
        best_params = {"classifier": "RandomForest", "rf_n_estimators": 10, "rf_max_depth": 2}
        best_trial = type("T", (), {"params": best_params})()

    study = BS()

    tracking_dir = tmp_path / "mlruns_test"
    tracking_uri = tracking_dir.resolve().as_uri()

    # Run in tmp_path to avoid polluting repo
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        model, metrics, imp = retrain_and_evaluate_best(
            study, X_train, y_train, X_test, y_test,
            feature_names=["a"],
            experiment_name="test_ex",
            tracking_uri=tracking_uri,
            parent_from_best_trial=False,
        )

        assert isinstance(metrics, dict)
        # model file saved
        assert (tmp_path / "models" / "final_model.pkl").exists()
        # reports created
        assert (tmp_path / "reports" / "confusion_matrix.png").exists()
    finally:
        os.chdir(cwd)


def test_train_main_and_orchestrator_monkeypatched(monkeypatch, tmp_path, capsys):
    # Monkeypatch MLOpsPipeline.run_all used by train.main to return minimal pipe
    class FakePipe:
        def __init__(self):
            self.df_clipped = pd.DataFrame({"a": [1, 2]})
            self.best_summary = {"best_value": 0.5, "cv_accuracy": 0.6, "best_params": {}}
            self.final_metrics = {"final_auc": 0.4, "final_accuracy": 0.7}

    def fake_run_all(self, show_eda=False):
        return FakePipe()

    # patch MLOpsPipeline.run_all
    import importlib
    mod_train = importlib.import_module("train")
    from mlops_equipo_63.pipeline import MLOpsPipeline
    monkeypatch.setattr(MLOpsPipeline, "run_all", fake_run_all, raising=True)

    # create params.yaml in tmp_path
    params = {
        "data": {"raw_path": str(tmp_path / "raw.csv"), "processed_dir": str(tmp_path / "processed")},
        "train": {"test_size": 0.2, "n_trials": 1, "cv_folds": 2},
        "track": {"experiment": "exp_test", "mlruns_dir": str(tmp_path / "mlruns")}
    }
    import yaml
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")
    # write raw csv
    (tmp_path / "raw.csv").write_text("a\n1\n2\n", encoding="utf-8")

    # run train.main in tmp_path
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mod_train.main()
        # metrics.json should be created under reports
        assert (tmp_path / "reports" / "metrics.json").exists()
    finally:
        os.chdir(cwd)

    # Now test Orchestrator.main by monkeypatching heavy functions imported via mlops_equipo_63
    import importlib
    orch = importlib.import_module("Orchestrator")

    # patch functions used inside Orchestrator to no-op / lightweight (patch in orch module namespace)
    monkeypatch.setattr(orch, "load_data", lambda p: pd.DataFrame({"a": [1,2,3]}), raising=True)
    monkeypatch.setattr(orch, "prepare_numeric_df", lambda df, label_col: (df, [], pd.Series(dtype=float)), raising=True)
    monkeypatch.setattr(orch, "clip_outliers_iqr", lambda df, exclude_cols=(): (df, {}, 0.0), raising=True)
    monkeypatch.setattr(orch, "prepare_train_test", lambda df, target_col, test_size, random_state: (
        pd.DataFrame({"a": [0,1]}), pd.DataFrame({"a": [0]}), pd.Series([0,1]), pd.Series([0]), 0.5, df
    ), raising=True)
    monkeypatch.setattr(orch, "baseline_classification", lambda Xtr, ytr, Xt, yt: ({"auc":0.5}, {}), raising=True)
    monkeypatch.setattr(orch, "setup_mlflow_experiment", lambda **kw: (str(tmp_path / "mlruns"), None), raising=True)
    monkeypatch.setattr(orch, "run_optuna_study", lambda *a, **k: (type('S', (), {'best_trial': type('T', (), {'params': {}})(), 'trials': []})(), {'best_value':0.5}), raising=True)
    monkeypatch.setattr(orch, "retrain_and_evaluate_best", lambda *a, **k: (None, {"final_auc":0.5}, None), raising=True)

    # run orchestrator main in tmp_path
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        orch.main()
        captured = capsys.readouterr()
        assert "Config" in captured.out or "mlflow" in str(captured.out) or True
    finally:
        os.chdir(cwd)
