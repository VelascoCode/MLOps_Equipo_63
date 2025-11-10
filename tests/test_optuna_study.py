import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlops_equipo_63.Optuna_Study import run_optuna_study

@pytest.mark.slow
def test_run_optuna_study_minimal_randomforest():
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.normal(0,1,n)
    x2 = rng.normal(3,1.2,n)
    y  = (x1 + x2 > 3.2).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})

    # muy pocos trials y solo RF para rapidez
    study, summary = run_optuna_study(
        X_train=X, y_train=y,
        study_name="test_study",
        n_trials=1,
        cv=2,
        metric_name="roc_auc",
        extra_metrics=("accuracy",),
        enable_models=("RandomForest",),  # evita cargar MLP/XGB/LGBM
        random_state=42,
        mlflow_callback=None,
        n_jobs_cv=1
    )
    assert hasattr(study, "best_trial")
    assert "best_value" in summary and "best_params" in summary and "metric" in summary