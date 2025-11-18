import numpy as np
from sklearn.datasets import make_classification

from mlops_equipo_63.Optuna_Study import run_optuna_study


def test_run_optuna_study_minimal():
    X, y = make_classification(n_samples=60, n_features=5, n_informative=3, random_state=0)
    # Run with 1 trial to keep it fast
    study, summary = run_optuna_study(X, y, study_name="test", n_trials=1, cv=2, enable_models=("RandomForest",), n_jobs_cv=1)
    assert 'best_value' in summary
    assert hasattr(study, 'trials')
