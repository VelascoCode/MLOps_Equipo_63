"""Tests for Configuration class and params.yaml loading."""
import os
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from mlops_equipo_63.Configuration import Config


def test_config_from_params_default():
    """Test that Config.from_params() loads the default params.yaml successfully."""
    cfg = Config.from_params("params.yaml")
    
    # Verify key attributes are loaded
    assert cfg.data_path == "data/raw/online_news_modified.csv"
    assert cfg.target_col == "shares"
    assert cfg.test_size == 0.2
    assert cfg.random_state == 42
    assert cfg.n_trials == 20
    assert cfg.cv_folds == 5
    assert cfg.mlflow_experiment == "Equipo63_Fase2"


def test_config_from_params_custom():
    """Test that Config.from_params() works with a custom params.yaml."""
    # Create a temporary params file
    custom_params = {
        "data": {
            "raw_path": "data/test.csv",
            "processed_dir": "data/test_processed"
        },
        "train": {
            "test_size": 0.25,
            "random_state": 123,
            "n_trials": 10,
            "cv_folds": 3,
            "n_jobs_cv": 1
        },
        "model": {
            "target_col": "target",
            "pos_label_threshold": 500,
            "study_name": "test_study"
        },
        "track": {
            "experiment": "test_experiment",
            "mlruns_dir": "test_mlruns"
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = Path(tmpdir) / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(custom_params, f)
        
        cfg = Config.from_params(str(params_path))
        
        assert cfg.data_path == "data/test.csv"
        assert cfg.target_col == "target"
        assert cfg.test_size == 0.25
        assert cfg.random_state == 123
        assert cfg.n_trials == 10
        assert cfg.cv_folds == 3


def test_config_env_var_overrides(monkeypatch):
    """Test that environment variables override params.yaml values."""
    custom_params = {
        "data": {"raw_path": "data/raw.csv", "processed_dir": "data/proc"},
        "train": {
            "test_size": 0.2,
            "random_state": 42,
            "n_trials": 20,
            "cv_folds": 5,
            "n_jobs_cv": -1
        },
        "model": {"target_col": "shares", "pos_label_threshold": 1400, "study_name": "study"},
        "track": {"experiment": "exp", "mlruns_dir": "mlruns"}
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = Path(tmpdir) / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(custom_params, f)
        
        # Set env var overrides
        monkeypatch.setenv("N_TRIALS", "50")
        monkeypatch.setenv("RANDOM_STATE", "999")
        monkeypatch.setenv("TARGET_COL", "custom_target")
        
        cfg = Config.from_params(str(params_path))
        
        assert cfg.n_trials == 50  # env override
        assert cfg.random_state == 999  # env override
        assert cfg.target_col == "custom_target"  # env override
        assert cfg.test_size == 0.2  # from params.yaml


def test_config_optuna_search_space():
    """Test that optuna search_space is loaded from params.yaml."""
    cfg = Config.from_params("params.yaml")
    
    # Verify search_space is loaded
    assert cfg.search_space is not None
    assert isinstance(cfg.search_space, dict)
    assert "RandomForest" in cfg.search_space
    assert "MLP" in cfg.search_space
    
    # Verify RandomForest search space
    rf_space = cfg.search_space["RandomForest"]
    assert "rf_n_estimators" in rf_space
    assert rf_space["rf_n_estimators"]["type"] == "int"
    assert rf_space["rf_n_estimators"]["low"] == 50
    assert rf_space["rf_n_estimators"]["high"] == 400


def test_config_enable_models():
    """Test that enable_models list is loaded from params.yaml."""
    cfg = Config.from_params("params.yaml")
    
    assert cfg.enable_models is not None
    assert isinstance(cfg.enable_models, list)
    assert "RandomForest" in cfg.enable_models
    assert "MLP" in cfg.enable_models


def test_config_api_paths():
    """Test that API paths are loaded correctly."""
    cfg = Config.from_params("params.yaml")
    
    assert cfg.api_model_path == "models/final_model.pkl"
    assert cfg.api_feature_names_path == "models/feature_names.json"


def test_config_feature_extraction_fill_random():
    """Test that feature_extraction fill_random flag is loaded."""
    cfg = Config.from_params("params.yaml")
    
    assert cfg.feature_extraction_fill_random is False  # Default is false


def test_config_missing_params_file_returns_defaults():
    """Test that Config.from_params() with missing file returns sensible defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config.from_params(str(Path(tmpdir) / "nonexistent.yaml"))
        
        # Verify defaults are used
        assert cfg.target_col == "shares"
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42

