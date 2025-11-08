import os
from mlops_equipo_63.Configuration import Config


def test_config_defaults():
    cfg = Config()
    assert hasattr(cfg, "data_path")
    assert hasattr(cfg, "target_col")
    assert isinstance(cfg.test_size, float)
    # mlflow_tracking_uri should default to 'mlruns' or an env var
    assert isinstance(cfg.mlflow_tracking_uri, str)

