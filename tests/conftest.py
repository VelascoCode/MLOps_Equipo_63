import sys
import os
import tempfile
from pathlib import Path

# Ensure the project root is on sys.path so tests can import the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Safety: ensure tests do not write to the repository-level mlruns/ directory
# Create a temporary directory for MLflow tracking for the duration of the
# test session and expose it via the standard environment variable. Use a
# plain path (not a file:// URI) because project code will convert the path
# to a URI via Path(...).as_uri().
_TEST_MLRUNS_DIR = tempfile.mkdtemp(prefix="pytest_mlruns_")
os.environ.setdefault("MLFLOW_TRACKING_URI", str(_TEST_MLRUNS_DIR))

# Provide a minimal stub for mlflow if it's not installed so imports succeed during tests
try:
    import mlflow  # type: ignore
except Exception:
    import types
    import pandas as pd
    fake_mlflow = types.ModuleType("mlflow")

    def _noop(*args, **kwargs):
        return None

    class _DummyRunCtx:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return types.SimpleNamespace()
        def __exit__(self, exc_type, exc, tb):
            return False

    fake_mlflow.set_tracking_uri = _noop
    fake_mlflow.set_registry_uri = _noop
    fake_mlflow.set_experiment = _noop
    fake_mlflow.get_tracking_uri = lambda: os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    fake_mlflow.get_experiment_by_name = lambda name: types.SimpleNamespace(experiment_id="1")
    fake_mlflow.search_runs = lambda experiment_ids=None: pd.DataFrame({"metrics.final_auc": [0.1], "run_id": ["r1"]})
    fake_mlflow.start_run = lambda *a, **k: _DummyRunCtx()
    fake_mlflow.log_param = _noop
    fake_mlflow.log_metric = _noop
    fake_mlflow.log_artifact = _noop
    fake_mlflow.set_registry_uri = _noop

    # minimal mlflow.models.signature.infer_signature
    fake_mlflow.models = types.ModuleType("mlflow.models")
    fake_mlflow.models.signature = types.ModuleType("mlflow.models.signature")
    fake_mlflow.models.signature.infer_signature = lambda *a, **k: None
    sys.modules["mlflow"] = fake_mlflow
    sys.modules["mlflow.models"] = fake_mlflow.models
    sys.modules["mlflow.models.signature"] = fake_mlflow.models.signature

# Provide a minimal stub for optuna.integration.mlflow if it's missing
try:
    import optuna  # type: ignore
    try:
        import optuna.integration.mlflow  # type: ignore
    except Exception:
        import types
        mod_mlflow = types.ModuleType("optuna.integration.mlflow")

        def MLflowCallback(*args, **kwargs):
            class Dummy:
                def __init__(self, *a, **k):
                    pass
            return Dummy()

        mod_mlflow.MLflowCallback = MLflowCallback
        sys.modules["optuna.integration.mlflow"] = mod_mlflow
except Exception:
    # optuna not installed at all: provide minimal package + mlflow integration stub
    import types
    mod_optuna = types.ModuleType("optuna")
    mod_integration = types.ModuleType("optuna.integration")
    mod_mlflow = types.ModuleType("optuna.integration.mlflow")

    def MLflowCallback(*args, **kwargs):
        class Dummy:
            def __init__(self, *a, **k):
                pass
        return Dummy()

    mod_mlflow.MLflowCallback = MLflowCallback
    sys.modules["optuna"] = mod_optuna
    sys.modules["optuna.integration"] = mod_integration
    sys.modules["optuna.integration.mlflow"] = mod_mlflow


