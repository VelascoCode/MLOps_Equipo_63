import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def _load_params(path: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from YAML file. Return defaults if file not found."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return loaded
    except Exception:
        return {}


@dataclass
class Config:
    """Configuration class that reads from params.yaml and env vars (with overrides)."""

    # Data
    data_path: str = "data/raw/online_news_modified.csv"
    processed_dir: str = "data/processed"

    # Train
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 20
    cv_folds: int = 5
    n_jobs_cv: int = -1

    # Model
    target_col: str = "shares"
    pos_label_threshold: int = 1400
    study_name: str = "optuna_study"

    # Track
    mlflow_experiment: str = "mlops_experiment"
    mlflow_tracking_uri: str = "mlruns"

    # Optuna (optional)
    optuna_random_state: Optional[int] = None
    optuna_n_jobs: Optional[int] = 1
    enable_models: Optional[list] = None
    search_space: Optional[Dict[str, Any]] = None

    # API
    api_model_path: str = "models/final_model.pkl"
    api_feature_names_path: str = "models/feature_names.json"

    # Feature extraction
    feature_extraction_fill_random: bool = False

    @classmethod
    def from_params(cls, params_path: str = "params.yaml") -> "Config":
        """Load config from params.yaml with env var overrides."""
        p = _load_params(params_path)
        
        # Data section
        data_path = os.getenv("DATA_RAW_PATH", p.get("data", {}).get("raw_path", "data/raw/online_news_modified.csv"))
        processed_dir = os.getenv("DATA_PROCESSED_DIR", p.get("data", {}).get("processed_dir", "data/processed"))
        
        # Train section
        test_size = float(os.getenv("TEST_SIZE", p.get("train", {}).get("test_size", 0.2)))
        random_state = int(os.getenv("RANDOM_STATE", p.get("train", {}).get("random_state", 42)))
        n_trials = int(os.getenv("N_TRIALS", p.get("train", {}).get("n_trials", 20)))
        cv_folds = int(os.getenv("CV_FOLDS", p.get("train", {}).get("cv_folds", 5)))
        n_jobs_cv = int(os.getenv("N_JOBS_CV", p.get("train", {}).get("n_jobs_cv", -1)))
        
        # Model section
        target_col = os.getenv("TARGET_COL", p.get("model", {}).get("target_col", "shares"))
        pos_label_threshold = int(os.getenv("POS_LABEL_THRESHOLD", p.get("model", {}).get("pos_label_threshold", 1400)))
        study_name = os.getenv("STUDY_NAME", p.get("model", {}).get("study_name", "optuna_study"))
        
        # Track section
        mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", p.get("track", {}).get("experiment", "mlops_experiment"))
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", p.get("track", {}).get("mlruns_dir", "mlruns"))
        
        # Optuna section (optional)
        optuna_random_state = int(os.getenv("OPTUNA_RANDOM_STATE", p.get("optuna", {}).get("random_state", random_state)))
        optuna_n_jobs = int(os.getenv("OPTUNA_N_JOBS", p.get("optuna", {}).get("n_jobs", 1)))
        enable_models = p.get("optuna", {}).get("enable_models", ["RandomForest", "MLP", "XGBoost", "LightGBM"])
        search_space = p.get("optuna", {}).get("search_space", {})
        
        # API section
        api_model_path = os.getenv("API_MODEL_PATH", p.get("api", {}).get("model_path", "models/final_model.pkl"))
        api_feature_names_path = os.getenv("API_FEATURE_NAMES_PATH", p.get("api", {}).get("feature_names_path", "models/feature_names.json"))
        
        # Feature extraction section
        feature_extraction_fill_random = os.getenv("FEATURE_EXTRACTION_FILL_RANDOM", str(p.get("feature_extraction", {}).get("fill_random", False))).lower() in ("true", "1", "yes")
        
        return cls(
            data_path=str(data_path),
            processed_dir=str(processed_dir),
            test_size=test_size,
            random_state=random_state,
            n_trials=n_trials,
            cv_folds=cv_folds,
            n_jobs_cv=n_jobs_cv,
            target_col=str(target_col),
            pos_label_threshold=pos_label_threshold,
            study_name=str(study_name),
            mlflow_experiment=str(mlflow_experiment),
            mlflow_tracking_uri=str(mlflow_tracking_uri),
            optuna_random_state=optuna_random_state,
            optuna_n_jobs=optuna_n_jobs,
            enable_models=enable_models,
            search_space=search_space,
            api_model_path=str(api_model_path),
            api_feature_names_path=str(api_feature_names_path),
            feature_extraction_fill_random=feature_extraction_fill_random,
        )