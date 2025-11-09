# mlops_equipo_63/Retrain_and_Evaluate.py
import os
import json
import tempfile
from typing import Optional, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models.signature import infer_signature  # opcional (ya no la usamos para log_model)

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Modelos opcionales
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from mlops_equipo_63.Outlier_Clipper import OutlierClipper, DataFrameTransformer


def retrain_and_evaluate_best(
    study,
    X_train, y_train, X_test, y_test,
    feature_names: Optional[list] = None,
    experiment_name: str = "Online_News_Popularity_Estudio_Optuna",
    tracking_uri: Optional[str] = None,   # ej: 'file:///C:/.../mlruns'
    parent_from_best_trial: bool = True,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], Optional[pd.DataFrame]]:

    # ---------- reconstruir el mejor estimador ----------
    best_params = study.best_params.copy()
    clf_name = best_params.pop("classifier")

    def _parse_mlp_hidden(val):
        if isinstance(val, tuple):
            return val
        if isinstance(val, str):
            s = val.strip().strip("()")
            if not s:
                return (50,)
            try:
                return tuple(int(x) for x in s.split(",") if x.strip())
            except Exception:
                return (50,)
        return (50,)

    model_params: Dict[str, Any] = {}
    # Preprocessing steps: imputer first (mirrors prepare_numeric_df imputation)
    steps = []
    # wrap SimpleImputer so it returns a DataFrame with column names preserved
    steps.append(("imputer", DataFrameTransformer(SimpleImputer(strategy="median"))))
    # outlier clipping step (custom transformer that returns a DataFrame)
    steps.append(("outlierclipper", OutlierClipper()))
    # scaling (wrap so we keep DataFrame/column names)
    steps.append(("scaler", DataFrameTransformer(StandardScaler())))  # default scaler

    if clf_name == "RandomForest":
        rename_map = {"rf_n_estimators": "n_estimators", "rf_max_depth": "max_depth"}
        model_params = {rename_map[k]: v for k, v in best_params.items() if k in rename_map}
        final_clf = RandomForestClassifier(random_state=random_state, n_jobs=-1, **model_params)

    elif clf_name == "MLP":
        if "mlp_hidden_layers" in best_params:
            best_params["hidden_layer_sizes"] = _parse_mlp_hidden(best_params.pop("mlp_hidden_layers"))
        allow = {"mlp_alpha": "alpha"}
        model_params = {allow[k]: v for k, v in best_params.items() if k in allow}
        final_clf = MLPClassifier(max_iter=300, early_stopping=True, random_state=random_state, **model_params)
        # scaling required for neural nets — scaler is already in the default steps

    elif clf_name == "XGBoost":
        if xgb is None:
            raise RuntimeError("XGBoost no está instalado, no puedo reconstruir el mejor modelo.")
        rename_map = {
            "xgb_n_estimators": "n_estimators",
            "xgb_learning_rate": "learning_rate",
            "xgb_max_depth": "max_depth",
            "xgb_subsample": "subsample",
            "xgb_colsample": "colsample_bytree",
        }
        model_params = {rename_map[k]: v for k, v in best_params.items() if k in rename_map}
        final_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            **model_params
        )

    elif clf_name == "LightGBM":
        if lgb is None:
            raise RuntimeError("LightGBM no está instalado, no puedo reconstruir el mejor modelo.")
        rename_map = {
            "lgbm_n_estimators": "n_estimators",
            "lgbm_learning_rate": "learning_rate",
            "lgbm_num_leaves": "num_leaves",
            "lgbm_subsample": "subsample",
            "lgbm_colsample": "feature_fraction",
        }
        model_params = {rename_map[k]: v for k, v in best_params.items() if k in rename_map}
        final_clf = lgb.LGBMClassifier(objective="binary", random_state=random_state, n_jobs=-1, **model_params)

    else:
        raise ValueError(f"Clasificador desconocido en best_params: {clf_name}")

    steps.append(("classifier", final_clf))
    final_pipeline = Pipeline(steps=steps)

    # ---------- MLflow (sólo métricas/artefactos pequeños) ----------
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        if tracking_uri.startswith("file:"):
            mlflow.set_registry_uri("")  # sin registry local

    mlflow.set_experiment(experiment_name)

    parent_run_id = None
    if parent_from_best_trial:
        parent_run_id = study.best_trial.system_attrs.get("mlflow_run_id")

    metrics: Dict[str, Any] = {}
    importance_df: Optional[pd.DataFrame] = None

    with mlflow.start_run(
        run_name=f"Final_{clf_name}_Model",
        nested=True if parent_run_id else False,
        tags={"stage": "final_evaluation"},
        parent_run_id=parent_run_id
    ):
        mlflow.log_param("classifier", clf_name)
        for p_k, p_v in model_params.items():
            mlflow.log_param(p_k, p_v)

        # Entrenar y evaluar
        final_pipeline.fit(X_train, y_train)

        # Intentar obtener los nombres de las características procesadas (útiles para serving)
        feature_names = None
        try:
            pre_clf = None
            if "scaler" in final_pipeline.named_steps:
                pre_clf = final_pipeline.named_steps["scaler"]
            elif "outlierclipper" in final_pipeline.named_steps:
                pre_clf = final_pipeline.named_steps["outlierclipper"]
            elif "imputer" in final_pipeline.named_steps:
                pre_clf = final_pipeline.named_steps["imputer"]

            if pre_clf is not None and hasattr(pre_clf, "get_feature_names_out"):
                feature_names = list(pre_clf.get_feature_names_out())
        except Exception:
            feature_names = None

        y_pred = final_pipeline.predict(X_test)

        if hasattr(final_pipeline, "predict_proba"):
            y_proba = final_pipeline.predict_proba(X_test)[:, 1]
            metrics["final_auc"] = float(roc_auc_score(y_test, y_proba))
        else:
            metrics["final_auc"] = float("nan")

        metrics["final_accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["classification_report"] = classification_report(
            y_test, y_pred, target_names=["Unpopular", "Popular"], zero_division=0
        )

        print(f"\nFinal Test AUC: {metrics['final_auc']:.4f}")
        print(f"Final Test Accuracy: {metrics['final_accuracy']:.4f}")
        print("\n--- Classification Report ---")
        print(metrics["classification_report"])

        mlflow.log_metric("final_auc", metrics["final_auc"])
        mlflow.log_metric("final_accuracy", metrics["final_accuracy"])

        # ---------- reportes locales + log de copias pequeñas ----------
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Unpopular", "Popular"], yticklabels=["Unpopular", "Popular"]
        )
        plt.title("Confusion Matrix on Test Set")
        plt.xlabel("Predicted"); plt.ylabel("True")
        cm_local = reports_dir / "confusion_matrix.png"
        plt.savefig(cm_local, bbox_inches="tight")
        plt.close()

        # Sube una copia pequeña a MLflow (opcional)
        try:
            mlflow.log_artifact(str(cm_local))
        except Exception:
            pass

        # Importancias (si existen)
        clf = final_pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_") and feature_names is not None:
            importances = clf.feature_importances_
            importance_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
            )
            plt.figure(figsize=(12, 8))
            sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
            plt.title(f"Top 20 Feature Importances — {clf.__class__.__name__}")
            fi_local = reports_dir / "feature_importance.png"
            plt.savefig(fi_local, bbox_inches="tight")
            plt.close()
            try:
                mlflow.log_artifact(str(fi_local))
                tmp_csv = reports_dir / "feature_importance_top20.csv"
                importance_df.head(20).to_csv(tmp_csv, index=False)
                mlflow.log_artifact(str(tmp_csv))
            except Exception:
                pass

        # Guarda el pipeline final SOLO en la carpeta 'models' (no en MLflow)
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "final_model.pkl"
        joblib.dump(final_pipeline, model_path)

        # Guardar lista de nombres de características procesadas (si se obtuvo)
        try:
            if feature_names:
                features_path = models_dir / "feature_names.json"
                with open(features_path, "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Artefacto con best_params (pequeño)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                params_path = Path(tmpdir) / "best_params.json"
                with open(params_path, "w", encoding="utf-8") as f:
                    json.dump(study.best_trial.params, f, indent=2)
                mlflow.log_artifact(str(params_path))
        except Exception:
            pass

    print("\n✅ Modelo final guardado en 'models/final_model.pkl' y métricas registradas en MLflow.")
    return final_pipeline, metrics, importance_df