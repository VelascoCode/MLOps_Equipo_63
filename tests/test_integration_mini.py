# tests/test_integration_mini.py
import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_equipo_63.load_and_preparation import (
    prepare_numeric_df,
    clip_outliers_iqr,
)
from mlops_equipo_63.Retrain_and_Evaluate import retrain_and_evaluate_best


class _FakeBestTrial:
    """Objeto mínimo para emular best_trial en la API usada por retrain_and_evaluate_best."""
    def __init__(self, params):
        self.params = params
        self.system_attrs = {}  # usado si parent_from_best_trial=True (aquí será False)


class FakeStudy:
    """Study falso con la interfaz mínima requerida por retrain_and_evaluate_best."""
    def __init__(self, params):
        # best_params: dict con al menos 'classifier' y los hiperparámetros del modelo
        self.best_params = params
        self.best_trial = _FakeBestTrial(params=params)


@pytest.mark.integration
def test_end_to_end_minimal(tmp_path, tiny_classification_df, isolated_mlflow_tmp, monkeypatch):
    """
    Flujo E2E mínimo:
    df -> prepare_numeric_df -> clip_outliers_iqr -> split -> FakeStudy -> retrain_and_evaluate_best
    Se registra en un MLflow local aislado (file:///...) y se guardan artefactos locales.
    """
    # Trabajamos en tmp para dejar limpio el repo
    monkeypatch.chdir(tmp_path)

    # 1) Preparación numérica
    df_num, _, _ = prepare_numeric_df(
        tiny_classification_df,
        exclude_cols=("url",),
        label_col="shares",
        drop_cols=("url", "timedelta"),
        impute_strategy="median",
    )

    # 2) Clipping por IQR (evita outliers fuertes que puedan afectar el split)
    df_clip, _, _ = clip_outliers_iqr(df_num, exclude_cols=("shares",), factor=1.5)

    # 3) Split
    X = df_clip.drop(columns=["shares"])
    y = df_clip["shares"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4) Estudio "falso" mínimo -> RandomForest
    study = FakeStudy(params={
        "classifier": "RandomForest",
        "rf_n_estimators": 120,
        "rf_max_depth": 6,
    })

    # 5) Ejecutar entrenamiento final y evaluación
    #    Nota: 'isolated_mlflow_tmp' setea MLFLOW_TRACKING_URI como URI (file:///...), la leemos del env
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    model, metrics, importance_df = retrain_and_evaluate_best(
        study,
        Xtr, ytr, Xte, yte,
        feature_names=list(X.columns),
        experiment_name="Integration_Exp",
        tracking_uri=tracking_uri,          # usar URI con esquema (no ruta Windows cruda)
        parent_from_best_trial=False,       # no necesitamos heredar run
        random_state=42
    )

    # 6) Asserts básicos de integridad
    assert model is not None, "El pipeline final no debe ser None"
    assert isinstance(metrics, dict), "Las métricas deben estar en un dict"
    assert "final_auc" in metrics, "Debe reportarse AUC final"
    assert "final_accuracy" in metrics, "Debe reportarse accuracy final"

    # 7) Artefactos locales esperados
    #    La función guarda el modelo en 'models/final_model.pkl' y gráficos en 'reports/...'
    expected_model = tmp_path / "models" / "final_model.pkl"
    assert expected_model.exists(), "No se encontró el modelo final en models/final_model.pkl"

    expected_cm = tmp_path / "reports" / "confusion_matrix.png"
    assert expected_cm.exists(), "No se generó la matriz de confusión en reports/confusion_matrix.png"