# tests/test_retrain_unit.py
import os
import pytest
import pandas as pd
from sklearn.datasets import make_classification
from mlops_equipo_63.Optuna_Study import run_optuna_study
from mlops_equipo_63.Retrain_and_Evaluate import retrain_and_evaluate_best

@pytest.fixture
def tiny_classification_df():
    """Crea un dataset diminuto para pruebas rápidas."""
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["shares"] = y  # se usa como target binario
    return df


@pytest.mark.unit
def test_retrain_and_evaluate_best_randomforest(tmp_path, tiny_classification_df, isolated_mlflow_tmp, monkeypatch):
    """Prueba un retraining completo usando RandomForest y MLflow local (aislado)."""
    # Cambiar al directorio temporal para no ensuciar el repo
    monkeypatch.chdir(tmp_path)

    # === Dividir el dataset ===
    from mlops_equipo_63.Split_and_Dummy import prepare_train_test
    Xtr, Xte, ytr, yte, _, _ = prepare_train_test(tiny_classification_df, target_col="shares", verbose=False)

    # === Crear un estudio Optuna diminuto ===
    study, summary = run_optuna_study(
        Xtr, ytr,
        study_name="unit_test_study",
        n_trials=2,        # pequeño para test
        cv=2,
        metric_name="roc_auc",
        extra_metrics=("accuracy",),
        enable_models=("RandomForest",),
        random_state=42,
        mlflow_callback=None,
        n_jobs_cv=1
    )

    #   Esta fixture ya configura MLFLOW_TRACKING_URI como un esquema file:// válido
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    # === Ejecutar retraining y evaluación ===
    model, metrics, importance_df = retrain_and_evaluate_best(
        study,
        Xtr, ytr, Xte, yte,
        feature_names=list(Xtr.columns),
        experiment_name="Test_Exp",
        tracking_uri=tracking_uri,
        parent_from_best_trial=False,
        random_state=42
    )

    # === Validaciones básicas ===
    assert model is not None, "El modelo final no debe ser None"
    assert isinstance(metrics, dict), "Las métricas deben ser un diccionario"
    assert "final_auc" in metrics, "Debe incluir AUC final"
    assert "final_accuracy" in metrics, "Debe incluir accuracy final"

    # Verificar que el modelo se guardó en carpeta local 'models'
    expected_model_path = tmp_path / "models" / "final_model.pkl"
    assert expected_model_path.exists(), "El modelo final no se guardó correctamente"

    # Verificar que los reportes se generaron
    reports_dir = tmp_path / "reports"
    assert (reports_dir / "confusion_matrix.png").exists(), "No se generó matriz de confusión"