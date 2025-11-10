import pandas as pd
import numpy as np
import pytest

from mlops_equipo_63.Split_and_Dummy import prepare_train_test, baseline_classification

@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "feat1": rng.normal(0,1,n),
        "feat2": rng.normal(4,2,n),
        "shares": rng.integers(0, 1000, n)  # objetivo continuo original
    })
    return df

@pytest.mark.unit
def test_prepare_train_test_shapes_and_columns(synthetic_df):
    Xtr, Xte, ytr, yte, threshold, df_clean = prepare_train_test(
        synthetic_df, target_col="shares", test_size=0.2, random_state=42, stratify=True, verbose=False
    )
    # tamaños consistentes
    assert len(Xtr) + len(Xte) == len(df_clean)
    assert len(ytr) + len(yte) == len(df_clean)
    # columnas: ni 'shares' ni 'popular' deben estar en X
    for X in (Xtr, Xte):
        assert "shares" not in X.columns
        assert "popular" not in X.columns
    # y debe ser binaria
    for y in (ytr, yte):
        assert set(y.unique()).issubset({0,1})
    # umbral mediana numérico
    assert isinstance(threshold, (int, float))

@pytest.mark.unit
def test_baseline_classification_returns_metrics(synthetic_df):
    Xtr, Xte, ytr, yte, _, _ = prepare_train_test(synthetic_df, verbose=False)
    metrics, pipe = baseline_classification(Xtr, ytr, Xte, yte, return_pipeline=True)
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert pipe is not None