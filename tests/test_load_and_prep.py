# tests/test_load_and_prep.py
import numpy as np
import pandas as pd
import pytest

from mlops_equipo_63.load_and_preparation import prepare_numeric_df, clip_outliers_iqr

@pytest.mark.unit
def test_prepare_numeric_df_returns_expected_shapes_and_no_nans(tiny_classification_df):
    df_num, non_numeric_cols, missing_pct = prepare_numeric_df(
        tiny_classification_df,
        exclude_cols=("url",),
        label_col="shares",
        drop_cols=("url", "timedelta"),
        impute_strategy="median",
    )

    # 1) Debe existir la columna 'shares' y no tener NaNs
    assert "shares" in df_num.columns
    assert df_num["shares"].isna().sum() == 0

    # 2) df_num no debe tener las columnas excluidas y sin NaNs
    assert "url" not in df_num.columns
    assert "timedelta" not in df_num.columns
    # non_numeric_cols debe contener las columnas no numéricas originales menos las eliminadas
    assert df_num.isna().sum().sum() == 0

    # 3) missing_pct debe tener índices coincidentes con las columnas numéricas
    assert set(missing_pct.index) == set(df_num.columns)
    assert (missing_pct >= 0).all() and (missing_pct <= 100).all()

@pytest.mark.unit
def test_clip_outliers_iqr_clips_and_reports(tiny_classification_df):
    # Fuerza algunos outliers
    df = tiny_classification_df.copy()
    df.loc[df.index[:5], "x1"] = 1000.0

    clipped, col_perc, mean_pct = clip_outliers_iqr(
        df, exclude_cols=("shares",), factor=1.5
    )

    # Misma forma, pero sin esos outliers extremos
    assert clipped.shape == df.shape
    assert (clipped["x1"].max() < 1000.0)
    # Reporte de porcentajes consistente
    assert isinstance(col_perc, dict)
    assert isinstance(mean_pct, float)
    assert mean_pct >= 0.0