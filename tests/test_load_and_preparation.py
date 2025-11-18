import pandas as pd
import numpy as np
from mlops_equipo_63.load_and_preparation import (
    prepare_numeric_df,
    clip_outliers_iqr,
    load_data,
)


def test_prepare_numeric_df_and_imputation(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        # make b numeric so imputer has observed values
        "b": [0.1, 0.2, 0.3, 0.4],
        "url": ["u1", "u2", "u3", "u4"],
        "shares": [10, 20, 30, None],
    })

    df_numeric, non_numeric, missing_pct = prepare_numeric_df(df, exclude_cols=("url",), label_col="shares")

    # numeric dataframe should contain numeric conversions
    assert "a" in df_numeric.columns
    # in this test all columns (except excluded ones) are numeric so non_numeric should be empty
    assert isinstance(non_numeric, list)
    assert len(non_numeric) == 0
    # missing_pct should be a pandas Series-like
    assert hasattr(missing_pct, "loc")


def test_clip_outliers_iqr():
    # create dataframe with an outlier
    df = pd.DataFrame({"x": [1, 2, 3, 1000], "shares": [0, 1, 0, 1]})
    dfc, outlier_map, mean_pct = clip_outliers_iqr(df, exclude_cols=("shares",), factor=1.5)
    # outlier should be clipped to some finite value
    assert dfc["x"].max() < 1000
    assert isinstance(outlier_map, dict)


def test_load_data_roundtrip(tmp_path):
    p = tmp_path / "tmp.csv"
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_csv(p, index=False)
    df2 = load_data(str(p))
    assert list(df2.columns) == ["a"]
