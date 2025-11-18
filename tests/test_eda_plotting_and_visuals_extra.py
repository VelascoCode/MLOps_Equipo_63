import matplotlib
matplotlib.use("Agg")
import pytest
import pandas as pd

from mlops_equipo_63.EDA_Plotting import EDAPlotter
from mlops_equipo_63.EDA_visuals import EDAVisualizer


def make_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "shares": [10, 20, 30, 40, 50]})


def test_eda_plotter_hist_and_corr_and_boxplots(capsys):
    df = make_df()
    p = EDAPlotter(df)

    # histogram for existing column should not raise
    p.plot_hist(col="shares", bins=5)

    # missing column raises ValueError
    with pytest.raises(ValueError):
        p.plot_hist(col="missing_col")

    # boxplots should skip missing columns and print a warning
    p.plot_boxplots(cols=["a", "missing_col"])  # should not raise
    captured = capsys.readouterr()
    assert "no está en el DataFrame" in captured.out or "se omite" in captured.out

    # correlation returns a DataFrame-like object with numeric columns
    corr = p.correlation(plot=False)
    assert isinstance(corr, type(df.corr()))
    assert "a" in corr.columns and "shares" in corr.columns


def test_eda_visualizer_hist_box_and_corr():
    df = make_df()
    v = EDAVisualizer(df)

    # hist and boxplots should run without exception
    v.hist("a", bins=3)
    v.boxplots(cols=["a"])  # should not raise

    # corr_matrix returns same correlation as pandas
    corr = v.corr_matrix(method="pearson")
    expected = df.corr(numeric_only=True, method="pearson")
    # numeric equivalence
    assert corr.equals(expected)
