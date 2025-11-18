import pandas as pd
import matplotlib
matplotlib.use('Agg')

from mlops_equipo_63.EDA_Plotting import EDAPlotter
from mlops_equipo_63.EDA_visuals import EDAVisualizer


def test_eda_correlation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    eda = EDAPlotter(df)
    # call with plot=True to ensure corr_to_plot is created and returned
    corr = eda.correlation(plot=True)
    assert corr is not None


def test_visualizer_corr():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ev = EDAVisualizer(df)
    corr = ev.corr_matrix(method="pearson")
    assert corr.shape[0] == corr.shape[1]
