import pandas as pd
import numpy as np
from mlops_equipo_63.Split_and_Dummy import prepare_train_test, baseline_classification


def make_sample_df(n=50):
    df = pd.DataFrame({
        'shares': np.random.randint(0, 1000, size=n),
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
    })
    return df


def test_prepare_train_test():
    df = make_sample_df(30)
    X_train, X_test, y_train, y_test, threshold, df_clean = prepare_train_test(df, verbose=False)
    assert hasattr(X_train, 'shape')
    assert isinstance(threshold, (int, float))


def test_baseline_classification():
    df = make_sample_df(40)
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test(df, verbose=False)
    metrics, pipe = baseline_classification(X_train, y_train, X_test, y_test, return_pipeline=True)
    assert 'accuracy' in metrics
    assert pipe is not None
