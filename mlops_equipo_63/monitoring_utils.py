import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute PSI between two arrays.

    expected, actual: 1D arrays of the same variable (baseline, monitoring)
    buckets: number of quantile buckets to use
    """
    eps = 1e-8
    try:
        expected = np.asarray(expected).astype(float)
        actual = np.asarray(actual).astype(float)
    except Exception:
        # If not numeric, return large PSI to indicate drift
        return float('inf')

    # create quantile bins on expected
    quantiles = np.linspace(0, 1, buckets + 1)
    bins = np.unique(np.quantile(expected, quantiles))
    if len(bins) <= 1:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_counts / (expected_counts.sum() + eps)
    actual_perc = actual_counts / (actual_counts.sum() + eps)

    # avoid zeros
    expected_perc = np.where(expected_perc == 0, eps, expected_perc)
    actual_perc = np.where(actual_perc == 0, eps, actual_perc)

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return float(psi)


def ks_test_pvalue(expected: np.ndarray, actual: np.ndarray) -> float:
    """Return KS test p-value between two numeric arrays."""
    try:
        stat, pvalue = stats.ks_2samp(expected, actual)
        return float(pvalue)
    except Exception:
        return 0.0


def categorical_kl_divergence(expected: pd.Series, actual: pd.Series) -> float:
    """Compute KL divergence between categorical distributions (expected || actual)."""
    eps = 1e-8
    exp_counts = expected.value_counts(normalize=True)
    act_counts = actual.value_counts(normalize=True)
    all_index = exp_counts.index.union(act_counts.index)
    p = exp_counts.reindex(all_index, fill_value=0).values + eps
    q = act_counts.reindex(all_index, fill_value=0).values + eps
    return float(np.sum(p * np.log(p / q)))


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)
