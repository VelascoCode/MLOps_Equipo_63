from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ----------------------------------------------------------------------
# Estructura para guardar métricas de performance
# ----------------------------------------------------------------------
@dataclass
class PerfMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc: float | None


# ----------------------------------------------------------------------
# Funciones para simular drift en los datos
# ----------------------------------------------------------------------
def simulate_mean_shift(df: pd.DataFrame, shift: float = 0.5) -> pd.DataFrame:
    """Aplica un desplazamiento de media a las columnas numéricas."""
    df_shifted = df.copy()
    num_cols = df_shifted.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        std = df_shifted[col].std()
        df_shifted[col] = df_shifted[col] + shift * std

    return df_shifted


def simulate_scale_change(df: pd.DataFrame, scale: float = 1.2) -> pd.DataFrame:
    """Escala las columnas numéricas por un factor fijo."""
    df_scaled = df.copy()
    num_cols = df_scaled.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        df_scaled[col] = df_scaled[col] * scale

    return df_scaled


def simulate_missingness(df: pd.DataFrame, missing_rate: float = 0.08) -> pd.DataFrame:
    """Inyecta NaNs aleatorios en columnas numéricas."""
    df_missing = df.copy()
    num_cols = df_missing.select_dtypes(include=[np.number]).columns

    n_rows = len(df_missing)
    n_to_nan = int(n_rows * missing_rate)

    rng = np.random.default_rng(42)

    for col in num_cols:
        if n_to_nan == 0:
            continue
        idx = rng.choice(n_rows, size=n_to_nan, replace=False)
        df_missing.loc[idx, col] = np.nan

    return df_missing


# ----------------------------------------------------------------------
# Cálculo de performance (binario o multiclase)
# ----------------------------------------------------------------------
def evaluate_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> PerfMetrics:
    """
    Calcula métricas de performance. Soporta casos binarios y multiclase.

    - Si hay 2 clases -> average='binary'
    - Si hay >2 clases -> average='weighted'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = np.unique(y_true).size
    if n_classes <= 2:
        avg = "binary"
    else:
        avg = "weighted"

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)

    auc: float | None = None
    if y_proba is not None:
        try:
            if n_classes <= 2:
                # Usamos la prob de la clase positiva
                if y_proba.ndim == 1:
                    proba_pos = y_proba
                else:
                    proba_pos = y_proba[:, 1]
                auc = roc_auc_score(y_true, proba_pos)
            else:
                # Multiclase
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            auc = None

    return PerfMetrics(
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        auc=auc,
    )


# ----------------------------------------------------------------------
# Cálculo de drift por feature (PSI + KS)
# ----------------------------------------------------------------------
def _psi_single(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index para una variable."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Quitamos NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.unique(np.percentile(expected, quantiles))

    if len(bins) <= 2:
        return 0.0

    expected_bins = np.histogram(expected, bins=bins)[0].astype(float)
    actual_bins = np.histogram(actual, bins=bins)[0].astype(float)

    expected_perc = expected_bins / (expected_bins.sum() + 1e-12)
    actual_perc = actual_bins / (actual_bins.sum() + 1e-12)

    psi = np.sum(
        (expected_perc - actual_perc)
        * np.log((expected_perc + 1e-12) / (actual_perc + 1e-12))
    )
    return float(psi)


def compute_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    cols: List[str] | None = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calcula PSI y KS para cada columna numérica.

    Devuelve:
        - dict feature -> PSI
        - dict feature -> KS statistic
    """
    if cols is None:
        cols = list(ref_df.columns)

    psi_scores: Dict[str, float] = {}
    ks_scores: Dict[str, float] = {}

    for col in cols:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue

        ref = ref_df[col].values
        cur = cur_df[col].values

        # Sólo numérico
        if not np.issubdtype(ref.dtype, np.number):
            continue

        psi_scores[col] = _psi_single(ref, cur)

        # KS
        ref_clean = ref[~np.isnan(ref)]
        cur_clean = cur[~np.isnan(cur)]
        if len(ref_clean) > 0 and len(cur_clean) > 0:
            ks_stat, _ = ks_2samp(ref_clean, cur_clean)
            ks_scores[col] = float(ks_stat)
        else:
            ks_scores[col] = 0.0

    return psi_scores, ks_scores


# ----------------------------------------------------------------------
# Lógica de alerta de drift
# ----------------------------------------------------------------------
def drift_alert(
    psi_by_feat: Dict[str, float],
    ks_by_feat: Dict[str, float],
    baseline: PerfMetrics,
    current: PerfMetrics,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.1,
    perf_drop_threshold: float = 0.05,
) -> Tuple[bool, List[str]]:
    """
    Regresa:
      - alert: bool
      - reasons: lista de strings explicando por qué
    """
    reasons: List[str] = []
    alert = False

    # 1) Drift en features (PSI / KS)
    strong_psi = [f for f, v in psi_by_feat.items() if v >= psi_threshold]
    strong_ks = [f for f, v in ks_by_feat.items() if v >= ks_threshold]

    if strong_psi:
        alert = True
        reasons.append(
            f"PSI alto (≥ {psi_threshold}) en features: {', '.join(strong_psi)}"
        )
    if strong_ks:
        alert = True
        reasons.append(
            f"KS alto (≥ {ks_threshold}) en features: {', '.join(strong_ks)}"
        )

    # 2) Caída de performance
    if baseline.accuracy > 0:
        acc_drop = baseline.accuracy - current.accuracy
        if acc_drop >= perf_drop_threshold:
            alert = True
            reasons.append(
                f"Caída de accuracy de {acc_drop:.3f} (threshold={perf_drop_threshold})."
            )

    if baseline.f1 > 0:
        f1_drop = baseline.f1 - current.f1
        if f1_drop >= perf_drop_threshold:
            alert = True
            reasons.append(
                f"Caída de F1 de {f1_drop:.3f} (threshold={perf_drop_threshold})."
            )

    return alert, reasons