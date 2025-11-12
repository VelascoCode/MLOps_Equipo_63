"""Drift simulation and detection script.

- Uses `data/processed/dataset_processed.csv` as baseline.
- Loads `models/final_model.pkl` for inference (user requested).
- Creates drifted datasets and writes all outputs to `data/monitoring/` (DVC trackable).

Run example:
    python -m mlops_equipo_63.drift_simulation --n-samples 2000

"""
import argparse
import os
import joblib
import json
from datetime import datetime
from typing import List, Dict, Any
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from mlops_equipo_63 import monitoring_utils as mu
from mlops_equipo_63.Split_and_Dummy import prepare_train_test


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# BASE_DIR now points to the repository root. Use paths relative to the repo root.
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed', 'dataset_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.pkl')
MONITORING_DIR = os.path.join(DATA_DIR, 'monitoring')
os.makedirs(MONITORING_DIR, exist_ok=True)


# thresholds (as requested defaults)
THRESHOLDS = {
    'roc_auc_abs_drop': 0.05,
    'roc_auc_rel_drop': 0.10,
    'accuracy_abs_drop': 0.05,
    'psi_feature_threshold': 0.1,
    'ks_pvalue_threshold': 0.01,
    'feature_count_alert': 3,
}


def load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_data(path: str, n_samples: int = None):
    df = pd.read_csv(path)
    if n_samples is not None and n_samples > 0 and len(df) > n_samples:
        df = df.sample(n_samples, random_state=42).reset_index(drop=True)
    return df


def baseline_split_and_threshold(df: pd.DataFrame):
    # Match behavior from project: dropna on target, median threshold to create 'popular'
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['shares'])
    threshold = df_clean['shares'].median()
    df_clean['popular'] = (df_clean['shares'] > threshold).astype(int)
    # We'll use the whole cleaned dataframe as baseline for monitoring calculations
    return df_clean, threshold


def evaluate_model(model, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
    result = {}
    try:
        y_pred = model.predict(X)
        result['accuracy'] = float(accuracy_score(y_true, y_pred))
    except Exception:
        result['accuracy'] = None

    # try predict_proba for roc_auc
    try:
        proba = model.predict_proba(X)
        if proba is not None and proba.shape[1] == 2:
            result['roc_auc'] = float(roc_auc_score(y_true, proba[:, 1]))
            # convert numpy array to list for JSON serialization
            try:
                result['y_proba'] = proba[:, 1].tolist()
            except Exception:
                result['y_proba'] = list(map(float, proba[:, 1]))
        else:
            result['roc_auc'] = None
    except Exception:
        result['roc_auc'] = None

    return result


def apply_mean_shift(df: pd.DataFrame, numeric_cols: List[str], delta: float) -> pd.DataFrame:
    df2 = df.copy()
    for c in numeric_cols:
        std = df2[c].std()
        df2[c] = df2[c] + delta * std
    return df2


def apply_missingness(df: pd.DataFrame, cols: List[str], rate: float) -> pd.DataFrame:
    df2 = df.copy()
    n = len(df2)
    for c in cols:
        mask = np.random.RandomState(42).rand(n) < rate
        df2.loc[mask, c] = np.nan
    return df2


def feature_drift_report(baseline: pd.DataFrame, current: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]):
    report = {}
    psi_map = {}
    ks_map = {}
    kl_map = {}

    for c in numeric_cols:
        try:
            psi = mu.population_stability_index(baseline[c].dropna().values, current[c].dropna().values)
            pval = mu.ks_test_pvalue(baseline[c].dropna().values, current[c].dropna().values)
        except Exception:
            psi = float('inf')
            pval = 0.0
        psi_map[c] = psi
        ks_map[c] = pval

    for c in cat_cols:
        try:
            kl = mu.categorical_kl_divergence(baseline[c].astype(str), current[c].astype(str))
        except Exception:
            kl = float('inf')
        kl_map[c] = kl

    report['psi'] = psi_map
    report['ks_pvalue'] = ks_map
    report['kl'] = kl_map
    return report


def detect_alerts(baseline_metrics: Dict[str, Any], current_metrics: Dict[str, Any], drift_report: Dict[str, Any]):
    alerts = []
    # performance
    if baseline_metrics.get('roc_auc') is not None and current_metrics.get('roc_auc') is not None:
        abs_drop = baseline_metrics['roc_auc'] - current_metrics['roc_auc']
        rel_drop = abs_drop / (baseline_metrics['roc_auc'] + 1e-8)
        if abs_drop >= THRESHOLDS['roc_auc_abs_drop'] or rel_drop >= THRESHOLDS['roc_auc_rel_drop']:
            alerts.append({'type': 'performance', 'metric': 'roc_auc', 'baseline': baseline_metrics['roc_auc'], 'current': current_metrics['roc_auc'], 'abs_drop': abs_drop, 'rel_drop': rel_drop})

    if baseline_metrics.get('accuracy') is not None and current_metrics.get('accuracy') is not None:
        abs_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
        if abs_drop >= THRESHOLDS['accuracy_abs_drop']:
            alerts.append({'type': 'performance', 'metric': 'accuracy', 'baseline': baseline_metrics['accuracy'], 'current': current_metrics['accuracy'], 'abs_drop': abs_drop})

    # feature drift
    psi_flags = [c for c, v in drift_report['psi'].items() if v > THRESHOLDS['psi_feature_threshold']]
    ks_flags = [c for c, p in drift_report['ks_pvalue'].items() if p < THRESHOLDS['ks_pvalue_threshold']]
    kl_flags = [c for c, v in drift_report['kl'].items() if v > 0.5]  # arbitrary

    feature_flags = set(psi_flags + ks_flags + kl_flags)
    if len(feature_flags) >= THRESHOLDS['feature_count_alert']:
        alerts.append({'type': 'feature_drift', 'features': list(feature_flags), 'count': len(feature_flags)})

    return alerts


def plot_feature_distribution(baseline: pd.Series, current: pd.Series, col: str, outdir: str):
    plt.figure(figsize=(6, 4))
    if mu.is_numeric_series(baseline):
        plt.hist(baseline.dropna(), bins=30, alpha=0.5, label='baseline', density=True)
        plt.hist(current.dropna(), bins=30, alpha=0.5, label='current', density=True)
    else:
        # categorical
        b = baseline.astype(str).value_counts(normalize=True)
        c = current.astype(str).value_counts(normalize=True)
        idx = sorted(set(b.index).union(set(c.index)))
        bvals = [b.get(i, 0) for i in idx]
        cvals = [c.get(i, 0) for i in idx]
        x = np.arange(len(idx))
        width = 0.35
        plt.bar(x - width/2, bvals, width=width, label='baseline')
        plt.bar(x + width/2, cvals, width=width, label='current')
        plt.xticks(x, idx, rotation=45, ha='right')

    plt.title(col)
    plt.legend()
    plt.tight_layout()
    fpath = os.path.join(outdir, f"feature_{col}.png")
    plt.savefig(fpath)
    plt.close()


def save_report_structure(outdir: str, summary: Dict[str, Any]):
    os.makedirs(outdir, exist_ok=True)
    # save JSON summary
    with open(os.path.join(outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)


def run(args):
    df = load_data(PROCESSED_PATH, args.n_samples)
    # use the project's train/test split to get a proper baseline test set
    X_train, X_test, y_train, y_test, threshold, df_clean = prepare_train_test(df, verbose=False)

    # numeric and categorical columns based on X_test
    numeric_cols = [c for c in X_test.columns if mu.is_numeric_series(X_test[c])]
    cat_cols = [c for c in X_test.columns if not mu.is_numeric_series(X_test[c])]

    model = load_model(MODEL_PATH)
    if model is None:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please place final_model.pkl under models/ and retry.")

    # baseline eval: evaluate on test split
    X_base = X_test
    y_base = y_test
    baseline_metrics = evaluate_model(model, X_base, y_base)

    # If a canonical metrics report exists (from training), prefer that as baseline to match project reports
    reports_metrics_path = os.path.join(BASE_DIR, 'reports', 'metrics.json')
    if os.path.exists(reports_metrics_path):
        try:
            with open(reports_metrics_path, 'r', encoding='utf-8') as f:
                rep = json.load(f)
            test_metrics = rep.get('test', {})
            # use reported final metrics if present
            if 'final_auc' in test_metrics or 'final_accuracy' in test_metrics:
                baseline_metrics = {
                    'roc_auc': test_metrics.get('final_auc', baseline_metrics.get('roc_auc')),
                    'accuracy': test_metrics.get('final_accuracy', baseline_metrics.get('accuracy'))
                }
        except Exception:
            pass

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_root = os.path.join(MONITORING_DIR, timestamp)
    os.makedirs(out_root, exist_ok=True)

    results = []
    dvc_results = []

    # mean shift scenarios
    for delta in args.mean_shifts:
        cur = apply_mean_shift(df_clean, numeric_cols, delta)
        X_cur = cur.drop(['shares', 'popular'], axis=1)
        y_cur = cur['popular']
        metrics = evaluate_model(model, X_cur, y_cur)
        drift_r = feature_drift_report(df_clean, cur, numeric_cols, cat_cols)
        alerts = detect_alerts(baseline_metrics, metrics, drift_r)

        # save dataset to data/monitoring
        fname = f"monitor_mean_shift_delta_{delta}.csv"
        cur.to_csv(os.path.join(out_root, fname), index=False)

        # save plots for top features (limit)
        plots_dir = os.path.join(out_root, f"plots_mean_shift_delta_{delta}")
        os.makedirs(plots_dir, exist_ok=True)
        for c in (numeric_cols[:10] + cat_cols[:5]):
            try:
                plot_feature_distribution(df_clean[c], cur[c], c, plots_dir)
            except Exception:
                pass

        # ROC overlay
        if 'y_proba' in baseline_metrics and 'y_proba' in metrics:
            pass
        else:
            # if we have probabilities for current
            if metrics.get('y_proba') is not None and baseline_metrics.get('y_proba') is not None:
                fpr_b, tpr_b, _ = roc_curve(y_base, baseline_metrics['y_proba'])
                fpr_c, tpr_c, _ = roc_curve(y_cur, metrics['y_proba'])
                plt.figure()
                plt.plot(fpr_b, tpr_b, label='baseline')
                plt.plot(fpr_c, tpr_c, label=f'delta_{delta}')
                plt.xlabel('fpr')
                plt.ylabel('tpr')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'roc_overlay.png'))
                plt.close()

        summary = {
            'scenario': 'mean_shift',
            'delta': delta,
            'metrics': metrics,
            'drift_report': drift_r,
            'alerts': alerts,
            'dataset': fname,
            'plots_dir': os.path.relpath(plots_dir, start=DATA_DIR)
        }
        results.append(summary)

    # missingness scenarios
    for rate in args.missing_rates:
        # for simplicity choose top numeric cols to introduce missingness
        cols = numeric_cols[:args.n_cols_missing]
        cur = apply_missingness(df_clean, cols, rate)
        X_cur = cur.drop(['shares', 'popular'], axis=1)
        y_cur = cur['popular']
        metrics = evaluate_model(model, X_cur, y_cur)
        drift_r = feature_drift_report(df_clean, cur, numeric_cols, cat_cols)
        alerts = detect_alerts(baseline_metrics, metrics, drift_r)

        fname = f"monitor_missing_rate_{int(rate*100)}.csv"
        cur.to_csv(os.path.join(out_root, fname), index=False)

        plots_dir = os.path.join(out_root, f"plots_missing_rate_{int(rate*100)}")
        os.makedirs(plots_dir, exist_ok=True)
        for c in (cols[:10] + cat_cols[:5]):
            try:
                plot_feature_distribution(df_clean[c], cur[c], c, plots_dir)
            except Exception:
                pass

        summary = {
            'scenario': 'missingness',
            'rate': rate,
            'affected_cols': cols,
            'metrics': metrics,
            'drift_report': drift_r,
            'alerts': alerts,
            'dataset': fname,
            'plots_dir': os.path.relpath(plots_dir, start=DATA_DIR)
        }
        results.append(summary)

    # Save overall summary
    save_report_structure(out_root, {
        'timestamp': timestamp,
        'baseline_metrics': baseline_metrics,
        'thresholds': THRESHOLDS,
        'results': results,
    })

    # create a lightweight markdown report
    md_lines = [f"# Drift Simulation Report - {timestamp}", "\n"]
    md_lines.append("## Baseline metrics")
    md_lines.append(json.dumps(baseline_metrics, indent=2))
    md_lines.append("\n## Scenarios\n")
    for r in results:
        md_lines.append(f"### {r['scenario']} - {r.get('delta', r.get('rate'))}")
        md_lines.append("Metrics:\n")
        md_lines.append(json.dumps(r['metrics'], indent=2))
        md_lines.append("Alerts:\n")
        md_lines.append(json.dumps(r['alerts'], indent=2))
        md_lines.append(f"Plots: {r['plots_dir']}\n")

    with open(os.path.join(out_root, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(md_lines))

        # save metrics summary CSV for easy ingestion
        metrics_rows = []
        for r in results:
            row = {
                'scenario': r.get('scenario'),
                'param': r.get('delta', r.get('rate')),
                'alerts': len(r.get('alerts', [])),
            }
            # flatten metrics if present
            for m_k, m_v in (r.get('metrics') or {}).items():
                if m_k != 'y_proba':
                    row[m_k] = m_v
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(out_root, 'metrics_summary.csv')
        metrics_df.to_csv(metrics_csv, index=False)

        # Optionally add artifacts to DVC
        dvc_status = {'attempted': False, 'results': []}
        if args.dvc_add:
            dvc_status['attempted'] = True
            for path in dvc_results + [os.path.join(out_root, 'report.md'), metrics_csv, os.path.join(out_root, 'summary.json')]:
                try:
                    # run dvc add; don't fail the whole run if dvc is missing
                    cp = subprocess.run(['dvc', 'add', path], capture_output=True, text=True)
                    ok = (cp.returncode == 0)
                    dvc_status['results'].append({'path': path, 'ok': ok, 'stdout': cp.stdout, 'stderr': cp.stderr})
                except FileNotFoundError as e:
                    dvc_status['results'].append({'path': path, 'ok': False, 'error': 'dvc-not-found'})
                    break

        # attach dvc status to summary
        with open(os.path.join(out_root, 'dvc_add_status.json'), 'w', encoding='utf-8') as f:
            json.dump(dvc_status, f, indent=2, default=str)

        print(f"Monitoring artifacts written to: {out_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--mean-shifts', type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0])
    parser.add_argument('--missing-rates', type=float, nargs='+', default=[0.05, 0.2, 0.5])
    parser.add_argument('--n-cols-missing', type=int, default=3)
    parser.add_argument('--dvc-add', dest='dvc_add', action='store_true', help='Automatically dvc add generated artifacts', default=True)
    args = parser.parse_args()
    run(args)
