import pandas as pd
import pytest

from mlops_equipo_63 import reporting


def test_print_missing_info_none_and_empty(capsys):
    # None -> prints (sin valores faltantes)
    reporting.print_missing_info(None)
    out = capsys.readouterr().out
    assert "(sin valores faltantes)" in out

    # empty Series -> same message
    s = pd.Series(dtype=float)
    reporting.print_missing_info(s)
    out = capsys.readouterr().out
    assert "(sin valores faltantes)" in out


def test_print_missing_info_with_values(capsys):
    s = pd.Series({"a": 0.12, "b": 0.0})
    reporting.print_missing_info(s)
    out = capsys.readouterr().out
    assert "a" in out


def test_print_outlier_summary_numeric_and_bad(capsys):
    reporting.print_outlier_summary(12.3456)
    out = capsys.readouterr().out
    assert "12.35%" in out or "12.35" in out

    reporting.print_outlier_summary("not-a-number")
    out = capsys.readouterr().out
    assert "(valor no numérico)" in out


def test_print_baseline_and_best_and_final(capsys):
    # baseline with non-dict
    reporting.print_baseline("just a string")
    out = capsys.readouterr().out
    assert "just a string" in out

    # baseline with dict including float and non-float
    metrics = {"auc": 0.87654321, "report": "ok"}
    reporting.print_baseline(metrics)
    out = capsys.readouterr().out
    assert "auc: 0.8765" in out or "auc:" in out

    # best summary and final metrics
    best = {"best_value": 0.99, "n_trials": 5}
    reporting.print_best_summary(best)
    out = capsys.readouterr().out
    assert "best_value" in out and "n_trials" in out

    final = {"final_auc": 0.91, "classification_report": "rpt"}
    reporting.print_final_metrics(final)
    out = capsys.readouterr().out
    assert "final_auc" in out and "classification_report" in out
