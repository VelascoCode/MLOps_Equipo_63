from mlops_equipo_63 import reporting
import pandas as pd


def test_print_non_numeric(capsys):
    reporting.print_non_numeric(['a', 'b', 'c'])
    captured = capsys.readouterr()
    assert 'Columnas no numéricas' in captured.out


def test_print_missing_info_empty(capsys):
    reporting.print_missing_info(pd.Series(dtype=float))
    captured = capsys.readouterr()
    assert '(sin valores faltantes)' in captured.out


def test_print_outlier_summary():
    reporting.print_outlier_summary(12.3456)


def test_print_baseline_and_best_and_final(capsys):
    reporting.print_baseline({'accuracy': 0.5})
    reporting.print_best_summary({'best_value': 0.6})
    reporting.print_final_metrics({'final_accuracy': 0.7})
    captured = capsys.readouterr()
    assert 'Rendimiento base' in captured.out
