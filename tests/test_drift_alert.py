# tests/test_drift_alerts.py
import numpy as np
from mlops_equipo_63.monitoring.drift import PerfMetrics, drift_alert

def test_drift_alert_basic():
    # PSI severo en una feature + caída de F1
    psi = {"x1": 0.35, "x2": 0.05}
    ks = {"x1": {"stat": 0.3, "pvalue": 0.01}, "x2": {"stat": 0.05, "pvalue": 0.6}}
    baseline = PerfMetrics(accuracy=0.90, f1=0.88, precision=0.9, recall=0.86, auc=0.92)
    current  = PerfMetrics(accuracy=0.84, f1=0.80, precision=0.85, recall=0.78, auc=0.89)
    alert, reasons = drift_alert(psi, ks, baseline, current)
    assert alert is True
    assert any("PSI crítico" in r for r in reasons) or any("KS p<" in r for r in reasons)