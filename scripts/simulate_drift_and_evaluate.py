from __future__ import annotations

# --- HACK: asegurar que la raíz del repo esté en sys.path ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------

import argparse
import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

from mlops_equipo_63.monitoring.drift import (
    simulate_mean_shift,
    simulate_missingness,
    simulate_scale_change,
    compute_feature_drift,
    evaluate_performance,
    drift_alert,
)


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    path.write_text(data, encoding="utf-8")

def main() -> None:
    print("[DEBUG] Entré a main() de simulate_drift_and_evaluate.py")
    parser = argparse.ArgumentParser(
        description="Simulación de data drift y evaluación de performance."
    )
    parser.add_argument(
        "--ref_csv",
        type=str,
        required=True,
        help="Ruta al CSV de referencia (ej. data/processed/valid.csv).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="shares",
        help="Nombre de la columna etiqueta.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/final_model.pkl",
        help="Ruta al modelo entrenado (.pkl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/drift",
        help="Directorio base para guardar reportes.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.6,
        help="Factor de desplazamiento de media (en múltiplos de la desviación estándar).",
    )
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.08,
        help="Proporción de filas donde se inyectan NaNs en variables numéricas.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.2,
        help="Factor de escala para simulate_scale_change.",
    )

    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Usando ref_csv={args.ref_csv}")
    print(f"[INFO] Usando model_path={args.model_path}")
    print(f"[INFO] Reportes se guardarán en {out_dir}")

    # 1) Cargar referencia (baseline)
    ref_df = pd.read_csv(args.ref_csv)

    if args.label_col not in ref_df.columns:
        raise ValueError(
            f"La columna label '{args.label_col}' no está en {args.ref_csv}. "
            f"Columnas disponibles: {list(ref_df.columns)}"
        )

    y_ref = ref_df[args.label_col].astype(int).values
    X_ref = ref_df.drop(columns=[args.label_col])

    # 2) Cargar modelo entrenado
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. Ajusta --model_path."
        )

    model = joblib.load(model_path)

    # 3) Definir features (usamos las columnas de X_ref)
    feature_names: List[str] = list(X_ref.columns)

    # 4) Simular drift sobre X_ref
    X_drift = simulate_mean_shift(X_ref, shift=args.shift)
    X_drift = simulate_scale_change(X_drift, scale=args.scale)
    X_drift = simulate_missingness(X_drift, missing_rate=args.missing_rate)

    # 5) Guardar dataset drifteado como "conjunto de monitoreo"
    drift_csv_path = out_dir / "monitoring_drifted.csv"
    X_drift_with_label = X_drift.copy()
    X_drift_with_label[args.label_col] = y_ref
    X_drift_with_label.to_csv(drift_csv_path, index=False)
    print(f"[INFO] Dataset de monitoreo con drift guardado en {drift_csv_path}")

     # 6) Predicciones y métricas en baseline y drift
    # ---------------------------------------------
    # === BINARIZAR shares para clasificación ===
    # Usamos un umbral consistente (ej. percentil 90)
    threshold = 1400
    print(f"[INFO] Umbral para clasificación = {threshold:.2f}")

    # Etiqueta binaria
    y_true_bin = (y_ref >= threshold).astype(int)

    # === BASELINE (sin drift) ===
    y_pred_ref_raw = model.predict(X_ref)

    # Si el modelo devuelve valores continuos, los binarizamos con el mismo umbral
    if isinstance(y_pred_ref_raw, (np.ndarray, list)):
        y_pred_ref_raw = np.asarray(y_pred_ref_raw)
    if y_pred_ref_raw.dtype.kind in ["f", "i"]:
        y_pred_ref_bin = (y_pred_ref_raw >= threshold).astype(int)
    else:
        # Caso raro en que el modelo ya devuelve 0/1
        y_pred_ref_bin = y_pred_ref_raw

    # Probabilidades si el modelo las soporta (para AUC)
    if hasattr(model, "predict_proba"):
        proba_ref = model.predict_proba(X_ref)
        # usamos la probabilidad de la clase positiva
        y_proba_ref = proba_ref[:, 1]
    else:
        y_proba_ref = None

    # Usamos la función general de métricas (PerfMetrics)
    baseline_perf = evaluate_performance(y_true_bin, y_pred_ref_bin, y_proba_ref)

    # === DRIFT (dataset drifteado) ===
    X_drift = X_drift.reindex(columns=feature_names, fill_value=np.nan)
    y_pred_drift_raw = model.predict(X_drift)
    if isinstance(y_pred_drift_raw, (np.ndarray, list)):
        y_pred_drift_raw = np.asarray(y_pred_drift_raw)
    if y_pred_drift_raw.dtype.kind in ["f", "i"]:
        y_pred_drift_bin = (y_pred_drift_raw >= threshold).astype(int)
    else:
        y_pred_drift_bin = y_pred_drift_raw

    if hasattr(model, "predict_proba"):
        proba_drift = model.predict_proba(X_drift)
        y_proba_drift = proba_drift[:, 1]
    else:
        y_proba_drift = None

    current_perf = evaluate_performance(y_true_bin, y_pred_drift_bin, y_proba_drift)

    # Imprimimos métricas en formato JSON-friendly
    print("[INFO] Métricas baseline:")
    print(json.dumps(baseline_perf.__dict__, indent=2, default=float))

    print("[INFO] Métricas con drift:")
    print(json.dumps(current_perf.__dict__, indent=2, default=float))

    # 7) Métricas de drift por feature (PSI/KS)
    psi_by_feat, ks_by_feat = compute_feature_drift(X_ref, X_drift, cols=feature_names)

    # 8) Alerta de drift
    alert, reasons = drift_alert(
        psi_by_feat,
        ks_by_feat,
        baseline=baseline_perf,
        current=current_perf,
    )

    # 9) Guardar reportes en JSON
    _save_json({"psi": psi_by_feat}, out_dir / "feature_psi.json")
    _save_json({"ks": ks_by_feat}, out_dir / "feature_ks.json")
    _save_json(
        {
            "baseline_perf": baseline_perf.__dict__,
            "current_perf": current_perf.__dict__,
            "alert": alert,
            "reasons": reasons,
        },
        out_dir / "summary.json",
    )

    print("[INFO] Reporte de drift guardado en:")
    print(f"   {out_dir / 'feature_psi.json'}")
    print(f"   {out_dir / 'feature_ks.json'}")
    print(f"   {out_dir / 'summary.json'}")

    if alert:
        print("\n[ALERTA] Se detectó data drift.")
        for r in reasons:
            print(f"  - {r}")
        print(
            "\n[ACCIÓN SUGERIDA] Revisar las features con PSI/KS más altos y considerar "
            "reentrenar el modelo con datos recientes."
        )
    else:
        print("\n[OK] No se detectó drift significativo según los umbrales configurados.")


if __name__ == "__main__":
    main()