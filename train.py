# train.py (raíz del repo)
import sys
import random
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlops_equipo_63.Configuration import Config
from mlops_equipo_63.pipeline import MLOpsPipeline
import json, yaml

def main():
    # Cargar configuración desde params.yaml (con overrides de env vars)
    cfg = Config.from_params("params.yaml")
    
    # Establecer semillas globales una única vez al inicio
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    
    # Ejecutar pipeline con la configuración
    pipe = MLOpsPipeline(cfg).run_all(show_eda=False)

    # === Persistir artefactos para DVC ===
    # 1) Dataset procesado
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if pipe.df_clipped is not None:
        (processed_dir / "dataset_processed.csv").write_text(
            pipe.df_clipped.to_csv(index=False, lineterminator="\n"),
            encoding="utf-8"
        )

    # 2) Métricas (CV + Test + params)
    Path("reports").mkdir(exist_ok=True)
    metrics = {
        "cv": {
            "roc_auc": pipe.best_summary.get("best_value") if pipe.best_summary else None,
            "accuracy": pipe.best_summary.get("cv_accuracy") if pipe.best_summary else None,
        },
        "test": {
            "final_auc": pipe.final_metrics.get("final_auc") if pipe.final_metrics else None,
            "final_accuracy": pipe.final_metrics.get("final_accuracy") if pipe.final_metrics else None,
        },
        "best_params": pipe.best_summary.get("best_params") if pipe.best_summary else None,
    }
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()