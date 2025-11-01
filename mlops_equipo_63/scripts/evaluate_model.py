"""
Script de evaluación con MLflow tracking para DVC pipeline.
"""
import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops_equipo_63.preprocessing import split_data
from mlops_equipo_63.evaluation import evaluate_model
import yaml

def main():
    print("\n" + "="*70)
    print("STAGE 3: EVALUACIÓN DEL MODELO")
    print("="*70)
    
    # Cargar parámetros
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Cargar datos y modelo
    df = pd.read_csv('data/processed/cleaned_data.csv')
    model = joblib.load('models/best_model.pkl')
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Iniciar run de MLflow
    with mlflow.start_run(run_name="DVC_Pipeline_Evaluation"):
        
        mlflow.set_tag("dvc_stage", "evaluate_model")
        mlflow.set_tag("git_commit", "HEAD")
        
        # Evaluación del modelo
        save_path = Path('reports/figures')
        metrics = evaluate_model(model, X_test, y_test, 
                                show_plots=False, 
                                save_path=save_path)
        
        # Log de métricas en MLflow
        mlflow.log_metrics(metrics)
        
        # Log de artefactos visuales
        mlflow.log_artifact('reports/figures/confusion_matrix.png')
        mlflow.log_artifact('reports/figures/roc_curve.png')
        
        print("✓ Artefactos registrados en MLflow")
        
        # Guardar resultados para DVC
        with open('reports/evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("✓ Resultados guardados en reports/evaluation_results.json")
    
    print("="*70)

if __name__ == "__main__":
    main()