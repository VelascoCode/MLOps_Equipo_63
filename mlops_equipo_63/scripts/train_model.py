"""
Script de entrenamiento con MLflow tracking para DVC pipeline.
"""
import sys
import yaml
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.utils.requirements_utils")
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops_equipo_63.preprocessing import split_data
from mlops_equipo_63.hyperparameter_optimization import run_optuna_optimization
from mlops_equipo_63.model_training import train_final_model
from sklearn.metrics import roc_auc_score, accuracy_score

def main():
    print("\n" + "="*70)
    print("STAGE 2: ENTRENAMIENTO DEL MODELO")
    print("="*70)
    
    # Cargar parámetros
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Cargar datos procesados
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Dividir datos
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=params['preprocessing']['test_size'],
        random_state=params['training']['random_state']
    )
    
    # Iniciar run de MLflow
    with mlflow.start_run(run_name="DVC_Pipeline_Training"):
        
        # Log de parámetros
        mlflow.log_params(params['preprocessing'])
        mlflow.log_params(params['training'])
        mlflow.set_tag("dvc_stage", "train_model")
        mlflow.set_tag("git_commit", "HEAD")  # Se registra automáticamente si hay git
        
        print("✓ MLflow run iniciado")
        
        # Optimización con Optuna
        study = run_optuna_optimization(
            X_train, y_train,
            n_trials=params['training']['n_trials'],
            cv_folds=params['training']['cv_folds']
        )
        
        # Log de mejores hiperparámetros
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_cv_auc", study.best_trial.value)
        mlflow.log_metric("best_cv_accuracy",study.best_trial.user_attrs.get('accuracy', 0))
        
        # Entrenamiento del modelo final
        final_model = train_final_model(X_train, y_train, study.best_trial.params)
        
        # Evaluación en test set
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Log de métricas de test
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        print(f"\nMétricas en Test Set:")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        
        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(
            final_model,
            "model",
            registered_model_name="NewsPopularityModel"
        )
        
        print("✓ Modelo registrado en MLflow")
        
        # Guardar modelo localmente para DVC
        models_path = Path('models')
        models_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(final_model, 'models/best_model.pkl')
        print("✓ Modelo guardado en models/best_model.pkl")
        
        # Guardar métricas para DVC
        training_metrics = {
            'best_cv_auc': float(study.best_trial.value),
            'best_cv_accuracy': float(study.best_trial.user_attrs.get('accuracy', 0)),
            'test_auc': float(test_auc),
            'test_accuracy': float(test_accuracy),
            'best_params': study.best_trial.params,
            'n_trials': len(study.trials)
        }
        
        reports_path = Path('reports')
        reports_path.mkdir(parents=True, exist_ok=True)
        
        with open('reports/metrics.json', 'w') as f:
            json.dump(training_metrics, f, indent=4)
        
        print("✓ Métricas guardadas en reports/metrics.json")
    
    print("="*70)

if __name__ == "__main__":
    main()