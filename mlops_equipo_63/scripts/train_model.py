"""
Script de entrenamiento con MLflow tracking para DVC pipeline.
Optimiza hiperparámetros, entrena el modelo final y registra métricas.
"""
import sys
import yaml
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.utils.requirements_utils")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar la raíz del proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops_equipo_63.preprocessing import split_data
from mlops_equipo_63.hyperparameter_optimization import run_optuna_optimization
from mlops_equipo_63.model_training import train_final_model
from sklearn.metrics import roc_auc_score, accuracy_score


def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    """Carga parámetros de configuración desde archivo YAML."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Parámetros cargados desde {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Archivo de parámetros no encontrado: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear YAML: {e}")
        raise


def setup_mlflow(params: Dict[str, Any]) -> None:
    """Configura MLflow con tracking URI y nombre de experimento."""
    try:
        mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
        mlflow.set_experiment(params['mlflow']['experiment_name'])
        logger.info(f"MLflow configurado: {params['mlflow']['experiment_name']}")
    except KeyError as e:
        logger.error(f"Parámetro MLflow faltante: {e}")
        raise


def load_and_split_data(params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga datos procesados y los divide en train/test."""
    data_path = 'data/processed/cleaned_data.csv'
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        X_train, X_test, y_train, y_test = split_data(
            df,
            test_size=params['preprocessing']['test_size'],
            random_state=params['training']['random_state']
        )
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        logger.error(f"Archivo de datos no encontrado: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar/dividir datos: {e}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evalúa el modelo en el conjunto de prueba."""
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Evaluación completada - AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
        
        return {
            'test_auc': float(test_auc),
            'test_accuracy': float(test_accuracy)
        }
    except Exception as e:
        logger.error(f"Error al evaluar modelo: {e}")
        raise


def save_model_and_metrics(model, study, test_metrics: Dict[str, float]) -> None:
    """Guarda el modelo y métricas localmente y en MLflow."""
    try:
        # Crear directorios necesarios
        Path('models').mkdir(parents=True, exist_ok=True)
        Path('reports').mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo localmente
        model_path = 'models/best_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado en {model_path}")
        
        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="NewsPopularityModel"
        )
        logger.info("Modelo registrado en MLflow")
        
        # Guardar métricas
        training_metrics = {
            'best_cv_auc': float(study.best_trial.value),
            'best_cv_accuracy': float(study.best_trial.user_attrs.get('accuracy', 0)),
            'test_auc': test_metrics['test_auc'],
            'test_accuracy': test_metrics['test_accuracy'],
            'best_params': study.best_trial.params,
            'n_trials': len(study.trials)
        }
        
        metrics_path = 'reports/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=4)
        logger.info(f"Métricas guardadas en {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error al guardar modelo/métricas: {e}")
        raise


def main():
    """
    Función principal que ejecuta el pipeline de entrenamiento completo:
    1. Carga parámetros y configura MLflow
    2. Carga y divide datos
    3. Optimiza hiperparámetros
    4. Entrena modelo final
    5. Evalúa y guarda resultados
    """
    logger.info("="*70)
    logger.info("STAGE 2: ENTRENAMIENTO DEL MODELO")
    logger.info("="*70)
    
    try:
        # 1. Cargar configuración
        params = load_params()
        
        # 2. Configurar MLflow
        setup_mlflow(params)
        
        # 3. Cargar y dividir datos
        X_train, X_test, y_train, y_test = load_and_split_data(params)
        
        # 4. Iniciar run de MLflow
        with mlflow.start_run(run_name="Pipeline_Training"):
            
            # Log de parámetros
            mlflow.log_params(params['preprocessing'])
            mlflow.log_params(params['training'])
            mlflow.set_tag("dvc_stage", "train_model")
            logger.info("MLflow run iniciado")
            
            # 5. Optimización de hiperparámetros
            logger.info("Iniciando optimización de hiperparámetros...")
            study = run_optuna_optimization(X_train, y_train)
            
            # Log mejores parámetros
            mlflow.log_params(study.best_trial.params)
            mlflow.log_metric("best_cv_auc", study.best_trial.value)
            mlflow.log_metric("best_cv_accuracy", study.best_trial.user_attrs.get('accuracy', 0))
            
            # 6. Entrenar modelo final
            logger.info("Entrenando modelo final...")
            final_model = train_final_model(X_train, y_train, study.best_trial.params)
            
            # 7. Evaluar modelo
            test_metrics = evaluate_model(final_model, X_test, y_test)
            
            # Log métricas de test
            mlflow.log_metric("test_auc", test_metrics['test_auc'])
            mlflow.log_metric("test_accuracy", test_metrics['test_accuracy'])
            
            # 8. Guardar todo
            save_model_and_metrics(final_model, study, test_metrics)
        
        logger.info("="*70)
        logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()
