"""
Script de evaluación con MLflow tracking para DVC pipeline.
Evalúa el modelo entrenado, genera reportes visuales y registra resultados.
"""
import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar raíz del proyecto al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops_equipo_63.preprocessing import split_data
from mlops_equipo_63.evaluation import evaluate_model


# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    """Carga parámetros desde archivo YAML."""
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
    """Configura MLflow con tracking URI y experimento."""
    try:
        mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
        mlflow.set_experiment(params['mlflow']['experiment_name'])
        logger.info(f"MLflow configurado: {params['mlflow']['experiment_name']}")
    except KeyError as e:
        logger.error(f"Parámetro MLflow faltante: {e}")
        raise


def load_data_and_model(
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    Carga datos procesados y modelo entrenado.
    
    Returns:
        Tupla (X_train, X_test, y_train, y_test, model)
    """
    data_path = 'data/processed/cleaned_data.csv'
    model_path = 'models/best_model.pkl'
    
    try:
        # Validar existencia de archivos
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Datos no encontrados: {data_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Cargar datos
        df = pd.read_csv(data_path)
        logger.info(f"Datos cargados: {df.shape}")
        
        # Cargar modelo
        model = joblib.load(model_path)
        logger.info(f"Modelo cargado desde {model_path}")
        
        # Dividir datos usando parámetros de params.yaml
        X_train, X_test, y_train, y_test = split_data(
            df,
            test_size=params['preprocessing']['test_size'],
            random_state=params['training']['random_state']
        )
        
        return X_train, X_test, y_train, y_test, model
        
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error al cargar datos/modelo: {e}")
        raise


def save_evaluation_results(metrics: Dict[str, float], output_path: str = 'reports/evaluation_results.json') -> None:
    """Guarda resultados de evaluación en JSON."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Resultados guardados en {output_path}")
        
    except Exception as e:
        logger.error(f"Error al guardar resultados: {e}")
        raise


def log_artifacts_to_mlflow(figures_path: Path) -> None:
    """Registra artefactos visuales en MLflow."""
    try:
        artifacts = [
            figures_path / 'confusion_matrix.png',
            figures_path / 'roc_curve.png'
        ]
        
        for artifact_path in artifacts:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
                logger.info(f"Artefacto registrado: {artifact_path.name}")
            else:
                logger.warning(f"Artefacto no encontrado: {artifact_path}")
        
    except Exception as e:
        logger.error(f"Error al registrar artefactos: {e}")
        raise


# =====================================================================
# FUNCIÓN PRINCIPAL
# =====================================================================

def main():
    """
    Función principal que ejecuta el pipeline de evaluación completo:
    1. Carga parámetros y configura MLflow
    2. Carga datos y modelo
    3. Evalúa el modelo
    4. Registra métricas y artefactos en MLflow
    5. Guarda resultados localmente para DVC
    """
    logger.info("="*70)
    logger.info("STAGE 3: EVALUACIÓN DEL MODELO")
    logger.info("="*70)
    
    try:
        # 1. Cargar configuración
        params = load_params()
        
        # 2. Configurar MLflow
        setup_mlflow(params)
        
        # 3. Cargar datos y modelo
        X_train, X_test, y_train, y_test, model = load_data_and_model(params)
        
        # 4. Iniciar run de MLflow
        with mlflow.start_run(run_name="DVC_Pipeline_Evaluation"):
            
            # Tags de contexto
            mlflow.set_tag("dvc_stage", "evaluate_model")
            mlflow.set_tag("test_samples", len(y_test))
            logger.info("MLflow run iniciado")
            
            # 5. Evaluación del modelo
            logger.info("Iniciando evaluación del modelo...")
            figures_path = Path('reports/figures')
            
            metrics = evaluate_model(
                model, 
                X_test, 
                y_test,
                show_plots=False,
                save_path=str(figures_path)
            )
            
            # 6. Log de métricas en MLflow
            mlflow.log_metrics(metrics)
            logger.info("Métricas registradas en MLflow")
            
            # 7. Log de artefactos visuales
            log_artifacts_to_mlflow(figures_path)
            
            # 8. Guardar resultados para DVC
            save_evaluation_results(metrics)
        
        logger.info("="*70)
        logger.info("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de evaluación: {e}")
        raise


if __name__ == "__main__":
    main()
