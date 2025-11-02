"""
Módulo para entrenamiento de modelos.
Construye pipelines con los mejores hiperparámetros encontrados por Optuna.
"""
import yaml
import logging
from typing import Dict, Any, Callable
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# FUNCIONES PARA CREAR CLASIFICADORES
# =====================================================================

def create_random_forest_from_params(params: Dict[str, Any], random_state: int) -> RandomForestClassifier:
    """Crea clasificador Random Forest desde parámetros optimizados."""
    model_params = {k.replace('rf_', ''): v for k, v in params.items() if k.startswith('rf_')}
    return RandomForestClassifier(random_state=random_state, n_jobs=-1, **model_params)


def create_mlp_from_params(params: Dict[str, Any], random_state: int) -> MLPClassifier:
    """Crea clasificador MLP desde parámetros optimizados."""
    model_params = {k.replace('mlp_', ''): v for k, v in params.items() if k.startswith('mlp_')}
    
    # Manejar hidden_layers especialmente
    if 'hidden_layers' in model_params:
        model_params['hidden_layer_sizes'] = eval(model_params.pop('hidden_layers'))
    
    return MLPClassifier(
        random_state=random_state,
        max_iter=300,
        early_stopping=True,
        **model_params
    )


def create_xgboost_from_params(params: Dict[str, Any], random_state: int) -> xgb.XGBClassifier:
    """Crea clasificador XGBoost desde parámetros optimizados."""
    model_params = {k.replace('xgb_', ''): v for k, v in params.items() if k.startswith('xgb_')}
    return xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
        **model_params
    )


def create_lightgbm_from_params(params: Dict[str, Any], random_state: int) -> lgb.LGBMClassifier:
    """Crea clasificador LightGBM desde parámetros optimizados."""
    model_params = {k.replace('lgbm_', ''): v for k, v in params.items() if k.startswith('lgbm_')}
    return lgb.LGBMClassifier(
        objective='binary',
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **model_params
    )


# Factory de clasificadores
CLASSIFIER_BUILDERS: Dict[str, Callable] = {
    'RandomForest': create_random_forest_from_params,
    'MLP': create_mlp_from_params,
    'XGBoost': create_xgboost_from_params,
    'LightGBM': create_lightgbm_from_params
}


# =====================================================================
# CONSTRUCCIÓN DE PIPELINE
# =====================================================================

def create_preprocessor(X_train) -> ColumnTransformer:
    """
    Crea el preprocesador para columnas numéricas.
    
    Args:
        X_train: DataFrame de entrenamiento.
        
    Returns:
        ColumnTransformer configurado.
    """
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    logger.info(f"Pipeline configurado para {len(numeric_cols)} columnas numéricas")
    
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols)
        ],
        remainder='drop'
    )


def build_classifier(best_params: Dict[str, Any], random_state: int):
    """
    Construye el clasificador según los parámetros optimizados.
    
    Args:
        best_params: Diccionario con mejores parámetros de Optuna.
        random_state: Semilla para reproducibilidad.
        
    Returns:
        Clasificador configurado.
        
    Raises:
        ValueError: Si el clasificador no es reconocido.
    """
    classifier_name = best_params.get('classifier')
    
    if classifier_name not in CLASSIFIER_BUILDERS:
        raise ValueError(f"Clasificador desconocido: {classifier_name}")
    
    # Usar factory para crear el clasificador
    classifier = CLASSIFIER_BUILDERS[classifier_name](best_params, random_state)
    logger.info(f"Clasificador creado: {classifier_name}")
    
    return classifier


def build_pipeline_from_params(
    best_params: Dict[str, Any], 
    X_train,
    params_path: str = 'params.yaml'
) -> Pipeline:
    """
    Construye un pipeline completo con preprocesamiento y clasificador.
    
    Args:
        best_params: Diccionario con los mejores parámetros de Optuna.
        X_train: DataFrame de entrenamiento.
        params_path: Ruta al archivo de parámetros.
    
    Returns:
        Pipeline configurado y listo para entrenar.
    """
    try:
        # Cargar random_state desde params.yaml
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        random_state = params.get('training', {}).get('random_state', 42)
        
        # Crear preprocesador
        preprocessor = create_preprocessor(X_train)
        
        # Crear clasificador
        classifier = build_classifier(best_params, random_state)
        
        # Armar pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        logger.info("Pipeline construido exitosamente")
        return pipeline
        
    except FileNotFoundError:
        logger.error(f"Archivo de parámetros no encontrado: {params_path}")
        raise
    except KeyError as e:
        logger.error(f"Parámetro faltante en best_params: {e}")
        raise
    except Exception as e:
        logger.error(f"Error al construir pipeline: {e}")
        raise


# =====================================================================
# ENTRENAMIENTO DEL MODELO FINAL
# =====================================================================

def train_final_model(
    X_train, 
    y_train, 
    best_params: Dict[str, Any],
    params_path: str = 'params.yaml'
) -> Pipeline:
    """
    Entrena el modelo final con los mejores parámetros.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        best_params: Mejores parámetros encontrados por Optuna.
        params_path: Ruta al archivo de parámetros.
    
    Returns:
        Pipeline entrenado.
    """
    logger.info("="*70)
    logger.info("ENTRENAMIENTO DEL MODELO FINAL")
    logger.info("="*70)
    
    try:
        # Obtener nombre del clasificador
        classifier_name = best_params.get('classifier', 'Desconocido')
        logger.info(f"Clasificador seleccionado: {classifier_name}")
        
        # Construir pipeline
        pipeline = build_pipeline_from_params(best_params, X_train, params_path)
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        pipeline.fit(X_train, y_train)
        logger.info(f"✓ Modelo entrenado exitosamente: {classifier_name}")
        logger.info("="*70)
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise
