"""
Módulo para optimización de hiperparámetros con Optuna.
Permite comparar múltiples familias de modelos y encontrar la mejor configuración.
Los rangos de hiperparámetros y parámetros de optimización se configuran desde params.yaml.
"""
import optuna
import numpy as np
import yaml
import logging
from typing import Dict, Any, Callable
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from optuna.integration.mlflow import MLflowCallback
import xgboost as xgb
import lightgbm as lgb

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# CARGA DE PARÁMETROS
# =====================================================================

def load_hyperparameter_config(params_path: str = 'params.yaml') -> Dict[str, Any]:
    """
    Carga la configuración de hiperparámetros desde params.yaml.
    
    Args:
        params_path: Ruta al archivo de parámetros.
        
    Returns:
        Diccionario con configuración de hiperparámetros.
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('hyperparameter_optimization', {})
    except FileNotFoundError:
        logger.error(f"Archivo de parámetros no encontrado: {params_path}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar parámetros: {e}")
        raise


# Cargar configuración global al importar el módulo
HYPERPARAM_CONFIG = load_hyperparameter_config()


# =====================================================================
# FUNCIONES PARA CREAR CLASIFICADORES
# =====================================================================

def create_random_forest(trial: optuna.Trial, random_state: int = 42) -> RandomForestClassifier:
    """Crea clasificador Random Forest con hiperparámetros desde params.yaml."""
    rf_params = HYPERPARAM_CONFIG.get('random_forest', {})
    
    return RandomForestClassifier(
        n_estimators=trial.suggest_int(
            'rf_n_estimators',
            rf_params.get('n_estimators_min', 50),
            rf_params.get('n_estimators_max', 400)
        ),
        max_depth=trial.suggest_int(
            'rf_max_depth',
            rf_params.get('max_depth_min', 5),
            rf_params.get('max_depth_max', 30)
        ),
        min_samples_split=trial.suggest_int(
            'rf_min_samples_split',
            rf_params.get('min_samples_split_min', 2),
            rf_params.get('min_samples_split_max', 20)
        ),
        min_samples_leaf=trial.suggest_int(
            'rf_min_samples_leaf',
            rf_params.get('min_samples_leaf_min', 1),
            rf_params.get('min_samples_leaf_max', 10)
        ),
        random_state=random_state,
        n_jobs=-1
    )


def create_mlp(trial: optuna.Trial, random_state: int = 42) -> MLPClassifier:
    """Crea clasificador MLP con hiperparámetros desde params.yaml."""
    mlp_params = HYPERPARAM_CONFIG.get('mlp', {})
    
    hidden_layer_config = trial.suggest_categorical(
        'mlp_hidden_layers',
        mlp_params.get('hidden_layers', ["(50,)", "(100,)", "(50, 50)", "(100, 50)"])
    )
    
    return MLPClassifier(
        hidden_layer_sizes=eval(hidden_layer_config),
        alpha=trial.suggest_float(
            'mlp_alpha',
            mlp_params.get('alpha_min', 1e-5),
            mlp_params.get('alpha_max', 1e-1),
            log=True
        ),
        learning_rate_init=trial.suggest_float(
            'mlp_learning_rate',
            mlp_params.get('learning_rate_min', 1e-4),
            mlp_params.get('learning_rate_max', 1e-2),
            log=True
        ),
        max_iter=300,
        early_stopping=True,
        random_state=random_state
    )


def create_xgboost(trial: optuna.Trial, random_state: int = 42) -> xgb.XGBClassifier:
    """Crea clasificador XGBoost con hiperparámetros desde params.yaml."""
    xgb_params = HYPERPARAM_CONFIG.get('xgboost', {})
    
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int(
            'xgb_n_estimators',
            xgb_params.get('n_estimators_min', 100),
            xgb_params.get('n_estimators_max', 800)
        ),
        learning_rate=trial.suggest_float(
            'xgb_learning_rate',
            xgb_params.get('learning_rate_min', 0.01),
            xgb_params.get('learning_rate_max', 0.3)
        ),
        max_depth=trial.suggest_int(
            'xgb_max_depth',
            xgb_params.get('max_depth_min', 3),
            xgb_params.get('max_depth_max', 10)
        ),
        min_child_weight=trial.suggest_int(
            'xgb_min_child_weight',
            xgb_params.get('min_child_weight_min', 1),
            xgb_params.get('min_child_weight_max', 7)
        ),
        subsample=trial.suggest_float(
            'xgb_subsample',
            xgb_params.get('subsample_min', 0.6),
            xgb_params.get('subsample_max', 1.0)
        ),
        colsample_bytree=trial.suggest_float(
            'xgb_colsample_bytree',
            xgb_params.get('colsample_bytree_min', 0.6),
            xgb_params.get('colsample_bytree_max', 1.0)
        ),
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1
    )


def create_lightgbm(trial: optuna.Trial, random_state: int = 42) -> lgb.LGBMClassifier:
    """Crea clasificador LightGBM con hiperparámetros desde params.yaml."""
    lgbm_params = HYPERPARAM_CONFIG.get('lightgbm', {})
    
    return lgb.LGBMClassifier(
        n_estimators=trial.suggest_int(
            'lgbm_n_estimators',
            lgbm_params.get('n_estimators_min', 100),
            lgbm_params.get('n_estimators_max', 800)
        ),
        learning_rate=trial.suggest_float(
            'lgbm_learning_rate',
            lgbm_params.get('learning_rate_min', 0.01),
            lgbm_params.get('learning_rate_max', 0.3)
        ),
        num_leaves=trial.suggest_int(
            'lgbm_num_leaves',
            lgbm_params.get('num_leaves_min', 20),
            lgbm_params.get('num_leaves_max', 100)
        ),
        min_child_samples=trial.suggest_int(
            'lgbm_min_child_samples',
            lgbm_params.get('min_child_samples_min', 5),
            lgbm_params.get('min_child_samples_max', 50)
        ),
        subsample=trial.suggest_float(
            'lgbm_subsample',
            lgbm_params.get('subsample_min', 0.6),
            lgbm_params.get('subsample_max', 1.0)
        ),
        colsample_bytree=trial.suggest_float(
            'lgbm_colsample_bytree',
            lgbm_params.get('colsample_bytree_min', 0.6),
            lgbm_params.get('colsample_bytree_max', 1.0)
        ),
        objective='binary',
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )


# Mapa de clasificadores disponibles
CLASSIFIER_FACTORY: Dict[str, Callable] = {
    'RandomForest': create_random_forest,
    'MLP': create_mlp,
    'XGBoost': create_xgboost,
    'LightGBM': create_lightgbm
}


# =====================================================================
# FUNCIÓN OBJETIVO Y AUXILIARES
# =====================================================================

def create_preprocessing_pipeline(X_train) -> ColumnTransformer:
    """
    Crea el pipeline de preprocesamiento para columnas numéricas.
    
    Args:
        X_train: DataFrame de entrenamiento para detectar columnas numéricas.
        
    Returns:
        ColumnTransformer configurado.
    """
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols)
        ],
        remainder='drop'
    )


def objective(trial: optuna.Trial, X_train, y_train, cv_folds: int = 3, random_state: int = 42) -> float:
    """
    Función objetivo para optimización con Optuna.
    
    Args:
        trial: Trial de Optuna para sugerir hiperparámetros.
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv_folds: Número de folds para validación cruzada.
    
    Returns:
        AUC promedio de validación cruzada (métrica a maximizar).
    """
    try:
        # Obtener lista de clasificadores desde params.yaml
        available_classifiers = HYPERPARAM_CONFIG.get('classifiers', list(CLASSIFIER_FACTORY.keys()))
        
        # Seleccionar clasificador
        classifier_name = trial.suggest_categorical('classifier', available_classifiers)
        
        # Crear clasificador usando la factory
        classifier = CLASSIFIER_FACTORY[classifier_name](trial, random_state)
        
        # Crear pipeline completo
        pipeline = Pipeline(steps=[
            ('preprocessor', create_preprocessing_pipeline(X_train)),
            ('classifier', classifier)
        ])
        
        # Validación cruzada
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_folds,
            scoring=['roc_auc', 'accuracy'],
            n_jobs=-1,
            error_score='raise'
        )
        
        mean_auc = np.mean(scores['test_roc_auc'])
        mean_accuracy = np.mean(scores['test_accuracy'])
        
        # Guardar accuracy como atributo del trial
        trial.set_user_attr('accuracy', mean_accuracy)
        
        logger.debug(f"Trial {trial.number}: {classifier_name} | AUC={mean_auc:.4f} | Acc={mean_accuracy:.4f}")
        
        return mean_auc
        
    except ValueError as e:
        logger.warning(f"Trial {trial.number} failed with ValueError: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with unexpected error: {e}")
        return 0.0


def print_metrics_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    """
    Callback para imprimir métricas durante la optimización.
    
    Args:
        study: Estudio de Optuna.
        trial: Trial completado.
    """
    print(f"Trial {trial.number:3d} | "
          f"AUC: {trial.value:.4f} | "
          f"Accuracy: {trial.user_attrs.get('accuracy', 0):.4f} | "
          f"Classifier: {trial.params.get('classifier', 'N/A')}")


# =====================================================================
# FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN
# =====================================================================

def run_optuna_optimization(
    X_train, 
    y_train, 
    params_path: str = 'params.yaml'
) -> optuna.Study:
    """
    Ejecuta la optimización de hiperparámetros con Optuna.
    Todos los parámetros se cargan desde params.yaml.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        params_path: Ruta al archivo de parámetros YAML.
    
    Returns:
        Estudio de Optuna completado con los mejores hiperparámetros.
    """
    # Cargar parámetros desde YAML
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    
    # Extraer parámetros de entrenamiento
    training_params = params.get('training', {})
    n_trials = training_params.get('n_trials', 50)
    cv_folds = training_params.get('cv_folds', 3)
    random_state = training_params.get('random_state', 42)
    
    # Nombre del estudio desde configuración de proyecto
    project_name = params.get('project', {}).get('name', 'optimization')
    study_name = f"{project_name}_optuna_study"
    
    logger.info("="*70)
    logger.info("OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    logger.info("="*70)
    logger.info(f"Configuración: {n_trials} trials, {cv_folds}-fold CV")
    logger.info(f"Clasificadores disponibles: {HYPERPARAM_CONFIG.get('classifiers', 'Todos')}")
    
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name
    )
    
    # Ejecutar optimización
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cv_folds,random_state),
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[print_metrics_callback],
        show_progress_bar=True
    )
    
    # Mostrar resultados
    logger.info("="*70)
    logger.info("OPTIMIZACIÓN COMPLETADA")
    logger.info("="*70)
    logger.info(f"Mejor AUC (CV): {study.best_trial.value:.4f}")
    logger.info(f"Mejor Accuracy (CV): {study.best_trial.user_attrs.get('accuracy', 0):.4f}")
    logger.info(f"Mejor Clasificador: {study.best_trial.params.get('classifier', 'N/A')}")
    logger.info("\nMejores Hiperparámetros:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    return study
