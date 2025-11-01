"""
Módulo para optimización de hiperparámetros con Optuna.
"""
import optuna
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from mlops_equipo_63.feature_engineering import DataFrameImputer, create_scaler

def objective(trial, X_train, y_train, cv_folds=3):
    """
    Función objetivo para Optuna.
    
    Args:
        trial: Trial de Optuna.
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv_folds: Número de folds para cross-validation.
    
    Returns:
        AUC promedio de validación cruzada.
    """
    # Selección del clasificador
    classifier_name = trial.suggest_categorical(
        'classifier', 
        ['RandomForest', 'XGBoost', 'LightGBM', 'MLP']
    )
    
    # Configuración según el clasificador
    if classifier_name == 'RandomForest':
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 50, 400),
            max_depth=trial.suggest_int('rf_max_depth', 5, 30),
            min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
            random_state=42,
            n_jobs=-1
        )
    
    elif classifier_name == 'MLP':
        hidden_layer_config = trial.suggest_categorical(
            'mlp_hidden_layers', 
            ["(50,)", "(100,)", "(50, 50)", "(100, 50)"]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=eval(hidden_layer_config),
            alpha=trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float('mlp_learning_rate', 1e-4, 1e-2, log=True),
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
    
    elif classifier_name == 'XGBoost':
        classifier = xgb.XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 100, 800),
            learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
            min_child_weight=trial.suggest_int('xgb_min_child_weight', 1, 7),
            subsample=trial.suggest_float('xgb_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
    
    elif classifier_name == 'LightGBM':
        classifier = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int('lgbm_n_estimators', 100, 800),
            learning_rate=trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
            num_leaves=trial.suggest_int('lgbm_num_leaves', 20, 100),
            min_child_samples=trial.suggest_int('lgbm_min_child_samples', 5, 50),
            subsample=trial.suggest_float('lgbm_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
            objective='binary',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # Seleccionar solo columnas numéricas
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    # Pipeline con imputación, escalado y clasificación
    pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_cols)
            ],
            remainder='drop'  #Eliminar columnas no numéricas
        )),
        ('classifier', classifier)
    ])
    
    # Validación cruzada
    try:
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
        
        return mean_auc
        
    except Exception as e:
        print(f"Error en trial {trial.number}: {e}")
        return 0.0

def print_metrics_callback(study, trial):
    """Callback para imprimir métricas durante optimización."""
    print(f"Trial {trial.number:3d} | "
          f"AUC: {trial.value:.4f} | "
          f"Accuracy: {trial.user_attrs.get('accuracy', 0):.4f} | "
          f"Classifier: {trial.params.get('classifier', 'N/A')}")

def run_optuna_optimization(X_train, y_train, n_trials=50, cv_folds=3):
    """
    Ejecuta la optimización con Optuna.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        n_trials: Número de trials a ejecutar.
        cv_folds: Número de folds para cross-validation.
    
    Returns:
        optuna.Study: Estudio completado.
    """
    print("\n" + "="*70)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    print("="*70)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='news_popularity_optimization'
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cv_folds),
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[print_metrics_callback],
        show_progress_bar=True
    )
    
    print("\n" + "="*70)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*70)
    print(f"Mejor AUC (CV): {study.best_trial.value:.4f}")
    print(f"Mejor Accuracy (CV): {study.best_trial.user_attrs.get('accuracy', 0):.4f}")
    print(f"\nMejores Hiperparámetros:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    return study
