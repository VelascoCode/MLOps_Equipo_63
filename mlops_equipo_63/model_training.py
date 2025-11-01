"""
Módulo para entrenamiento de modelos.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb


from mlops_equipo_63.feature_engineering import DataFrameImputer, create_scaler

def build_pipeline_from_params(best_params, X_train):
    """
    Construye un pipeline de Scikit-Learn con los mejores parámetros.
    
    Args:
        best_params: Diccionario con los mejores parámetros de Optuna.
    
    Returns:
        Pipeline configurado.
    """
    # Seleccionar sólo columnas numéricas
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    print(f"  Pipeline configurado para {len(numeric_cols)} columnas numéricas")
    
    # Crear preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols)
        ],
        remainder='drop' 
    )
    
    # Selección y construcción del clasificador según best_params (como ya tienes: RandomForest, MLP, etc.)
    params = best_params.copy()
    classifier_name = params.pop('classifier')

    if classifier_name == 'RandomForest':
        model_params = {k.replace('rf_', ''): v for k, v in best_params.items()}
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1, **model_params)
    elif classifier_name == 'MLP':
        model_params = {k.replace('mlp_', ''): v for k, v in best_params.items()}
        if 'hidden_layers' in model_params:
            model_params['hidden_layer_sizes'] = eval(model_params.pop('hidden_layers'))
        classifier = MLPClassifier(random_state=42, max_iter=300, early_stopping=True, **model_params)
    elif classifier_name == 'XGBoost':
        model_params = {k.replace('xgb_', ''): v for k, v in best_params.items()}
        classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            **model_params
        )
    elif classifier_name == 'LightGBM':
        model_params = {k.replace('lgbm_', ''): v for k, v in best_params.items()}
        classifier = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **model_params
        )
    
    # Armar pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def train_final_model(X_train, y_train, best_params):
    """
    Entrena el modelo final con los mejores parámetros.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        best_params: Mejores parámetros de Optuna.
    
    Returns:
        Pipeline entrenado.
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL MODELO FINAL")
    print("="*70)
    
    # Guardar el nombre del clasificador antes de que build_pipeline lo elimine
    classifier_name = best_params.get('classifier', 'Desconocido')
    
    pipeline = build_pipeline_from_params(best_params, X_train)
    pipeline.fit(X_train, y_train)
    
    print(f"✓ Modelo entrenado: {classifier_name}")
    print("="*70)
    
    return pipeline
