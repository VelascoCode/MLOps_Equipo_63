"""
Módulo para preprocesamiento de datos.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple

def clean_data(df: pd.DataFrame, target_column: str = 'shares') -> pd.DataFrame:
    """
    Limpia el dataset eliminando NaNs en el target y columnas innecesarias.
    """
    print(f"Limpieza de datos:")
    print(f"  Filas iniciales: {len(df)}")
    
    # Convertir columna target a numérico
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    
    # Eliminar filas con NaN en el target
    df = df.dropna(subset=[target_column])
    
    # Eliminar columnas URL y timedelta si existen
    columns_to_drop = ['url', 'timedelta']
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"  Columnas eliminadas: {existing_cols}")
    
    # Convertir TODAS las columnas restantes a numérico
    print("  Convirtiendo todas las columnas a tipo numérico...")
    for col in df.columns:
        if col != target_column:  # Ya convertimos el target antes
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas donde TODAS las features sean NaN
    df = df.dropna(how='all', subset=[c for c in df.columns if c != target_column])
    
    print(f"  Filas finales: {len(df)}")
    print(f"  Valores nulos restantes por columna:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Imputa valores faltantes en columnas numéricas.
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) == 0:
        print(f"⚠ No hay columnas numéricas para imputar")
        return df
    
    if df[numeric_cols].isnull().sum().sum() == 0:
        print(f"✓ No hay valores faltantes que imputar")
        return df
    
    print(f"  Imputando {df[numeric_cols].isnull().sum().sum()} valores faltantes...")
    
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    print(f"✓ Valores faltantes imputados con estrategia: {strategy}")
    return df


def create_binary_target(df: pd.DataFrame, 
                         target_column: str = 'shares', 
                         threshold: float = None) -> pd.DataFrame:
    """
    Crea variable objetivo binaria.
    
    Args:
        df: DataFrame original.
        target_column: Columna objetivo continua.
        threshold: Umbral para clasificación. Si None, usa mediana.
    
    Returns:
        DataFrame con columna 'popular'.
    """
    
    #Asegurar que la columna sea numérica
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df = df.dropna(subset=[target_column])
    
    if threshold is None:
        threshold = df[target_column].median()
    
    df['popular'] = (df[target_column] >= threshold).astype(int)
    
    print(f"✓ Variable objetivo binaria 'popular' creada")
    print(f"  Umbral: {threshold:.2f}")
    print(f"  Distribución de clases:")
    print(f"    Unpopular (0): {(df['popular'] == 0).sum()} ({(df['popular'] == 0).mean()*100:.1f}%)")
    print(f"    Popular (1): {(df['popular'] == 1).sum()} ({(df['popular'] == 1).mean()*100:.1f}%)")
    
    return df

def split_data(df: pd.DataFrame, 
               target_column: str = 'popular',
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en train y test con estratificación.
    
    Args:
        df: DataFrame completo.
        target_column: Columna objetivo.
        test_size: Proporción de datos para test.
        random_state: Semilla para reproducibilidad.
    
    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    # Eliminar columnas no necesarias para el modelo
    columns_to_drop = [target_column, 'shares', 'url', 'timedelta']
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=existing_cols, errors='ignore')
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"✓ Datos divididos exitosamente")
    print(f"  Train: {len(X_train)} muestras ({(1-test_size)*100:.0f}%)")
    print(f"  Test: {len(X_test)} muestras ({test_size*100:.0f}%)")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test
