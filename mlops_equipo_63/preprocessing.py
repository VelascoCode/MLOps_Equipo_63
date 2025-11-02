"""
Módulo para preprocesamiento de datos.
Incluye funciones de limpieza, imputación, creación de variable objetivo binaria y división de datos.

"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clean_data(df: pd.DataFrame, target_column: str = 'shares', drop_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Limpia el dataset eliminando filas con NaN en el target, columnas innecesarias y convierte features a numérico.
    Args:
        df: DataFrame de entrada.
        target_column: nombre de la columna objetivo (default: 'shares').
        drop_columns: lista de columnas adicionales a eliminar.
    Returns:
        DataFrame limpio.
    """
    df = df.copy()
    logger.info(f"Iniciando limpieza: {df.shape[0]} filas, {df.shape[1]} columnas.")

    # Convertir target a numérico y eliminar filas NaN en target
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    before = df.shape[0]
    df = df.dropna(subset=[target_column])
    logger.info(f"Filas eliminadas por NaN en target: {before - df.shape[0]}.")

    # Eliminar columnas extra si es necesario
    columns_to_drop = ['url', 'timedelta']
    if drop_columns:
        columns_to_drop.extend(drop_columns)
    cols_existing = [col for col in columns_to_drop if col in df.columns]
    if cols_existing:
        df = df.drop(columns=cols_existing)
        logger.info(f"Columnas eliminadas: {cols_existing}")

    # Convertir todas las demás columnas numéricas (pero no el target)
    feature_cols = [c for c in df.columns if c != target_column]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    logger.info("Features convertidas a numérico si fue posible.")

    # Eliminar filas donde todas las features son NaN
    before2 = df.shape[0]
    df = df.dropna(how='all', subset=feature_cols)
    logger.info(f"Filas eliminadas por tener todas sus features NaN: {before2 - df.shape[0]}.")

    # Diagnóstico final
    nulls = df.isnull().sum()
    n_null_cols = nulls[nulls > 0].to_dict()
    if n_null_cols:
        logger.warning(f"Columnas con valores nulos restantes: {n_null_cols}")
    else:
        logger.info("No quedan valores nulos en features ni target.")

    logger.info(f"Final de limpieza: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df

def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Imputa valores faltantes solo en columnas numéricas.
    Args:
        df: DataFrame.
        strategy: estrategia de imputación (default: 'median').
    Returns:
        DataFrame con imputación aplicada.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        logger.warning("No hay columnas numéricas para imputar.")
        return df
    if df[numeric_cols].isnull().sum().sum() == 0:
        logger.info("No hay valores numéricos faltantes que imputar.")
        return df
    logger.info(f"Imputando {df[numeric_cols].isnull().sum().sum()} valores faltantes en columnas numéricas...")
    imputer = SimpleImputer(strategy=strategy)
    df.loc[:, numeric_cols] = imputer.fit_transform(df[numeric_cols])
    logger.info(f"Imputación completada con estrategia: {strategy}")
    return df

def create_binary_target(
    df: pd.DataFrame, 
    target_column: str = 'shares', 
    threshold: Optional[float] = None,
    new_column: str = 'popular'
) -> pd.DataFrame:
    """
    Crea una columna binaria como target.
    Args:
        df: DataFrame original.
        target_column: columna objetivo.
        threshold: umbral. Si None, usa la mediana.
        new_column: nombre de la columna binaria (default: 'popular').
    Returns:
        DataFrame con nueva columna.
    """
    df = df.copy()
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df = df.dropna(subset=[target_column])
    if threshold is None:
        threshold = df[target_column].median()
    df[new_column] = (df[target_column] >= threshold).astype(int)
    class0 = (df[new_column] == 0).mean() * 100
    class1 = (df[new_column] == 1).mean() * 100
    logger.info(f"Variable objetivo '{new_column}' creada. Umbral: {threshold:.2f}")
    logger.info(f"Distribución: 0 (no popular): {class0:.1f}%, 1 (popular): {class1:.1f}%")
    return df

def split_data(
    df: pd.DataFrame, 
    target_column: str = 'popular',
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide datos en train/test con estratificación por el target.
    Args:
        df: DataFrame.
        target_column: target binario.
        test_size: proporción para test.
        random_state: semilla.
    Returns:
        X_train, X_test, y_train, y_test.
    """
    # Eliminar columnas no necesarias
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
    logger.info(f"Datos divididos: train={len(X_train)}, test={len(X_test)}, features={X_train.shape[1]}")
    return X_train, X_test, y_train, y_test
