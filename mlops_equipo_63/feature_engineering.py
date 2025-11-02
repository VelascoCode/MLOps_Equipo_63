"""
Módulo para ingeniería de features.

Proporciona herramientas para preprocesamiento de datos:
- DataFrameImputer: Imputa valores faltantes preservando estructura de DataFrame
- create_scaler: Crea escalador estándar para normalización

Ejemplo de uso:
    >>> from feature_engineering import DataFrameImputer, create_scaler
    >>> imputer = DataFrameImputer()
    >>> df_clean = imputer.fit_transform(df)
    >>> scaler = create_scaler()
    >>> X_scaled = scaler.fit_transform(X)
"""
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrameImputer:
    """
    Imputer que preserva nombres de columnas y índices en DataFrames.
    
    Extiende la funcionalidad de SimpleImputer de scikit-learn para mantener
    la estructura de DataFrame después de la imputación.
    
    Attributes:
        imputer: Instancia de SimpleImputer para realizar la imputación.
        columns: Lista de nombres de columnas del DataFrame original.
    
    Examples:
        >>> imputer = DataFrameImputer()
        >>> df_clean = imputer.fit_transform(df)
        >>> print(df_clean.columns)  # Columnas preservadas
    """
    
    def __init__(self, imputer: Optional[SimpleImputer] = None):
        """
        Inicializa el DataFrameImputer.
        
        Args:
            imputer: Instancia de SimpleImputer. Si None, usa estrategia 'median'.
        """
        self.imputer = imputer if imputer else SimpleImputer(strategy='median')
        self.columns = None
        logger.debug(f"DataFrameImputer inicializado con estrategia: {self.imputer.strategy}")
    
    def fit(self, X, y=None):
        """
        Ajusta el imputer a los datos.
        
        Args:
            X: DataFrame o array de entrada.
            y: Target (no usado, presente para compatibilidad con sklearn).
        
        Returns:
            self: Instancia ajustada.
        """
        try:
            # Guardar nombres de columnas si es DataFrame
            self.columns = X.columns.tolist() if hasattr(X, 'columns') else None
            
            # Ajustar imputer
            self.imputer.fit(X)
            
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X.columns)
            logger.info(f"DataFrameImputer ajustado con {n_features} features")
            
            return self
            
        except Exception as e:
            logger.error(f"Error al ajustar DataFrameImputer: {e}")
            raise
    
    def transform(self, X):
        """
        Transforma los datos imputando valores faltantes.
        
        Args:
            X: DataFrame o array de entrada.
        
        Returns:
            DataFrame o array con valores imputados.
        """
        try:
            # Verificar si hay valores nulos antes de la imputación
            if hasattr(X, 'isnull'):
                n_nulls_before = X.isnull().sum().sum()
                logger.debug(f"Valores nulos antes de imputación: {n_nulls_before}")
            
            # Realizar imputación
            X_imputed = self.imputer.transform(X)
            
            # Si teníamos columnas guardadas, reconstruir DataFrame
            if self.columns is not None:
                result = pd.DataFrame(X_imputed, columns=self.columns, index=X.index)
                logger.info(f"Imputación completada. Shape: {result.shape}")
                return result
            
            logger.info(f"Imputación completada. Shape: {X_imputed.shape}")
            return X_imputed
            
        except Exception as e:
            logger.error(f"Error al transformar con DataFrameImputer: {e}")
            raise
    
    def fit_transform(self, X, y=None):
        """
        Ajusta y transforma los datos en un solo paso.
        
        Args:
            X: DataFrame o array de entrada.
            y: Target (no usado, presente para compatibilidad con sklearn).
        
        Returns:
            DataFrame o array con valores imputados.
        
        Examples:
            >>> imputer = DataFrameImputer()
            >>> df_clean = imputer.fit_transform(df)
        """
        logger.debug("Ejecutando fit_transform")
        return self.fit(X, y).transform(X)


def create_scaler(with_mean: bool = True, with_std: bool = True) -> StandardScaler:
    """
    Crea un StandardScaler para normalización de features.
    
    El StandardScaler estandariza features restando la media y dividiendo
    por la desviación estándar, resultando en features con media 0 y 
    desviación estándar 1.
    
    Args:
        with_mean: Si True, centra los datos antes de escalar.
        with_std: Si True, escala los datos a varianza unitaria.
    
    Returns:
        StandardScaler configurado.
    
    Examples:
        >>> scaler = create_scaler()
        >>> X_scaled = scaler.fit_transform(X)
        >>> print(X_scaled.mean())  # Aproximadamente 0
        >>> print(X_scaled.std())   # Aproximadamente 1
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    logger.debug(f"StandardScaler creado (with_mean={with_mean}, with_std={with_std})")
    return scaler
