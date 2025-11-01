"""
Módulo para ingeniería de features.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataFrameImputer:
    """
    Imputer que preserva nombres de columnas y índices en DataFrames.
    """
    
    def __init__(self, imputer=None):
        """
        Args:
            imputer: Instancia de SimpleImputer. Si None, usa median.
        """
        self.imputer = imputer if imputer else SimpleImputer(strategy='median')
        self.columns = None
    
    def fit(self, X, y=None):
        """Ajusta el imputer a los datos."""
        self.columns = X.columns if hasattr(X, 'columns') else None
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        """Transforma los datos."""
        X_imputed = self.imputer.transform(X)
        
        if self.columns is not None:
            return pd.DataFrame(X_imputed, columns=self.columns, index=X.index)
        return X_imputed
    
    def fit_transform(self, X, y=None):
        """Ajusta y transforma en un solo paso."""
        return self.fit(X, y).transform(X)

def create_scaler():
    """
    Crea un StandardScaler para normalización.
    
    Returns:
        StandardScaler configurado.
    """
    return StandardScaler()
