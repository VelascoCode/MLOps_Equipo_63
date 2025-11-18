from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# --- [COMPONENTES DEL PIPELINE] - Transformadores Personalizados ---
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Un transformador personalizado de scikit-learn para recortar valores atípicos (outliers)
    usando el método de Rango Intercuartílico (IQR).
    
    Este paso es crucial para hacer que los modelos (especialmente los lineales) y los
    escaladores sean más robustos frente a valores extremos.
    """
    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """Calcula y almacena los límites inferiores y superiores para cada columna."""
        X_df = pd.DataFrame(X)
        # preserve input feature names
        if hasattr(X_df, 'columns'):
            self.feature_names_in_ = list(X_df.columns)
        else:
            self.feature_names_in_ = [f"f{i}" for i in range(X_df.shape[1])]
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.iqr_multiplier * IQR
            self.upper_bounds_[col] = Q3 + self.iqr_multiplier * IQR
        return self

    def transform(self, X, y=None):
        """Recorta los datos en X basándose en los límites calculados en 'fit'.

        Devuelve un DataFrame para preservar nombres de columnas a lo largo del pipeline.
        """
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_.get(col),
                upper=self.upper_bounds_.get(col)
            )
        return X_df

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is None:
            return np.array([])
        return np.array(self.feature_names_in_)


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Wrapper genérico para que transformadores de sklearn devuelvan DataFrame
    y preserven nombres de columnas (útil para que el Pipeline mantenga feature names).
    """
    def __init__(self, transformer):
        self.transformer = transformer
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # store column names if present
        X_df = pd.DataFrame(X)
        if hasattr(X_df, 'columns'):
            self.feature_names_in_ = list(X_df.columns)
        else:
            self.feature_names_in_ = [f"f{i}" for i in range(X_df.shape[1])]
        self.transformer.fit(X_df, y)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        Xt = self.transformer.transform(X_df)
        # If transformer returns 1d array, convert to 2d
        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)
        return pd.DataFrame(Xt, columns=self.feature_names_in_, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)
