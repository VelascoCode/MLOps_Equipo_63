"""
Módulo para carga de datos.
"""
import pandas as pd
from pathlib import Path

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        filepath: Ruta al archivo CSV.
    
    Returns:
        DataFrame con los datos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"El archivo {filepath} no existe")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Datos cargados exitosamente")
        print(f"  Shape: {df.shape}")
        print(f"  Columnas: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Error al cargar datos: {e}")
        raise