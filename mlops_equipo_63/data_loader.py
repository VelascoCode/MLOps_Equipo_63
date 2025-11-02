"""
Módulo para carga de datos.

Proporciona funciones para cargar datasets desde archivos CSV
con validación y diagnóstico básico de calidad de datos.
"""
import pandas as pd
from pathlib import Path
import logging

# Configurar logger para este módulo
logger = logging.getLogger(__name__)


def load_data(
    filepath: str, 
    verbose: bool = True,
    encoding: str = 'utf-8',
    sep: str = ','
) -> pd.DataFrame:
    """
    Carga un dataset desde un archivo CSV con validación y diagnóstico.
    
    Args:
        filepath (str): Ruta al archivo CSV.
        verbose (bool): Si True, imprime información sobre los datos cargados.
        encoding (str): Codificación del archivo (default: 'utf-8').
        sep (str): Separador de columnas (default: ',').
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        pd.errors.EmptyDataError: Si el archivo está vacío.
        pd.errors.ParserError: Si hay errores al parsear el CSV.
        Exception: Para cualquier otro error inesperado.
        
    Examples:
        >>> df = load_data('data/raw/dataset.csv')
        >>> df = load_data('data/raw/dataset.csv', verbose=False)
        >>> df = load_data('data/raw/dataset.tsv', sep='\\t')
    """
    # Convertir a Path para mejor manejo de rutas
    filepath = Path(filepath)
    
    # Validar que el archivo existe
    if not filepath.exists():
        error_msg = f"El archivo no existe: {filepath.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validar que es un archivo (no directorio)
    if not filepath.is_file():
        error_msg = f"La ruta especificada no es un archivo: {filepath.absolute()}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Cargar datos
        df = pd.read_csv(filepath, encoding=encoding, sep=sep)
        
        # Validar que el DataFrame no esté vacío
        if df.empty:
            logger.warning(f"El archivo {filepath.name} está vacío (0 filas)")
        
        # Mostrar información si verbose=True
        if verbose:
            print(f"✓ Datos cargados exitosamente desde: {filepath.name}")
            print(f"  Ruta completa: {filepath.absolute()}")
            print(f"  Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
            print(f"  Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Información adicional sobre tipos de datos
            numeric_cols = df.select_dtypes(include=['number']).columns
            object_cols = df.select_dtypes(include=['object']).columns
            print(f"  Columnas numéricas: {len(numeric_cols)}")
            print(f"  Columnas de texto/objeto: {len(object_cols)}")
            
            # Advertencia sobre valores nulos
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                print(f"  ⚠️  Valores nulos detectados: {null_count:,} ({(null_count / df.size * 100):.2f}%)")
        
        logger.info(f"Dataset cargado: {filepath.name} - Shape: {df.shape}")
        return df
        
    except pd.errors.EmptyDataError as e:
        error_msg = f"El archivo {filepath.name} está vacío o no contiene datos válidos"
        logger.error(error_msg)
        raise pd.errors.EmptyDataError(error_msg) from e
        
    except pd.errors.ParserError as e:
        error_msg = f"Error al parsear el archivo CSV {filepath.name}. Verifica el formato y el separador."
        logger.error(f"{error_msg}: {str(e)}")
        raise pd.errors.ParserError(error_msg) from e
        
    except UnicodeDecodeError as e:
        error_msg = f"Error de codificación al leer {filepath.name}. Prueba con encoding='latin-1' o 'cp1252'"
        logger.error(f"{error_msg}: {str(e)}")
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end, error_msg
        ) from e
        
    except Exception as e:
        error_msg = f"Error inesperado al cargar {filepath.name}: {type(e).__name__}"
        logger.error(f"{error_msg}: {str(e)}")
        print(f"✗ {error_msg}")
        print(f"  Detalles: {str(e)}")
        raise


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Genera un resumen estadístico básico del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar.
        
    Returns:
        dict: Diccionario con estadísticas del dataset.
        
    Examples:
        >>> summary = get_data_summary(df)
        >>> print(summary['total_rows'])
    """
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=['number']).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'total_missing': int(df.isnull().sum().sum()),
        'missing_percentage': round((df.isnull().sum().sum() / df.size) * 100, 2),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'duplicate_rows': int(df.duplicated().sum())
    }
