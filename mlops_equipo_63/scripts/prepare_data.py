"""
Script de preparación de datos para el pipeline DVC.

Este script ejecuta la primera etapa del pipeline de ML:
- Carga los datos crudos desde data/raw/
- Aplica limpieza, imputación y transformación
- Genera la variable objetivo binaria
- Guarda los datos procesados y métricas de calidad
"""
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar la raíz del proyecto al path para importar módulos personalizados
sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops_equipo_63.data_loader import load_data
from mlops_equipo_63.preprocessing import (
    clean_data,
    impute_missing_values,
    create_binary_target
)


# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def load_params(params_path: str = 'params.yaml') -> Dict[str, Any]:
    """Carga parámetros de configuración desde archivo YAML."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"✓ Parámetros cargados desde {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Archivo de parámetros no encontrado: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear YAML: {e}")
        raise


def validate_data_path(data_path: str) -> Path:
    """
    Valida que el archivo de datos existe.
    
    Args:
        data_path: Ruta al archivo de datos.
        
    Returns:
        Path validado.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_path}")
    return path


def verify_data_quality(df) -> None:
    """
    Verifica y reporta la calidad de los datos procesados.
    
    Args:
        df: DataFrame procesado.
    """
    logger.info("\n--- Verificación de tipos de datos ---")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    
    logger.info(f"  Columnas numéricas: {len(numeric_cols)}")
    logger.info(f"  Columnas no numéricas: {len(non_numeric_cols)}")
    
    # Advertencia si hay columnas no numéricas
    if len(non_numeric_cols) > 0:
        logger.warning(f"  ⚠️ Columnas no numéricas detectadas: {non_numeric_cols.tolist()}")


def save_processed_data(df, output_path: str = 'data/processed/cleaned_data.csv') -> None:
    """
    Guarda datos procesados en CSV.
    
    Args:
        df: DataFrame procesado.
        output_path: Ruta de salida.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Datos procesados guardados en {output_path}")
        
    except Exception as e:
        logger.error(f"Error al guardar datos procesados: {e}")
        raise


def calculate_quality_metrics(df) -> Dict[str, Any]:
    """
    Calcula métricas de calidad de los datos.
    
    Args:
        df: DataFrame procesado.
        
    Returns:
        Diccionario con métricas de calidad.
    """
    try:
        metrics = {
            'total_samples': int(len(df)),
            'num_features': int(len(df.columns)),
            'missing_values': int(df.isnull().sum().sum()),
            'class_distribution': {
                'unpopular': int((df['popular'] == 0).sum()),
                'popular': int((df['popular'] == 1).sum())
            },
            'class_balance': {
                'unpopular_pct': round((df['popular'] == 0).mean() * 100, 2),
                'popular_pct': round((df['popular'] == 1).mean() * 100, 2)
            }
        }
        
        logger.info("Métricas de calidad calculadas")
        return metrics
        
    except Exception as e:
        logger.error(f"Error al calcular métricas: {e}")
        raise


def save_quality_metrics(metrics: Dict[str, Any], output_path: str = 'reports/data_quality.json') -> None:
    """
    Guarda métricas de calidad en archivo JSON.
    
    Args:
        metrics: Diccionario con métricas.
        output_path: Ruta de salida.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"✓ Métricas de calidad guardadas en {output_path}")
        
    except Exception as e:
        logger.error(f"Error al guardar métricas: {e}")
        raise


def print_summary(metrics: Dict[str, Any]) -> None:
    """
    Imprime resumen final de preparación de datos.
    
    Args:
        metrics: Diccionario con métricas de calidad.
    """
    logger.info("\n" + "="*70)
    logger.info("RESUMEN DE PREPARACIÓN DE DATOS")
    logger.info("="*70)
    logger.info(f"  Total de muestras: {metrics['total_samples']:,}")
    logger.info(f"  Total de features: {metrics['num_features']}")
    logger.info(f"  Valores faltantes: {metrics['missing_values']}")
    logger.info(f"  Distribución de clases:")
    logger.info(f"    - No popular (0): {metrics['class_distribution']['unpopular']:,} "
                f"({metrics['class_balance']['unpopular_pct']}%)")
    logger.info(f"    - Popular (1): {metrics['class_distribution']['popular']:,} "
                f"({metrics['class_balance']['popular_pct']}%)")
    logger.info("="*70)


# =====================================================================
# FUNCIÓN PRINCIPAL
# =====================================================================

def main():
    """
    Función principal que ejecuta el flujo completo de preparación de datos:
    1. Carga parámetros de configuración
    2. Carga datos crudos
    3. Aplica limpieza, imputación y transformación
    4. Genera variable objetivo binaria
    5. Verifica calidad de datos
    6. Guarda datos procesados y métricas
    """
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: PREPARACIÓN DE DATOS")
    logger.info("="*70)
    
    try:
        # 1. Cargar configuración
        params = load_params()
        
        # 2. Validar y cargar datos crudos
        data_path = 'data/raw/online_news_modified.csv'
        validate_data_path(data_path)
        df = load_data(data_path)
        
        # 3. Preprocesamiento de datos
        logger.info("\n--- Iniciando preprocesamiento ---")
        
        # Limpiar datos
        df = clean_data(df)
        
        # Imputar valores faltantes
        df = impute_missing_values(
            df,
            strategy=params['preprocessing']['imputation_strategy']
        )
        
        # Crear variable objetivo binaria
        df = create_binary_target(
            df,
            threshold=params['preprocessing']['target_threshold']
        )
        
        # 4. Verificación de calidad
        verify_data_quality(df)
        
        # 5. Guardar datos procesados
        save_processed_data(df)
        
        # 6. Calcular y guardar métricas
        metrics = calculate_quality_metrics(df)
        save_quality_metrics(metrics)
        
        # 7. Mostrar resumen
        print_summary(metrics)
        
        logger.info("✅ STAGE 1 COMPLETADO EXITOSAMENTE\n")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Archivo no encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error en preparación de datos: {e}")
        raise


if __name__ == "__main__":
    main()
