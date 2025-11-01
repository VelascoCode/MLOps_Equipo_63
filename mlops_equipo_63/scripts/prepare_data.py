"""
Script de preparación de datos para DVC pipeline.
"""
import sys
import yaml
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops_equipo_63.data_loader import load_data
from mlops_equipo_63.preprocessing import (clean_data, impute_missing_values, 
                                create_binary_target)

def main():
    print("\n" + "="*70)
    print("STAGE 1: PREPARACIÓN DE DATOS")
    print("="*70)
    
    # Cargar parámetros
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Cargar datos
    df = load_data('data/raw/online_news_modified.csv')
    
    # Preprocesamiento
    df = clean_data(df)
    df = impute_missing_values(df, strategy=params['preprocessing']['imputation_strategy'])
    df = create_binary_target(df, threshold=params['preprocessing']['target_threshold'])
    
     # AGREGAR: Verificación final de tipos de datos
    print("\n  Verificación de tipos de datos:")
    print(f"  Columnas numéricas: {len(df.select_dtypes(include=['float64', 'int64']).columns)}")
    print(f"  Columnas no numéricas: {len(df.select_dtypes(exclude=['float64', 'int64']).columns)}")
    
    non_numeric = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
    if non_numeric:
        print(f"  ⚠ ADVERTENCIA: Columnas no numéricas detectadas: {non_numeric}")
    
    # Guardar datos procesados
    output_path = Path('data/processed')
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    print(f"✓ Datos procesados guardados en data/processed/cleaned_data.csv")
    
    # Guardar métricas de calidad de datos
    quality_metrics = {
        'total_samples': int(len(df)),
        'num_features': int(len(df.columns)),
        'missing_values': int(df.isnull().sum().sum()),
        'class_distribution': {
            'unpopular': int((df['popular'] == 0).sum()),
            'popular': int((df['popular'] == 1).sum())
        }
    }
    
    reports_path = Path('reports')
    reports_path.mkdir(parents=True, exist_ok=True)
    
    with open('reports/data_quality.json', 'w') as f:
        json.dump(quality_metrics, f, indent=4)
    
    print("✓ Métricas de calidad guardadas en reports/data_quality.json")
    print("="*70)

if __name__ == "__main__":
    main()