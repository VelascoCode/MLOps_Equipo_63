"""
Módulo para evaluación de modelos de clasificación.

Proporciona funciones para calcular métricas, generar reportes visuales
y guardar resultados de evaluación.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report, 
    confusion_matrix,
    roc_curve
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# =====================================================================
# CÁLCULO DE MÉTRICAS
# =====================================================================

def calculate_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """
    Calcula métricas de clasificación binaria.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones binarias.
        y_prob: Probabilidades de clase positiva.
    
    Returns:
        Diccionario con todas las métricas calculadas.
    """
    try:
        metrics = {
            'auc': float(roc_auc_score(y_true, y_prob)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred))
        }
        
        logger.info(f"Métricas calculadas: AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error al calcular métricas: {e}")
        raise


def print_evaluation_report(metrics: Dict[str, float], y_true, y_pred) -> None:
    """
    Imprime reporte detallado de evaluación.
    
    Args:
        metrics: Diccionario con métricas calculadas.
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones binarias.
    """
    logger.info("="*70)
    logger.info("MÉTRICAS DE EVALUACIÓN - HOLD-OUT TEST SET")
    logger.info("="*70)
    logger.info(f"AUC-ROC:   {metrics['auc']:.4f}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info("\nReporte de Clasificación:")
    
    report = classification_report(
        y_true, y_pred,
        target_names=['Unpopular', 'Popular']
    )
    print(report)
    logger.info("="*70)


# =====================================================================
# VISUALIZACIONES
# =====================================================================

def plot_confusion_matrix(
    y_true, 
    y_pred, 
    save_path: Optional[Path] = None,
    show_plot: bool = False
) -> None:
    """
    Genera y guarda matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones binarias.
        save_path: Directorio donde guardar la figura.
        show_plot: Si True, muestra el gráfico.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Unpopular', 'Popular'],
            yticklabels=['Unpopular', 'Popular'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            file_path = save_path / 'confusion_matrix.png'
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Confusion matrix guardada en {file_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error al crear confusion matrix: {e}")
        raise


def plot_roc_curve(
    y_true, 
    y_prob, 
    auc: float,
    save_path: Optional[Path] = None,
    show_plot: bool = False
) -> None:
    """
    Genera y guarda curva ROC.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_prob: Probabilidades de clase positiva.
        auc: Valor de AUC ya calculado.
        save_path: Directorio donde guardar la figura.
        show_plot: Si True, muestra el gráfico.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}', color='#1f77b4')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        if save_path:
            file_path = save_path / 'roc_curve.png'
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ ROC curve guardada en {file_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error al crear ROC curve: {e}")
        raise


# =====================================================================
# FUNCIÓN PRINCIPAL DE EVALUACIÓN
# =====================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    show_plots: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evalúa el modelo de clasificación y genera reportes completos.
    
    Args:
        model: Modelo entrenado con métodos predict y predict_proba.
        X_test: Features de prueba.
        y_test: Target de prueba.
        show_plots: Si True, muestra los gráficos generados.
        save_path: Directorio donde guardar figuras y reportes.
    
    Returns:
        Diccionario con todas las métricas de evaluación.
    
    Raises:
        ValueError: Si el modelo no tiene predict_proba.
        Exception: Si hay errores en predicciones o guardado de archivos.
    
    Examples:
        >>> metrics = evaluate_model(model, X_test, y_test, save_path='reports')
        >>> print(f"AUC: {metrics['auc']:.4f}")
    """
    try:
        logger.info("Iniciando evaluación del modelo")
        
        # Validar que el modelo tenga predict_proba
        if not hasattr(model, 'predict_proba'):
            raise ValueError("El modelo no tiene método predict_proba")
        
        # Generar predicciones
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        logger.info(f"Predicciones generadas para {len(y_test)} muestras")
        
        # Calcular métricas
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Imprimir reporte
        print_evaluation_report(metrics, y_test, y_pred)
        
        # Preparar directorio de guardado si es necesario
        save_path_obj = None
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Guardando resultados en {save_path_obj}")
        
        # Generar visualizaciones
        if show_plots or save_path_obj:
            plot_confusion_matrix(y_test, y_pred, save_path_obj, show_plots)
            plot_roc_curve(y_test, y_prob, metrics['auc'], save_path_obj, show_plots)
        
        logger.info("Evaluación completada exitosamente")
        return metrics
        
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise
    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        raise
