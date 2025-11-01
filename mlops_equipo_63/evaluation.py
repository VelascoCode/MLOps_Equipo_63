"""
Módulo para evaluación de modelos.
"""
from sklearn.metrics import (roc_auc_score, accuracy_score, 
                             classification_report, confusion_matrix,
                             roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_model(model, X_test, y_test, show_plots=True, save_path=None):
    """
    Evalúa el modelo y muestra métricas.
    
    Args:
        model: Modelo entrenado.
        X_test: Features de prueba.
        y_test: Target de prueba.
        show_plots: Mostrar gráficos.
        save_path: Ruta para guardar figuras.
    
    Returns:
        Diccionario con métricas.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print("MÉTRICAS DE EVALUACIÓN - HOLD-OUT TEST SET")
    print("="*70)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Unpopular', 'Popular']))
    
    if show_plots or save_path:
        # Crear directorio si no existe
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Unpopular', 'Popular'],
                    yticklabels=['Unpopular', 'Popular'],
                    ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        if save_path:
            fig.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix guardada en {save_path / 'confusion_matrix.png'}")
        
        if show_plots:
            plt.show()
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve guardada en {save_path / 'roc_curve.png'}")
        
        if show_plots:
            plt.show()
        plt.close()
    
    print("="*70)
    
    return {
        'auc': float(auc),
        'accuracy': float(accuracy)
    }