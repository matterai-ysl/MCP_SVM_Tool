"""Comprehensive Metrics Evaluation Module"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """Comprehensive metrics evaluation for ML models."""
    
    def __init__(self, task_type: str = "auto"):
        self.task_type = task_type
        self.results_history = []
        
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive regression metrics."""
        metrics = {}
        
        try:
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            metrics['root_mean_squared_error'] = np.sqrt(metrics['mean_squared_error'])
            metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
            
            # Additional metrics
            residuals = y_true - y_pred
            metrics['mean_residual'] = np.mean(residuals)
            metrics['std_residual'] = np.std(residuals)
            
        except Exception as e:
            logger.error(f"Error computing regression metrics: {e}")
            raise
            
        return metrics
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute comprehensive classification metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # ROC AUC for binary classification
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Error computing classification metrics: {e}")
            raise
            
        return metrics
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: Optional[np.ndarray] = None,
                      task_type: Optional[str] = None) -> Dict[str, Any]:
        """Automatically evaluate model based on task type."""
        if task_type is None:
            task_type = self.task_type
        
        if task_type == "auto":
            task_type = self._detect_task_type(y_true, y_pred)
        
        if task_type == "regression":
            return self.evaluate_regression(y_true, y_pred)
        elif task_type == "classification":
            return self.evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               output_dir: str = "evaluation_plots",
                               task_type: Optional[str] = None) -> Dict[str, str]:
        """Create comprehensive evaluation plots."""
        if task_type is None:
            task_type = self.task_type
        
        if task_type == "auto":
            task_type = self._detect_task_type(y_true, y_pred)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        if task_type == "regression":
            # Predicted vs Actual scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.grid(True, alpha=0.3)
            
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
            
            pred_vs_actual_path = output_dir / "predicted_vs_actual.png"
            plt.savefig(pred_vs_actual_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['predicted_vs_actual'] = str(pred_vs_actual_path)
            
        elif task_type == "classification":
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            confusion_matrix_path = output_dir / "confusion_matrix.png"
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['confusion_matrix'] = str(confusion_matrix_path)
        
        return plot_paths
    
    def _detect_task_type(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Automatically detect task type based on target values."""
        unique_true = len(np.unique(y_true))
        unique_pred = len(np.unique(y_pred))
        
        if unique_true <= 20 and unique_pred <= 20:
            if (np.allclose(y_true, np.round(y_true)) and 
                np.allclose(y_pred, np.round(y_pred))):
                return "classification"
        
        return "regression"
    
    def save_metrics_to_csv(self, metrics: Dict[str, Any], output_path: str) -> str:
        """Save metrics to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten nested dictionaries
        flattened_metrics = self._flatten_dict(metrics)
        
        # Add timestamp
        import datetime
        flattened_metrics['evaluation_timestamp'] = datetime.datetime.now().isoformat()
        
        # Save to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=flattened_metrics.keys())
            writer.writeheader()
            writer.writerow(flattened_metrics)
        
        logger.info(f"Metrics saved to: {output_path}")
        return str(output_path)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                for i, row in enumerate(v):
                    for j, val in enumerate(row):
                        items.append((f"{new_key}_row{i}_col{j}", val))
            else:
                items.append((new_key, v))
        return dict(items)
 