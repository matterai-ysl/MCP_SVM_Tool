"""Training Process Monitoring Module"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from sklearn.model_selection import validation_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 添加ROC曲线相关导入
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training process and record metrics."""
    
    def __init__(self):
        self.training_history = []
        self.validation_history = []
        self.metrics_history = []
        
    
    
    def create_cross_validation_scatter_plot(self,
                                           cv_results: Dict[str, Any] = None,
                                           y_true: np.ndarray = None,
                                           y_pred: np.ndarray = None,
                                           output_path: str = None,
                                           plot_title: str = None,
                                           target_names: Union[str, List[str]] = None) -> str:
        """Create scatter plot for cross-validation results showing predicted vs actual values."""
        
        # Set default title
        if plot_title is None:
            plot_title = 'Cross-Validation: Predicted vs Actual Values'
        
        # Priority: use direct y_true and y_pred if provided
        if y_true is not None and y_pred is not None:
            # Check if this is multi-target regression
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                # Multi-target case
                n_targets = y_true.shape[1]
                n_cols = min(3, n_targets)
                n_rows = (n_targets + n_cols - 1) // n_cols
                
                plt.figure(figsize=(6*n_cols, 6*n_rows))
                
                # Get target names for plot titles
                if isinstance(target_names, list) and len(target_names) == n_targets:
                    target_name_list = target_names
                elif isinstance(target_names, str):
                    target_name_list = [f"{target_names}_{i}" for i in range(n_targets)]
                else:
                    # Fallback to generic names
                    target_name_list = [f"Target {i}" for i in range(n_targets)]
                
                for i in range(n_targets):
                    plt.subplot(n_rows, n_cols, i+1)
                    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=50, c='blue', 
                              edgecolors='k', linewidth=0.5, label='Predictions')
                    
                    # Add perfect prediction line
                    min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
                    max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                            label='Perfect Prediction')
                    
                    # Calculate multiple metrics
                    r2 = r2_score(y_true[:, i], y_pred[:, i])
                    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
                    rmse = np.sqrt(mse)
                    
                    # Add metrics text with improved formatting
                    metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}'
                    plt.text(0.05, 0.95, metrics_text, 
                            transform=plt.gca().transAxes, fontsize=10,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    
                    # Use actual target name in axis labels and title
                    target_name = target_name_list[i]
                    plt.xlabel(f'True Values', fontsize=11)
                    plt.ylabel(f'Predicted Values', fontsize=11)
                    plt.title(f'{target_name}: R² = {r2:.4f}', fontsize=12)
                    plt.legend(fontsize=9)
                    plt.grid(True, alpha=0.3)
                    
                    # Set equal limits for better interpretation
                    lims = [min_val, max_val]
                    plt.xlim(lims)
                    plt.ylim(lims)
                    plt.gca().set_aspect('equal', adjustable='box')
                
                plt.tight_layout()
                
            else:
                # Single target case
                # Handle multi-dimensional targets by flattening
                if y_true.ndim > 1:
                    y_true_flat = y_true.flatten()
                    y_pred_flat = y_pred.flatten()
                else:
                    y_true_flat = y_true
                    y_pred_flat = y_pred
                
                plt.figure(figsize=(12, 8))
                
                # Create predicted vs actual scatter plot
                plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=50, c='blue', label='Predictions')
                
                # Add perfect prediction line
                min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
                max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                        label='Perfect Prediction')
                
                # Calculate and display multiple metrics
                r2 = r2_score(y_true_flat, y_pred_flat)
                mae = mean_absolute_error(y_true_flat, y_pred_flat)
                mse = mean_squared_error(y_true_flat, y_pred_flat)
                rmse = np.sqrt(mse)
                
                # Add metrics text with improved formatting
                metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}'
                plt.text(0.05, 0.95, metrics_text, 
                        transform=plt.gca().transAxes, fontsize=12,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(plot_title)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Set equal limits for better interpretation
                lims = [min_val, max_val]
                plt.xlim(lims)
                plt.ylim(lims)
                
                # Add diagonal line for reference
                plt.gca().set_aspect('equal', adjustable='box')
            
        elif cv_results and 'cv_scores' in cv_results:
            # Fallback: original CV scores plot if no prediction data available
            plt.figure(figsize=(12, 8))
            scores = cv_results['cv_scores']
            folds = list(range(1, len(scores) + 1))
            
            plt.scatter(folds, scores, s=100, alpha=0.7, c='blue')
            plt.plot(folds, scores, '--', alpha=0.5, c='blue')
            
            # Add mean line
            mean_score = np.mean(scores)
            plt.axhline(y=mean_score, color='red', linestyle='-', linewidth=2, 
                       label=f'Mean: {mean_score:.4f}')
            
            # Add std band
            std_score = np.std(scores)
            plt.axhline(y=mean_score + std_score, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=mean_score - std_score, color='red', linestyle='--', alpha=0.5)
            plt.fill_between(folds, mean_score - std_score, mean_score + std_score, 
                           alpha=0.2, color='red', label=f'±1 Std: {std_score:.4f}')
            
            plt.xlabel('Cross-Validation Fold')
            plt.ylabel('Score')
            plt.title(plot_title if plot_title != 'Cross-Validation: Predicted vs Actual Values' 
                     else 'Cross-Validation Scores Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Annotate points
            for i, score in enumerate(scores):
                plt.annotate(f'{score:.3f}', (folds[i], score), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        else:
            # No data available
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'No prediction data available for scatter plot', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)
            plt.title('Cross-Validation Results - No Data')
        
        # Save plot
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cross-validation scatter plot saved to: {output_path}")
            return str(output_path)
        else:
            plt.show()
            return "plot_displayed"
    
    def save_training_metrics_csv(self, 
                                 metrics_data: Dict[str, Any],
                                 output_path: str) -> str:
        """Save training metrics to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV - generic metrics
        df = pd.DataFrame([metrics_data])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Training metrics saved to: {output_path}")
        return str(output_path)
    
    def save_cv_results_csv(self, 
                           cv_results: Dict[str, Any],
                           output_path: str) -> str:
        """Save cross-validation results to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare CV data
        cv_data = []
        if 'cv_scores' in cv_results:
            scores = cv_results['cv_scores']
            for i, score in enumerate(scores):
                cv_data.append({
                    'fold': i + 1,
                    'score': score,
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                })
        
        # Add metadata
        if cv_data:
            for entry in cv_data:
                entry.update({
                    'total_folds': len(cv_results.get('cv_scores', [])),
                    'best_score': cv_results.get('best_score', None),
                    'scoring_metric': cv_results.get('scoring_metric', 'unknown')
                })
        
        df = pd.DataFrame(cv_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Cross-validation results saved to: {output_path}")
        return str(output_path)

    def create_roc_curve_plot(self,
                             y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             output_path: str = None,
                             plot_title: str = None) -> str:
        """
        Create ROC curve plot for classification tasks.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            class_names: List of class names
            output_path: Path to save the plot
            plot_title: Title for the plot
            
        Returns:
            Path to saved plot
        """
        
        if plot_title is None:
            plot_title = 'ROC Curves'
        
        # Get unique classes
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        # Set up class names
        if class_names is None:
            class_names = [f'Class {cls}' for cls in classes]
        elif len(class_names) != n_classes:
            class_names = [f'Class {cls}' for cls in classes]
        
        plt.figure(figsize=(12, 8))
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            
        else:
            # Multi-class classification (One-vs-Rest)
            # Binarize the output
            y_true_binarized = label_binarize(y_true, classes=classes)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves for each class
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.4f})')
            
            # Compute and plot micro-average ROC curve
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"],
                    label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.4f})',
                    color='deeppink', linestyle=':', linewidth=4)
            
            # Compute and plot macro-average ROC curve
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
            plt.plot(fpr["macro"], tpr["macro"],
                    label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.4f})',
                    color='navy', linestyle=':', linewidth=4)
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        
        if n_classes == 2:
            plt.title(f'{plot_title} - Binary Classification')
        else:
            plt.title(f'{plot_title} - Multi-class Classification (One-vs-Rest)')
        
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve plot saved to: {output_path}")
            return str(output_path)
        else:
            plt.show()
            return "plot_displayed"
