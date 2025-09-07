"""
Visualization Generator

This module provides visualization generation functionality for training results.
Handles feature importance plots and other visualizations.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generates visualizations for training results.
    
    Features:
    - Feature importance plots (tree-based and permutation)
    
    - Cross-validation visualizations
    - Comparison plots
    """
    
    def __init__(self):
        """Initialize visualization generator."""
        logger.info("Initialized VisualizationGenerator")
    
    def generate_feature_importance_plots(self, model_directory: Path, feature_importance: Dict[str, Any]) -> None:
        """Generate feature importance visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style for better looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Extract data for plotting
            features = list(feature_importance.keys())
            tree_importance = [data.get('gain', 0) for data in feature_importance.values()]
            perm_importance_mean = [data.get('permutation_mean', 0) for data in feature_importance.values()]
            perm_importance_std = [data.get('permutation_std', 0) for data in feature_importance.values()]
            
            # Sort by tree importance for consistency
            sorted_indices = np.argsort(tree_importance)[::-1]  # Descending order
            sorted_features = [features[i] for i in sorted_indices]
            sorted_tree_imp = [tree_importance[i] for i in sorted_indices]
            sorted_perm_imp = [perm_importance_mean[i] for i in sorted_indices]
            sorted_perm_std = [perm_importance_std[i] for i in sorted_indices]
            
            # Limit to top 15 features for better visualization
            top_n = min(15, len(sorted_features))
            top_features = sorted_features[:top_n]
            top_tree_imp = sorted_tree_imp[:top_n]
            top_perm_imp = sorted_perm_imp[:top_n]
            top_perm_std = sorted_perm_std[:top_n]
            
            # 1. Tree-based importance plot
            self._create_tree_importance_plot(
                model_directory, top_features, top_tree_imp
            )
            
            # 2. Permutation importance plot with error bars
            if any(imp > 0 for imp in top_perm_imp):
                self._create_permutation_importance_plot(
                    model_directory, top_features, top_perm_imp, top_perm_std
                )
            
            # 3. Comparison plot (Tree vs Permutation)
            if any(imp > 0 for imp in top_perm_imp):
                self._create_importance_comparison_plot(
                    model_directory, top_features, top_tree_imp, top_perm_imp
                )
            
            logger.info("Feature importance plots generated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to generate feature importance plots: {e}")
    
    def _create_tree_importance_plot(self, model_directory: Path, features: List[str], importance: List[float]) -> None:
        """Create tree-based importance plot."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(features))
            
            bars = plt.barh(y_pos, importance, color='skyblue', alpha=0.8)
            plt.yticks(y_pos, features)
            plt.xlabel('Tree-based Feature Importance')
            plt.title('Feature Importance (Tree-based)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, importance)):
                plt.text(value + max(importance) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.4f}', va='center', fontsize=9)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            # Save tree importance plot
            tree_importance_plot_path = model_directory / "feature_importance_tree.png"
            plt.savefig(tree_importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Tree-based feature importance plot saved to: {tree_importance_plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create tree importance plot: {e}")
    
    def _create_permutation_importance_plot(self, model_directory: Path, features: List[str], 
                                          importance_mean: List[float], importance_std: List[float]) -> None:
        """Create permutation importance plot with error bars."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(features))
            
            bars = plt.barh(y_pos, importance_mean, xerr=importance_std, 
                           color='lightcoral', alpha=0.8, capsize=4)
            plt.yticks(y_pos, features)
            plt.xlabel('Permutation Feature Importance')
            plt.title('Feature Importance (Permutation-based with Standard Deviation)', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, mean_val, std_val) in enumerate(zip(bars, importance_mean, importance_std)):
                plt.text(mean_val + std_val + max(importance_mean) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{mean_val:.4f}±{std_val:.4f}', va='center', fontsize=9)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            # Save permutation importance plot
            perm_importance_plot_path = model_directory / "feature_importance_permutation.png"
            plt.savefig(perm_importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Permutation feature importance plot saved to: {perm_importance_plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create permutation importance plot: {e}")
    
    def _create_importance_comparison_plot(self, model_directory: Path, features: List[str],
                                         tree_importance: List[float], perm_importance: List[float]) -> None:
        """Create comparison plot between tree-based and permutation importance."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(features))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, tree_importance, width, label='Tree-based', 
                           color='skyblue', alpha=0.8)
            bars2 = plt.bar(x + width/2, perm_importance, width, label='Permutation-based', 
                           color='lightcoral', alpha=0.8)
            
            plt.xlabel('Features')
            plt.ylabel('Feature Importance')
            plt.title('Feature Importance Comparison: Tree-based vs Permutation-based', 
                     fontsize=14, fontweight='bold')
            plt.xticks(x, features, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(tree_importance) * 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                        
            for bar in bars2:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(perm_importance) * 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save comparison plot
            comparison_plot_path = model_directory / "feature_importance_comparison.png"
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Feature importance comparison plot saved to: {comparison_plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create importance comparison plot: {e}")
    
    def generate_cross_validation_plots(self, model_directory: Path, cv_results: Dict[str, Any], 
                                      task_type: str, target_names: Optional[List[str]] = None) -> None:
        """Generate cross-validation related plots."""
        try:
            from .training_monitor import TrainingMonitor
            
            # Create cross_validation_data directory for all CV-related files
            cv_data_dir = model_directory / "cross_validation_data"
            cv_data_dir.mkdir(exist_ok=True)
            
            # Initialize monitor for plot generation
            monitor = TrainingMonitor()
            
            # Get predictions from cv_results for visualization
            cv_predictions = cv_results.get('cv_predictions', {})
            if cv_predictions:
                y_true_original = cv_predictions.get('y_true_original')
                y_pred_original = cv_predictions.get('y_pred_original')
                y_true_processed = cv_predictions.get('y_true_processed')
                y_pred_processed = cv_predictions.get('y_pred_processed')
                y_pred_proba_processed = cv_predictions.get('y_pred_proba_processed')
                
                if task_type == "regression":
                    # Generate scatter plots for regression tasks
                    if y_true_original is not None and y_pred_original is not None:
                        original_scatter_path = cv_data_dir / "cross_validation_scatter_original.png"
                        monitor.create_cross_validation_scatter_plot(
                            cv_results=cv_results,
                            y_true=y_true_original,
                            y_pred=y_pred_original,
                            output_path=str(original_scatter_path),
                            plot_title="Cross-Validation: Predicted vs Actual (Original Scale)",
                            target_names=target_names
                        )
                    
                    # Generate scatter plot for normalized/processed data
                    if y_true_processed is not None and y_pred_processed is not None:
                        normalized_scatter_path = cv_data_dir / "cross_validation_scatter_normalized.png"
                        monitor.create_cross_validation_scatter_plot(
                            cv_results=cv_results,
                            y_true=y_true_processed,
                            y_pred=y_pred_processed,
                            output_path=str(normalized_scatter_path),
                            plot_title="Cross-Validation: Predicted vs Actual (Normalized Data)",
                            target_names=target_names
                        )
                        
                        # Also create a simple named version for easier detection
                        simple_scatter_path = cv_data_dir / "cross_validation_scatter.png"
                        monitor.create_cross_validation_scatter_plot(
                            cv_results=cv_results,
                            y_true=y_true_processed,
                            y_pred=y_pred_processed,
                            output_path=str(simple_scatter_path),
                            plot_title="Cross-Validation: Predicted vs Actual Values",
                            target_names=target_names
                        )
                        
                elif task_type == "classification":
                    # Generate ROC curves for classification tasks
                    if y_true_processed is not None and y_pred_proba_processed is not None:
                        roc_curve_path = cv_data_dir / "cross_validation_roc_curves.png"
                        
                        # Get class names from original target data if available
                        classes = np.unique(y_true_processed)
                        class_names = [str(cls) for cls in classes]
                        
                        monitor.create_roc_curve_plot(
                            y_true=y_true_processed,
                            y_pred_proba=y_pred_proba_processed,
                            class_names=class_names,
                            output_path=str(roc_curve_path),
                            plot_title="Cross-Validation ROC Curves"
                        )
                        
                        # Also create a simple named version for easier detection
                        simple_roc_path = cv_data_dir / "cross_validation_visualization.png"
                        monitor.create_roc_curve_plot(
                            y_true=y_true_processed,
                            y_pred_proba=y_pred_proba_processed,
                            class_names=class_names,
                            output_path=str(simple_roc_path),
                            plot_title="Cross-Validation Performance"
                        )
            
            logger.info("Cross-validation plots generated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to generate cross-validation plots: {e}")
    

    
    def generate_additional_scatter_plots(self, model_directory: Path, metadata: Dict[str, Any]) -> None:
        """Generate additional scatter plots for regression tasks."""
        try:
            task_type = metadata.get('task_type')
            if task_type != "regression":
                logger.info("Skipping scatter plot generation for non-regression task")
                return
            
            from .training_monitor import TrainingMonitor
            
            X = metadata.get('X')
            y = metadata.get('y')
            y_true = metadata.get('y_true')
            y_pred = metadata.get('y_pred')
            
            if not all([X is not None, y is not None, y_true is not None, y_pred is not None]):
                logger.warning("Missing data for scatter plot generation")
                return
            
            cv_data_dir = model_directory / "cross_validation_data"
            cv_data_dir.mkdir(exist_ok=True)
            
            monitor = TrainingMonitor()
            
            # Create normalized (processed) scatter plot
            normalized_plot_path = cv_data_dir / "cross_validation_scatter_normalized.png"
            
            # Get target names from metadata
            target_names = metadata.get('target_name', None)
            if not target_names:
                # Fallback to target_column if target_name not available
                target_col = metadata.get('target_column', None)
                if isinstance(target_col, list):
                    target_names = target_col
                elif isinstance(target_col, str):
                    target_names = [target_col]
            
            monitor.create_cross_validation_scatter_plot(
                y_true=y_true,
                y_pred=y_pred,
                output_path=str(normalized_plot_path),
                plot_title="Cross-Validation: Predicted vs Actual (Normalized Data)",
                target_names=target_names
            )
            
            # Try to create original scale scatter plot if preprocessing pipeline is available
            preprocessing_pipeline_path = metadata.get('preprocessing_pipeline_path')
            if preprocessing_pipeline_path:
                try:
                    import joblib
                    pipeline_path = Path(preprocessing_pipeline_path)
                    if pipeline_path.exists():
                        pipeline = joblib.load(pipeline_path)
                        
                        # Get target scaler if available
                        if hasattr(pipeline, 'target_scaler') and pipeline.target_scaler is not None:
                            # Inverse transform the normalized data
                            y_true_original = pipeline.target_scaler.inverse_transform(
                                y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
                            )
                            y_pred_original = pipeline.target_scaler.inverse_transform(
                                y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
                            )
                            
                            # Create original scale scatter plot
                            original_plot_path = cv_data_dir / "cross_validation_scatter_original.png"
                            monitor.create_cross_validation_scatter_plot(
                                y_true=y_true_original if y_true_original.ndim > 1 and y_true_original.shape[1] > 1 else y_true_original.flatten(),
                                y_pred=y_pred_original if y_pred_original.ndim > 1 and y_pred_original.shape[1] > 1 else y_pred_original.flatten(),
                                output_path=str(original_plot_path),
                                plot_title="Cross-Validation: Predicted vs Actual (Original Scale)",
                                target_names=target_names
                            )
                except Exception as e:
                    logger.warning(f"Could not generate original scale scatter plot: {e}")
            
            logger.info("Additional scatter plots generated successfully")
            
        except Exception as e:
            logger.warning(f"Could not generate additional scatter plots: {e}")
    
    def generate_evaluation_metrics_csv(self, model_directory: Path, y_true, y_pred, task_type: str) -> None:
        """Generate evaluation metrics CSV file."""
        try:
            from .metrics_evaluator import MetricsEvaluator
            
            # Ensure arrays are 1-dimensional for evaluation
            y_true_eval = y_true
            y_pred_eval = y_pred
            
            if hasattr(y_true_eval, 'flatten'):
                y_true_eval = y_true_eval.flatten()
            if hasattr(y_pred_eval, 'flatten'):
                y_pred_eval = y_pred_eval.flatten()
            
            # Save evaluation metrics to CSV in model root directory
            metrics_evaluator = MetricsEvaluator()
            eval_metrics = metrics_evaluator.evaluate_model(
                y_true_eval, y_pred_eval,
                task_type=task_type
            )
            
            eval_metrics_csv_path = model_directory / "evaluation_metrics.csv"
            metrics_evaluator.save_metrics_to_csv(eval_metrics, str(eval_metrics_csv_path))
            
            logger.info("Evaluation metrics CSV generated successfully")
            
        except Exception as e:
            logger.warning(f"Could not generate evaluation metrics: {e}") 


class DataValidationVisualizer:
    """
    Generates visualizations for data validation and feature analysis.
    
    Features:
    - Correlation heatmaps for continuous and categorical features
    - Feature distribution plots
    - Data quality visualizations
    - Multicollinearity analysis plots
    """
    
    def __init__(self):
        """Initialize data validation visualizer."""
        logger.info("Initialized DataValidationVisualizer")
    
    def generate_correlation_heatmap(
        self, 
        model_directory: Path, 
        data: pd.DataFrame, 
        target_column: str, 
        categorical_features: List[str],
        task_type: str
    ) -> None:
        """
        Generate correlation heatmap for features.
        
        Args:
            model_directory: Path to model directory
            data: DataFrame containing features and target
            target_column: Name of target column(s)
            categorical_features: List of categorical feature names
            task_type: Type of ML task ('classification' or 'regression')
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import spearmanr
            import warnings
            warnings.filterwarnings('ignore')
            
            # Create feature_plots directory
            feature_plots_dir = model_directory / "feature_plots" / "correlations"
            feature_plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for correlation analysis
            numeric_data = data.copy()
            
            # Handle target column(s)
            target_cols = target_column if isinstance(target_column, list) else [target_column]
            
            # Encode categorical features for correlation analysis
            categorical_cols = [col for col in categorical_features if col in numeric_data.columns]
            
            for col in categorical_cols:
                if numeric_data[col].dtype == 'object' or numeric_data[col].dtype.name == 'category':
                    # Label encode categorical variables for correlation
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    numeric_data[col] = le.fit_transform(numeric_data[col].astype(str))
            
            # Ensure all columns are numeric
            for col in numeric_data.columns:
                if numeric_data[col].dtype == 'object':
                    try:
                        numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                    except:
                        numeric_data = numeric_data.drop(columns=[col])
            
            # Remove rows with NaN values for correlation calculation
            clean_data = numeric_data.dropna()
            
            if clean_data.empty or clean_data.shape[1] < 2:
                logger.warning("Insufficient data for correlation analysis")
                return
            
            # Calculate correlations
            self._create_pearson_correlation_heatmap(feature_plots_dir, clean_data, target_cols, categorical_cols)
            self._create_spearman_correlation_heatmap(feature_plots_dir, clean_data, target_cols, categorical_cols)
            
            # Create focused correlation plot for highly correlated features
            self._create_high_correlation_heatmap(feature_plots_dir, clean_data, target_cols, threshold=0.7)
            
            logger.info("Correlation heatmaps generated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to generate correlation heatmaps: {e}")
    
    def _create_pearson_correlation_heatmap(
        self, 
        feature_plots_dir: Path, 
        data: pd.DataFrame, 
        target_cols: List[str],
        categorical_cols: List[str]
    ) -> None:
        """Create Pearson correlation heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate Pearson correlation
            corr_matrix = data.corr(method='pearson')
            
            # Create figure with appropriate size
            n_features = len(corr_matrix.columns)
            fig_size = max(10, min(20, n_features * 0.6))
            
            plt.figure(figsize=(fig_size, fig_size))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
            
            ax = sns.heatmap(
                corr_matrix, 
                mask=mask,
                annot=n_features <= 20,  # Only annotate if not too many features
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            
            # Highlight target columns
            target_indices = [corr_matrix.columns.get_loc(col) for col in target_cols if col in corr_matrix.columns]
            for idx in target_indices:
                ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='gold', lw=3))
            
            # Highlight categorical features
            cat_indices = [corr_matrix.columns.get_loc(col) for col in categorical_cols if col in corr_matrix.columns]
            for idx in cat_indices:
                ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='orange', lw=2, linestyle='--'))
            
            plt.title('Pearson Correlation Matrix\n(Gold: Target, Orange Dash: Categorical)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save plot
            pearson_path = feature_plots_dir / "correlation_heatmap_pearson.png"
            plt.savefig(pearson_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pearson correlation heatmap saved to: {pearson_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create Pearson correlation heatmap: {e}")
    
    def _create_spearman_correlation_heatmap(
        self, 
        feature_plots_dir: Path, 
        data: pd.DataFrame, 
        target_cols: List[str],
        categorical_cols: List[str]
    ) -> None:
        """Create Spearman correlation heatmap (better for non-linear relationships)."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import spearmanr
            
            # Calculate Spearman correlation
            corr_matrix = data.corr(method='spearman')
            
            # Create figure with appropriate size
            n_features = len(corr_matrix.columns)
            fig_size = max(10, min(20, n_features * 0.6))
            
            plt.figure(figsize=(fig_size, fig_size))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
            
            ax = sns.heatmap(
                corr_matrix, 
                mask=mask,
                annot=n_features <= 20,  # Only annotate if not too many features
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            
            # Highlight target columns
            target_indices = [corr_matrix.columns.get_loc(col) for col in target_cols if col in corr_matrix.columns]
            for idx in target_indices:
                ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='gold', lw=3))
            
            # Highlight categorical features
            cat_indices = [corr_matrix.columns.get_loc(col) for col in categorical_cols if col in corr_matrix.columns]
            for idx in cat_indices:
                ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='orange', lw=2, linestyle='--'))
            
            plt.title('Spearman Correlation Matrix (Rank-based)\n(Gold: Target, Orange Dash: Categorical)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save plot
            spearman_path = feature_plots_dir / "correlation_heatmap_spearman.png"
            plt.savefig(spearman_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spearman correlation heatmap saved to: {spearman_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create Spearman correlation heatmap: {e}")
    
    def _create_high_correlation_heatmap(
        self, 
        feature_plots_dir: Path, 
        data: pd.DataFrame, 
        target_cols: List[str],
        threshold: float = 0.7
    ) -> None:
        """Create focused heatmap for highly correlated feature pairs."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate correlation matrix
            corr_matrix = data.corr(method='pearson').abs()  # Use absolute values
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            if not high_corr_pairs:
                logger.info(f"No feature pairs with correlation >= {threshold} found")
                return
            
            # Extract relevant features for focused heatmap
            high_corr_features = set()
            for feat1, feat2, _ in high_corr_pairs:
                high_corr_features.add(feat1)
                high_corr_features.add(feat2)
            
            # Add target columns to the analysis
            high_corr_features.update([col for col in target_cols if col in data.columns])
            high_corr_features = list(high_corr_features)
            
            if len(high_corr_features) < 2:
                logger.info("Insufficient highly correlated features for focused heatmap")
                return
            
            # Create focused correlation matrix
            focused_corr = data[high_corr_features].corr(method='pearson')
            
            # Create heatmap
            fig_size = max(8, min(16, len(high_corr_features) * 0.8))
            plt.figure(figsize=(fig_size, fig_size))
            
            ax = sns.heatmap(
                focused_corr,
                annot=True,
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                fmt='.3f'
            )
            
            # Highlight target columns
            target_indices = [focused_corr.columns.get_loc(col) for col in target_cols if col in focused_corr.columns]
            for idx in target_indices:
                ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='gold', lw=4))
            
            plt.title(f'High Correlation Features (≥{threshold})\n{len(high_corr_pairs)} pairs found', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save plot
            high_corr_path = feature_plots_dir / "correlation_heatmap_high_correlations.png"
            plt.savefig(high_corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"High correlation heatmap saved to: {high_corr_path}")
            
            # Also save a summary of high correlation pairs
            summary_path = feature_plots_dir / "high_correlation_pairs.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"High Correlation Pairs (>={threshold}):\n")
                f.write("=" * 50 + "\n")
                for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                    f.write(f"{feat1} <-> {feat2}: {corr_val:.4f}\n")
            
            logger.info(f"High correlation pairs summary saved to: {summary_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create high correlation heatmap: {e}")
    
    def generate_continuous_correlation_plots(
        self,
        pearson_corr: pd.DataFrame,
        spearman_corr: pd.DataFrame,
        continuous_features: List[str],
        model_directory: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Generate correlation plots for continuous features."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            model_path = Path(model_directory)
            feature_plots_dir = model_path / "feature_plots" / "correlations"
            feature_plots_dir.mkdir(parents=True, exist_ok=True)
            
            plots_generated = []
            
            # 1. Pearson correlation heatmap
            if len(continuous_features) >= 2:
                fig_size = max(8, min(16, len(continuous_features) * 0.8))
                
                # Pearson correlation plot
                plt.figure(figsize=(fig_size, fig_size))
                mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
                
                ax = sns.heatmap(
                    pearson_corr,
                    mask=mask,
                    annot=len(continuous_features) <= 15,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8}
                )
                
                plt.title('Continuous Features - Pearson Correlation Matrix', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Features', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                pearson_path = feature_plots_dir / "continuous_pearson_correlation.png"
                plt.savefig(pearson_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('continuous_pearson_correlation')
                
                # Spearman correlation plot
                plt.figure(figsize=(fig_size, fig_size))
                ax = sns.heatmap(
                    spearman_corr,
                    mask=mask,
                    annot=len(continuous_features) <= 15,
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8}
                )
                
                plt.title('Continuous Features - Spearman Correlation Matrix', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Features', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                spearman_path = feature_plots_dir / "continuous_spearman_correlation.png"
                plt.savefig(spearman_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('continuous_spearman_correlation')
            
            # 2. High correlation pairs summary
            high_corr_pairs = []
            for i in range(len(continuous_features)):
                for j in range(i + 1, len(continuous_features)):
                    feature1 = continuous_features[i]
                    feature2 = continuous_features[j]
                    pearson_val = abs(pearson_corr.iloc[i, j])
                    spearman_val = abs(spearman_corr.iloc[i, j])
                    
                    if pearson_val >= threshold or spearman_val >= threshold:
                        high_corr_pairs.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'pearson': pearson_corr.iloc[i, j],
                            'spearman': spearman_corr.iloc[i, j],
                            'max_abs': max(pearson_val, spearman_val)
                        })
            
            if high_corr_pairs:
                summary_path = feature_plots_dir / "continuous_high_correlations.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"High Correlation Pairs in Continuous Features (>={threshold}):\n")
                    f.write("=" * 60 + "\n")
                    for pair in sorted(high_corr_pairs, key=lambda x: x['max_abs'], reverse=True):
                        f.write(f"{pair['feature1']} <-> {pair['feature2']}:\n")
                        f.write(f"  Pearson: {pair['pearson']:.4f}\n")
                        f.write(f"  Spearman: {pair['spearman']:.4f}\n\n")
                
                plots_generated.append('continuous_high_correlations_summary')
            
            return {
                'plots_generated': plots_generated,
                'high_correlations_count': len(high_corr_pairs),
                'features_analyzed': len(continuous_features)
            }
            
        except Exception as e:
            logger.error(f"Error generating continuous correlation plots: {e}")
            return {'error': str(e)}
    
    def generate_categorical_correlation_plots(
        self,
        cramers_v_matrix: pd.DataFrame,
        categorical_features: List[str],
        model_directory: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Generate correlation plots for categorical features using Cramér's V."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            model_path = Path(model_directory)
            feature_plots_dir = model_path / "feature_plots" / "correlations"
            feature_plots_dir.mkdir(parents=True, exist_ok=True)
            
            plots_generated = []
            
            if len(categorical_features) >= 2:
                fig_size = max(8, min(16, len(categorical_features) * 0.8))
                
                plt.figure(figsize=(fig_size, fig_size))
                mask = np.triu(np.ones_like(cramers_v_matrix, dtype=bool))
                
                ax = sns.heatmap(
                    cramers_v_matrix,
                    mask=mask,
                    annot=len(categorical_features) <= 15,
                    cmap='YlOrRd',
                    vmin=0, vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8}
                )
                
                plt.title("Categorical Features - Cramér's V Association Matrix", 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Features', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                cramers_path = feature_plots_dir / "categorical_cramers_v_correlation.png"
                plt.savefig(cramers_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('categorical_cramers_v_correlation')
            
            # High correlation pairs summary
            high_corr_pairs = []
            for i in range(len(categorical_features)):
                for j in range(i + 1, len(categorical_features)):
                    feature1 = categorical_features[i]
                    feature2 = categorical_features[j]
                    cramers_val = cramers_v_matrix.iloc[i, j]
                    
                    if cramers_val >= threshold:
                        high_corr_pairs.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'cramers_v': cramers_val
                        })
            
            if high_corr_pairs:
                summary_path = feature_plots_dir / "categorical_high_associations.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"High Association Pairs in Categorical Features (>={threshold}):\n")
                    f.write("=" * 60 + "\n")
                    for pair in sorted(high_corr_pairs, key=lambda x: x['cramers_v'], reverse=True):
                        f.write(f"{pair['feature1']} <-> {pair['feature2']}: {pair['cramers_v']:.4f}\n")
                
                plots_generated.append('categorical_high_associations_summary')
            
            return {
                'plots_generated': plots_generated,
                'high_associations_count': len(high_corr_pairs),
                'features_analyzed': len(categorical_features)
            }
            
        except Exception as e:
            logger.error(f"Error generating categorical correlation plots: {e}")
            return {'error': str(e)}
    
    def generate_mixed_correlation_plots(
        self,
        correlation_matrix: pd.DataFrame,
        continuous_features: List[str],
        categorical_features: List[str],
        model_directory: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Generate correlation plots for mixed continuous-categorical features."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            model_path = Path(model_directory)
            feature_plots_dir = model_path / "feature_plots" / "correlations"
            feature_plots_dir.mkdir(parents=True, exist_ok=True)
            
            plots_generated = []
            
            if len(continuous_features) > 0 and len(categorical_features) > 0:
                fig_width = max(8, min(16, len(categorical_features) * 1.2))
                fig_height = max(6, min(12, len(continuous_features) * 0.8))
                
                plt.figure(figsize=(fig_width, fig_height))
                
                ax = sns.heatmap(
                    correlation_matrix,
                    annot=len(continuous_features) * len(categorical_features) <= 50,
                    cmap='RdYlBu_r',
                    vmin=0, vmax=1,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8}
                )
                
                plt.title('Mixed Features - Correlation Ratio Matrix\n(Continuous vs Categorical)', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Categorical Features', fontsize=12)
                plt.ylabel('Continuous Features', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                mixed_path = feature_plots_dir / "mixed_correlation_ratio.png"
                plt.savefig(mixed_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('mixed_correlation_ratio')
            
            # High correlation pairs summary
            high_corr_pairs = []
            for i, cont_feature in enumerate(continuous_features):
                for j, cat_feature in enumerate(categorical_features):
                    corr_val = correlation_matrix.iloc[i, j]
                    
                    if corr_val >= threshold:
                        high_corr_pairs.append({
                            'continuous_feature': cont_feature,
                            'categorical_feature': cat_feature,
                            'correlation_ratio': corr_val
                        })
            
            if high_corr_pairs:
                summary_path = feature_plots_dir / "mixed_high_correlations.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"High Mixed Correlations (>={threshold}):\n")
                    f.write("=" * 50 + "\n")
                    for pair in sorted(high_corr_pairs, key=lambda x: x['correlation_ratio'], reverse=True):
                        f.write(f"{pair['continuous_feature']} <-> {pair['categorical_feature']}: {pair['correlation_ratio']:.4f}\n")
                
                plots_generated.append('mixed_high_correlations_summary')
            
            return {
                'plots_generated': plots_generated,
                'high_correlations_count': len(high_corr_pairs),
                'continuous_features_analyzed': len(continuous_features),
                'categorical_features_analyzed': len(categorical_features)
            }
            
        except Exception as e:
            logger.error(f"Error generating mixed correlation plots: {e}")
            return {'error': str(e)}
    
    def generate_feature_target_correlation_plots(
        self,
        target_correlations: Dict[str, Any],
        target_column: str,
        model_directory: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Generate feature-target correlation visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            model_path = Path(model_directory)
            feature_plots_dir = model_path / "feature_plots" / "correlations"
            feature_plots_dir.mkdir(parents=True, exist_ok=True)
            
            plots_generated = []
            
            # Extract correlation data
            continuous_corr = target_correlations.get('continuous_features', {})
            categorical_corr = target_correlations.get('categorical_features', {})
            
            # 1. Continuous features with target correlation bar plot
            if continuous_corr:
                features = list(continuous_corr.keys())
                values = [abs(continuous_corr[f]['value']) for f in features]
                methods = [continuous_corr[f]['method'] for f in features]
                
                # Sort by correlation strength
                sorted_data = sorted(zip(features, values, methods), key=lambda x: x[1], reverse=True)
                features, values, methods = zip(*sorted_data)
                
                fig_height = max(6, len(features) * 0.4)
                plt.figure(figsize=(10, fig_height))
                
                colors = ['darkblue' if method == 'pearson' else 'darkgreen' for method in methods]
                bars = plt.barh(range(len(features)), values, color=colors, alpha=0.7)
                
                plt.yticks(range(len(features)), features)
                plt.xlabel('Correlation Strength (Absolute Value)', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title(f'Feature-Target Correlation Strength\nTarget: {target_column} ({task_type})', 
                         fontsize=14, fontweight='bold', pad=20)
                
                # Add correlation values as text
                for i, (bar, value) in enumerate(zip(bars, values)):
                    plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
                
                # Add legend for methods
                from matplotlib.patches import Patch
                legend_elements = []
                if 'pearson' in methods:
                    legend_elements.append(Patch(facecolor='darkblue', alpha=0.7, label='Pearson Correlation'))
                if 'correlation_ratio' in methods:
                    legend_elements.append(Patch(facecolor='darkgreen', alpha=0.7, label='Correlation Ratio'))
                
                if legend_elements:
                    plt.legend(handles=legend_elements, loc='lower right')
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                continuous_target_path = feature_plots_dir / f"feature_target_continuous_correlation.png"
                plt.savefig(continuous_target_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('feature_target_continuous_correlation')
                
                # Generate continuous feature-target summary
                summary_path = feature_plots_dir / f"feature_target_continuous_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"Feature-Target Correlation Analysis\n")
                    f.write(f"Target: {target_column} ({task_type})\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("CONTINUOUS FEATURES:\n")
                    f.write("-" * 30 + "\n")
                    for feature, value, method in sorted_data:
                        original_value = continuous_corr[feature]['value']  # Get original signed value
                        f.write(f"{feature}:\n")
                        f.write(f"  Method: {method}\n")
                        f.write(f"  Correlation: {original_value:.4f}\n")
                        f.write(f"  Strength: {abs(original_value):.4f}\n\n")
                
                plots_generated.append('feature_target_continuous_summary')
            
            # 2. Categorical features with target correlation bar plot
            if categorical_corr:
                features = list(categorical_corr.keys())
                values = [categorical_corr[f]['value'] for f in features]
                methods = [categorical_corr[f]['method'] for f in features]
                
                # Sort by correlation strength
                sorted_data = sorted(zip(features, values, methods), key=lambda x: x[1], reverse=True)
                features, values, methods = zip(*sorted_data)
                
                fig_height = max(6, len(features) * 0.4)
                plt.figure(figsize=(10, fig_height))
                
                colors = ['darkred' if method == 'cramers_v' else 'darkorange' for method in methods]
                bars = plt.barh(range(len(features)), values, color=colors, alpha=0.7)
                
                plt.yticks(range(len(features)), features)
                plt.xlabel('Association Strength', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title(f'Categorical Feature-Target Association Strength\nTarget: {target_column} ({task_type})', 
                         fontsize=14, fontweight='bold', pad=20)
                
                # Add correlation values as text
                for i, (bar, value) in enumerate(zip(bars, values)):
                    plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)
                
                # Add legend for methods
                from matplotlib.patches import Patch
                legend_elements = []
                if 'cramers_v' in methods:
                    legend_elements.append(Patch(facecolor='darkred', alpha=0.7, label="Cramér's V"))
                if 'correlation_ratio' in methods:
                    legend_elements.append(Patch(facecolor='darkorange', alpha=0.7, label='Correlation Ratio'))
                
                if legend_elements:
                    plt.legend(handles=legend_elements, loc='lower right')
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                categorical_target_path = feature_plots_dir / f"feature_target_categorical_association.png"
                plt.savefig(categorical_target_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('feature_target_categorical_association')
                
                # Generate categorical feature-target summary
                summary_path = feature_plots_dir / f"feature_target_categorical_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"Feature-Target Association Analysis\n")
                    f.write(f"Target: {target_column} ({task_type})\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("CATEGORICAL FEATURES:\n")
                    f.write("-" * 30 + "\n")
                    for feature, value, method in sorted_data:
                        f.write(f"{feature}:\n")
                        f.write(f"  Method: {method}\n")
                        f.write(f"  Association: {value:.4f}\n\n")
                
                plots_generated.append('feature_target_categorical_summary')
            
            # Note: Combined ranking plot removed because different statistical methods 
            # (Pearson correlation, Cramér's V, correlation ratio) cannot be directly compared
            # as they measure different aspects of relationships with different scales and interpretations
            
            return {
                'plots_generated': plots_generated,
                'continuous_features_analyzed': len(continuous_corr),
                'categorical_features_analyzed': len(categorical_corr),
                'target_column': target_column,
                'task_type': task_type
            }
            
        except Exception as e:
            logger.error(f"Error generating feature-target correlation plots: {e}")
            return {'error': str(e)}
    
    def generate_feature_distribution_plots(
        self,
        data: pd.DataFrame,
        continuous_analysis: Dict[str, Any],
        categorical_analysis: Dict[str, Any],
        target_column: Union[str, List[str]],
        task_type: str,
        model_directory: str
    ) -> Dict[str, Any]:
        """Generate comprehensive feature distribution visualizations."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            model_path = Path(model_directory)
            feature_plots_dir = model_path / "feature_plots" / "distributions"
            feature_plots_dir.mkdir(parents=True, exist_ok=True)
            
            plots_generated = []
            
            # Handle multi-target case
            if isinstance(target_column, list):
                primary_target = target_column[0]
                target_is_multi = True
            else:
                primary_target = target_column
                target_is_multi = False
            
            # Determine if target is categorical based on task type
            target_is_categorical = task_type == 'classification'
            
            # 1. Continuous feature distributions
            if continuous_analysis and continuous_analysis.get('distribution_statistics'):
                plots_generated.extend(self._generate_continuous_distribution_plots(
                    data, continuous_analysis, primary_target, target_is_categorical, 
                    feature_plots_dir, task_type
                ))
            
            # 2. Categorical feature distributions
            if categorical_analysis and categorical_analysis.get('distribution_statistics'):
                plots_generated.extend(self._generate_categorical_distribution_plots(
                    data, categorical_analysis, primary_target, target_is_categorical,
                    feature_plots_dir, task_type
                ))
            
            # 3. Generate summary report
            summary_path = feature_plots_dir / "feature_distributions_summary.txt"
            self._generate_distribution_summary_file(
                continuous_analysis, categorical_analysis, target_column, 
                task_type, summary_path
            )
            plots_generated.append('feature_distributions_summary')
            
            return {
                'plots_generated': plots_generated,
                'continuous_features_analyzed': len(continuous_analysis.get('distribution_statistics', {})),
                'categorical_features_analyzed': len(categorical_analysis.get('distribution_statistics', {})),
                'target_column': target_column,
                'task_type': task_type,
                'target_is_multi': target_is_multi
            }
            
        except Exception as e:
            logger.error(f"Error generating feature distribution plots: {e}")
            return {'error': str(e)}
    
    def _generate_continuous_distribution_plots(
        self,
        data: pd.DataFrame,
        continuous_analysis: Dict[str, Any],
        target_column: str,
        target_is_categorical: bool,
        feature_plots_dir: Path,
        task_type: str
    ) -> List[str]:
        """Generate distribution plots for continuous features."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        plots_generated = []
        distribution_stats = continuous_analysis.get('distribution_statistics', {})
        skewness_analysis = continuous_analysis.get('skewness_analysis', {})
        outlier_detection = continuous_analysis.get('outlier_detection', {})
        
        if not distribution_stats:
            return plots_generated
        
        continuous_features = list(distribution_stats.keys())
        
        # 1. Individual distribution plots (histograms with target overlay if classification)
        n_features = len(continuous_features)
        if n_features > 0:
            # Determine subplot layout
            if n_features <= 4:
                cols = 2
                rows = (n_features + 1) // 2
            elif n_features <= 9:
                cols = 3
                rows = (n_features + 2) // 3
            else:
                cols = 4
                rows = (n_features + 3) // 4
            
            fig_height = max(8, rows * 3)
            fig_width = max(12, cols * 4)
            
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(continuous_features):
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    break
                
                feature_data = data[feature].dropna()
                
                if target_is_categorical and target_column in data.columns:
                    # Classification: Overlay distributions by target class
                    target_data = data[target_column].dropna()
                    unique_targets = target_data.unique()
                    
                    if len(unique_targets) <= 10:  # Don't overlay if too many classes
                        for target_val in unique_targets:
                            mask = data[target_column] == target_val
                            subset_data = data.loc[mask, feature].dropna()
                            if len(subset_data) > 0:
                                ax.hist(subset_data, alpha=0.6, label=f'{target_column}={target_val}', bins=20)
                        ax.legend()
                    else:
                        ax.hist(feature_data, bins=30, alpha=0.7, color='steelblue')
                else:
                    # Regression: Simple histogram
                    ax.hist(feature_data, bins=30, alpha=0.7, color='steelblue')
                
                # Add statistics annotations
                stats = distribution_stats[feature]
                skew_info = skewness_analysis.get(feature, {})
                outlier_info = outlier_detection.get(feature, {})
                
                # Add text box with key statistics
                stats_text = f"Mean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\n"
                stats_text += f"Skew: {stats['skewness']:.2f} ({skew_info.get('interpretation', 'N/A')})\n"
                stats_text += f"Outliers: {outlier_info.get('count', 0)} ({outlier_info.get('ratio', 0)*100:.1f}%)"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
                
                ax.set_title(f'{feature} Distribution', fontsize=10, fontweight='bold')
                ax.set_xlabel(feature, fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
                ax.grid(alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Continuous Features Distribution ({task_type.title()})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            dist_path = feature_plots_dir / "continuous_feature_distributions.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('continuous_feature_distributions')
        
        # 2. Box plots for outlier visualization
        if n_features > 0:
            fig_height = max(6, n_features * 0.5)
            plt.figure(figsize=(12, fig_height))
            
            # Prepare data for box plot
            box_data = []
            labels = []
            
            for feature in continuous_features:
                feature_data = data[feature].dropna()
                box_data.append(feature_data)
                labels.append(feature)
            
            plt.boxplot(box_data, labels=labels, vert=False, patch_artist=True)
            plt.title(f'Continuous Features - Outlier Detection (Box Plots)', fontsize=14, fontweight='bold')
            plt.xlabel('Feature Values', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            box_path = feature_plots_dir / "continuous_feature_outliers.png"
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('continuous_feature_outliers')
        
        # 3. Violin plots for distribution shape visualization
        if n_features > 0:
            # Include target variable if it's continuous (regression task)
            violin_features = continuous_features.copy()
            violin_data = []
            violin_labels = []
            
            # Add continuous features
            for feature in continuous_features:
                feature_data = data[feature].dropna()
                if len(feature_data) > 0:
                    violin_data.append(feature_data)
                    violin_labels.append(feature)
            
            # Add target variable if it's continuous (regression)
            if not target_is_categorical and target_column in data.columns:
                target_data = data[target_column].dropna()
                if len(target_data) > 0:
                    violin_data.append(target_data)
                    violin_labels.append(f"{target_column} (Target)")
            
            if violin_data:
                fig_height = max(8, len(violin_data) * 0.8)
                plt.figure(figsize=(12, fig_height))
                
                # Create violin plot
                positions = range(1, len(violin_data) + 1)
                parts = plt.violinplot(violin_data, positions=positions, vert=False, 
                                     showmeans=True, showmedians=True)
                
                # Customize violin plot colors
                for i, pc in enumerate(parts['bodies']):
                    if i < len(continuous_features):
                        pc.set_facecolor('lightblue')
                        pc.set_alpha(0.7)
                    else:
                        # Target variable - different color
                        pc.set_facecolor('lightcoral')
                        pc.set_alpha(0.8)
                
                # Customize other plot elements
                parts['cmeans'].set_color('red')
                parts['cmedians'].set_color('black')
                
                plt.yticks(positions, violin_labels)
                plt.xlabel('Feature Values', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title(f'Continuous Features - Distribution Shapes (Violin Plot)\n{task_type.title()} Task', 
                         fontsize=14, fontweight='bold')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='lightblue', alpha=0.7, label='Features'),
                ]
                if not target_is_categorical:
                    legend_elements.append(Patch(facecolor='lightcoral', alpha=0.8, label='Target Variable'))
                legend_elements.extend([
                    plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
                    plt.Line2D([0], [0], color='black', lw=2, label='Median')
                ])
                plt.legend(handles=legend_elements, loc='upper right')
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                violin_path = feature_plots_dir / "continuous_feature_violin_plots.png"
                plt.savefig(violin_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('continuous_feature_violin_plots')
        
        # 4. Normality assessment plot
        if n_features > 0:
            normality_tests = continuous_analysis.get('normality_tests', {})
            
            # Create normality summary plot
            features_with_tests = [f for f in continuous_features if f in normality_tests]
            if features_with_tests:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: P-values from normality tests
                test_names = []
                p_values = []
                feature_names = []
                
                for feature in features_with_tests:
                    tests = normality_tests[feature]
                    for test_name, test_result in tests.items():
                        if isinstance(test_result, dict) and 'p_value' in test_result:
                            test_names.append(f"{feature}\n({test_name})")
                            p_values.append(test_result['p_value'])
                            feature_names.append(feature)
                
                if p_values:
                    colors = ['green' if p > 0.05 else 'red' for p in p_values]
                    ax1.barh(range(len(p_values)), p_values, color=colors, alpha=0.7)
                    ax1.axvline(x=0.05, color='black', linestyle='--', alpha=0.8, label='α = 0.05')
                    ax1.set_yticks(range(len(test_names)))
                    ax1.set_yticklabels(test_names, fontsize=8)
                    ax1.set_xlabel('P-value', fontsize=12)
                    ax1.set_title('Normality Test P-values', fontsize=12, fontweight='bold')
                    ax1.legend()
                    ax1.grid(axis='x', alpha=0.3)
                
                # Plot 2: Skewness values
                skewness_values = []
                skew_labels = []
                for feature in continuous_features:
                    if feature in skewness_analysis:
                        skewness_values.append(abs(distribution_stats[feature]['skewness']))
                        skew_labels.append(feature)
                
                if skewness_values:
                    colors = ['red' if abs(s) > 1 else 'orange' if abs(s) > 0.5 else 'green' 
                             for s in skewness_values]
                    ax2.barh(range(len(skewness_values)), skewness_values, color=colors, alpha=0.7)
                    ax2.set_yticks(range(len(skew_labels)))
                    ax2.set_yticklabels(skew_labels)
                    ax2.set_xlabel('|Skewness|', fontsize=12)
                    ax2.set_title('Feature Skewness (Absolute)', fontsize=12, fontweight='bold')
                    ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.8, label='Moderate (0.5)')
                    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='High (1.0)')
                    ax2.legend()
                    ax2.grid(axis='x', alpha=0.3)
                
                plt.suptitle('Continuous Features - Normality & Skewness Assessment', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                normality_path = feature_plots_dir / "continuous_feature_normality.png"
                plt.savefig(normality_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots_generated.append('continuous_feature_normality')
        
        return plots_generated
    
    def _generate_categorical_distribution_plots(
        self,
        data: pd.DataFrame,
        categorical_analysis: Dict[str, Any],
        target_column: str,
        target_is_categorical: bool,
        feature_plots_dir: Path,
        task_type: str
    ) -> List[str]:
        """Generate distribution plots for categorical features."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        plots_generated = []
        distribution_stats = categorical_analysis.get('distribution_statistics', {})
        imbalance_analysis = categorical_analysis.get('imbalance_analysis', {})
        
        if not distribution_stats:
            return plots_generated
        
        categorical_features = list(distribution_stats.keys())
        n_features = len(categorical_features)
        
        # 1. Individual categorical distribution plots
        if n_features > 0:
            # Determine subplot layout
            if n_features <= 4:
                cols = 2
                rows = (n_features + 1) // 2
            elif n_features <= 9:
                cols = 3
                rows = (n_features + 2) // 3
            else:
                cols = 4
                rows = (n_features + 3) // 4
            
            fig_height = max(8, rows * 4)
            fig_width = max(12, cols * 4)
            
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(categorical_features):
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    break
                
                # Get value counts
                value_counts = data[feature].value_counts()
                
                if target_is_categorical and target_column in data.columns:
                    # Classification: Create stacked bar chart by target
                    crosstab = pd.crosstab(data[feature], data[target_column])
                    crosstab.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
                    ax.legend(title=target_column, bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Regression or simple count: Regular bar chart
                    value_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
                
                # Add statistics annotations
                stats = distribution_stats[feature]
                imbalance_info = imbalance_analysis.get(feature, {})
                
                stats_text = f"Unique: {stats['unique_count']}\n"
                stats_text += f"Most common: {stats['most_common_value']} ({stats['most_common_ratio']*100:.1f}%)\n"
                if 'imbalance_ratio' in imbalance_info:
                    stats_text += f"Imbalance: {imbalance_info['imbalance_ratio']:.2f}\n"
                stats_text += f"Severity: {imbalance_info.get('severity', 'N/A')}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
                
                ax.set_title(f'{feature} Distribution', fontsize=10, fontweight='bold')
                ax.set_xlabel(feature, fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Categorical Features Distribution ({task_type.title()})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            cat_dist_path = feature_plots_dir / "categorical_feature_distributions.png"
            plt.savefig(cat_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('categorical_feature_distributions')
        
        # 2. Cardinality and imbalance summary
        if n_features > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Cardinality (number of unique values)
            cardinalities = []
            feature_names = []
            for feature in categorical_features:
                stats = distribution_stats[feature]
                cardinalities.append(stats['unique_count'])
                feature_names.append(feature)
            
            colors_card = ['red' if c > 50 else 'orange' if c > 10 else 'green' for c in cardinalities]
            ax1.barh(range(len(cardinalities)), cardinalities, color=colors_card, alpha=0.7)
            ax1.set_yticks(range(len(feature_names)))
            ax1.set_yticklabels(feature_names)
            ax1.set_xlabel('Number of Unique Values', fontsize=12)
            ax1.set_title('Categorical Features Cardinality', fontsize=12, fontweight='bold')
            ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.8, label='Medium (10)')
            ax1.axvline(x=50, color='red', linestyle='--', alpha=0.8, label='High (50)')
            ax1.legend()
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: Imbalance ratios
            imbalance_ratios = []
            imbalance_labels = []
            for feature in categorical_features:
                if feature in imbalance_analysis:
                    imbalance_info = imbalance_analysis[feature]
                    if 'imbalance_ratio' in imbalance_info:
                        imbalance_ratios.append(imbalance_info['imbalance_ratio'])
                        imbalance_labels.append(feature)
            
            if imbalance_ratios:
                colors_imb = ['red' if r > 10 else 'orange' if r > 5 else 'green' for r in imbalance_ratios]
                ax2.barh(range(len(imbalance_ratios)), imbalance_ratios, color=colors_imb, alpha=0.7)
                ax2.set_yticks(range(len(imbalance_labels)))
                ax2.set_yticklabels(imbalance_labels)
                ax2.set_xlabel('Imbalance Ratio (Most/Least Common)', fontsize=12)
                ax2.set_title('Categorical Features Imbalance', fontsize=12, fontweight='bold')
                ax2.axvline(x=5, color='orange', linestyle='--', alpha=0.8, label='Moderate (5:1)')
                ax2.axvline(x=10, color='red', linestyle='--', alpha=0.8, label='High (10:1)')
                ax2.legend()
                ax2.grid(axis='x', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No imbalance data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Categorical Features Imbalance', fontsize=12, fontweight='bold')
            
            plt.suptitle('Categorical Features - Cardinality & Imbalance Assessment', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            cardinality_path = feature_plots_dir / "categorical_feature_cardinality.png"
            plt.savefig(cardinality_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_generated.append('categorical_feature_cardinality')
        
        return plots_generated
    
    def _generate_distribution_summary_file(
        self,
        continuous_analysis: Dict[str, Any],
        categorical_analysis: Dict[str, Any],
        target_column: Union[str, List[str]],
        task_type: str,
        summary_path: Path
    ) -> None:
        """Generate a comprehensive text summary of feature distributions."""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("FEATURE DISTRIBUTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Task Type: {task_type.title()}\n")
            f.write(f"Target Column(s): {target_column}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Continuous features summary
            if continuous_analysis and continuous_analysis.get('distribution_statistics'):
                f.write("CONTINUOUS FEATURES ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                cont_stats = continuous_analysis['distribution_statistics']
                skew_analysis = continuous_analysis.get('skewness_analysis', {})
                outlier_analysis = continuous_analysis.get('outlier_detection', {})
                normality_tests = continuous_analysis.get('normality_tests', {})
                
                for feature, stats in cont_stats.items():
                    f.write(f"\n{feature}:\n")
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Std: {stats['std']:.4f}\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    f.write(f"  Quartiles: Q1={stats['q25']:.4f}, Q2={stats['median']:.4f}, Q3={stats['q75']:.4f}\n")
                    
                    # Skewness
                    if feature in skew_analysis:
                        skew_info = skew_analysis[feature]
                        f.write(f"  Skewness: {skew_info['value']:.4f} ({skew_info.get('interpretation', 'N/A')})\n")
                        f.write(f"  Skew Severity: {skew_info.get('severity', 'N/A')}\n")
                    
                    # Outliers
                    if feature in outlier_analysis:
                        outlier_info = outlier_analysis[feature]
                        f.write(f"  Outliers: {outlier_info['count']} ({outlier_info['ratio']*100:.2f}%)\n")
                        f.write(f"  Outlier Severity: {outlier_info.get('severity', 'N/A')}\n")
                    
                    # Normality
                    if feature in normality_tests:
                        f.write(f"  Normality Tests:\n")
                        for test_name, test_result in normality_tests[feature].items():
                            if isinstance(test_result, dict) and 'p_value' in test_result:
                                is_normal = "Yes" if test_result['is_normal'] else "No"
                                f.write(f"    {test_name}: p={test_result['p_value']:.4f} (Normal: {is_normal})\n")
            
            # Categorical features summary
            if categorical_analysis and categorical_analysis.get('distribution_statistics'):
                f.write(f"\n\nCATEGORICAL FEATURES ANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                cat_stats = categorical_analysis['distribution_statistics']
                imbalance_analysis = categorical_analysis.get('imbalance_analysis', {})
                
                for feature, stats in cat_stats.items():
                    f.write(f"\n{feature}:\n")
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Unique Values: {stats['unique_count']}\n")
                    f.write(f"  Most Common: '{stats['most_common_value']}' ({stats['most_common_ratio']*100:.2f}%)\n")
                    f.write(f"  Least Common: '{stats['least_common_value']}' ({stats['least_common_ratio']*100:.2f}%)\n")
                    
                    # Cardinality assessment
                    cardinality = stats['unique_count']
                    if cardinality > 50:
                        cardinality_level = "Very High (>50)"
                    elif cardinality > 10:
                        cardinality_level = "High (10-50)"
                    elif cardinality > 5:
                        cardinality_level = "Medium (5-10)"
                    else:
                        cardinality_level = "Low (≤5)"
                    f.write(f"  Cardinality: {cardinality_level}\n")
                    
                    # Imbalance
                    if feature in imbalance_analysis:
                        imbalance_info = imbalance_analysis[feature]
                        if 'imbalance_ratio' in imbalance_info:
                            f.write(f"  Imbalance Ratio: {imbalance_info['imbalance_ratio']:.2f}:1\n")
                            f.write(f"  Imbalance Severity: {imbalance_info.get('severity', 'N/A')}\n")
            
            # Overall recommendations
            f.write(f"\n\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            recommendations = []
            
            # Continuous feature recommendations
            if continuous_analysis:
                cont_issues = continuous_analysis.get('issues', [])
                for issue in cont_issues:
                    if 'skew' in issue.lower():
                        recommendations.append("Consider applying log transformation or Box-Cox transformation to highly skewed features")
                    elif 'outlier' in issue.lower():
                        recommendations.append("Investigate and handle outliers using robust scaling or outlier detection methods")
                    elif 'normal' in issue.lower():
                        recommendations.append("Consider normality transformations for features that will be used in algorithms assuming normality")
            
            # Categorical feature recommendations
            if categorical_analysis:
                cat_issues = categorical_analysis.get('issues', [])
                for issue in cat_issues:
                    if 'cardinality' in issue.lower():
                        recommendations.append("Consider feature engineering for high-cardinality categorical features (target encoding, frequency encoding)")
                    elif 'imbalance' in issue.lower():
                        recommendations.append("Address categorical imbalance through sampling techniques or feature grouping")
            
            # Task-specific recommendations
            if task_type == 'classification':
                recommendations.append("For classification tasks, ensure feature distributions don't indicate data leakage")
                recommendations.append("Consider feature scaling for distance-based algorithms")
            else:  # regression
                recommendations.append("For regression tasks, check if target transformation is needed based on feature-target relationships")
                recommendations.append("Consider polynomial features for non-linear relationships")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("No specific recommendations identified.\n")
    
    def generate_vif_plots(
        self,
        vif_results: Dict[str, Any],
        model_directory: str
    ) -> Dict[str, Any]:
        """
        Generate VIF (Variance Inflation Factor) visualization plots.
        
        Args:
            vif_results: VIF analysis results from detect_multicollinearity
            model_directory: Directory to save plots
            
        Returns:
            Dictionary with plot generation results
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from pathlib import Path
        
        model_path = Path(model_directory)
        feature_plots_dir = model_path / "feature_plots" / "correlations"
        feature_plots_dir.mkdir(parents=True, exist_ok=True)
        
        plots_generated = []
        
        # Extract VIF data
        vif_scores = vif_results.get('vif_scores', {})
        high_vif_features = vif_results.get('high_vif_features', [])
        categorical_vif_details = vif_results.get('categorical_vif_details', {})
        encoding_details = vif_results.get('encoding_details', {})
        vif_threshold = vif_results.get('vif_threshold', 5.0)
        
        if not vif_scores:
            return {
                'plots_generated': plots_generated,
                'message': 'No VIF scores available for visualization'
            }
        
        # 1. Main VIF scores bar plot
        self._create_vif_scores_plot(
            vif_scores, high_vif_features, vif_threshold, 
            encoding_details, feature_plots_dir
        )
        plots_generated.append('vif_scores')
        
        # 2. VIF threshold analysis plot
        if len(vif_scores) > 1:
            self._create_vif_threshold_analysis_plot(
                vif_scores, vif_threshold, feature_plots_dir
            )
            plots_generated.append('vif_threshold_analysis')
        
        # 3. Categorical VIF details plot (if any one-hot encoded features)
        if categorical_vif_details:
            self._create_categorical_vif_details_plot(
                categorical_vif_details, vif_threshold, feature_plots_dir
            )
            plots_generated.append('categorical_vif_details')
        
        # 4. Encoding strategy summary plot
        if encoding_details:
            self._create_encoding_strategy_plot(
                encoding_details, vif_scores, feature_plots_dir
            )
            plots_generated.append('encoding_strategy_summary')
        
        return {
            'plots_generated': plots_generated,
            'plot_paths': [feature_plots_dir / f"{plot}.png" for plot in plots_generated],
            'summary': f"Generated {len(plots_generated)} VIF visualization plots"
        }
    
    def _create_vif_scores_plot(
        self,
        vif_scores: Dict[str, float],
        high_vif_features: List[Dict],
        vif_threshold: float,
        encoding_details: Dict[str, Any],
        feature_plots_dir: Path
    ) -> None:
        """Create main VIF scores bar plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        features = list(vif_scores.keys())
        scores = list(vif_scores.values())
        
        # Sort by VIF score (descending)
        sorted_data = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_data)
        
        # Create colors based on encoding method and VIF level
        colors = []
        for feature in features:
            vif_score = vif_scores[feature]
            encoding_method = encoding_details.get(feature, {}).get('method', 'continuous')
            
            if vif_score >= vif_threshold:
                if encoding_method == 'one_hot':
                    colors.append('#FF6B6B')  # Red for high VIF one-hot
                else:
                    colors.append('#FF4757')  # Dark red for high VIF others
            else:
                if encoding_method == 'one_hot':
                    colors.append('#4ECDC4')  # Teal for normal one-hot
                elif encoding_method == 'target_encoding':
                    colors.append('#45B7D1')  # Blue for target encoded
                else:
                    colors.append('#96CEB4')  # Green for continuous
        
        # Create horizontal bar plot
        fig_height = max(8, len(features) * 0.5)
        plt.figure(figsize=(12, fig_height))
        
        bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
        
        # Add threshold line
        plt.axvline(x=vif_threshold, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'VIF Threshold ({vif_threshold})')
        
        # Customize plot
        plt.yticks(range(len(features)), features)
        plt.xlabel('VIF Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Variance Inflation Factor (VIF) Analysis\nMulticollinearity Detection', 
                 fontsize=14, fontweight='bold')
        
        # Add VIF score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', va='center', fontsize=10)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#96CEB4', alpha=0.8, label='Continuous Features'),
            Patch(facecolor='#4ECDC4', alpha=0.8, label='One-hot Encoded'),
            Patch(facecolor='#45B7D1', alpha=0.8, label='Target Encoded'),
            Patch(facecolor='#FF6B6B', alpha=0.8, label=f'High VIF (≥{vif_threshold})'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='VIF Threshold')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        vif_path = feature_plots_dir / "vif_scores.png"
        plt.savefig(vif_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_vif_threshold_analysis_plot(
        self,
        vif_scores: Dict[str, float],
        vif_threshold: float,
        feature_plots_dir: Path
    ) -> None:
        """Create VIF threshold analysis plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        scores = list(vif_scores.values())
        
        # Create threshold analysis
        thresholds = np.arange(1, 15, 0.5)
        high_vif_counts = []
        
        for threshold in thresholds:
            count = sum(1 for score in scores if score >= threshold)
            high_vif_counts.append(count)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(thresholds, high_vif_counts, 'b-', linewidth=2, marker='o', markersize=4)
        plt.axvline(x=vif_threshold, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'Current Threshold ({vif_threshold})')
        
        # Highlight common thresholds
        common_thresholds = [2.5, 5.0, 10.0]
        for ct in common_thresholds:
            if ct != vif_threshold:
                plt.axvline(x=ct, color='gray', linestyle=':', alpha=0.5)
                
        plt.xlabel('VIF Threshold', fontsize=12)
        plt.ylabel('Number of Features with High VIF', fontsize=12)
        plt.title('VIF Threshold Sensitivity Analysis\nHow threshold choice affects feature selection', 
                 fontsize=14, fontweight='bold')
        
        # Add annotations for common thresholds
        current_count = sum(1 for score in scores if score >= vif_threshold)
        plt.annotate(f'{current_count} features', 
                    xy=(vif_threshold, current_count), 
                    xytext=(vif_threshold + 1, current_count + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        threshold_path = feature_plots_dir / "vif_threshold_analysis.png"
        plt.savefig(threshold_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_categorical_vif_details_plot(
        self,
        categorical_vif_details: Dict[str, Any],
        vif_threshold: float,
        feature_plots_dir: Path
    ) -> None:
        """Create detailed VIF plot for one-hot encoded categorical features."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not categorical_vif_details:
            return
        
        # Prepare data
        categories = list(categorical_vif_details.keys())
        n_categories = len(categories)
        
        fig_height = max(8, n_categories * 3)
        fig, axes = plt.subplots(n_categories, 1, figsize=(12, fig_height))
        
        if n_categories == 1:
            axes = [axes]
        
        for i, (category, details) in enumerate(categorical_vif_details.items()):
            ax = axes[i]
            
            # Get individual VIF scores for dummy variables
            individual_vifs = details['individual_vifs']
            dummy_names = list(individual_vifs.keys())
            dummy_scores = list(individual_vifs.values())
            
            # Create bar plot
            colors = ['red' if score >= vif_threshold else 'steelblue' for score in dummy_scores]
            bars = ax.bar(range(len(dummy_names)), dummy_scores, color=colors, alpha=0.7)
            
            # Add threshold line
            ax.axhline(y=vif_threshold, color='red', linestyle='--', alpha=0.8)
            
            # Add summary statistics
            max_vif = details['max_vif']
            mean_vif = details['mean_vif']
            ax.text(0.02, 0.98, f'Max VIF: {max_vif:.2f}\nMean VIF: {mean_vif:.2f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Customize subplot
            ax.set_xticks(range(len(dummy_names)))
            ax.set_xticklabels(dummy_names, rotation=45, ha='right')
            ax.set_ylabel('VIF Score')
            ax.set_title(f'{category} - Dummy Variables VIF Analysis', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, dummy_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(dummy_scores) * 0.01,
                       f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('One-hot Encoded Features - Detailed VIF Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        categorical_path = feature_plots_dir / "categorical_vif_details.png"
        plt.savefig(categorical_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_encoding_strategy_plot(
        self,
        encoding_details: Dict[str, Any],
        vif_scores: Dict[str, float],
        feature_plots_dir: Path
    ) -> None:
        """Create encoding strategy summary plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Count features by encoding method
        encoding_counts = {}
        encoding_vif_stats = {}
        
        for feature, details in encoding_details.items():
            method = details['method']
            if method not in encoding_counts:
                encoding_counts[method] = 0
                encoding_vif_stats[method] = []
            
            encoding_counts[method] += 1
            if feature in vif_scores:
                encoding_vif_stats[method].append(vif_scores[feature])
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Encoding method distribution
        methods = list(encoding_counts.keys())
        counts = list(encoding_counts.values())
        colors = ['#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(methods)]
        
        wedges, texts, autotexts = ax1.pie(counts, labels=methods, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Feature Encoding Strategy Distribution', fontweight='bold')
        
        # Plot 2: VIF by encoding method
        method_names = []
        vif_means = []
        vif_stds = []
        
        for method in methods:
            if encoding_vif_stats[method]:
                method_names.append(method)
                vif_means.append(np.mean(encoding_vif_stats[method]))
                vif_stds.append(np.std(encoding_vif_stats[method]))
        
        if method_names:
            x_pos = np.arange(len(method_names))
            bars = ax2.bar(x_pos, vif_means, yerr=vif_stds, capsize=5, 
                          color=colors[:len(method_names)], alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(method_names, rotation=45, ha='right')
            ax2.set_ylabel('Mean VIF Score')
            ax2.set_title('Average VIF by Encoding Method', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, vif_means):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(vif_means) * 0.01,
                        f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Cardinality vs VIF scatter
        cardinalities = []
        vif_values = []
        method_colors = []
        feature_labels = []
        
        color_map = {'one_hot': '#4ECDC4', 'target_encoding': '#45B7D1', 
                    'label_encoding_fallback': '#FFA07A'}
        
        for feature, details in encoding_details.items():
            if feature in vif_scores:
                cardinalities.append(details['unique_count'])
                vif_values.append(vif_scores[feature])
                method_colors.append(color_map.get(details['method'], '#98D8C8'))
                feature_labels.append(f"{feature}\n({details['method']})")
        
        if cardinalities:
            scatter = ax3.scatter(cardinalities, vif_values, c=method_colors, 
                                alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            ax3.set_xlabel('Feature Cardinality (Unique Values)')
            ax3.set_ylabel('VIF Score')
            ax3.set_title('Feature Cardinality vs VIF Score', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add threshold lines
            ax3.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='VIF = 5.0')
            ax3.axvline(x=10, color='blue', linestyle='--', alpha=0.7, label='Cardinality = 10')
            ax3.legend()
        
        # Plot 4: Feature cardinality distribution
        all_cardinalities = [details['unique_count'] for details in encoding_details.values()]
        if all_cardinalities:
            ax4.hist(all_cardinalities, bins=min(20, len(set(all_cardinalities))), 
                    color='skyblue', alpha=0.7, edgecolor='black')
            ax4.axvline(x=10, color='blue', linestyle='--', alpha=0.7, 
                       label='One-hot threshold (10)')
            ax4.axvline(x=50, color='red', linestyle='--', alpha=0.7, 
                       label='Exclusion threshold (50)')
            ax4.set_xlabel('Feature Cardinality')
            ax4.set_ylabel('Number of Features')
            ax4.set_title('Distribution of Feature Cardinality', fontweight='bold')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Feature Encoding Strategy & VIF Analysis Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        strategy_path = feature_plots_dir / "encoding_strategy_summary.png"
        plt.savefig(strategy_path, dpi=300, bbox_inches='tight')
        plt.close()