"""
Cross-Validation Strategy Module

This module provides comprehensive cross-validation functionality for 
model evaluation using scikit-learn's cross_val_score with support for
different metrics for regression and classification tasks.
"""

import logging
from typing import Dict, Any, List, Union, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    train_test_split, validation_curve
)
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score, 
    # Classification metrics  
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
import warnings

logger = logging.getLogger(__name__)

class CrossValidationStrategy:
    """
    Comprehensive cross-validation strategy for model evaluation.
    
    Supports different evaluation metrics for regression and classification tasks
    with flexible folding strategies and detailed results reporting.
    """
    
    # Supported metrics for different task types
    REGRESSION_METRICS = {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error', 
        'RMSE': 'neg_root_mean_squared_error',
        'R2': 'r2'
    }
    
    CLASSIFICATION_METRICS = {
        'accuracy': 'accuracy',
        'f1': 'f1_weighted',  # Use weighted for multiclass
        'f1_macro': 'f1_macro',
        'f1_micro': 'f1_micro',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'auc': 'roc_auc_ovr_weighted'  # For multiclass AUC
    }
    
    def __init__(self, 
                 cv_folds: int = 5,
                 random_state: Optional[int] = 42,
                 shuffle: bool = True,
                 stratify: bool = True):
        """
        Initialize CrossValidationStrategy.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            shuffle: Whether to shuffle data before splitting
            stratify: Whether to use stratified folding for classification
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        
        logger.info(f"Initialized CrossValidationStrategy with {cv_folds} folds")
    
    def _get_cv_splitter(self, X: np.ndarray, y: np.ndarray, task_type: str):
        """
        Get appropriate cross-validation splitter based on task type.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: "regression" or "classification"
            
        Returns:
            Cross-validation splitter
        """
        if task_type == "classification" and self.stratify:
            # Check if stratification is possible
            unique_classes, counts = np.unique(y, return_counts=True)
            min_class_count = np.min(counts)
            
            if min_class_count >= self.cv_folds:
                return StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state
                )
            else:
                logger.warning(f"Cannot stratify: min class count ({min_class_count}) < cv_folds ({self.cv_folds}). Using regular KFold.")
                
        # Use regular KFold for regression or when stratification isn't possible
        return KFold(
            n_splits=self.cv_folds,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Automatically detect task type based on target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Detected task type: "regression" or "classification"
        """
        unique_values = np.unique(y)
        
        # If target has few unique values and they are integers/strings, likely classification
        if len(unique_values) <= 50 and (
            np.all(unique_values == unique_values.astype(int)) or 
            np.issubdtype(y.dtype, np.integer) or
            np.issubdtype(y.dtype, np.str_) or
            np.issubdtype(y.dtype, np.object_)
        ):
            return "classification"
        else:
            return "regression"
    
    def _get_scoring_metrics(self, task_type: str, metrics: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get scoring metrics for cross-validation based on task type.
        
        Args:
            task_type: "regression" or "classification"
            metrics: Optional list of specific metrics to use
            
        Returns:
            Dictionary of metric names and scikit-learn scoring strings
        """
        if task_type == "regression":
            available_metrics = self.REGRESSION_METRICS
            default_metrics = ['MAE', 'MSE', 'R2']
        else:
            available_metrics = self.CLASSIFICATION_METRICS
            default_metrics = ['accuracy', 'f1', 'precision', 'recall']
            
        if metrics is None:
            metrics = default_metrics
            
        scoring = {}
        for metric in metrics:
            if metric in available_metrics:
                scoring[metric] = available_metrics[metric]
            else:
                logger.warning(f"Metric '{metric}' not supported for {task_type}. Skipping.")
                
        return scoring
    
    def cross_validate_model(self,
                           estimator: BaseEstimator,
                           X: Union[np.ndarray, pd.DataFrame],
                           y: np.ndarray,
                           task_type: Optional[str] = None,
                           metrics: Optional[List[str]] = None,
                           return_train_score: bool = True,
                           save_data: bool = False,
                           output_dir: str = "cv_results",
                           data_name: str = "cv_data",
                           preprocessor: Optional[Any] = None,
                           feature_names: Optional[List[str]] = None,
                           original_X: Optional[pd.DataFrame] = None,
                           original_y: Optional[np.ndarray] = None,
                           original_feature_names: Optional[List[str]] = None,
                           original_target_name: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on a model with comprehensive metrics.
        
        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix
            y: Target vector
            task_type: Optional task type ("regression"/"classification"). Auto-detected if None.
            metrics: Optional list of metrics to evaluate
            return_train_score: Whether to return training scores
            save_data: Whether to save cross-validation data to CSV files
            output_dir: Directory to save CSV files and plots
            data_name: Base name for saved files
            preprocessor: Optional data preprocessor for inverse transform
            feature_names: Optional feature names for saved data
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            # Use provided original data or create from current data
            if original_X is not None and original_y is not None:
                # Use provided original data with proper column names
                original_X_df = original_X.copy()
                original_y_values = original_y.copy()
                if original_feature_names is not None:
                    original_X_df.columns = original_feature_names[:original_X_df.shape[1]]
                original_target_col = original_target_name if original_target_name else 'target'
            else:
                # Fallback: Store current data as "original" (this is the processed data case)
                original_X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
                original_y_values = y.copy()
                if feature_names is not None:
                    original_X_df.columns = feature_names[:original_X_df.shape[1]]
                elif not isinstance(X, pd.DataFrame):
                    original_X_df.columns = [f"feature_{i}" for i in range(original_X_df.shape[1])]
                original_target_col = 'target'
            
            # Convert DataFrame to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
                
            # Auto-detect task type if not provided
            if task_type is None:
                task_type = self._detect_task_type(y)
                
            logger.info(f"Starting cross-validation for {task_type} task with {self.cv_folds} folds")
            
            # Get appropriate CV splitter
            cv_splitter = self._get_cv_splitter(X_array, y, task_type)
            
            # Get scoring metrics
            scoring = self._get_scoring_metrics(task_type, metrics)
            
            if not scoring:
                raise ValueError(f"No valid metrics specified for {task_type} task")
            
            # Perform cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                cv_results = cross_validate(
                    estimator=estimator,
                    X=X_array,
                    y=y,
                    cv=cv_splitter,
                    scoring=scoring,
                    return_train_score=return_train_score,
                    n_jobs=1,  # Avoid nested parallelism
                    error_score='raise'
                )
            
            # Get cross-validation predictions for scatter plot
            try:
                from sklearn.model_selection import cross_val_predict
                y_pred_cv = cross_val_predict(estimator, X_array, y, cv=cv_splitter, n_jobs=1)
                logger.info("Generated cross-validation predictions for scatter plot")
                
                # For classification tasks, also get prediction probabilities for ROC curves
                y_pred_proba_cv = None
                if task_type == "classification":
                    try:
                        # Check if estimator is an SVM
                        is_svm = 'SVC' in str(type(estimator)) or 'SVM' in str(type(estimator))
                        
                        # Check if estimator has predict_proba method
                        if hasattr(estimator, 'predict_proba'):
                            y_pred_proba_cv = cross_val_predict(estimator, X_array, y, cv=cv_splitter, 
                                                              method='predict_proba', n_jobs=1)
                            logger.info("Generated cross-validation prediction probabilities for ROC curves")
                        # Special handling for SVM without probability support
                        elif is_svm and hasattr(estimator, 'decision_function'):
                            logger.info("SVM classifier detected without probability support. Using decision function for ROC curves.")
                            
                            # Get decision function values
                            decisions = cross_val_predict(estimator, X_array, y, cv=cv_splitter, 
                                                        method='decision_function', n_jobs=1)
                            
                            # Convert decision function to pseudo-probabilities
                            # For binary classification
                            if len(np.unique(y)) == 2:
                                # Apply sigmoid/softmax to decision values
                                from scipy.special import expit  # Sigmoid function
                                probs_pos = expit(decisions)
                                y_pred_proba_cv = np.column_stack((1 - probs_pos, probs_pos))
                            # For multiclass
                            else:
                                # Apply softmax to decision values
                                from scipy.special import softmax
                                if decisions.ndim == 1:
                                    decisions = decisions.reshape(-1, 1)
                                y_pred_proba_cv = softmax(decisions, axis=1)
                            
                            logger.info("Generated pseudo-probabilities from decision function for SVM")
                        else:
                            logger.warning("Estimator does not have predict_proba or decision_function method, cannot generate ROC curves")
                    except Exception as e:
                        logger.warning(f"Could not generate cross-validation prediction probabilities: {e}")
                        
            except Exception as e:
                logger.warning(f"Could not generate cross-validation predictions: {e}")
                y_pred_cv = None
                y_pred_proba_cv = None
            
            # Save data files if requested
            saved_files = {}
            if save_data:
                saved_files = self._save_cv_data(
                    original_X=original_X_df,
                    original_y=original_y_values,
                    original_target_name=original_target_col,
                    X_processed=X_array,
                    y_true=y,
                    y_pred=y_pred_cv,
                    y_pred_proba=y_pred_proba_cv,
                    preprocessor=preprocessor,
                    output_dir=output_dir,
                    data_name=data_name,
                    task_type=task_type
                )
            
            # Calculate original scale predictions if preprocessor is available
            cv_predictions_data = {}
            if y_pred_cv is not None:
                # Processed scale data
                cv_predictions_data['y_true_processed'] = y
                cv_predictions_data['y_pred_processed'] = y_pred_cv
                
                # Add prediction probabilities for classification
                if y_pred_proba_cv is not None:
                    cv_predictions_data['y_pred_proba_processed'] = y_pred_proba_cv
                
                # Original scale data (if inverse transform is possible)
                if preprocessor is not None and hasattr(preprocessor, 'inverse_transform_target'):
                    try:
                        y_true_original = preprocessor.inverse_transform_target(y)
                        y_pred_original = preprocessor.inverse_transform_target(y_pred_cv)
                        cv_predictions_data['y_true_original'] = y_true_original
                        cv_predictions_data['y_pred_original'] = y_pred_original
                        logger.info("Added original scale predictions to CV results")
                    except Exception as e:
                        logger.warning(f"Could not generate original scale predictions: {e}")
            
            # Process results
            results = self._process_cv_results(cv_results, task_type)
            results['task_type'] = task_type
            results['cv_folds'] = self.cv_folds
            results['data_shape'] = X_array.shape
            results['y_true'] = y
            results['y_pred_cv'] = y_pred_cv  # Add cross-validation predictions
            results['y_pred_proba_cv'] = y_pred_proba_cv  # Add cross-validation prediction probabilities
            results['saved_files'] = saved_files  # Add saved file information
            results['cv_predictions'] = cv_predictions_data  # Add prediction data for scatter plots
            
            logger.info(f"Cross-validation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise
    
    def _save_cv_data(self,
                      original_X: pd.DataFrame,
                      original_y: np.ndarray,
                      original_target_name: Union[str, List[str]],  # Support both string and list
                      X_processed: np.ndarray,
                      y_true: np.ndarray,
                      y_pred: Optional[np.ndarray],
                      y_pred_proba: Optional[np.ndarray],
                      preprocessor: Optional[Any],
                      output_dir: str,
                      data_name: str,
                      task_type: str) -> Dict[str, str]:
        """
        Save cross-validation data to CSV files and generate plots.
        
        Args:
            original_X: Original feature data before preprocessing
            X_processed: Processed feature data
            y_true: True target values
            y_pred: Predicted target values
            y_pred_proba: Predicted probabilities
            preprocessor: Data preprocessor for inverse transform
            output_dir: Output directory
            data_name: Base name for files
            task_type: Task type for plot generation
            
        Returns:
            Dictionary with saved file paths
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        try:
            # 1. Save original data as CSV
            original_data_path = os.path.join(output_dir, f"{data_name}_original_data.csv")
            original_with_targets = original_X.copy()
            
            # Add target columns to original data
            if original_y.ndim == 1 or (original_y.ndim > 1 and original_y.shape[1] == 1):
                # Single target or column vector
                y_original_flat = original_y.flatten() if original_y.ndim > 1 else original_y
                target_col_name = original_target_name if isinstance(original_target_name, str) else original_target_name[0] if original_target_name else 'target'
                original_with_targets[target_col_name] = y_original_flat
            else:
                # Multi-target - use actual target names if available
                if isinstance(original_target_name, list) and len(original_target_name) == original_y.shape[1]:
                    # Use actual target names
                    for i, target_name in enumerate(original_target_name):
                        original_with_targets[target_name] = original_y[:, i]
                else:
                    # Fallback to generic names
                    base_name = original_target_name if isinstance(original_target_name, str) else 'target'
                    for i in range(original_y.shape[1]):
                        original_with_targets[f'{base_name}_{i}'] = original_y[:, i]
            
            original_with_targets.to_csv(original_data_path, index=False)
            saved_files['original_data'] = original_data_path
            logger.info(f"Saved original data to: {original_data_path}")
            
            # 2. Save preprocessed data as CSV
            processed_data_path = os.path.join(output_dir, f"{data_name}_preprocessed_data.csv")
            processed_df = pd.DataFrame(X_processed)
            
            # Get correct feature names from preprocessor if available
            if preprocessor is not None and hasattr(preprocessor, 'get_feature_names'):
                try:
                    # Use the DataPreprocessor's feature names method
                    correct_feature_names = preprocessor.get_feature_names()
                    if correct_feature_names and len(correct_feature_names) == X_processed.shape[1]:
                        processed_df.columns = correct_feature_names
                        logger.info(f"Used correct feature names from preprocessor: {correct_feature_names}")
                    else:
                        # Fallback to generic names
                        processed_df.columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
                        logger.warning("Could not get correct feature names, using generic names")
                except Exception as e:
                    logger.warning(f"Error getting feature names from preprocessor: {e}")
                    # Fallback to generic names
                    processed_df.columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
            else:
                # Use original feature names if available and shapes match, otherwise use generic names  
                if len(original_X.columns) == X_processed.shape[1]:
                    processed_df.columns = original_X.columns
                    logger.warning("Using original column names - this may be incorrect if ColumnTransformer changed order")
                else:
                    processed_df.columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
            # Add processed target columns
            if y_true.ndim == 1 or (y_true.ndim > 1 and y_true.shape[1] == 1):
                # Single target or column vector
                y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
                target_col_name = original_target_name if isinstance(original_target_name, str) else original_target_name[0] if original_target_name else 'target'
                processed_df[target_col_name] = y_true_flat
            else:
                # Multi-target - use actual target names if available
                if isinstance(original_target_name, list) and len(original_target_name) == y_true.shape[1]:
                    # Use actual target names
                    for i, target_name in enumerate(original_target_name):
                        processed_df[target_name] = y_true[:, i]
                else:
                    # Fallback to generic names
                    base_name = original_target_name if isinstance(original_target_name, str) else 'target'
                    for i in range(y_true.shape[1]):
                        processed_df[f'{base_name}_{i}'] = y_true[:, i]
            
            processed_df.to_csv(processed_data_path, index=False)
            saved_files['preprocessed_data'] = processed_data_path
            logger.info(f"Saved preprocessed data to: {processed_data_path}")
            
            if y_pred is not None:
                # 3. Save cross-validation predictions (processed scale)
                cv_predictions_path = os.path.join(output_dir, f"{data_name}_cv_predictions_processed.csv")
                cv_df = pd.DataFrame(X_processed)
                
                # Get correct feature names from preprocessor if available
                if preprocessor is not None and hasattr(preprocessor, 'get_feature_names'):
                    try:
                        # Use the DataPreprocessor's feature names method
                        correct_feature_names = preprocessor.get_feature_names()
                        if correct_feature_names and len(correct_feature_names) == X_processed.shape[1]:
                            cv_df.columns = correct_feature_names
                            logger.info(f"Used correct feature names for CV predictions: {correct_feature_names}")
                        else:
                            # Fallback to generic names
                            cv_df.columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
                            logger.warning("Could not get correct feature names for CV predictions, using generic names")
                    except Exception as e:
                        logger.warning(f"Error getting feature names from preprocessor for CV predictions: {e}")
                        # Fallback to generic names
                        cv_df.columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
                else:
                    # Use original feature names if available and shapes match, otherwise use generic names
                    if len(original_X.columns) == X_processed.shape[1]:
                        cv_df.columns = original_X.columns
                        logger.warning("Using original column names for CV predictions - this may be incorrect if ColumnTransformer changed order")
                    else:
                        cv_df.columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
                
                # Add true and predicted values
                if y_true.ndim == 1 or (y_true.ndim > 1 and y_true.shape[1] == 1):
                    # Single target or column vector
                    y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
                    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
                    cv_df['y_true'] = y_true_flat
                    cv_df['y_pred'] = y_pred_flat
                else:
                    # Multi-target - use actual target names if available
                    if isinstance(original_target_name, list) and len(original_target_name) == y_true.shape[1]:
                        # Use actual target names
                        for i, target_name in enumerate(original_target_name):
                            cv_df[f'y_true_{target_name}'] = y_true[:, i]
                            cv_df[f'y_pred_{target_name}'] = y_pred[:, i]
                    else:
                        # Fallback to indexed names
                        for i in range(y_true.shape[1]):
                            cv_df[f'y_true_{i}'] = y_true[:, i]
                            cv_df[f'y_pred_{i}'] = y_pred[:, i]
                
                cv_df.to_csv(cv_predictions_path, index=False)
                saved_files['cv_predictions_processed'] = cv_predictions_path
                logger.info(f"Saved CV predictions (processed scale) to: {cv_predictions_path}")
                
                # 4. Save cross-validation predictions with inverse transform (original scale)
                cv_original_path = os.path.join(output_dir, f"{data_name}_cv_predictions_original.csv")
                cv_original_df = original_X.copy()
                
                # Inverse transform targets if preprocessor is available
                if preprocessor is not None and hasattr(preprocessor, 'inverse_transform_target'):
                    try:
                        y_true_original = preprocessor.inverse_transform_target(y_true)
                        y_pred_original = preprocessor.inverse_transform_target(y_pred)
                        logger.info("Applied inverse transform to targets")
                    except Exception as e:
                        logger.warning(f"Could not apply inverse transform: {e}")
                        y_true_original = y_true
                        y_pred_original = y_pred
                else:
                    y_true_original = y_true
                    y_pred_original = y_pred
                
                # Add true and predicted values (original scale)
                if y_true_original.ndim == 1 or (y_true_original.ndim > 1 and y_true_original.shape[1] == 1):
                    # Single target or column vector
                    y_true_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
                    y_pred_flat = y_pred_original.flatten() if y_pred_original.ndim > 1 else y_pred_original
                    cv_original_df['y_true'] = y_true_flat
                    cv_original_df['y_pred'] = y_pred_flat
                else:
                    # Multi-target - use actual target names if available
                    if isinstance(original_target_name, list) and len(original_target_name) == y_true_original.shape[1]:
                        # Use actual target names
                        for i, target_name in enumerate(original_target_name):
                            cv_original_df[f'y_true_{target_name}'] = y_true_original[:, i]
                            cv_original_df[f'y_pred_{target_name}'] = y_pred_original[:, i]
                    else:
                        # Fallback to indexed names
                        for i in range(y_true_original.shape[1]):
                            cv_original_df[f'y_true_{i}'] = y_true_original[:, i]
                            cv_original_df[f'y_pred_{i}'] = y_pred_original[:, i]
                
                cv_original_df.to_csv(cv_original_path, index=False)
                saved_files['cv_predictions_original'] = cv_original_path
                logger.info(f"Saved CV predictions (original scale) to: {cv_original_path}")
                
                # 5. Generate and save plots based on task type
                if task_type == "regression":
                    # Generate scatter plots for regression
                    try:
                        scatter_plot_path = os.path.join(output_dir, f"{data_name}_cv_scatter_plot.png")
                        
                        # Import metrics for detailed analysis
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                        
                        # Create scatter plot(s)
                        if y_true_original.ndim == 1:
                            # Single target
                            plt.figure(figsize=(12, 8))
                            plt.scatter(y_true_original, y_pred_original, alpha=0.6, s=50, c='blue', 
                                      edgecolors='k', linewidth=0.5, label='Predictions')
                            
                            # Add perfect prediction line
                            min_val = min(np.min(y_true_original), np.min(y_pred_original))
                            max_val = max(np.max(y_true_original), np.max(y_pred_original))
                            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                            
                            # Calculate multiple metrics
                            r2 = r2_score(y_true_original, y_pred_original)
                            mae = mean_absolute_error(y_true_original, y_pred_original)
                            mse = mean_squared_error(y_true_original, y_pred_original)
                            rmse = np.sqrt(mse)
                            
                            # Add metrics text with improved formatting
                            metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}'
                            plt.text(0.05, 0.95, metrics_text, 
                                    transform=plt.gca().transAxes, fontsize=12,
                                    verticalalignment='top', 
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                            
                            plt.xlabel('True Values (Original Scale)', fontsize=12)
                            plt.ylabel('Predicted Values (Original Scale)', fontsize=12)
                            plt.title(f'Cross-Validation Predictions vs True Values\nR² = {r2:.4f}', fontsize=14)
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            # Set equal limits for better interpretation
                            lims = [min_val, max_val]
                            plt.xlim(lims)
                            plt.ylim(lims)
                            plt.gca().set_aspect('equal', adjustable='box')
                            
                        else:
                            # Multi-target case
                            n_targets = y_true_original.shape[1]
                            if n_targets > 1:
                                # Create subplots for each target
                                n_cols = min(3, n_targets)
                                n_rows = (n_targets + n_cols - 1) // n_cols
                                
                                plt.figure(figsize=(6*n_cols, 6*n_rows))
                                
                                # Get target names for plot titles
                                if isinstance(original_target_name, list) and len(original_target_name) == n_targets:
                                    target_names = original_target_name
                                elif isinstance(original_target_name, str):
                                    target_names = [f"{original_target_name}_{i}" for i in range(n_targets)]
                                else:
                                    # Fallback to generic names
                                    target_names = [f"Target {i}" for i in range(n_targets)]
                                
                                for i in range(n_targets):
                                    plt.subplot(n_rows, n_cols, i+1)
                                    plt.scatter(y_true_original[:, i], y_pred_original[:, i], alpha=0.6, s=50, c='blue',
                                              edgecolors='k', linewidth=0.5, label='Predictions')
                                    
                                    # Add perfect prediction line
                                    min_val = min(np.min(y_true_original[:, i]), np.min(y_pred_original[:, i]))
                                    max_val = max(np.max(y_true_original[:, i]), np.max(y_pred_original[:, i]))
                                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                                    
                                    # Calculate multiple metrics
                                    r2 = r2_score(y_true_original[:, i], y_pred_original[:, i])
                                    mae = mean_absolute_error(y_true_original[:, i], y_pred_original[:, i])
                                    mse = mean_squared_error(y_true_original[:, i], y_pred_original[:, i])
                                    rmse = np.sqrt(mse)
                                    
                                    # Add metrics text with improved formatting
                                    metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}'
                                    plt.text(0.05, 0.95, metrics_text, 
                                            transform=plt.gca().transAxes, fontsize=10,
                                            verticalalignment='top', 
                                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                                    
                                    # Use actual target name in axis labels and title
                                    target_name = target_names[i]
                                    plt.xlabel(f'True Values {target_name} (Original Scale)', fontsize=12)
                                    plt.ylabel(f'Predicted Values {target_name} (Original Scale)', fontsize=12)
                                    plt.title(f'{target_name}: R² = {r2:.4f}', fontsize=14)
                                    plt.legend(fontsize=9)
                                    plt.grid(True, alpha=0.3)
                                    
                                    # Set equal limits for better interpretation
                                    lims = [min_val, max_val]
                                    plt.xlim(lims)
                                    plt.ylim(lims)
                                    plt.gca().set_aspect('equal', adjustable='box')
                            else:
                                # Actually single target, treat as such
                                y_true_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
                                y_pred_flat = y_pred_original.flatten() if y_pred_original.ndim > 1 else y_pred_original
                                
                                plt.figure(figsize=(12, 8))
                                plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=50, c='blue',
                                          edgecolors='k', linewidth=0.5, label='Predictions')
                                
                                # Add perfect prediction line
                                min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
                                max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
                                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                                
                                # Calculate multiple metrics
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
                                
                                plt.xlabel('True Values (Original Scale)', fontsize=12)
                                plt.ylabel('Predicted Values (Original Scale)', fontsize=12)
                                plt.title(f'Cross-Validation Predictions vs True Values\nR² = {r2:.4f}', fontsize=14)
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                
                                # Set equal limits for better interpretation
                                lims = [min_val, max_val]
                                plt.xlim(lims)
                                plt.ylim(lims)
                                plt.gca().set_aspect('equal', adjustable='box')
                        
                        plt.tight_layout()
                        plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        saved_files['scatter_plot'] = scatter_plot_path
                        logger.info(f"Saved scatter plot to: {scatter_plot_path}")
                        
                    except Exception as e:
                        logger.warning(f"Could not create scatter plot: {e}")
                        
                elif task_type == "classification" and y_pred_proba is not None:
                    # Generate ROC curves for classification
                    try:
                        from .training_monitor import TrainingMonitor
                        monitor = TrainingMonitor()
                        
                        roc_plot_path = os.path.join(output_dir, f"{data_name}_roc_curves.png")
                        
                        # Get class names from unique values in original data
                        classes = np.unique(original_y)
                        class_names = [str(cls) for cls in classes]
                        
                        # Use processed scale data for ROC curves (standard practice)
                        monitor.create_roc_curve_plot(
                            y_true=y_true,
                            y_pred_proba=y_pred_proba,
                            class_names=class_names,
                            output_path=roc_plot_path,
                            plot_title="Cross-Validation ROC Curves"
                        )
                        
                        saved_files['roc_curves'] = roc_plot_path
                        logger.info(f"Saved ROC curves to: {roc_plot_path}")
                        
                    except Exception as e:
                        logger.warning(f"Could not create ROC curves: {e}")
                        
                else:
                    logger.info(f"No visualization generated for {task_type} task (insufficient data)")
                    
                logger.info("Applied inverse transform to targets")
            
        except Exception as e:
            logger.error(f"Error saving cross-validation data: {e}")
            
        return saved_files
    
    def _process_cv_results(self, cv_results: Dict, task_type: str) -> Dict[str, Any]:
        """
        Process cross-validation results into a standardized format.
        
        Args:
            cv_results: Results from sklearn's cross_validate
            task_type: Task type for appropriate processing
            
        Returns:
            Processed results dictionary
        """
        processed = {
            'test_scores': {},
            'train_scores': {},
            'fit_times': cv_results['fit_time'],
            'score_times': cv_results['score_time']
        }
        
        # Process test scores
        for key, scores in cv_results.items():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '')
                
                # Handle negative scoring (for regression metrics) - ALWAYS convert to positive
                if metric_name and metric_name.startswith('neg_'):
                    scores = -scores  # Convert negative scores to positive
                    metric_name = metric_name.replace('neg_', '').replace('mean_', '').upper()
                elif metric_name in ['MAE', 'MSE', 'RMSE']:
                    scores = np.abs(scores)  # Ensure positive values
                    metric_name = metric_name.upper() if metric_name else 'UNKNOWN'
                else:
                    # For metrics that should already be positive (R2, accuracy, etc.)
                    metric_name = metric_name.upper() if metric_name else 'UNKNOWN'
                    
                processed['test_scores'][metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
                
            elif key.startswith('train_'):
                metric_name = key.replace('train_', '')
                
                # Handle negative scoring - ALWAYS convert to positive
                if metric_name and metric_name.startswith('neg_'):
                    scores = -scores  # Convert negative scores to positive
                    metric_name = metric_name.replace('neg_', '').replace('mean_', '').upper()
                elif metric_name in ['MAE', 'MSE', 'RMSE']:
                    scores = np.abs(scores)  # Ensure positive values
                    metric_name = metric_name.upper() if metric_name else 'UNKNOWN'
                else:
                    # For metrics that should already be positive
                    metric_name = metric_name.upper() if metric_name else 'UNKNOWN'
                    
                processed['train_scores'][metric_name] = {
                    'scores': scores.tolist(),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
        
        # Add timing statistics
        processed['timing'] = {
            'fit_time_mean': float(np.mean(processed['fit_times'])),
            'fit_time_std': float(np.std(processed['fit_times'])),
            'score_time_mean': float(np.mean(processed['score_times'])),
            'score_time_std': float(np.std(processed['score_times']))
        }
        
        return processed
    
    def simple_cross_val_score(self,
                             estimator: BaseEstimator,
                             X: Union[np.ndarray, pd.DataFrame],
                             y: np.ndarray,
                             metric: str = 'default',
                             task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Simple cross-validation using a single metric (wrapper around cross_val_score).
        
        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix
            y: Target vector
            metric: Metric to use ('default' uses MAE for regression, accuracy for classification)
            task_type: Optional task type. Auto-detected if None.
            
        Returns:
            Dictionary with cross-validation scores
        """
        try:
            # Convert DataFrame to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X = X.values
                
            # Auto-detect task type if not provided
            if task_type is None:
                task_type = self._detect_task_type(y)
            
            # Set default metric
            if metric == 'default':
                metric = 'MAE' if task_type == 'regression' else 'accuracy'
            
            # Get scoring string
            scoring_metrics = self._get_scoring_metrics(task_type, [metric])
            if not scoring_metrics:
                raise ValueError(f"Metric '{metric}' not supported for {task_type}")
            
            scoring = scoring_metrics[metric]
            
            # Get CV splitter
            cv_splitter = self._get_cv_splitter(X, y, task_type)
            
            # Perform cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                scores = cross_val_score(
                    estimator=estimator,
                    X=X,
                    y=y,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=1
                )
            
            # Handle negative scoring
            if scoring.startswith('neg_'):
                scores = -scores
            
            return {
                'metric': metric,
                'task_type': task_type,
                'scores': scores.tolist(),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'cv_folds': self.cv_folds
            }
            
        except Exception as e:
            logger.error(f"Error in simple cross-validation: {str(e)}")
            raise
    
    def validation_curve_analysis(self,
                                estimator: BaseEstimator,
                                X: Union[np.ndarray, pd.DataFrame],
                                y: np.ndarray,
                                param_name: str,
                                param_range: List[Any],
                                task_type: Optional[str] = None,
                                metric: str = 'default') -> Dict[str, Any]:
        """
        Generate validation curve for hyperparameter analysis.
        
        Args:
            estimator: Scikit-learn compatible estimator
            X: Feature matrix
            y: Target vector
            param_name: Name of parameter to vary
            param_range: Range of parameter values to test
            task_type: Optional task type. Auto-detected if None.
            metric: Metric to use for evaluation
            
        Returns:
            Dictionary with validation curve results
        """
        try:
            # Convert DataFrame to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X = X.values
                
            # Auto-detect task type if not provided
            if task_type is None:
                task_type = self._detect_task_type(y)
            
            # Set default metric
            if metric == 'default':
                metric = 'MAE' if task_type == 'regression' else 'accuracy'
            
            # Get scoring string
            scoring_metrics = self._get_scoring_metrics(task_type, [metric])
            scoring = scoring_metrics[metric]
            
            # Get CV splitter
            cv_splitter = self._get_cv_splitter(X, y, task_type)
            
            # Generate validation curve
            train_scores, test_scores = validation_curve(
                estimator=estimator,
                X=X,
                y=y,
                param_name=param_name,
                param_range=param_range,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=1
            )
            
            # Handle negative scoring
            if scoring.startswith('neg_'):
                train_scores = -train_scores
                test_scores = -test_scores
            
            return {
                'param_name': param_name,
                'param_range': param_range,
                'train_scores': train_scores.tolist(),
                'test_scores': test_scores.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
                'test_scores_std': np.std(test_scores, axis=1).tolist(),
                'metric': metric,
                'task_type': task_type
            }
            
        except Exception as e:
            logger.error(f"Error in validation curve analysis: {str(e)}")
            raise
    
    def get_supported_metrics(self, task_type: str) -> List[str]:
        """
        Get list of supported metrics for a given task type.
        
        Args:
            task_type: "regression" or "classification"
            
        Returns:
            List of supported metric names
        """
        if task_type == "regression":
            return list(self.REGRESSION_METRICS.keys())
        elif task_type == "classification":
            return list(self.CLASSIFICATION_METRICS.keys())
        else:
            raise ValueError(f"Unsupported task type: {task_type}") 