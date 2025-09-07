"""
SVM Wrapper Module

This module provides the SVMWrapper class that encapsulates 
SVM algorithms with enhanced functionality, consistent with XGBoostWrapper.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
from .cross_validation import CrossValidationStrategy

logger = logging.getLogger(__name__)


class SVMWrapper:
    """
    Wrapper class for SVM algorithms with enhanced functionality.
    
    Supports both regression and classification tasks with automatic
    feature importance calculation and cross-validation capabilities.
    """
    
    def __init__(self, 
                 task_type: str = "auto",
                 kernel: str = "rbf",
                 C: float = 1.0,
                 gamma: str = "scale",
                 degree: int = 3,
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 probability: bool = False,
                 tol: float = 1e-3,
                 cache_size: float = 200,
                 class_weight: Optional[str] = None,
                 verbose: bool = False,
                 max_iter: int = -1,
                 decision_function_shape: str = "ovr",
                 break_ties: bool = False,
                 random_state: Optional[int] = 42,
                 epsilon: float = 0.1,  # For SVR
                 **kwargs):
        """
        Initialize SVMWrapper.
        
        Args:
            task_type: "auto", "regression", or "classification"
            kernel: SVM kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
            degree: Degree of polynomial kernel
            coef0: Independent term in kernel function
            shrinking: Whether to use shrinking heuristic
            tol: Tolerance for stopping criterion
            cache_size: Kernel cache size in MB
            class_weight: Class weights for imbalanced datasets
            verbose: Enable verbose output
            max_iter: Maximum number of iterations
            decision_function_shape: Decision function shape for multiclass
            break_ties: Break ties for multiclass
            random_state: Random state for reproducibility
            epsilon: Epsilon for SVR
            **kwargs: Additional parameters
        """
        self.task_type = task_type.lower()
        
        self.model_params = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'shrinking': shrinking,
            'tol': tol,
            'cache_size': cache_size,
            'class_weight': class_weight,
            'verbose': verbose,
            'max_iter': max_iter,
            'decision_function_shape': decision_function_shape,
            'break_ties': break_ties,
            'epsilon': epsilon,  # For SVR
            **kwargs
        }
        if task_type == "classification":
            self.model_params['probability'] = probability
        # SVM specific parameters
        self.model = None
        self.feature_names = None
        self.feature_importances = None
        self.permutation_importances = None
        self.is_fitted = False
        self.actual_task_type = None
        
        logger.info(f"Initialized SVMWrapper with task_type={task_type}, kernel={kernel}, C={C}")
        
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
            
    def _initialize_model(self, task_type: str):
        """
        Initialize the appropriate SVM model based on task type.
        
        Args:
            task_type: "regression" or "classification"
        """
        # Set model-specific parameters
        model_params = self.model_params.copy()
        
        if task_type == "regression":
            # Remove classification-specific parameters
            model_params.pop('class_weight', None)
            model_params.pop('decision_function_shape', None)
            model_params.pop('break_ties', None)
            
            self.model = SVR(**model_params)
        elif task_type == "classification":
            # Remove regression-specific parameters
            model_params.pop('epsilon', None)
            
            self.model = SVC(**model_params)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        logger.info(f"Initialized SVM {task_type} model")
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            compute_permutation_importance: bool = True,
            verbose: bool = True) -> 'SVMWrapper':
        """
        Fit the SVM model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names
            compute_permutation_importance: Whether to compute permutation importance
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        try:
            # Convert to numpy array if pandas DataFrame
            if isinstance(X, pd.DataFrame):
                if feature_names is None:
                    feature_names = X.columns.tolist()
                X = X.values
                
            # Store feature names
            if feature_names is not None:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Detect task type
            self.actual_task_type = self._detect_task_type(y)
            
            # Initialize appropriate model
            self._initialize_model(self.actual_task_type)
            
            # Fit the model
            logger.info(f"Training SVM {self.actual_task_type} model with {X.shape[0]} samples and {X.shape[1]} features")
            
            self.model.fit(X, y)
            
            # Calculate feature importances
            self._calculate_feature_importances(X, y, compute_permutation_importance)
            
            self.is_fitted = True
            
            logger.info("Model training completed successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray, 
                                     compute_permutation_importance: bool = True):
        """
        Calculate feature importances for SVM.
        
        Args:
            X: Feature matrix
            y: Target vector
            compute_permutation_importance: Whether to compute permutation importance
        """
        # Initialize feature importances dictionary
        self.feature_importances = {}
        
        # Linear kernel has coefficients (like linear regression)
        if self.model.kernel == 'linear':
            try:
                if self.actual_task_type == "classification":
                    # For classification, use absolute values of coefficients
                    if hasattr(self.model, 'coef_'):
                        if self.model.coef_.ndim == 1:
                            # Binary classification
                            coef_abs = np.abs(self.model.coef_)
                        else:
                            # Multi-class classification - take mean of absolute coefficients
                            coef_abs = np.mean(np.abs(self.model.coef_), axis=0)
                        
                        self.feature_importances['linear_coef'] = {
                            name: float(coef) 
                            for name, coef in zip(self.feature_names, coef_abs)
                        }
                else:
                    # For regression, use absolute values of coefficients
                    if hasattr(self.model, 'coef_'):
                        coef_abs = np.abs(self.model.coef_)
                        self.feature_importances['linear_coef'] = {
                            name: float(coef) 
                            for name, coef in zip(self.feature_names, coef_abs)
                        }
                        
            except Exception as e:
                logger.warning(f"Failed to compute linear coefficients: {str(e)}")
                self.feature_importances['linear_coef'] = {name: 0.0 for name in self.feature_names}
        
        # Permutation importance (works for all kernels)
        if compute_permutation_importance:
            try:
                logger.info("Computing permutation importance...")
                perm_importance = permutation_importance(
                    self.model, X, y, 
                    n_repeats=5, 
                    random_state=self.model_params.get('random_state', 42),
                    n_jobs=1  # Use single job to avoid nested parallelism
                )
                
                self.permutation_importances = {
                    name: {
                        'importance_mean': float(mean),
                        'importance_std': float(std)
                    }
                    for name, mean, std in zip(
                        self.feature_names, 
                        perm_importance.importances_mean,
                        perm_importance.importances_std
                    )
                }
                logger.info("Permutation importance calculation completed")
                
            except Exception as e:
                logger.warning(f"Failed to compute permutation importance: {str(e)}")
                self.permutation_importances = None
                
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Prediction array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities (only for classification with probability=True).
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Probability array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if self.actual_task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
            
        if not self.model.probability:
            raise ValueError("Model must be trained with probability=True for predict_proba")
            
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)
        
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on given data.
        
        Args:
            X: Feature matrix
            y_true: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        y_pred = self.predict(X)
        
        if self.actual_task_type == "regression":
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
            }
        else:  # classification
            # Handle different averaging strategies for multiclass
            n_classes = len(np.unique(y_true))
            average = 'binary' if n_classes == 2 else 'weighted'
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                    'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0))
                }
                
        return metrics
        
    def get_all_feature_importances(self) -> Dict[str, Any]:
        """
        Get all calculated feature importances.
        
        Returns:
            Dictionary with all feature importance types
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importances")
            
        result = {}
        
        # Add standard feature importances
        if self.feature_importances:
            result.update(self.feature_importances)
            
        # Add permutation importances
        if self.permutation_importances:
            result['permutation'] = self.permutation_importances
            
        return result
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'task_type': self.task_type,
                'model_params': self.model_params
            }
            
        info = {
            'is_fitted': True,
            'task_type': self.actual_task_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma,
            'degree': self.model.degree,
            'n_support': self.model.n_support_.tolist() if hasattr(self.model, 'n_support_') else None,
            'support_vectors_count': self.model.support_vectors_.shape[0] if hasattr(self.model, 'support_vectors_') else None
        }
        
        if hasattr(self.model, 'classes_'):
            info['classes'] = self.model.classes_.tolist()
            info['n_classes'] = len(self.model.classes_)
            
        return info
        
    def cross_validate(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: np.ndarray,
                      cv_folds: int = 5,
                      metrics: Optional[List[str]] = None,
                      stratify: bool = True,
                      return_train_score: bool = True,
                      random_state: Optional[int] = None,
                      save_data: bool = False,
                      output_dir: str = "cv_results",
                      data_name: str = "cv_data",
                      preprocessor: Optional[Any] = None,
                      feature_names: Optional[List[str]] = None,
                      original_X: Optional[pd.DataFrame] = None,
                      original_y: Optional[np.ndarray] = None,
                      original_feature_names: Optional[List[str]] = None,
                      original_target_name: Optional[str] = None,
                      task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Cross-validate the model, allowing explicit task_type passing.
        
        This method creates a fresh model instance to avoid data leakage.
        """
        # Use external task_type if provided
        if task_type is not None:
            use_task_type = task_type
        else:
            use_task_type = self.actual_task_type if self.is_fitted else self._detect_task_type(y)
            
        # Create a fresh model instance with current parameters for CV
        fresh_model = self._create_fresh_model(use_task_type)
        
        # Create cross-validation strategy
        cv_strategy = CrossValidationStrategy(
            cv_folds=cv_folds,
            random_state=random_state,
            stratify=stratify
        )
        
        # Perform cross-validation
        results = cv_strategy.cross_validate_model(
            estimator=fresh_model,
            X=X,
            y=y,
            task_type=use_task_type,
            metrics=metrics,
            return_train_score=return_train_score,
            save_data=save_data,
            output_dir=output_dir,
            data_name=data_name,
            preprocessor=preprocessor,
            feature_names=feature_names,
            original_X=original_X,
            original_y=original_y,
            original_feature_names=original_feature_names,
            original_target_name=original_target_name
        )
        
        return results
        
    def _create_fresh_model(self, task_type: str):
        """
        Create a fresh model instance with current parameters.
        
        Args:
            task_type: "regression" or "classification"
            
        Returns:
            Fresh model instance
        """
        # Get the most up-to-date parameters
        current_params = self.model_params.copy()
        
        if task_type == "regression":
            # Remove classification-specific parameters
            current_params.pop('class_weight', None)
            current_params.pop('decision_function_shape', None)
            current_params.pop('break_ties', None)
            current_params.pop('probability', None)  # SVR doesn't support probability
            current_params.pop('random_state', None)  # SVR doesn't support random_state
            
            return SVR(**current_params)
        elif task_type == "classification":
            # Remove regression-specific parameters
            current_params.pop('epsilon', None)
            
            return SVC(**current_params)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
    def get_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of current model parameters
        """
        if self.is_fitted and self.model is not None:
            # Return the actual fitted model's parameters
            return self.model.get_params()
        else:
            # Return the stored parameters
            return self.model_params.copy()
            
    def set_params(self, **params) -> 'SVMWrapper':
        """
        Set model parameters and update internal model_params.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        # Update internal model_params
        self.model_params.update(params)
        
        # If model is already fitted, update its parameters too
        if self.is_fitted and self.model is not None:
            self.model.set_params(**params)
            logger.info(f"Updated model parameters: {params}")
        
        return self
        
    def save_model(self, filepath: str) -> None:
        """
        Save the trained SVM model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        # Save using joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str) -> 'SVMWrapper':
        """
        Load a trained SVM model.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        try:
            # Load using joblib
            self.model = joblib.load(filepath)
            self.is_fitted = True
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        return self 