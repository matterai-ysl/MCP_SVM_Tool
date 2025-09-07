"""
Training Engine with Hyperparameter Optimization

This module provides training functionality with integrated 
Optuna hyperparameter optimization and cross-validation.
"""

import logging
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
import shutil
import zipfile
import os

from .hyperparameter_optimizer import HyperparameterOptimizer
from .cross_validation import CrossValidationStrategy
from .data_utils import DataProcessor
from .data_validator import DataValidator
from .model_manager import ModelManager
from .data_preprocessing import DataPreprocessor
from .metrics_evaluator import MetricsEvaluator
from .training_monitor import TrainingMonitor
from .academic_report_generator import AcademicReportGenerator
from .html_report_generator import HTMLReportGenerator
from .visualization_generator import VisualizationGenerator
from .svm_wrapper import SVMWrapper
base_url = "http://127.0.0.1:8080"
logger = logging.getLogger(__name__)

class TrainingEngine:
    """
    Handles model training with hyperparameter optimization.
    
    Features:
    - Optuna integration for hyperparameter optimization
    - Cross-validation strategies
    - Parallel training support
    - Training progress monitoring
    - Performance metrics calculation
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        """Initialize TrainingEngine."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.data_processor = DataProcessor()
        self.data_validator = DataValidator()
        self.model_manager = ModelManager(str(self.models_dir))
        self.data_preprocessor = DataPreprocessor()
        self.metrics_evaluator = MetricsEvaluator()
        self.training_monitor = TrainingMonitor()
        self.academic_report_generator = AcademicReportGenerator()
        self.report_generator = HTMLReportGenerator()
        self.visualization_generator = VisualizationGenerator()
        
        logger.info("Initialized TrainingEngine")
    
    def _generate_html_training_report(self, model_directory: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate comprehensive training report and save to model directory.
        
        Args:
            model_directory: Path to model directory
            metadata: Model metadata and training results
            
        Returns:
            str: Path to the generated HTML report
        """
        try:
            # Generate HTML report using dedicated generator
            html_report_path = self.report_generator.generate_training_report_for_model_directory(model_directory, metadata)
            
            # Generate cross-validation visualizations if available
            cv_results = metadata.get('cv_results') or metadata.get('cross_validation_results')
            if cv_results:
                task_type = metadata.get('task_type', 'regression')
                target_names = self._extract_target_names(metadata)
                
                self.visualization_generator.generate_cross_validation_plots(
                    model_directory, cv_results, task_type, target_names
                )
            
            # Generate feature importance plots
            if 'feature_importance' in metadata:
                self.visualization_generator.generate_feature_importance_plots(
                    model_directory, metadata['feature_importance']
                )
            

            
            # Generate evaluation metrics CSV
            if 'y_true' in metadata and 'y_pred' in metadata:
                task_type = metadata.get('task_type', 'regression')
                self.visualization_generator.generate_evaluation_metrics_csv(
                    model_directory, metadata['y_true'], metadata['y_pred'], task_type
                )
            
            # Generate additional scatter plots for regression tasks
            if metadata.get('task_type') == 'regression':
                self.visualization_generator.generate_additional_scatter_plots(model_directory, metadata)
            
            logger.info(f"Training reports and visualizations generated in {model_directory}")
            return html_report_path
            
        except Exception as e:
            logger.warning(f"Failed to generate training reports: {e}")
            return None
    
    def _extract_target_names(self, metadata: Dict[str, Any]) -> Optional[List[str]]:
        """Extract target names from metadata."""
        target_names = None
        if isinstance(metadata.get('target_column'), list):
            target_names = metadata['target_column']
        elif isinstance(metadata.get('target_column'), str):
            target_col = metadata['target_column']
            if target_col.startswith('[') and target_col.endswith(']'):
                import ast
                try:
                    target_names = ast.literal_eval(target_col)
                except:
                    target_names = [target_col]
            else:
                target_names = [target_col]
        else:
            target_names = ['target']
        return target_names
    
    def _get_scaling_method_description(self, scaling_method: str) -> str:
        """Get description of the scaling method."""
        descriptions = {
            "standard": "StandardScaler (mean=0, std=1)",
            "minmax": "MinMaxScaler (range 0-1)",
            "robust": "RobustScaler (median-based, outlier-resistant)",
            "quantile": "QuantileTransformer (normal distribution)",
            "power": "PowerTransformer (Yeo-Johnson)"
        }
        return descriptions.get(scaling_method, f"Unknown scaling method: {scaling_method}")
    
    def _create_model_archive(self, model_directory: Path, model_id: str) -> str:
        """
        Create a ZIP archive of the complete model directory.
        
        Args:
            model_directory: Path to the model directory to archive
            model_id: Model identifier for naming the archive
            
        Returns:
            Path to the created ZIP file
        """
        try:
            # Create archives directory if it doesn't exist
            archives_dir = self.models_dir / "archives"
            archives_dir.mkdir(exist_ok=True)
            
            # Create ZIP file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"{model_id}_{timestamp}.zip"
            zip_path = archives_dir / zip_filename
            
            # Create ZIP archive
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through all files in the model directory
                for root, dirs, files in os.walk(model_directory):
                    for file in files:
                        file_path = Path(root) / file
                        # Calculate relative path from model directory
                        arcname = file_path.relative_to(model_directory)
                        zipf.write(file_path, arcname)
                        
            # Calculate file size
            file_size = zip_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"Model archive created: {zip_path} ({file_size_mb:.2f} MB)")
            
            # Update model metadata with archive information
            try:
                metadata_path = model_directory / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 计算相对于项目根目录的路径
                    relative_path = zip_path.relative_to(self.models_dir.parent)
                    
                    metadata['archive_info'] = {
                        'archive_path': str(zip_path),
                        'archive_filename': zip_filename,
                        'archive_size_bytes': file_size,
                        'archive_size_mb': round(file_size_mb, 2),
                        'created_at': datetime.now().isoformat(),
                        'download_url_by_model': f"/download/{model_id}",
                        'download_url_by_path': f"/download/file/{relative_path.as_posix()}",
                        'relative_path': str(relative_path.as_posix())
                    }
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                        
                    logger.info(f"Archive information added to model metadata")
                    
            except Exception as e:
                logger.warning(f"Could not update metadata with archive info: {e}")
            
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Failed to create model archive: {e}")
            return None

    def _generate_performance_summary(self, cv_results: Dict[str, Any], task_type: str, scoring_metric: str) -> str:
        """
        Generate a concise one-sentence performance summary based on cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            task_type: 'classification' or 'regression'
            scoring_metric: Primary scoring metric used
            
        Returns:
            A one-sentence performance summary string
        """
        try:
            test_scores = cv_results.get('test_scores', {})
            
            # Create case-insensitive lookup for metrics
            metrics_lookup = {k.lower(): k for k in test_scores.keys()}
            
            if task_type == "classification":
                # For classification, prioritize common metrics
                if 'accuracy' in metrics_lookup and test_scores[metrics_lookup['accuracy']].get('mean') is not None:
                    acc_key = metrics_lookup['accuracy']
                    acc_mean = test_scores[acc_key]['mean']
                    acc_std = test_scores[acc_key].get('std', 0)
                    summary = f"Classification model achieved {acc_mean:.3f}±{acc_std:.3f} accuracy"
                    
                    # Add F1 score if available
                    if 'f1' in metrics_lookup and test_scores[metrics_lookup['f1']].get('mean') is not None:
                        f1_key = metrics_lookup['f1']
                        f1_mean = test_scores[f1_key]['mean']
                        summary += f" and {f1_mean:.3f} F1-score"
                    elif 'f1_weighted' in metrics_lookup and test_scores[metrics_lookup['f1_weighted']].get('mean') is not None:
                        f1_key = metrics_lookup['f1_weighted']
                        f1_mean = test_scores[f1_key]['mean']
                        summary += f" and {f1_mean:.3f} weighted F1-score"
                        
                elif 'f1' in metrics_lookup and test_scores[metrics_lookup['f1']].get('mean') is not None:
                    f1_key = metrics_lookup['f1']
                    f1_mean = test_scores[f1_key]['mean']
                    f1_std = test_scores[f1_key].get('std', 0)
                    summary = f"Classification model achieved {f1_mean:.3f}±{f1_std:.3f} F1-score"
                    
                elif 'f1_weighted' in metrics_lookup and test_scores[metrics_lookup['f1_weighted']].get('mean') is not None:
                    f1_key = metrics_lookup['f1_weighted']
                    f1_mean = test_scores[f1_key]['mean']
                    f1_std = test_scores[f1_key].get('std', 0)
                    summary = f"Classification model achieved {f1_mean:.3f}±{f1_std:.3f} weighted F1-score"
                    
                elif scoring_metric.lower() in metrics_lookup and test_scores[metrics_lookup[scoring_metric.lower()]].get('mean') is not None:
                    metric_key = metrics_lookup[scoring_metric.lower()]
                    score_mean = test_scores[metric_key]['mean']
                    score_std = test_scores[metric_key].get('std', 0)
                    metric_name = scoring_metric.replace('_', ' ').title()
                    summary = f"Classification model achieved {score_mean:.3f}±{score_std:.3f} {metric_name}"
                else:
                    summary = "Classification model training completed (metrics unavailable)"
                    
            else:  # regression
                # For regression, prioritize R², MAE, MSE
                if 'r2' in metrics_lookup and test_scores[metrics_lookup['r2']].get('mean') is not None:
                    r2_key = metrics_lookup['r2']
                    r2_mean = test_scores[r2_key]['mean']
                    r2_std = test_scores[r2_key].get('std', 0)
                    summary = f"Regression model achieved R² of {r2_mean:.3f}±{r2_std:.3f}"
                    
                    # Add MAE if available
                    if 'neg_mean_absolute_error' in metrics_lookup and test_scores[metrics_lookup['neg_mean_absolute_error']].get('mean') is not None:
                        mae_key = metrics_lookup['neg_mean_absolute_error']
                        mae_mean = -test_scores[mae_key]['mean']  # Convert back to positive
                        summary += f" with MAE of {mae_mean:.3f}"
                    elif 'mean_absolute_error' in metrics_lookup and test_scores[metrics_lookup['mean_absolute_error']].get('mean') is not None:
                        mae_key = metrics_lookup['mean_absolute_error']
                        mae_mean = test_scores[mae_key]['mean']
                        summary += f" with MAE of {mae_mean:.3f}"
                    elif 'mae' in metrics_lookup and test_scores[metrics_lookup['mae']].get('mean') is not None:
                        mae_key = metrics_lookup['mae']
                        mae_mean = test_scores[mae_key]['mean']
                        summary += f" with MAE of {mae_mean:.3f}"
                        
                elif 'neg_mean_absolute_error' in metrics_lookup and test_scores[metrics_lookup['neg_mean_absolute_error']].get('mean') is not None:
                    mae_key = metrics_lookup['neg_mean_absolute_error']
                    mae_mean = -test_scores[mae_key]['mean']
                    mae_std = test_scores[mae_key].get('std', 0)
                    summary = f"Regression model achieved MAE of {mae_mean:.3f}±{mae_std:.3f}"
                    
                elif 'mean_absolute_error' in metrics_lookup and test_scores[metrics_lookup['mean_absolute_error']].get('mean') is not None:
                    mae_key = metrics_lookup['mean_absolute_error']
                    mae_mean = test_scores[mae_key]['mean']
                    mae_std = test_scores[mae_key].get('std', 0)
                    summary = f"Regression model achieved MAE of {mae_mean:.3f}±{mae_std:.3f}"
                    
                elif 'mae' in metrics_lookup and test_scores[metrics_lookup['mae']].get('mean') is not None:
                    mae_key = metrics_lookup['mae']
                    mae_mean = test_scores[mae_key]['mean']
                    mae_std = test_scores[mae_key].get('std', 0)
                    summary = f"Regression model achieved MAE of {mae_mean:.3f}±{mae_std:.3f}"
                    
                elif 'neg_mean_squared_error' in metrics_lookup and test_scores[metrics_lookup['neg_mean_squared_error']].get('mean') is not None:
                    mse_key = metrics_lookup['neg_mean_squared_error']
                    mse_mean = -test_scores[mse_key]['mean']
                    rmse_mean = np.sqrt(mse_mean)
                    summary = f"Regression model achieved RMSE of {rmse_mean:.3f}"
                    
                elif scoring_metric.lower() in metrics_lookup and test_scores[metrics_lookup[scoring_metric.lower()]].get('mean') is not None:
                    metric_key = metrics_lookup[scoring_metric.lower()]
                    score_mean = test_scores[metric_key]['mean']
                    score_std = test_scores[metric_key].get('std', 0)
                    # Handle negative metrics (convert to positive for display)
                    if scoring_metric.startswith('neg_'):
                        score_mean = -score_mean
                        metric_name = scoring_metric.replace('neg_', '').replace('_', ' ').title()
                        summary = f"Regression model achieved {metric_name} of {score_mean:.3f}±{score_std:.3f}"
                    else:
                        metric_name = scoring_metric.replace('_', ' ').title()
                        summary = f"Regression model achieved {metric_name} of {score_mean:.3f}±{score_std:.3f}"
                else:
                    summary = "Regression model training completed (metrics unavailable)"
            
            # Add cross-validation info
            cv_folds = cv_results.get('cv_folds', 5)
            summary += f" ({cv_folds}-fold CV)"
            
            return summary
            
        except Exception as e:
            logger.warning(f"Could not generate performance summary: {e}")
            return f"{task_type.title()} model training completed successfully"
    
    def _load_and_prepare_data(self, 
                              data_source: Union[str, pd.DataFrame],
                              target_column: Optional[str] = None,
                              validate_data: bool = True) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Load and prepare data for training.
        
        Args:
            data_source: File path or DataFrame
            target_column: Name of target column
            validate_data: Whether to validate data
            
        Returns:
            Tuple of (features_df, target_array, feature_names)
        """
        # Load data
        if isinstance(data_source, str):
            logger.info(f"Loading data from file: {data_source}")
            df = self.data_processor.load_data(data_source)
        elif isinstance(data_source, pd.DataFrame):
            logger.info("Using provided DataFrame")
            df = data_source.copy()
        else:
            raise ValueError("data_source must be a file path or pandas DataFrame")
        
        # Data validation is handled in train_random_forest method, not here
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Original data columns: {list(df.columns)}")
        logger.info(f"Original data dtypes: {df.dtypes.to_dict()}")
        
        # Separate features and target (no preprocessing here)
        if target_column:
            # Handle multi-target case (list of column names)
            if isinstance(target_column, list):
                missing_cols = [col for col in target_column if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Target columns not found: {missing_cols}")
                
                X = df.drop(columns=target_column)
                y = df[target_column].values
                logger.info(f"Using multiple target columns: {target_column}")
            else:
                # Single target case
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                
                X = df.drop(columns=[target_column])
                y = df[target_column].values
        else:
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1].values
            target_column = df.columns[-1]
            logger.info(f"Using last column '{target_column}' as target variable")
        
        feature_names = X.columns.tolist()
        
        logger.info(f"Features and target separated: X shape={X.shape}, y shape={y.shape}")
        logger.info(f"Feature columns: {feature_names}")
        logger.info(f"Target unique values: {len(np.unique(y)) if hasattr(y, '__len__') else 'N/A'}")
        
        return X, y, feature_names
    
    def train_svm_classification(
        self,
        data_source: Union[str, pd.DataFrame],
        target_column: Optional[str] = None,
        model_id: Optional[str] = None,
        kernel: str = None,
        class_weight: str = "balanced",
        optimize_hyperparameters: bool = True,
        n_trials: int = 50,
        cv_folds: int = 5,
        optimization_algorithm: str = "TPE",
        scoring_metric: str = "f1_weighted",
        validate_data: bool = True,
        save_model: bool = True,
        apply_preprocessing: bool = True,
        scaling_method: str = "standard",
        **svm_params
    ) -> Dict[str, Any]:
        """
        Train an SVM classification model with hyperparameter optimization.
        
        Args:
            data_source: File path or DataFrame with training data
            target_column: Name of target column (if None, uses last column)
            model_id: Name for the saved model (if None, auto-generates)
            kernel: SVM kernel type ("rbf", "linear", "poly", "sigmoid")
            class_weight: Weights associated with classes ("balanced" or None)
            optimize_hyperparameters: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            optimization_algorithm: Optimization algorithm ("TPE" or "CmaES")
            scoring_metric: Scoring metric for optimization
            validate_data: Whether to validate data quality
            save_model: Whether to save the trained model
            apply_preprocessing: Whether to apply data preprocessing
            scaling_method: Method for feature scaling ("standard", "minmax", "robust")
            **svm_params: Additional parameters for the SVM model
            
        Returns:
            Dictionary with training results
        """
        
        return self._train_svm_model(
            data_source=data_source,
            target_column=target_column,
            model_id=model_id,
            task_type="classification",
            kernel=kernel,
            class_weight=class_weight,
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            cv_folds=cv_folds,
            optimization_algorithm=optimization_algorithm,
            scoring_metric=scoring_metric,
            validate_data=validate_data,
            save_model=save_model,
            apply_preprocessing=apply_preprocessing,
            scaling_method=scaling_method,
            **svm_params
        )

    def train_svm_regression(
        self,
        data_source: Union[str, pd.DataFrame],
        target_column: Optional[str] = None,
        model_id: Optional[str] = None,
        kernel: str = None,
        optimize_hyperparameters: bool = True,
        n_trials: int = 50,
        cv_folds: int = 5,
        optimization_algorithm: str = "TPE",
        scoring_metric: str = "neg_mean_squared_error",
        validate_data: bool = True,
        save_model: bool = True,
        apply_preprocessing: bool = True,
        scaling_method: str = "standard",
        **svm_params
    ) -> Dict[str, Any]:
        """
        Train an SVM regression model with hyperparameter optimization.
        
        Args:
            data_source: File path or DataFrame with training data
            target_column: Name of target column (if None, uses last column)
            model_id: Name for the saved model (if None, auto-generates)
            kernel: SVM kernel type ("rbf", "linear", "poly", "sigmoid")
            optimize_hyperparameters: Whether to run hyperparameter optimization
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            optimization_algorithm: Optimization algorithm ("TPE" or "CmaES")
            scoring_metric: Scoring metric for optimization
            validate_data: Whether to validate data quality
            save_model: Whether to save the trained model
            apply_preprocessing: Whether to apply data preprocessing
            scaling_method: Method for feature scaling ("standard", "minmax", "robust")
            **svm_params: Additional parameters for the SVM model
            
        Returns:
            Dictionary with training results
        """
        return self._train_svm_model(
            data_source=data_source,
            target_column=target_column,
            model_id=model_id,
            task_type="regression",
            kernel=kernel,
            class_weight=None,  # Not used for regression
            optimize_hyperparameters=optimize_hyperparameters,
            n_trials=n_trials,
            cv_folds=cv_folds,
            optimization_algorithm=optimization_algorithm,
            scoring_metric=scoring_metric,
            validate_data=validate_data,
            save_model=save_model,
            apply_preprocessing=apply_preprocessing,
            scaling_method=scaling_method,
            **svm_params
        )

    def _train_svm_model(
        self,
        data_source: Union[str, pd.DataFrame],
        target_column: Optional[str] = None,
        model_id: Optional[str] = None,
        task_type: str = "classification",
        kernel: str = None,
        class_weight: Optional[str] = None,
        optimize_hyperparameters: bool = True,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring_metric: str = None,
        validate_data: bool = True,
        save_model: bool = True,
        apply_preprocessing: bool = True,
        scaling_method: str = "standard",
        optimization_algorithm: str = "TPE"
    ) -> Dict[str, Any]:
        """
        Core SVM training method that handles both classification and regression.
        """

        try:
            start_time = datetime.now()
            print(f"n_trials: {n_trials}")
            # Generate model ID if not provided
            if model_id is None:
                model_id = str(uuid.uuid4())
            
            logger.info(f"Starting SVM {task_type} training with model_id: {model_id}")
            
            # Load and prepare data (reuse SVM infrastructure)
            X, y, feature_names = self._load_and_prepare_data(
                data_source, target_column, validate_data
            )
            
            # Initialize preprocessing variables early to avoid undefined variable errors
            preprocessing_applied = False
            target_preprocessing_applied = False
            
            # Save data validation report if validation was performed
            validation_results = None
            if validate_data:
                try:
                    # Re-run validation to get results for saving
                    if isinstance(data_source, str):
                        df = self.data_processor.load_data(data_source)
                    else:
                        df = data_source.copy()
                    # Create model directory for validation report and visualizations
                    temp_model_dir = self.models_dir / model_id
                    temp_model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save original raw data for future feature importance analysis
                    raw_data_path = temp_model_dir / "raw_data.csv"
                    df.to_csv(raw_data_path, index=False)
                    logger.info(f"Raw training data saved to: {raw_data_path}")
                    
                    reports_dir = temp_model_dir / "reports"
                    reports_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Run validation with model directory for visualizations
                    logger.info(f"🔍 Starting data validation...")
                    logger.info(f"  DataFrame shape: {df.shape}")
                    logger.info(f"  Target column: {target_column}")
                    logger.info(f"  Task type: {task_type}")
                    
                    validation_results = self.data_validator.validate_dataset(
                        df, target_column, task_type, 
                        model_directory=str(temp_model_dir)
                    )
                    
                    logger.info(f"✅ Data validation completed")
                    
                    validation_report_path = reports_dir / "data_validation_report.json"
                    logger.info(f"💾 Attempting to save validation report to: {validation_report_path}")
                    
                    self.data_validator.save_validation_report(validation_results, str(validation_report_path))
                    logger.info(f"✅ Data validation report saved successfully to: {validation_report_path}")
                except Exception as e:
                    logger.error(f"❌ Could not save data validation report: {e}")
                    import traceback
                    logger.error(f"   Full traceback:\n{traceback.format_exc()}")
                    validation_results = None
            else:
                # Save raw data even if validation is disabled
                try:
                    if isinstance(data_source, str):
                        df = self.data_processor.load_data(data_source)
                    else:
                        df = data_source.copy()
                    # Create model directory for raw data saving
                    temp_model_dir = self.models_dir / model_id
                    temp_model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save original raw data for future feature importance analysis
                    raw_data_path = temp_model_dir / "raw_data.csv"
                    df.to_csv(raw_data_path, index=False)
                    logger.info(f"Raw training data saved to: {raw_data_path}")
                except Exception as e:
                    logger.warning(f"Could not save raw training data: {e}")
            
            # Set up base SVM parameters
            base_params = {
                'kernel': "rbf",
                'C': 1.0,
                'gamma': "scale",
                'degree': 3
            }
            svm_params = {}
            if task_type == "classification":
                from sklearn.svm import SVC
                if class_weight:
                    svm_params['class_weight'] = class_weight
                model_class = SVC
                # Set default scoring metric for classification
                if scoring_metric is None:
                    scoring_metric = "f1_weighted"
            else:  # regression
                from sklearn.svm import SVR
                model_class = SVR
                # Set default scoring metric for regression
                if scoring_metric is None:
                    scoring_metric = "neg_mean_squared_error"
            
            
            # Apply data preprocessing if requested
            X_processed = X.copy()
            y_processed = y.copy()
            
            # 调试信息
            logger.info(f"📋 Preprocessing Check:")
            logger.info(f"  apply_preprocessing: {apply_preprocessing}")
            logger.info(f"  len(X): {len(X)}")
            logger.info(f"  X shape: {X.shape}")
            logger.info(f"  X dtypes: {X.dtypes if hasattr(X, 'dtypes') else 'No dtypes'}")
            
            if apply_preprocessing and len(X) > 0:
                try:
                    logger.info("Applying data preprocessing...")
                    enhanced_preprocessor = DataPreprocessor(use_one_hot=True)
                    X_processed = enhanced_preprocessor.fit_transform_features(
                        X, scaling_method=scaling_method
                    )
                    preprocessing_applied = True
                    logger.info(f"Feature preprocessing completed. Shape: {X.shape} -> {X_processed.shape}")
                    
                    # 目标预处理始终传递task_type，无论分类还是回归都需要处理
                    y_processed = enhanced_preprocessor.fit_transform_target(
                        y, task_type=task_type, target_scaling_method=scaling_method
                    )
                    target_preprocessing_applied = True
                    logger.info(f"Target preprocessing applied for {task_type} task")
                    self.data_preprocessor = enhanced_preprocessor
                    
                    # Save preprocessed data for direct feature importance analysis
                    if hasattr(self, 'data_preprocessor') and self.data_preprocessor:
                        try:
                            # Save preprocessed features and target
                            processed_data_dir = temp_model_dir / "processed_data"
                            processed_data_dir.mkdir(exist_ok=True)
                            
                            # Convert processed data to DataFrame with proper feature names
                            processed_feature_names = self.data_preprocessor.get_feature_names()
                            if not processed_feature_names or len(processed_feature_names) != X_processed.shape[1]:
                                processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                            
                            processed_X_df = pd.DataFrame(X_processed, columns=processed_feature_names)
                            processed_y_df = pd.DataFrame(y_processed, columns=[target_column] if not isinstance(target_column, list) else target_column)
                            
                            # Save processed features and target separately
                            processed_X_path = processed_data_dir / "processed_features.csv"
                            processed_y_path = processed_data_dir / "processed_target.csv"
                            
                            processed_X_df.to_csv(processed_X_path, index=False)
                            processed_y_df.to_csv(processed_y_path, index=False)
                            
                            # Save combined processed data
                            processed_combined = pd.concat([processed_X_df, processed_y_df], axis=1)
                            processed_combined_path = processed_data_dir / "processed_data.csv"
                            processed_combined.to_csv(processed_combined_path, index=False)
                            
                            logger.info(f"Processed training data saved to: {processed_data_dir}")
                            logger.info(f"Processed features shape: {processed_X_df.shape}")
                            logger.info(f"Processed target shape: {processed_y_df.shape}")
                            
                        except Exception as e:
                            logger.warning(f"Could not save processed training data: {e}")
                    
                except Exception as e:
                    logger.error(f"Data preprocessing failed: {e}")
                    logger.exception("Full preprocessing error traceback:")
                    raise RuntimeError(f"Data preprocessing failed: {e}") from e
            else:
                logger.warning(f"Preprocessing SKIPPED! Conditions:")
                logger.warning(f"  apply_preprocessing: {apply_preprocessing}")
                logger.warning(f"  len(X) > 0: {len(X) > 0}")
                logger.warning(f"  This means data will contain raw strings and cause training to fail!")
            
            # Keep original data for saving
            X_original = X.copy()
            y_original = y.copy()
            original_feature_names = feature_names.copy()
            
            # Set original target name based on actual target names for multi-target case
            if isinstance(target_column, list):
                original_target_column_name = target_column  # Keep as list for multi-target
            else:
                original_target_column_name = target_column if target_column else (feature_names[-1] if feature_names else 'target')
            
            # Use processed data for training
            X = X_processed
            y = y_processed
            
            # Update feature_names to match the processed data order
            if preprocessing_applied and hasattr(self, 'data_preprocessor') and self.data_preprocessor:
                if hasattr(self.data_preprocessor, 'get_feature_names'):
                    processed_feature_names = self.data_preprocessor.get_feature_names()
                    if processed_feature_names and len(processed_feature_names) == X.shape[1]:
                        feature_names = processed_feature_names
                        logger.info(f"Updated feature_names to match processed data order: {feature_names}")
                    else:
                        logger.warning(f"Could not update feature_names: preprocessor returned {len(processed_feature_names) if processed_feature_names else 0} names for {X.shape[1]} features")
                else:
                    logger.warning("Preprocessor does not have get_feature_names method, keeping original feature names")
            
            # Ensure y is 1-dimensional to avoid sklearn warnings
            if hasattr(y, 'ndim') and y.ndim > 1 and y.shape[1] == 1:
                y = y.ravel()
                logger.info(f"Converted y from shape {y_processed.shape} to 1D array with shape {y.shape}")
            
            logger.info(f"Starting training for model: {model_id} (ID: {model_id})")
            
            # Hyperparameter optimization
            optimization_results = {}
            optimizer = None
            optimized_params = None
            
            if optimize_hyperparameters:
                try:
                    logger.info("Starting hyperparameter optimization...")
                    logger.info(f"Data diagnostics: X shape={X.shape}, y shape={y.shape}")
                    logger.info(f"X data types: {X.dtypes if hasattr(X, 'dtypes') else 'numpy array'}")
                    logger.info(f"Y unique values: {len(np.unique(y))} (sample: {np.unique(y)[:10]})")
                    logger.info(f"X contains NaN: {np.any(np.isnan(X))}")
                    try:
                        logger.info(f"Y contains NaN: {np.any(np.isnan(y))}")
                    except TypeError:
                        # For categorical targets, use pandas to check for NaN
                        logger.info(f"Y contains NaN: {pd.Series(y).isna().any()}")
                
                    optimizer = HyperparameterOptimizer(
                        sampler_type=optimization_algorithm.upper(),
                        n_trials=n_trials,
                        cv_folds=cv_folds,
                        random_state=42
                    )
                    
                    # Create save directory for optimization history
                    save_dir = temp_model_dir / "optimization" if 'temp_model_dir' in locals() else None
                    
                    # Run SVM hyperparameter optimization
                    optimized_params, optimization_results = optimizer.optimize_svm(
                        X=X, 
                        y=y,
                        kernel=kernel,
                        task_type=task_type,
                        scoring_metric=scoring_metric,  
                        svm_params=svm_params,
                        save_dir=str(save_dir) if save_dir else None
                    )
                    final_model = optimizer.create_optimized_model()
                    logger.info(f"Optimization completed. Best score: {optimization_results.get('best_score', 'N/A'):.4f}")
                    
                except Exception as e:
                    logger.warning(f"Hyperparameter optimization failed: {str(e)}")
                    logger.warning("Falling back to default hyperparameters...")
                    final_model = model_class(**base_params)
                    # Fall back to default parameters
                    optimized_params = base_params.copy()
                    optimization_results = {
                        'error': str(e),
                        'fallback_used': True,
                        'best_params': optimized_params,
                        'n_trials': n_trials
                    }
                    
                    # Try to save failed optimization history
                    if optimizer and hasattr(optimizer, 'study') and optimizer.study:
                        try:
                            # Get trials dataframe even from failed optimization
                            trials_df = optimizer.study.trials_dataframe()
                            optimization_results['trials_dataframe'] = trials_df
                            optimization_results['optimization_history'] = optimizer.get_optimization_results().get('optimization_history', [])
                            logger.info(f"Saved failed optimization history with {len(trials_df)} trials")
                            
                        except Exception as save_error:
                            logger.warning(f"Could not save failed optimization history: {save_error}")
            else:
                optimized_params = base_params
            
            # Train final model with optimized parameters
            logger.info("Training final SVM model")
            print(f"》》》Final parameters: {optimized_params}")
            final_model.fit(X, y)
            
            # Perform cross-validation evaluation using sklearn's cross_validate
            logger.info("Performing cross-validation evaluation...")
            # Use SVMWrapper for cross-validation to avoid data leakage
            logger.info("Performing cross-validation evaluation using SVMWrapper...")
            
            # Create SVMWrapper for cross-validation
            svm_wrapper = SVMWrapper(
                task_type=task_type,
                **optimized_params
            )
            
            # Perform cross-validation with SVMWrapper
            cv_results = svm_wrapper.cross_validate(
                X, y,
                cv_folds=cv_folds,
                return_train_score=True,
                task_type=task_type,
                random_state=42,
                save_data=save_model,
                output_dir="cv_results",
                data_name=model_id,
                preprocessor=self.data_preprocessor if hasattr(self, 'data_preprocessor') else None,
                feature_names=feature_names,
                original_X=X_original,
                original_y=y_original,
                original_feature_names=original_feature_names,
                original_target_name=target_column
            )
            
            # Get feature importance from trained model
            logger.info("Computing feature importance...")
            try:
                # Create a temporary wrapper to get feature importance
                temp_wrapper = SVMWrapper(task_type=task_type, **optimized_params)
                temp_wrapper.fit(X, y, feature_names=feature_names)
                feature_importance = temp_wrapper.get_all_feature_importances()
                logger.info("Feature importance calculation completed")
            except Exception as e:
                logger.warning(f"Failed to compute feature importance: {e}")
                feature_importance = {}
            
            # Save model if requested
            model_path = None
            model_directory = None
            if save_model:
                # Create model directory
                model_directory = self.model_manager.create_model_directory(model_id)
                
                # Update cross-validation data directory for saving scatter plots
                if save_model:
                    cv_output_dir = model_directory / "cross_validation_data"
                    cv_output_dir.mkdir(exist_ok=True)
                    # Re-run cross-validation with correct output directory for scatter plots
                    try:
                        enhanced_cv_results = svm_wrapper.cross_validate(
                            X, y,
                            cv_folds=cv_folds,
                            return_train_score=True,
                            task_type=task_type,
                            random_state=42,
                            save_data=True,
                            output_dir=str(cv_output_dir),
                            data_name=model_id,
                            preprocessor=self.data_preprocessor if hasattr(self, 'data_preprocessor') else None,
                            feature_names=feature_names,
                            original_X=X_original,
                            original_y=y_original,
                            original_feature_names=original_feature_names,
                            original_target_name=target_column
                        )
                        # Update cv_results with enhanced results
                        if 'saved_files' in enhanced_cv_results:
                            cv_results['saved_files'] = enhanced_cv_results['saved_files']
                        if 'cv_predictions' in enhanced_cv_results:
                            cv_results['cv_predictions'] = enhanced_cv_results['cv_predictions']
                        logger.info(f"Cross-validation scatter plots saved to: {cv_output_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to save cross-validation scatter plots: {e}")
                
                # Save preprocessing pipeline if preprocessing was applied
                preprocessing_pipeline_path = None
                if preprocessing_applied and hasattr(self, 'data_preprocessor') and self.data_preprocessor:
                    preprocessing_pipeline_path = model_directory / "preprocessing_pipeline.pkl"
                    self.data_preprocessor.save_pipeline(str(preprocessing_pipeline_path))
                    logger.info(f"Complete preprocessing pipeline saved to: {preprocessing_pipeline_path}")
                    
                    # Save preprocessing info as JSON for easy inspection
                    preprocessing_info_path = model_directory / "preprocessing_info.json"
                    with open(preprocessing_info_path, 'w') as f:
                        json.dump(self.data_preprocessor.get_preprocessing_info(), f, indent=2)
                    logger.info(f"Preprocessing info saved to: {preprocessing_info_path}")
                
                # Get predictions for evaluation plots
                try:
                    y_pred = final_model.predict(X)
                    logger.info("Generated predictions for evaluation plots")
                except Exception as e:
                    logger.warning(f"Could not generate predictions: {e}")
                    y_pred = None
                
                # Get label mapping for classification tasks
                label_mapping = None
                if task_type == 'classification' and hasattr(self, 'data_preprocessor') and self.data_preprocessor:
                    if hasattr(self.data_preprocessor, 'label_mapping') and self.data_preprocessor.label_mapping:
                        label_mapping = self.data_preprocessor.label_mapping
                        logger.info(f"Saved label mapping for classification: {label_mapping['class_to_label']}")
                
                # Compile comprehensive metadata for the model
                comprehensive_metadata = {
                    'model_name': model_id,
                    'model_id': model_id,
                    'algorithm': f'SVM_{task_type.title()}',
                    'task_type': task_type,
                    'kernel': kernel,
                    'target_column': target_column,
                    'target_name': target_column if isinstance(target_column, list) else [target_column] if target_column else ['target'],
                    'target_dimension': len(target_column) if isinstance(target_column, list) else (1 if target_column else -1),
                    'training_time_seconds': (datetime.now() - start_time).total_seconds(),
                    'data_shape': {
                        'n_samples': X.shape[0],
                        'n_features': X.shape[1]
                    },
                    'feature_names': feature_names,
                    'hyperparameters': optimized_params or base_params,
                    'performance_metrics': cv_results['test_scores'],
                    'cross_validation_results': cv_results,
                    'optimization_results': optimization_results,
                    'training_completed_at': datetime.now().isoformat(),
                    'preprocessing_applied': preprocessing_applied,
                    'preprocessing_pipeline_path': str(preprocessing_pipeline_path) if preprocessing_pipeline_path else None,
                    'scaling_method': scaling_method if preprocessing_applied else None,
                    'label_mapping': label_mapping,
                    'tool_input_details': {
                        'data_source': data_source if isinstance(data_source, str) else 'DataFrame',
                        'optimize_hyperparameters': optimize_hyperparameters,
                        'n_trials': n_trials,
                        'cv_folds': cv_folds,
                        'optimization_algorithm': optimization_algorithm,
                        'scoring_metric': scoring_metric,
                        'validate_data': validate_data,
                        'save_model': save_model,
                        'apply_preprocessing': apply_preprocessing,
                        'scaling_method': scaling_method,
                        'task_type': task_type,
                        'kernel': kernel,
                        'additional_svm_params': {k: v for k, v in svm_params.items()},
                        'base_model_params': {
                            'random_state': base_params.get('random_state'),
                            'C': base_params.get('C'),
                            'gamma': base_params.get('gamma'),
                            'degree': base_params.get('degree')
                        }
                    },
                    'preprocessing_details': {
                        'applied': preprocessing_applied,
                        'target_preprocessing_applied': target_preprocessing_applied,
                        'scaling_method': scaling_method if preprocessing_applied else None,
                        'scaling_method_description': self._get_scaling_method_description(scaling_method) if preprocessing_applied else None,
                        'pipeline_saved': preprocessing_pipeline_path is not None,
                        'pipeline_components': {
                            'numerical_imputer': preprocessing_applied,
                            'categorical_one_hot_encoder': preprocessing_applied,
                            'categorical_label_encoder': preprocessing_applied,
                            'feature_scaler': preprocessing_applied,
                            'target_scaler': target_preprocessing_applied
                        } if preprocessing_applied else {},
                        'original_feature_count': len(feature_names),
                        'processed_feature_count': X.shape[1],
                        'preprocessing_info': self.data_preprocessor.get_preprocessing_info() if hasattr(self, 'data_preprocessor') and self.data_preprocessor else None
                    },
                    'model': final_model,
                    'X': X,
                    'y': y,
                    'y_true': y,
                    'y_pred': y_pred
                }
                
                # Save model using model manager
                model_path = self.model_manager.save_model(
                    model=final_model,  # Save the actual SVM model
                    model_id=model_id,
                    metadata=comprehensive_metadata
                )
                
                # Generate and save training report
                html_report_path = self._generate_html_training_report(model_directory, comprehensive_metadata)
                
                # Generate academic report
                try:
                    # Prepare training results for academic report
                    academic_training_results = {
                        'cv_scores': cv_results,
                        'test_scores': cv_results.get('test_scores', {}),
                        'train_scores': cv_results.get('train_scores', {}),
                        'timing': cv_results.get('timing', {}),
                        'task_type': task_type,
                        'cv_folds': cv_folds
                    }
                    
                    # Calculate model_score based on the primary scoring metric
                    model_score = 0
                    if cv_results.get('test_scores'):
                        test_scores = cv_results['test_scores']
                        # For classification, try to get the score for the specified metric
                        if task_type == 'classification':
                            # Map scoring metrics to their uppercase equivalents in test_scores
                            metric_mapping = {
                                'f1_weighted': 'F1',
                                'f1': 'F1', 
                                'accuracy': 'ACCURACY',
                                'precision': 'PRECISION',
                                'recall': 'RECALL'
                            }
                            metric_key = metric_mapping.get(scoring_metric, scoring_metric.upper())
                            if metric_key in test_scores:
                                model_score = test_scores[metric_key].get('mean', 0)
                            elif 'F1' in test_scores:  # Fallback to F1
                                model_score = test_scores['F1'].get('mean', 0)
                            elif 'ACCURACY' in test_scores:  # Fallback to accuracy
                                model_score = test_scores['ACCURACY'].get('mean', 0)
                        else:  # regression
                            # For regression, try common metrics
                            if scoring_metric in test_scores:
                                model_score = test_scores[scoring_metric].get('mean', 0)
                            elif 'R2' in test_scores:
                                model_score = test_scores['R2'].get('mean', 0)
                            elif 'MAE' in test_scores:
                                model_score = test_scores['MAE'].get('mean', 0)
                    
                    academic_training_results['model_score'] = model_score
                    
                    # Prepare hyperparameter optimization results if available
                    academic_hyperopt_results = None
                    if optimization_results and optimization_results.get('best_params'):
                        academic_hyperopt_results = {
                            'best_params': optimization_results['best_params'],
                            'best_score': optimization_results.get('best_score', 0),
                            'n_trials': optimization_results.get('n_trials', 0)
                        }
                    
                    # Prepare comprehensive metadata for academic report
                    academic_metadata = {
                        'model_id': model_id,
                        'data_source': data_source if isinstance(data_source, str) else 'DataFrame',
                        'data_shape': comprehensive_metadata['data_shape'],
                        'n_features': comprehensive_metadata['data_shape']['n_features'],
                        'target_dimension': comprehensive_metadata['target_dimension'],
                        'n_targets': comprehensive_metadata['target_dimension'],
                        'n_samples': comprehensive_metadata['data_shape']['n_samples'],
                        'feature_names': feature_names,
                        'target_name': comprehensive_metadata['target_name'],
                        'target_names': comprehensive_metadata['target_name'],
                        'algorithm': f'SVM_{task_type.title()}',
                        'cv_folds': cv_folds,
                        'random_state': base_params.get('random_state'),
                        'task_type': task_type,
                        'scoring_metric': scoring_metric,
                        'preprocessing_applied': preprocessing_applied,
                        'model_params': optimized_params or base_params
                    }
                    
                    # Generate academic report
                    academic_report_path = self.academic_report_generator.generate_academic_report(model_directory)

                    if academic_report_path:
                        logger.info(f"Academic report generated: {academic_report_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate academic report: {e}")
                
                logger.info(f"Model and reports saved to directory: {model_directory}")
                
                # Create ZIP archive of the model directory
                zip_path = self._create_model_archive(model_directory, model_id)
            
            # Set default paths for missing variables (in case report generation failed or save_model=False)
            if 'model_directory' not in locals() or model_directory is None:
                model_directory = self.models_dir / model_id
            if 'html_report_path' not in locals():
                html_report_path = model_directory / "reports" / "training_report.html"
            if 'zip_path' not in locals():
                zip_path = model_directory / f"{model_id}_archive.zip"
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store optimizer instance for later access (even if optimization failed)
            self.optimizer = optimizer
            
            # Compile results
            results = {
                'model_id': model_id,
                'algorithm': f'SVM_{task_type.title()}',
                'task_type': task_type,
                'kernel': kernel,
                'training_time_seconds': training_time,
                'data_shape': {
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1]
                },
                'feature_names': feature_names,
                'target_names': target_column,
                'best_hyperparameters': optimized_params or base_params,
                'feature_importance': feature_importance,
                'cv_results': cv_results,
                'performance_summary': self._generate_performance_summary(cv_results, task_type, scoring_metric),
                "trained_report_summary": f"You can find the html trained report summary in {base_url}/static/{Path(html_report_path).relative_to(self.models_dir).as_posix()}" if html_report_path else None,
                "trained_details": f"""All detailed training data are saved in {base_url}/download/file/{Path(zip_path).relative_to(self.models_dir.parent).as_posix()},
                which can be downloaded by users for reproducibility and academic research reference.""" if zip_path else None
            }
            
            # Log performance summary
            logger.info(f"📊 SVM {task_type} Performance Summary: {results['performance_summary']}")
            logger.info(f"SVM {task_type} training completed successfully in {training_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"SVM {task_type} training failed: {str(e)}")
            raise
