
import logging
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import json,uuid
import numpy as np
import asyncio
from datetime import datetime
import zipfile
import os
import re

# FastMCP import
from fastmcp import FastMCP
from fastmcp import Context
from .config import BASE_URL,get_download_url,get_static_url
# Internal modules
from .training import TrainingEngine
from .prediction import PredictionEngine
from .model_manager import ModelManager
from .data_validator import DataValidator
from .training_queue import get_queue_manager, initialize_queue_manager

from starlette.requests import Request
from starlette.responses import PlainTextResponse
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User directory management functions
def get_user_id(ctx: Optional[Context] = None) -> Optional[str]:
    """Extract user ID from MCP context."""
    if ctx is not None and hasattr(ctx, 'request_context') and hasattr(ctx.request_context, 'request'):
        return ctx.request_context.request.headers.get("user_id", None) # type: ignore
    return None

def get_user_models_dir(user_id: Optional[str] = None) -> str:
    """
    Get user-specific models directory with security cleaning.

    Args:
        user_id: User identifier, defaults to "default" if None

    Returns:
        Path to user-specific models directory
    """
    if user_id is None or user_id.strip() == "":
        user_id = "default"

    # Clean user ID to prevent path traversal attacks
    user_id = re.sub(r'[^\w\-_]', '_', user_id)

    user_dir = Path("trained_models") / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return str(user_dir)

# Initialize MCP server with detailed instructions
mcp = FastMCP(
    name="SVM Machine Learning Tool",
    instructions="""
    This is a comprehensive Machine Learning server providing advanced SVM capabilities with asynchronous training queue.

    Available tools:
    1. train_svm_classifier - Train an SVM classification model with automatic model_id generation for tracking
    2. train_svm_regressor - Train an SVM regression model with automatic model_id generation for tracking
    3. get_svm_task_status - Get training task status using model_id
    4. list_svm_training_tasks - List all training tasks with optional filters
    5. cancel_svm_training_task - Cancel a running or queued training task using model_id
    6. get_svm_queue_status - Get overall training queue status
    7. predict_from_file_svm - Make batch predictions from a data file using model_id
    8. predict_from_values_svm - Make real-time predictions from feature values using model_id
    9. list_svm_models - List all available trained models
    10. get_svm_model_info - Get detailed model information using model_id
    11. delete_svm_model - Delete a trained model using model_id

    Workflow:
    - Training: Submit training task → Immediately receive model_id → Use model_id to track status and manage model
    - Prediction: Use model_id to make predictions with trained models
    - Management: Use model_id to query information, cancel tasks, or delete models

    The SVM tools support multiple kernels (linear, rbf, poly, sigmoid), hyperparameter optimization with Optuna,
    and comprehensive feature importance analysis using permutation importance.

    The asynchronous training queue allows concurrent training jobs without blocking operations.
    Each training task immediately returns a model_id for unified management and tracking.
    """
)

# Initialize engines
root_dir = Path("./trained_models")
training_engine = TrainingEngine("trained_models")
prediction_engine = PredictionEngine("trained_models")
model_manager = ModelManager("trained_models")

@mcp.tool()
async def train_svm_classifier(
    data_source: str,
    kernel: str = "linear",
    optimize_hyperparameters: bool = True,
    n_trials: int = 20,
    cv_folds: int = 5,
    scoring_metric: str = "f1_weighted",
    validate_data: bool = True,
    save_model: bool = True,
    apply_preprocessing: bool = True,
    scaling_method: str = "standard",
    use_async_queue: bool = True,
    user_id: str = None, # type: ignore
    ctx: Context = None # type: ignore
) -> Dict[str, Any]:
    """
    Train an SVM classification model with automatic hyperparameter optimization.
    
    Args:
        data_source: Path to training data file (CSV, Excel, etc.)
        kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid'), if None, the kernel will be automatically selected based on the data.
        optimize_hyperparameters: Whether to run hyperparameter optimization (default: True)
        n_trials: Number of optimization trials (default: 50)
        cv_folds: Number of cross-validation folds (default: 5)
        scoring_metric: Scoring metric for classification optimization:
                       - 'f1_weighted': Weighted F1 score (default)
                       - 'f1_macro': Macro F1 score
                       - 'f1_micro': Micro F1 score
                       - 'accuracy': Classification accuracy
                       - 'precision_weighted': Weighted precision
                       - 'recall_weighted': Weighted recall
                       - 'roc_auc': ROC AUC (binary classification only)
        validate_data: Whether to validate data quality (default: True)
        save_model: Whether to save the trained model (default: True)
        apply_preprocessing: Whether to apply data preprocessing (default: True)
        scaling_method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
        use_async_queue: Whether to use asynchronous training queue (default: False)
        user_id: Optional user identifier for tracking (used with async queue)
        
    Returns:
        Training results including model performance, metadata, and SVM-specific information
        Or task submission details if using async queue
    """
    try:
        logger.info(f"Training SVM classification model from: {data_source}")

        # Get user ID and user-specific directory
        if user_id is None:
            user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Convert scoring metric to sklearn format first
        if scoring_metric == 'accuracy':
            scoring_metric = 'accuracy'
        elif scoring_metric == 'f1_weighted':
            scoring_metric = 'f1_weighted'
        elif scoring_metric == 'f1':
            scoring_metric = 'f1'
        elif scoring_metric == 'precision':
            scoring_metric = 'precision_weighted'
        elif scoring_metric == 'recall':
            scoring_metric = 'recall_weighted'
        elif scoring_metric == 'roc_auc':
            scoring_metric = 'roc_auc'
        
        # Validate parameters first (before any async queue logic)
        valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if kernel not in valid_kernels and kernel is not None:
            raise ValueError(f"Invalid kernel: {kernel}. Supported kernels: {valid_kernels}")
            
        valid_metrics = ['f1_weighted', 'f1_macro', 'f1_micro', 'accuracy', 'precision_weighted', 'recall_weighted', 'roc_auc']
        if scoring_metric not in valid_metrics:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}. Supported metrics: {valid_metrics}")
        
        # Load and validate data for classification to determine target column
        from .data_utils import DataProcessor
        data_processor = DataProcessor()
        df = data_processor.load_data(data_source)
        
        # For classification, assume last column is target (target_dimension = 1)
        target_dimension = 1
        if target_dimension > len(df.columns):
            raise ValueError(f"Target dimension {target_dimension} exceeds number of columns {len(df.columns)}")
        
        # Get target columns (last N columns based on target_dimension)
        target_columns = df.columns[-target_dimension:].tolist()
        target_column = target_columns[0]  # Single target for classification
        
        logger.info(f"SVM classification with target column: {target_column}")
        
        # Check if using async queue (after parameter validation and data loading)
        if use_async_queue:
            logger.info("Using asynchronous training queue")
            return await submit_svm_training_task(
                task_type="svm_classification",
                data_source=data_source,
                target_column=target_column,  # Use the determined target column
                kernel=kernel,
                optimize_hyperparameters=optimize_hyperparameters,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                validate_data=validate_data,
                save_model=save_model,
                apply_preprocessing=apply_preprocessing,
                scaling_method=scaling_method,
                user_id=user_id
            )
        
        # Continue with direct training
        logger.info("Using direct synchronous training")

        model_id = str(uuid.uuid4())

        # Create user-specific training engine
        user_training_engine = TrainingEngine(user_models_dir)

        # Run training in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: user_training_engine.train_svm_classification(
                data_source=data_source,
                target_column=target_column,  # Use the determined target column
                model_id=model_id,
                kernel=kernel,
                optimize_hyperparameters=optimize_hyperparameters,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                validate_data=validate_data,
                save_model=save_model,
                apply_preprocessing=apply_preprocessing,
                scaling_method=scaling_method
            )
        )
        
        logger.info("SVM classification training completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"SVM classification training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def train_svm_regressor(
    data_source: str,
    target_dimension: int = 1,
    kernel: str = None, # type: ignore
    optimize_hyperparameters: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    scoring_metric: str = "r2",
    validate_data: bool = True,
    save_model: bool = True,
    apply_preprocessing: bool = True,
    scaling_method: str = "standard",
    use_async_queue: bool = True,
    user_id: str = None, # type: ignore
    ctx: Context = None # type: ignore
) -> Dict[str, Any]:
    """
    Train an SVM regression model with automatic hyperparameter optimization.
    
    Args:
        data_source: Path to training data file (CSV, Excel, etc.)
        target_dimension: Number of target columns for multi-target regression (positive integer)
        kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid'), if None, the kernel will be automatically selected based on the data.
        optimize_hyperparameters: Whether to run hyperparameter optimization (default: True)
        n_trials: Number of optimization trials (default: 50)
        cv_folds: Number of cross-validation folds (default: 5)
        scoring_metric: Scoring metric for regression optimization:
                       - 'r2': R-squared score (default)
                       - 'neg_mean_squared_error': Negative MSE
                       - 'neg_mean_absolute_error': Negative MAE
                       - 'neg_root_mean_squared_error': Negative RMSE
                       - 'neg_mean_absolute_percentage_error': Negative MAPE
                       - 'explained_variance': Explained Variance Score
                       - 'max_error': Maximum Residual Error
                       - 'neg_median_absolute_error': Negative MAD
        validate_data: Whether to validate data quality (default: True)
        save_model: Whether to save the trained model (default: True)
        apply_preprocessing: Whether to apply data preprocessing (default: True)
        scaling_method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
        use_async_queue: Whether to use asynchronous training queue (default: True)
        user_id: Optional user identifier for tracking (used with async queue)
        
    Returns:
        Training results including model performance, metadata, and SVM-specific information
        Or task submission details if using async queue
    """
    try:
        logger.info(f"Training SVM regression model from: {data_source}")

        # Get user ID and user-specific directory
        if user_id is None:
            user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Validate parameters first (before any async queue logic)
        valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if kernel not in valid_kernels and kernel is not None:
            raise ValueError(f"Invalid kernel: {kernel}. Supported kernels: {valid_kernels}")
            
        valid_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 
                        'r2', 'neg_mean_absolute_percentage_error', 'explained_variance', 
                        'max_error', 'neg_median_absolute_error']
        if scoring_metric not in valid_metrics:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}. Supported metrics: {valid_metrics}")
        
        # Validate target_dimension parameter and load data to determine target column
        from .data_utils import DataProcessor
        data_processor = DataProcessor()
        df = data_processor.load_data(data_source)       
        if target_dimension <= 0:
            raise ValueError(f"Target dimension must be a positive integer, got: {target_dimension}")
        
        if target_dimension > len(df.columns):
            raise ValueError(f"Target dimension {target_dimension} exceeds number of columns {len(df.columns)}")
        
        # Get target columns (last N columns based on target_dimension)
        target_columns = df.columns[-target_dimension:].tolist()
        target_column = target_columns if target_dimension > 1 else target_columns[0]
        
        logger.info(f"SVM regression with {target_dimension} target(s): {target_columns}")
        
        # Check if using async queue (after parameter validation and data loading)
        if use_async_queue:
            logger.info("Using asynchronous training queue")
            return await submit_svm_training_task(
                task_type="svm_regression",
                data_source=data_source,
                target_column=target_column,  # Use the determined target column
                kernel=kernel,
                optimize_hyperparameters=optimize_hyperparameters,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                validate_data=validate_data,
                save_model=save_model,
                apply_preprocessing=apply_preprocessing,
                scaling_method=scaling_method,
                user_id=user_id
            )
        
        # Continue with direct training
        logger.info("Using direct synchronous training")

        model_id = str(uuid.uuid4())

        # Create user-specific training engine
        user_training_engine = TrainingEngine(user_models_dir)

        # Run training in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: user_training_engine.train_svm_regression(
                data_source=data_source,
                target_column=target_column, # type: ignore # Use the determined target column
                model_id=model_id,
                kernel=kernel,
                optimize_hyperparameters=optimize_hyperparameters,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring_metric=scoring_metric,
                validate_data=validate_data,
                save_model=save_model,
                apply_preprocessing=apply_preprocessing,
                scaling_method=scaling_method
            )
        )
        
        logger.info("SVM regression training completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"SVM regression training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def predict_from_file_svm(
    model_id: str,
    data_source: str,
    output_path: str = None, # type: ignore
    include_confidence: bool = True,
    generate_report: bool = True,
    ctx: Context = None # type: ignore
) -> Dict[str, Any]:
    """
    Make batch predictions from a data file using a trained SVM model.
    
    Args:
        model_id: Unique identifier for the trained SVM model
        data_source: Path to prediction data file (CSV, Excel, etc.)
        output_path: Path to save prediction results (if None, uses default)
        include_confidence: Whether to include confidence scores
        generate_report: Whether to generate detailed report
        
    Returns:
        Prediction results and analysis
    """
    try:
        logger.info(f"Making batch predictions with model {model_id} from file: {data_source}")

        # Get user ID and user-specific directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Create user-specific prediction engine
        user_prediction_engine = PredictionEngine(user_models_dir)

        # Run prediction in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: user_prediction_engine.predict_from_file(
                model_id=model_id,
                data_source=data_source,
                output_path=output_path,
                include_confidence=include_confidence,
                generate_report=generate_report
            )
        )
        
        logger.info("Batch prediction completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Batch prediction failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def predict_from_values_svm(
    model_id: str,
    feature_values: Union[List[float], List[List[float]], Dict[str, float], List[Dict[str, float]]],
    feature_names: List[str] = None, # type: ignore
    include_confidence: bool = True,
    save_intermediate_files: bool = True,
    generate_report: bool = True,
    output_path: str = None, # type: ignore
    ctx: Context = None # type: ignore
) -> Dict[str, Any]:
    """
    Make real-time predictions from feature values using a trained SVM model.
    
    Supports both single and batch predictions:
    - Single: [1, 2, 3] or {'feature1': 1, 'feature2': 2}
    - Batch: [[1, 2, 3], [4, 5, 6]] or [{'feature1': 1}, {'feature1': 4}]
    
    Args:
        model_id: Unique identifier for the trained SVM model
        feature_values: Feature values in various formats (single or batch)
        feature_names: Names of features (required if feature_values is a list of lists),if not provided, the feature names will be inferred from the model metadata.
        include_confidence: Whether to include confidence scores
        save_intermediate_files: Whether to save CSV files with processed features, predictions, and confidence scores
        generate_report: Whether to generate detailed experiment report
        output_path: Custom output path for prediction files (optional)
        
    Returns:
        Prediction results, CSV file paths, and analysis
    """
    try:
        logger.info(f"Making real-time prediction with model {model_id}")

        # Get user ID and user-specific directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Create user-specific prediction engine
        user_prediction_engine = PredictionEngine(user_models_dir)

        # Run prediction in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: user_prediction_engine.predict_from_values(
                model_id=model_id,
                feature_values=feature_values,
                feature_names=feature_names,
                include_confidence=include_confidence,
                save_intermediate_files=save_intermediate_files,
                generate_report=generate_report,
                output_path=output_path
            )
        )
        
        logger.info("Real-time prediction completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Real-time prediction failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def list_svm_models(ctx: Context = None) -> List[Dict[str, Any]]: # type: ignore
    """
    List all available trained SVM models.
    
    Returns:
        List of model information including IDs, names, and metadata
    """
    try:
        logger.info("Listing all available SVM models")

        # Get user ID and user-specific directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Create user-specific model manager
        user_model_manager = ModelManager(user_models_dir)

        # Run in executor to avoid blocking
        models = await asyncio.get_event_loop().run_in_executor(
            None, user_model_manager.list_models
        )
        
        logger.info(f"Found {len(models)} available models")
        return models
        
    except Exception as e:
        error_msg = f"Failed to list models: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()} # type: ignore

@mcp.tool()
async def get_svm_model_info(model_id: str, ctx: Context = None) -> Dict[str, Any]: # type: ignore
    """
    Get detailed information about a specific SVM model.
    
    Args:
        model_id: Unique identifier for the SVM model
        
    Returns:
        Detailed model information including performance metrics and metadata
    """
    try:
        logger.info(f"Getting information for model: {model_id}")

        # Get user ID and user-specific directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Create user-specific model manager
        user_model_manager = ModelManager(user_models_dir)

        # Run in executor to avoid blocking
        model_info = await asyncio.get_event_loop().run_in_executor(
            None, user_model_manager.get_model_info, model_id
        )
        
        logger.info("Model information retrieved successfully")
        return model_info
        
    except Exception as e:
        error_msg = f"Failed to get model info: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def delete_svm_model(model_id: str, ctx: Context = None) -> Dict[str, Any]: # type: ignore
    """
    Delete a trained SVM model.
    
    Args:
        model_id: Unique identifier for the SVM model to delete
        
    Returns:
        Deletion status and information
    """
    try:
        logger.info(f"Deleting model: {model_id}")

        # Get user ID and user-specific directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Create user-specific model manager
        user_model_manager = ModelManager(user_models_dir)

        # Run in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, user_model_manager.delete_model, model_id
        )
        
        if result.get('success', False):
            logger.info("Model deleted successfully")
        else:
            logger.warning(f"Model deletion failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to delete model: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

# Asynchronous Training Queue Tools


async def submit_svm_training_task(
    task_type: str,
    data_source: str,
    target_column: Optional[Union[str, List[str]]] = None,
    kernel: str = None, # type: ignore
    optimize_hyperparameters: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    scoring_metric: str = None, # type: ignore
    validate_data: bool = True,
    save_model: bool = True,
    apply_preprocessing: bool = True,
    scaling_method: str = "standard",
    user_id: str = None, # type: ignore
) -> Dict[str, Any]:
    """
    Submit a training task to the asynchronous queue.

    Args:
        task_type: Type of training task ('svm_regression' or 'svm_classification')
        data_source: Path to training data file (CSV, Excel, etc.)
        target_column: Target column(s) for training
        kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid'), if None, auto-selected
        optimize_hyperparameters: Whether to run hyperparameter optimization
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        scoring_metric: Scoring metric for optimization (auto-selected if None)
        validate_data: Whether to validate data quality
        save_model: Whether to save the trained model
        apply_preprocessing: Whether to apply data preprocessing
        scaling_method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
        user_id: Optional user identifier for tracking

    Returns:
        Task submission result with model_id for tracking
    """
    try:
        logger.info(f"Submitting {task_type} training task to queue")

        # Generate model_id immediately upon task submission
        model_id = str(uuid.uuid4())
        logger.info(f"Generated model_id: {model_id}")

        # Initialize queue manager if not already running
        queue_manager = await initialize_queue_manager()

        # Get user directory for the task
        user_models_dir = get_user_models_dir(user_id)

        # Prepare task parameters with model_id
        task_params = {
            'data_source': data_source,
            'target_column': target_column,
            'kernel': kernel,
            'optimize_hyperparameters': optimize_hyperparameters,
            'n_trials': n_trials,
            'cv_folds': cv_folds,
            'scoring_metric': scoring_metric,
            'validate_data': validate_data,
            'save_model': save_model,
            'apply_preprocessing': apply_preprocessing,
            'scaling_method': scaling_method,
            'models_dir': user_models_dir,
            'model_id': model_id  # Include model_id in task parameters
        }

        # Submit task to queue using model_id as task_id
        await queue_manager.submit_task(
            task_type=task_type,
            params=task_params,
            user_id=user_id,
            task_id=model_id  # Use model_id as task_id
        )

        logger.info(f"Training task submitted successfully with model_id: {model_id}")
        return {
            'success': True,
            'model_id': model_id,
            'task_type': task_type,
            'message': f'Training task submitted to queue. Use model_id {model_id} to track progress and manage the model.',
            'queue_status': await queue_manager.get_queue_status()
        }

    except Exception as e:
        error_msg = f"Failed to submit training task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def get_svm_task_status(model_id: str) -> Dict[str, Any]:
    """
    Get the status of a specific training task using model_id.

    Args:
        model_id: Unique identifier (model_id) of the training task to check

    Returns:
        Task status information including training progress, or error if task not found
    """
    try:
        logger.info(f"Getting status for model_id: {model_id}")

        queue_manager = get_queue_manager()
        task_status = await queue_manager.get_task_status(model_id)

        if task_status is None:
            return {
                'error': f'Task with model_id {model_id} not found',
                'success': False
            }

        logger.info(f"Task status for model_id {model_id} retrieved successfully")
        return {
            'success': True,
            **task_status
        }

    except Exception as e:
        error_msg = f"Failed to get task status: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def list_svm_training_tasks(
    user_id: str = None, # type: ignore
    status: str = None, # type: ignore 
    limit: int = 50
) -> Dict[str, Any]:
    """
    List training tasks with optional filters.
    
    Args:
        user_id: Optional user ID filter
        status: Optional status filter ('queued', 'running', 'completed', 'failed', 'cancelled')
        limit: Maximum number of tasks to return
        
    Returns:
        List of training tasks
    """
    try:
        logger.info("Listing training tasks")
        
        queue_manager = get_queue_manager()
        
        # Convert status string to enum if provided
        status_filter = None
        if status:
            from .training_queue import TaskStatus
            try:
                status_filter = TaskStatus(status.lower())
            except ValueError:
                return {
                    'error': f'Invalid status: {status}. Valid options: queued, running, completed, failed, cancelled',
                    'success': False
                }
        
        tasks = await queue_manager.list_tasks(
            user_id=user_id,
            status=status_filter
        )
        
        # Apply limit
        if limit and len(tasks) > limit:
            tasks = tasks[:limit]
        
        logger.info(f"Retrieved {len(tasks)} training tasks")
        return {
            'success': True,
            'tasks': tasks,
            'total_count': len(tasks),
            'filters_applied': {
                'user_id': user_id,
                'status': status,
                'limit': limit
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to list training tasks: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def cancel_svm_training_task(model_id: str) -> Dict[str, Any]:
    """
    Cancel a training task using model_id.

    Args:
        model_id: Unique identifier (model_id) of the task to cancel

    Returns:
        Cancellation status and information
    """
    try:
        logger.info(f"Cancelling training task with model_id: {model_id}")

        queue_manager = get_queue_manager()
        success = await queue_manager.cancel_task(model_id)

        if success:
            logger.info(f"Task with model_id {model_id} cancelled successfully")
            return {
                'success': True,
                'model_id': model_id,
                'message': f'Task with model_id {model_id} has been cancelled'
            }
        else:
            return {
                'success': False,
                'model_id': model_id,
                'message': f'Task with model_id {model_id} could not be cancelled (not found or already completed)'
            }

    except Exception as e:
        error_msg = f"Failed to cancel training task: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def get_svm_queue_status() -> Dict[str, Any]:
    """
    Get overall SVM training queue status information.
    
    Returns:
        Queue status including active tasks, queue length, etc.
    """
    try:
        logger.info("Getting training queue status")
        
        queue_manager = get_queue_manager()
        queue_status = await queue_manager.get_queue_status()
        
        logger.info("Queue status retrieved successfully")
        return {
            'success': True,
            **queue_status
        }
        
    except Exception as e:
        error_msg = f"Failed to get queue status: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}


def _create_analysis_archive(output_dir: Path, analysis_type: str, model_id: str) -> str:
    """
    创建分析结果的ZIP压缩包

    Args:
        output_dir: Output directory containing analysis results
        analysis_type: Type of analysis (e.g., 'feature', 'performance')
        model_id: Unique model identifier

    Returns:
        Path to created ZIP archive
    """
    try:
        # 创建archives目录
        archives_dir = Path("trained_models") / model_id / "feature_analysis" / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)

        # 创建ZIP文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{analysis_type}_analysis_{model_id}_{timestamp}.zip"
        zip_path = archives_dir / zip_filename

        # 创建ZIP压缩包
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)

        return str(zip_path)
    except Exception as e:
        logger.error(f"Failed to create analysis archive: {e}")
        return None # type: ignore

# async def test_predict_from_file():
#     """Test the predict_from_file function asynchronously."""
#     try:
#         model_id = "1ada0b3e-0271-4012-b53b-4a626786b08d"
#         data_source = r"D:\SLM预测.xls"
#         include_confidence = True
#         generate_report = True
        
#         print("Starting prediction test...")
#         result = await predict_from_file(
#             model_id=model_id,
#             data_source=data_source,
#             include_confidence=include_confidence,
#             generate_report=generate_report
#         )
#         print("Prediction completed successfully!")
#     except Exception as e:
#         print(f"Prediction failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# async def test_predict_form_values():

#     try:
#         model_id = "6b937211-5454-4a4f-b5cc-79514cb3cf48"
#         feature_values = [5.1, 3.5, 1.4, 0.2]
#         # model_id = "6299d812-41b2-4e40-aeb0-0133c6f3ba5f"
#         # feature_values = [35, 80, 130, 800]
#         print("Starting prediction test...")
#         result = await predict_from_values(
#             model_id=model_id,
#             feature_values=feature_values
#         )
#         print("Prediction completed successfully!")
#     except Exception as e:
#         print(f"Prediction failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None


# async def test_train_svm_regressor():
#     """Test the train_svm_regressor function asynchronously."""
#     try:
#         data_source = r"D:\SLM.xls"
#         kernel = None  # 使用 RBF 核
#         optimize_hyperparameters = True
#         n_trials = 2  # 减少试验次数以快速测试
#         cv_folds = 5
#         scoring_metric = "r2"
#         validate_data = True
#         save_model = True
#         apply_preprocessing = True
#         scaling_method = "standard"
        
#         print("Starting SVM regression training test...")
#         print(f"Data source: {data_source}")
#         print(f"Kernel: {kernel}")
#         print(f"Hyperparameter optimization: {optimize_hyperparameters}")
#         print(f"Number of trials: {n_trials}")
        
#         result = await train_svm_regressor(
#             data_source=data_source,
#             target_dimension=1,
#             kernel=kernel,
#             optimize_hyperparameters=optimize_hyperparameters,
#             n_trials=n_trials,
#             cv_folds=cv_folds,
#             scoring_metric=scoring_metric,
#             validate_data=validate_data,
#             save_model=save_model,
#             apply_preprocessing=apply_preprocessing,
#             scaling_method=scaling_method
#         )
        
#         print("SVM regression training completed successfully!")
#         print(f"Training result keys: {list(result.keys())}")
        
#         if 'error' not in result:
#             print("✅ SVM regression training test PASSED")
#             # 打印一些关键信息
#             if 'model_id' in result:
#                 print(f"Model ID: {result['model_id']}")
#             if 'performance_metrics' in result:
#                 print(f"Performance metrics: {result['performance_metrics']}")
#         else:
#             print("❌ SVM regression training test FAILED")
#             print(f"Error: {result['error']}")
            
#         return result
        
#     except Exception as e:
#         print(f"❌ SVM regression training test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# async def test_train_svm_classifier():
#     """Test the train_svm_classifier function asynchronously."""
#     try:
#         data_source = r"D:\鸢尾花多分类数据集.xlsx"
#         kernel = None
#         optimize_hyperparameters = True
#         n_trials = 2
#         cv_folds = 5
#         scoring_metric = "f1_weighted"
#         validate_data = True
#         save_model = True
#         apply_preprocessing = True
#         scaling_method = "standard"
#         result = await train_svm_classifier(
#             data_source=data_source,
#             kernel=kernel,
#             optimize_hyperparameters=optimize_hyperparameters,
#             n_trials=n_trials,
#             cv_folds=cv_folds,
#             scoring_metric=scoring_metric,
#             validate_data=validate_data,
#             save_model=save_model,
#             apply_preprocessing=apply_preprocessing,
#             scaling_method=scaling_method
#         )
        
#         print("Starting SVM classifier training test...")
#         print(f"Data source: {data_source}")
#         print(f"Kernel: {kernel}")
#         print(f"Hyperparameter optimization: {optimize_hyperparameters}")
#         print(f"Number of trials: {n_trials}")
#     except Exception as e:
#         print(f"SVM classifier training test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(test_predict_form_values())