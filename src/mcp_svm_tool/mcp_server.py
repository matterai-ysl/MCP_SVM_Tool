"""
MCP Server for SVM Machine Learning Tool

This module implements the MCP server using FastMCP and provides core tool functions:
1. train_svm_classifier - SVM classification model training with comprehensive features
2. train_svm_regressor - SVM regression model training
3. train_xgboost_regressor - XGBoost regression model training with comprehensive scoring metrics
4. train_xgboost_classifier - XGBoost classification model training  
5. predict_from_file - Batch prediction from file
6. predict_from_values - Real-time prediction from values
7. analyze_feature_importance - Feature importance analysis
8. list_models - List all trained models
9. get_model_info - Get model detailed information
10. delete_model - Delete trained model
"""

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

# FastMCP import
from mcp.server import FastMCP

# Internal modules
from .training import TrainingEngine
from .prediction import PredictionEngine
from .feature_importance import  FeatureImportanceAnalyzer
from .model_manager import ModelManager
from .data_validator import DataValidator

from starlette.requests import Request
from starlette.responses import PlainTextResponse
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server with detailed instructions
mcp = FastMCP(
    name="SVM Machine Learning Tool",
    instructions="""
    This is a comprehensive Machine Learning server providing advanced SVM and XGBoost capabilities.
    
    Available tools:
    1. train_svm_classifier - Train an SVM classification model with kernel options and hyperparameter optimization
    2. train_svm_regressor - Train an SVM regression model with kernel options and hyperparameter optimization
    3. train_xgboost_regressor - Train an XGBoost regression model with advanced features
    4. train_xgboost_classifier - Train an XGBoost classifier for classification tasks
    5. predict_from_file - Make batch predictions from a data file
    6. predict_from_values - Make real-time predictions from feature values
    7. analyze_feature_importance - Analyze feature importance of a trained model
    8. list_models - List all available trained models
    9. get_model_info - Get detailed information about a specific model
    10. delete_model - Delete a trained model
    
    Use these tools to build complete ML workflows from training to deployment.
    The SVM tools support multiple kernels (linear, rbf, poly, sigmoid), hyperparameter optimization with Optuna,
    and comprehensive feature importance analysis using permutation importance.
    """
)

# Initialize engines
root_dir = Path("./trained_models")
training_engine = TrainingEngine("trained_models")
prediction_engine = PredictionEngine("trained_models")
model_manager = ModelManager("trained_models")
base_url = "http://localhost:8080"
# @mcp.custom_route("/hello/{name}", methods=["GET"])
# async def simple_hello(request: Request) -> PlainTextResponse:
#     name = request.path_params["name"]
#     return PlainTextResponse(f"Hello, {name}!")

@mcp.tool()
async def train_svm_classifier(
    data_source: str,
    kernel: str = None,
    optimize_hyperparameters: bool = True,
    n_trials: int = 20,
    cv_folds: int = 5,
    scoring_metric: str = "f1_weighted",
    validate_data: bool = True,
    save_model: bool = True,
    apply_preprocessing: bool = True,
    scaling_method: str = "standard"
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
        
    Returns:
        Training results including model performance, metadata, and SVM-specific information
    """
    try:
        logger.info(f"Training SVM classification model from: {data_source}")
        # Convert scoring metric to sklearn format
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
        
        # Validate parameters
        valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if kernel not in valid_kernels and kernel is not None:
            raise ValueError(f"Invalid kernel: {kernel}. Supported kernels: {valid_kernels}")
            
        valid_metrics = ['f1_weighted', 'f1_macro', 'f1_micro', 'accuracy', 'precision_weighted', 'recall_weighted', 'roc_auc']
        if scoring_metric not in valid_metrics:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}. Supported metrics: {valid_metrics}")
        
        model_id = str(uuid.uuid4())
        # Load and validate data for classification
        from .data_utils import DataProcessor
        data_processor = DataProcessor()
        df = data_processor.load_data(data_source)
        # Validate target_dimension parameter
        target_dimension = 1
        if target_dimension <= 0:
            raise ValueError(f"Target dimension must be a positive integer, got: {target_dimension}")
        
        if target_dimension > len(df.columns):
            raise ValueError(f"Target dimension {target_dimension} exceeds number of columns {len(df.columns)}")
        
        # Get target columns (last N columns based on target_dimension)
        target_columns = df.columns[-target_dimension:].tolist()
        logger.info(f"XGBoost classification with {target_dimension} target(s): {target_columns}")
        
        # Run training in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: training_engine.train_svm_classification(
                data_source=data_source,
                model_id=model_id,
                kernel=kernel,
                target_column=target_columns if target_dimension > 1 else target_columns[0],
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
    kernel: str = None,
    optimize_hyperparameters: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    scoring_metric: str = "r2",
    validate_data: bool = True,
    save_model: bool = True,
    apply_preprocessing: bool = True,
    scaling_method: str = "standard"
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
        
    Returns:
        Training results including model performance, metadata, and SVM-specific information
    """
    try:
        logger.info(f"Training SVM regression model from: {data_source}")
        
        # Validate parameters
        valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if kernel not in valid_kernels and kernel is not None:
            raise ValueError(f"Invalid kernel: {kernel}. Supported kernels: {valid_kernels}")
            
        valid_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 
                        'r2', 'neg_mean_absolute_percentage_error', 'explained_variance', 
                        'max_error', 'neg_median_absolute_error']
        if scoring_metric not in valid_metrics:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}. Supported metrics: {valid_metrics}")
        # Validate target_dimension parameter



        from .data_utils import DataProcessor

        data_processor = DataProcessor()
        df = data_processor.load_data(data_source)       
        if target_dimension <= 0:
            raise ValueError(f"Target dimension must be a positive integer, got: {target_dimension}")
        
        if target_dimension > len(df.columns):
            raise ValueError(f"Target dimension {target_dimension} exceeds number of columns {len(df.columns)}")
        
        # Get target columns (last N columns based on target_dimension)
        target_columns = df.columns[-target_dimension:].tolist()
        
        logger.info(f"XGBoost multi-target regression with {target_dimension} target(s): {target_columns}")
        
        model_id = str(uuid.uuid4())
    
        
        
        # Run training in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: training_engine.train_svm_regression(
                data_source=data_source,
                target_column=target_columns if target_dimension > 1 else target_columns[0],
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
async def predict_from_file(
    model_id: str,
    data_source: str,
    output_path: str = None,
    include_confidence: bool = True,
    generate_report: bool = True
) -> Dict[str, Any]:
    """
    Make batch predictions from a data file.
    
    Args:
        model_id: Unique identifier for the trained model
        data_source: Path to prediction data file (CSV, Excel, etc.)
        output_path: Path to save prediction results (if None, uses default)
        include_confidence: Whether to include confidence scores
        generate_report: Whether to generate detailed report
        
    Returns:
        Prediction results and analysis
    """
    try:
        logger.info(f"Making batch predictions with model {model_id} from file: {data_source}")
        
        # Run prediction in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: prediction_engine.predict_from_file(
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
async def predict_from_values(
    model_id: str,
    feature_values: Union[List[float], List[List[float]], Dict[str, float], List[Dict[str, float]]],
    feature_names: List[str] = None,
    include_confidence: bool = True,
    save_intermediate_files: bool = True,
    generate_report: bool = True,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Make real-time predictions from feature values with CSV export and reporting.
    
    Supports both single and batch predictions:
    - Single: [1, 2, 3] or {'feature1': 1, 'feature2': 2}
    - Batch: [[1, 2, 3], [4, 5, 6]] or [{'feature1': 1}, {'feature1': 4}]
    
    Args:
        model_id: Unique identifier for the trained model
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
        
        # Run prediction in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: prediction_engine.predict_from_values(
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
async def list_models() -> List[Dict[str, Any]]:
    """
    List all available trained models.
    
    Returns:
        List of model information including IDs, names, and metadata
    """
    try:
        logger.info("Listing all available models")
        
        # Run in executor to avoid blocking
        models = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.list_models
        )
        
        logger.info(f"Found {len(models)} available models")
        return models
        
    except Exception as e:
        error_msg = f"Failed to list models: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Unique identifier for the model
        
    Returns:
        Detailed model information including performance metrics and metadata
    """
    try:
        logger.info(f"Getting information for model: {model_id}")
        
        # Run in executor to avoid blocking
        model_info = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.get_model_info, model_id
        )
        
        logger.info("Model information retrieved successfully")
        return model_info
        
    except Exception as e:
        error_msg = f"Failed to get model info: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg, "traceback": traceback.format_exc()}

@mcp.tool()
async def delete_model(model_id: str) -> Dict[str, Any]:
    """
    Delete a trained model.
    
    Args:
        model_id: Unique identifier for the model to delete
        
    Returns:
        Deletion status and information
    """
    try:
        logger.info(f"Deleting model: {model_id}")
        
        # Run in executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, model_manager.delete_model, model_id
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


def _create_analysis_archive(output_dir: Path, analysis_type: str, task_id: str, model_id: str) -> str:

    """创建分析结果的ZIP压缩包"""
    try:
        # 创建archives目录
        archives_dir = Path("trained_models") / model_id / "feature_analysis" / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建ZIP文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{analysis_type}_analysis_{task_id}_{timestamp}.zip"
        zip_path = archives_dir / zip_filename
        
        # 创建ZIP压缩包
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    except Exception as e:
        logger.error(f"Failed to create analysis archive: {e}")
        return None

async def test_predict_from_file():
    """Test the predict_from_file function asynchronously."""
    try:
        model_id = "1ada0b3e-0271-4012-b53b-4a626786b08d"
        data_source = r"D:\SLM预测.xls"
        include_confidence = True
        generate_report = True
        
        print("Starting prediction test...")
        result = await predict_from_file(
            model_id=model_id,
            data_source=data_source,
            include_confidence=include_confidence,
            generate_report=generate_report
        )
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_predict_form_values():

    try:
        model_id = "6b937211-5454-4a4f-b5cc-79514cb3cf48"
        feature_values = [5.1, 3.5, 1.4, 0.2]
        # model_id = "6299d812-41b2-4e40-aeb0-0133c6f3ba5f"
        # feature_values = [35, 80, 130, 800]
        print("Starting prediction test...")
        result = await predict_from_values(
            model_id=model_id,
            feature_values=feature_values
        )
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_train_svm_regressor():
    """Test the train_svm_regressor function asynchronously."""
    try:
        data_source = r"D:\SLM.xls"
        kernel = None  # 使用 RBF 核
        optimize_hyperparameters = True
        n_trials = 2  # 减少试验次数以快速测试
        cv_folds = 5
        scoring_metric = "r2"
        validate_data = True
        save_model = True
        apply_preprocessing = True
        scaling_method = "standard"
        
        print("Starting SVM regression training test...")
        print(f"Data source: {data_source}")
        print(f"Kernel: {kernel}")
        print(f"Hyperparameter optimization: {optimize_hyperparameters}")
        print(f"Number of trials: {n_trials}")
        
        result = await train_svm_regressor(
            data_source=data_source,
            target_dimension=1,
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
        
        print("SVM regression training completed successfully!")
        print(f"Training result keys: {list(result.keys())}")
        
        if 'error' not in result:
            print("✅ SVM regression training test PASSED")
            # 打印一些关键信息
            if 'model_id' in result:
                print(f"Model ID: {result['model_id']}")
            if 'performance_metrics' in result:
                print(f"Performance metrics: {result['performance_metrics']}")
        else:
            print("❌ SVM regression training test FAILED")
            print(f"Error: {result['error']}")
            
        return result
        
    except Exception as e:
        print(f"❌ SVM regression training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_train_svm_classifier():
    """Test the train_svm_classifier function asynchronously."""
    try:
        data_source = r"D:\鸢尾花多分类数据集.xlsx"
        kernel = None
        optimize_hyperparameters = True
        n_trials = 2
        cv_folds = 5
        scoring_metric = "f1_weighted"
        validate_data = True
        save_model = True
        apply_preprocessing = True
        scaling_method = "standard"
        result = await train_svm_classifier(
            data_source=data_source,
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
        
        print("Starting SVM classifier training test...")
        print(f"Data source: {data_source}")
        print(f"Kernel: {kernel}")
        print(f"Hyperparameter optimization: {optimize_hyperparameters}")
        print(f"Number of trials: {n_trials}")
    except Exception as e:
        print(f"SVM classifier training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_predict_form_values())