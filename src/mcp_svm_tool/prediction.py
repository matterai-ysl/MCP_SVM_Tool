"""
Prediction Engine

This module provides prediction functionality for both 
batch and real-time inference scenarios.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import uuid
import zipfile
import os

from .model_manager import ModelManager
from .data_utils import DataProcessor
from .data_preprocessing import DataPreprocessor
from .academic_report_generator import AcademicReportGenerator
from .config import BASE_URL,get_static_url,get_download_url
logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Handles model predictions for various input formats.
    
    Features:
    - Batch prediction from files
    - Real-time prediction from values
    - Confidence/probability estimation
    - Feature contribution analysis
    - Detailed prediction reports
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        """
        Initialize PredictionEngine.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.model_manager = ModelManager(models_dir)
        self.data_processor = DataProcessor()
        self.academic_report_generator = AcademicReportGenerator()
        logger.info("Initialized PredictionEngine")
        
    def predict_from_file(
        self,
        model_id: str,
        data_source: Union[str, pd.DataFrame],
        output_path: Optional[str] = None,
        include_confidence: bool = True,
        generate_report: bool = True,
        save_intermediate_files: bool = True
    ) -> Dict[str, Any]:
        """
        Make batch predictions from a file.
        
        Args:
            model_id: Unique identifier for the trained model
            data_source: Path to prediction data file or DataFrame
            output_path: Path to save prediction results
            include_confidence: Whether to include confidence scores
            generate_report: Whether to generate detailed report
            save_intermediate_files: Whether to save all intermediate files (original, processed, etc.)
            
        Returns:
            Prediction results and analysis
        """
        try:
            start_time = datetime.now()
            prediction_id = str(uuid.uuid4())
            
            logger.info(f"Starting batch prediction with model {model_id}")
            
            # Create prediction folder structure
            prediction_folder = self._create_prediction_folder(prediction_id, model_id, save_intermediate_files)
            
            # Load model
            model = self._load_model(model_id)
            model_info = self.model_manager.get_model_info(model_id)
            
            # Load preprocessing pipeline if available
            preprocessor = self._load_preprocessing_pipeline(model_id)
            
            # Load and prepare data
            if isinstance(data_source, str):
                logger.info(f"Loading data from file: {data_source}")
                df_original = self.data_processor.load_data(data_source)
                original_file_path = data_source
            else:
                logger.info("Using provided DataFrame")
                df_original = data_source.copy()
                original_file_path = None
            
            # Save original data if requested
            if save_intermediate_files and prediction_folder:
                original_data_path = prediction_folder / "01_original_data.csv"
                df_original.to_csv(original_data_path, index=False)
                logger.info(f"Saved original data to: {original_data_path}")
            
            # Apply preprocessing if available
            if preprocessor:
                logger.info("Applying preprocessing pipeline to input data")
                expected_features = preprocessor.feature_names_in
                self._validate_features(df_original, expected_features)
                
                # Use only the original features for preprocessing
                X_raw = df_original[expected_features]
                X_processed = preprocessor.transform_features(X_raw)
                feature_names_used = preprocessor.feature_names_out
                
                # Save processed features if requested
                if save_intermediate_files and prediction_folder:
                    df_processed = pd.DataFrame(X_processed, columns=feature_names_used)
                    processed_data_path = prediction_folder / "02_processed_features.csv"
                    df_processed.to_csv(processed_data_path, index=False)
                    logger.info(f"Saved processed features to: {processed_data_path}")
                
            else:
                # Fallback to original feature handling
                expected_features = model_info['feature_names']
                self._validate_features(df_original, expected_features)
                X_processed = df_original[expected_features].values
                feature_names_used = expected_features
            
            logger.info(f"Making predictions for {len(df_original)} samples")
            
            # Make predictions on processed features
            predictions_processed = model.predict(X_processed)
            
            # Apply inverse target transformation if available
            predictions_original_scale = predictions_processed.copy()
            if preprocessor and preprocessor.target_is_fitted and model_info['task_type'] == 'regression':
                logger.info("Applying inverse target transformation to predictions")
                predictions_original_scale = preprocessor.inverse_transform_target(predictions_processed)
            
            # Get confidence scores if requested
            confidence_scores = None
            if include_confidence:
                confidence_scores = self._get_confidence_scores(model, X_processed, model_info['task_type'])
            

            
            # Get label mapping for classification tasks
            label_mapping = None
            if model_info['task_type'] == 'classification':
                # Try to get label mapping from model metadata first (preferred)
                if 'label_mapping' in model_info and model_info['label_mapping']:
                    label_mapping = model_info['label_mapping']
                    logger.info(f"Retrieved label mapping from metadata: {label_mapping['class_to_label'] if label_mapping else 'None'}")
                # Fallback to preprocessor if not in metadata
                elif preprocessor and hasattr(preprocessor, 'label_mapping') and preprocessor.label_mapping:
                    label_mapping = preprocessor.label_mapping
                    logger.info(f"Retrieved label mapping from preprocessor: {label_mapping['class_to_label'] if label_mapping else 'None'}")
                else:
                    logger.warning("No label mapping found in metadata or preprocessor for classification task")

            # Save detailed prediction data if requested
            saved_files = {}
            if save_intermediate_files and prediction_folder:
                saved_files = self._save_prediction_details(
                    prediction_folder, df_original, X_processed, 
                    predictions_processed, predictions_original_scale, 
                    confidence_scores, feature_names_used, expected_features,
                    model_info,  # Pass model metadata for target names
                    label_mapping  # Pass label mapping for classification tasks
                )
            
            # Prepare results
            predict_metadata = {
                'prediction_id': prediction_id,
                'model_id': model_id,
                'task_type': model_info['task_type'],
                'predictions': predictions_processed.tolist(),
                'raw_predictions': predictions_original_scale.tolist(),  # Keep raw numeric predictions for label mapping
                'num_predictions': len(predictions_original_scale),
                'confidence_scores': confidence_scores,
                'label_mapping': label_mapping,  # Add label mapping for classification tasks
                'prediction_folder': str(prediction_folder) if save_intermediate_files else None,
                'saved_files': saved_files if save_intermediate_files else {},
                'model_metadata': model_info,  # Include full model metadata for target names
                'prediction_metadata': {
                    'prediction_timestamp': datetime.now().isoformat(),
                    'data_shape': df_original.shape,
                    'feature_names': expected_features,
                    'processed_feature_names': feature_names_used,
                    'original_file_path': original_file_path,
                    'preprocessing_applied': preprocessor is not None,
                    'prediction_time_seconds': (datetime.now() - start_time).total_seconds(),
                    'tool_input_details': {
                        'model_id': model_id,
                        'data_source': data_source if isinstance(data_source, str) else 'DataFrame',
                        'output_path': output_path,
                        'include_confidence': include_confidence,
                        'generate_report': generate_report,
                        'save_intermediate_files': save_intermediate_files
                    }
                }
            }
            
            # Save prediction metadata if intermediate files were requested
            if save_intermediate_files and prediction_folder:
                try:
                    self._save_prediction_form_values_results_metadata(predict_metadata, prediction_folder)
                except Exception as save_error:
                    logger.error(f"Failed to save prediction metadata: {str(save_error)}")

            # Save results if output path provided
            results = {
                'model_id': model_id,
                'task_type': model_info['task_type'] if model_info else 'unknown',
                'target_name': model_info['target_name'] if model_info else 'unknown',
                'predictions': predictions_original_scale.tolist()[:10],
                'confidence_scores': confidence_scores[:10],
                'num_predictions': len(predictions_original_scale),
                'input_data': df_original.to_dict(orient='records')[:10],
            }
            if output_path:
                self._save_predictions(results, output_path)
                results['output_path'] = output_path
            
            # Generate detailed report if requested
            if generate_report:
                try:
                    # Generate both HTML and markdown reports
                    html_report_path = self._generate_prediction_report(results, df_original, prediction_folder, model_info)
                    # Calculate relative path for HTML report
                    try:
                        html_relative_path = get_static_url(html_report_path) #type: ignore
                        results['predicttion_report_summary'] = f"result only show the first 10 predictions, You can find the html prediction report summary in {html_relative_path}"
                    except ValueError as e:
                        # Fallback if relative path calculation fails
                        logger.error(f"Failed to get static url: {e}")
                    # Generate markdown experiment report and create archive (only if prediction_folder exists)
                    if prediction_folder:
                        markdown_report_path = self.academic_report_generator.generate_prediction_experiment_report_from_folder(prediction_folder)
                        
                        # Create prediction archive
                        prediction_archive_path = self._create_prediction_archive(prediction_folder, prediction_id)
                        if prediction_archive_path:
                            # Use the archive path instead of folder path
                            try:
                                archive_relative_path = get_download_url(prediction_archive_path) #type: ignore
                                results['prediction_details'] = f"""All detailed prediction data are saved in {archive_relative_path},
                            which can be downloaded by users for reproducibility and academic research reference.  """
                            except ValueError as e:
                                # Fallback if relative path calculation fails
                                logger.error(f"Failed to get download url: {e}")
                                results['prediction_details'] = str(e)
                        else:
                            # Fallback to folder path if archive creation fails
                            try:
                                folder_relative_path = get_download_url(prediction_folder) #type: ignore
                                results['prediction_details'] = f"""All detailed prediction data are saved in {folder_relative_path},
                            which can be downloaded by users for reproducibility and academic research reference.  """
                            except ValueError as e:
                                logger.error(f"Failed to get download url: {e}")
                                results['prediction_details'] = str(e)

                        logger.info(f"Generated Markdown experiment report: {markdown_report_path}")
                    
                    logger.info(f"Generated HTML report: {html_report_path}")
                except Exception as report_error:
                    logger.error(f"Failed to generate reports: {str(report_error)}")
                    results['report_error'] = str(report_error)
            
            logger.info(f"Batch prediction completed successfully in {predict_metadata['prediction_metadata']['prediction_time_seconds']:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
        
    def predict_from_values(
        self,
        model_id: str,
        feature_values: Union[List[float], List[List[float]], Dict[str, float], List[Dict[str, float]]],
        feature_names: Optional[List[str]] = None,
        include_confidence: bool = True,
        save_intermediate_files: bool = True,
        generate_report: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make predictions from feature values for real-time inference.
        
        Supports both single and batch predictions:
        - Single: [1, 2, 3] or {'feature1': 1, 'feature2': 2}
        - Batch: [[1, 2, 3], [4, 5, 6]] or [{'feature1': 1}, {'feature1': 4}]
        
        Args:
            model_id: Unique identifier for the trained model
            feature_values: Feature values in various formats (single or batch)
            feature_names: Names of features (required if feature_values is a list of lists)
            include_confidence: Whether to include confidence scores
            save_intermediate_files: Whether to save CSV files and detailed prediction data
            generate_report: Whether to generate prediction experiment report
            output_path: Custom output path for prediction files (optional)
            
        Returns:
            Prediction results and analysis
        """
        start_time = datetime.now()
        prediction_id = str(uuid.uuid4())
        

        logger.info(f"Starting real-time prediction with model {model_id}")
        
        # Initialize variables to avoid scope issues
        prediction_folder = None
        model_info = None
        expected_features = None
        feature_names_used = None
        df_original = None
        X = None
        confidence_scores = None
        formatted_predictions = None
        formatted_raw_predictions = None
        display_predictions = None
        prediction_processed = None
        prediction_original = None
        label_mapping = None
        is_batch = False
        
        try:
            error_occurred = False
            error_message = ""
            
            # Load model and get info
            model = self._load_model(model_id)
            model_info = self.model_manager.get_model_info(model_id)
            logger.info(f"Successfully loaded model: {model_id}")

            # Get expected features from model metadata
            if 'feature_names' in model_info:
                expected_features = model_info['feature_names']
            elif 'training_features' in model_info:
                expected_features = model_info['training_features']
            else:
                raise ValueError("Model metadata does not contain feature information")
            
            # Load preprocessing pipeline if available
            preprocessor = self._load_preprocessing_pipeline(model_id)
            
            # Determine if this is batch prediction and prepare data
            is_batch = self._is_batch_input(feature_values)
            
            # Create prediction folder early for error handling
            if save_intermediate_files:
                prediction_folder = self._create_prediction_folder(prediction_id, model_id, save_intermediate_files)
            
            # Prepare original data DataFrame
            if preprocessor:
                df_original = self._prepare_dataframe_for_preprocessing(feature_values, feature_names, expected_features, is_batch)
            else:
                # Create DataFrame from direct feature values
                if is_batch:
                    if isinstance(feature_values[0], dict):
                        df_original = pd.DataFrame(feature_values)
                    else:
                        df_original = pd.DataFrame(feature_values, columns=expected_features)
                else:
                    if isinstance(feature_values, dict):
                        df_original = pd.DataFrame([feature_values])
                    else:
                        df_original = pd.DataFrame([feature_values], columns=expected_features)
            
            # Save original data early
            if prediction_folder:
                original_data_path = prediction_folder / "01_original_data.csv"
                df_original.to_csv(original_data_path, index=False)
                logger.info(f"Saved original data to: {original_data_path}")
            
            if preprocessor:
                # Use preprocessing pipeline
                # Validate features
                self._validate_features(df_original, expected_features)
                
                # Apply preprocessing with proper type enforcement
                logger.info("Applying preprocessing pipeline to input values")
                df_typed = self._enforce_training_data_types(df_original[expected_features], preprocessor)
                X = preprocessor.transform_features(df_typed)
                feature_names_used = preprocessor.feature_names_out
            else:
                # No preprocessing - use features directly
                X = self._prepare_feature_array_batch(feature_values, feature_names, expected_features, is_batch)
                feature_names_used = expected_features
            
            # Save processed features
            if prediction_folder:
                df_processed = pd.DataFrame(X, columns=feature_names_used)
                processed_data_path = prediction_folder / "02_processed_features.csv"
                df_processed.to_csv(processed_data_path, index=False)
                logger.info(f"Saved processed features to: {processed_data_path}")
            
            # Make prediction
            if is_batch:
                logger.info(f"Making batch prediction for {X.shape[0]} samples")
            else:
                logger.info("Making single prediction")
            prediction_processed = model.predict(X)  # This is in processed/normalized scale
            
            # Save numeric predictions before any label transformation
            prediction_numeric = prediction_processed.copy()
            
            # Apply inverse target transformation if available
            if preprocessor and hasattr(preprocessor, 'inverse_transform_target'):
                logger.info("Applying inverse target transformation to prediction")
                prediction_original = preprocessor.inverse_transform_target(prediction_processed)
                print("*"*100)
                print("prediction_original",prediction_original)
            else:
                prediction_original = prediction_processed  # No transformation needed
          
  
            # Use original scale prediction for final results (backward compatibility)
            prediction = prediction_original
            
            # For classification tasks, get label mapping for later use in reports
            # Get label mapping for classification tasks
            label_mapping = None
            if model_info['task_type'] == 'classification':
                # Try to get label mapping from model metadata first (preferred)
                if 'label_mapping' in model_info and model_info['label_mapping']:
                    label_mapping = model_info['label_mapping']
                    logger.info(f"Retrieved label mapping from metadata: {label_mapping['class_to_label'] if label_mapping else 'None'}")
                # Fallback to preprocessor if not in metadata
                elif preprocessor and hasattr(preprocessor, 'label_mapping') and preprocessor.label_mapping:
                    label_mapping = preprocessor.label_mapping
                    logger.info(f"Retrieved label mapping from preprocessor: {label_mapping['class_to_label'] if label_mapping else 'None'}")
                else:
                    logger.warning("No label mapping found in metadata or preprocessor for classification task")
            
            # Get confidence scores

            if include_confidence:
                confidence_scores = self._get_confidence_scores(model, X, model_info['task_type'])
                if confidence_scores:
                        logger.info("Confidence calculated using SVM")
                
            # Format predictions based on task type and whether it's single/batch
            # Use numeric predictions for formatting to ensure we get integers for classification
            formatted_predictions = self._format_predictions(prediction_numeric, model_info['task_type'], is_batch)
            formatted_raw_predictions = self._format_predictions(prediction_original, model_info['task_type'], is_batch)

            # For classification, apply label mapping to formatted predictions for display
            display_predictions = formatted_predictions

            
            if model_info['task_type'] == 'classification' and label_mapping:
                class_to_label = label_mapping.get('class_to_label', {})
                
                def safe_label_lookup(pred_value):
                    """Safely lookup label for prediction value, handling both string and numeric predictions."""
                    # If prediction is already a string (like 'Iris-setosa'), return it directly
                    if isinstance(pred_value, str):
                        return pred_value
                    
                    # For numeric predictions, try to convert and lookup
                    try:
                        # Try both string and integer keys
                        return (class_to_label.get(str(int(pred_value))) or 
                                class_to_label.get(int(pred_value)) or 
                                f'Class_{int(pred_value)}')
                    except (ValueError, TypeError):
                        # If conversion fails, return the original value or a default
                        return str(pred_value) if pred_value is not None else 'Unknown'
                
                if is_batch:
                    if isinstance(formatted_predictions[0], list):
                        # Multi-target classification batch
                        display_predictions = [[safe_label_lookup(p) for p in sample] for sample in formatted_predictions]
                    else:
                        # Single-target classification batch
                        display_predictions = [safe_label_lookup(p) for p in formatted_predictions]
                else:
                    if isinstance(formatted_predictions, list):
                        # Multi-target classification single
                        display_predictions = [safe_label_lookup(p) for p in formatted_predictions]
                    else:
                        # Single-target classification single
                        display_predictions = safe_label_lookup(formatted_predictions)
            
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            logger.error(f"Error during prediction: {error_message}")
            
            # Set default values for failed predictions
            if model_info is None:
                model_info = {'task_type': 'unknown', 'model_id': model_id}
            if formatted_predictions is None:
                formatted_predictions = []
            if display_predictions is None:
                display_predictions = []
            if formatted_raw_predictions is None:
                formatted_raw_predictions = []
        
        # Create prediction folder and save detailed files if requested
        saved_files = {}
  


        
        if save_intermediate_files and prediction_folder:
            try:
                # Get detailed model metadata for proper target names
                model_metadata = model_info if model_info else {'task_type': 'unknown'}
                
                # Save detailed prediction files including CSVs
                saved_files = self._save_prediction_details(
                    prediction_folder=prediction_folder,
                    df_original=df_original,
                    X_processed=X,
                    predictions_processed=prediction_processed if (preprocessor and hasattr(preprocessor, 'inverse_transform_target')) else None,
                    predictions_original=prediction_original,  # Use numeric predictions for CSV saving
                    confidence_scores=confidence_scores,
                    processed_feature_names=feature_names_used if feature_names_used else [],
                    original_feature_names=expected_features if expected_features else [],
                    model_metadata=model_metadata,
                    label_mapping=label_mapping,  # Pass label mapping for classification tasks
                    error_occurred=error_occurred,
                    error_message=error_message,
                )
                
                logger.info(f"Saved prediction files: {list(saved_files.keys())}")

                # Save prediction results to file

            except Exception as save_error:
                logger.error(f"Failed to save prediction details: {str(save_error)}")
        
        # Prepare metadata for reports (always needed)
        predict_metadata = {
            'model_id': model_id,
            'task_type': model_info['task_type'] if model_info else 'unknown',
            'is_batch': is_batch,
            'predictions': display_predictions,  # Use display predictions (with labels for classification)
            'raw_predictions': formatted_raw_predictions,  # Keep raw numeric predictions for label mapping in reports
            'confidence_scores': confidence_scores,
            'label_mapping': label_mapping if label_mapping else None,  # Add label mapping for classification tasks
            'error_occurred': error_occurred,
            'error_message': error_message if error_occurred else None,
            'prediction_id': prediction_id,  # Add prediction_id field
            'prediction_metadata': {
                'prediction_timestamp': datetime.now().isoformat(),
                'feature_names': expected_features if expected_features else [],
                'feature_values': X.tolist() if X is not None else [],
                'num_samples': X.shape[0] if X is not None else 0,
                'prediction_time_seconds': (datetime.now() - start_time).total_seconds(),
                'tool_input_details': {
                    'model_id': model_id,
                    'feature_values': feature_values,
                    'feature_names': feature_names,
                    'include_confidence': include_confidence,
                    'save_intermediate_files': save_intermediate_files,
                    'generate_report': generate_report,
                    'output_path': output_path
                }
            },
            'saved_files': saved_files if save_intermediate_files else {}
        }

        # Save prediction metadata if intermediate files were requested
        if save_intermediate_files and prediction_folder:
            try:
                self._save_prediction_form_values_results_metadata(predict_metadata, prediction_folder)
            except Exception as save_error:
                logger.error(f"Failed to save prediction metadata: {str(save_error)}")

        #精简返回结果
        results = {
            'model_id': model_id,
            'task_type': model_info['task_type'] if model_info else 'unknown',
            'is_batch': is_batch,
            'target_name': model_info['target_name'] if model_info else 'unknown',
            'predictions': display_predictions,  # Use display predictions (with labels for classification)
            'raw_predictions': formatted_raw_predictions,  # Keep raw numeric predictions for label mapping in reports
            'confidence_scores': confidence_scores,
            'prediction_metadata': {
                'prediction_timestamp': datetime.now().isoformat(),
                'feature_names': expected_features if expected_features else [],
                'feature_values': X.tolist() if X is not None else [],
                'num_samples': X.shape[0] if X is not None else 0,
                'prediction_time_seconds': (datetime.now() - start_time).total_seconds(),
            }
        }


        #Generate report if requested (always try to generate, even on errors)
        if generate_report:
            try:
                # Create input DataFrame for report generation
                input_df = df_original if df_original is not None else pd.DataFrame()
                
                # Generate both HTML and markdown reports
                html_report_path = self._generate_prediction_report(predict_metadata, input_df, prediction_folder,model_info)
                # Calculate relative path for HTML report
                try:
                    html_relative_path = get_static_url(html_report_path) #type: ignore
                    results['predicttion_report_summary_html_path'] = f"You can find the html prediction report summary in {html_relative_path}"
                except ValueError as e:
                    # Fallback if relative path calculation fails
                    logger.error(f"Failed to get static url: {e}")
                    results['predicttion_report_summary_html_path'] = str(e)
                
                # Generate markdown experiment report (only if prediction_folder exists)
                if prediction_folder:
                    markdown_report_path = self.academic_report_generator.generate_prediction_experiment_report_from_folder(prediction_folder)
                    # results['experiment_report'] = markdown_report_path
                    # results['markdown_report'] = markdown_report_path

                    # Create prediction archive
                    prediction_archive_path = self._create_prediction_archive(prediction_folder, prediction_id)
                    if prediction_archive_path:
                        # Use the archive path instead of folder path
                        archive_relative_path = get_download_url(prediction_archive_path) #type: ignore
                        results['prediction_details'] = f"""All detailed prediction data are saved in {archive_relative_path},
                        which can be downloaded by users for reproducibility and academic research reference.  """

                    else:
                        # Fallback to folder path if archive creation fails
                        try:
                            folder_relative_path = get_download_url(prediction_folder) #type: ignore
                            results['prediction_details'] = f"""All detailed prediction data are saved in {folder_relative_path},
                        which can be downloaded by users for reproducibility and academic research reference.  """
                        except ValueError as e:
                            logger.error(f"Failed to get download url: {e}")
                            results['prediction_details'] = str(e)

                    logger.info(f"Generated Markdown experiment report: {markdown_report_path}")
                
                logger.info(f"Generated HTML report: {html_report_path}")
            except Exception as report_error:
                logger.error(f"Failed to generate reports: {str(report_error)}")
                results['report_error'] = str(report_error)
        
        # Maintain backward compatibility for single predictions
        # if not is_batch and display_predictions is not None:
        #     results['prediction'] = display_predictions[0] if isinstance(display_predictions, list) else display_predictions
        #     results['confidence_score'] = confidence_scores[0] if confidence_scores else None
        
        if error_occurred:
            logger.error(f"Real-time prediction completed with errors in {results['prediction_metadata']['prediction_time_seconds']:.4f} seconds: {error_message}")
        else:
            logger.info(f"Real-time prediction completed successfully in {results['prediction_metadata']['prediction_time_seconds']:.4f} seconds")
                #精简返回结果


        return results
    
    def _load_model(self, model_id: str) -> Any:
        """Load model from model manager."""
        try:
            model = self.model_manager.load_model(model_id)
            logger.info(f"Successfully loaded model: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise ValueError(f"Model {model_id} not found or failed to load")
    
    def _load_preprocessing_pipeline(self, model_id: str) -> Optional[DataPreprocessor]:
        """Load preprocessing pipeline if available."""
        try:
            # Get model directory
            model_info = self.model_manager.get_model_info(model_id)
            model_dir = Path(self.model_manager.models_dir) / model_id
            
            # Check for preprocessing pipeline file
            pipeline_path = model_dir / "preprocessing_pipeline.pkl"
            
            if pipeline_path.exists():
                logger.info(f"Loading preprocessing pipeline from: {pipeline_path}")
                preprocessor = DataPreprocessor()
                preprocessor.load_pipeline(str(pipeline_path))
                return preprocessor
            else:
                logger.info("No preprocessing pipeline found, using original features")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load preprocessing pipeline: {e}")
            return None
    
    def _validate_features(self, df: pd.DataFrame, expected_features: List[str]):
        """Validate that DataFrame contains expected features."""
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        extra_features = set(df.columns) - set(expected_features)
        if extra_features:
            logger.warning(f"Extra features found (will be ignored): {extra_features}")
    
    def _prepare_feature_array(
        self, 
        feature_values: Union[List[float], Dict[str, float]], 
        feature_names: Optional[List[str]], 
        expected_features: List[str]
    ) -> np.ndarray:
        """Prepare feature array from various input formats."""
        if isinstance(feature_values, dict):
            # Dictionary format: {feature_name: value}
            missing_features = set(expected_features) - set(feature_values.keys())
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            X = np.array([feature_values[feature] for feature in expected_features])
            
        elif isinstance(feature_values, list):
            # List format: [value1, value2, ...]
            if len(feature_values) != len(expected_features):
                raise ValueError(f"Expected {len(expected_features)} features, got {len(feature_values)}")
            
            X = np.array(feature_values)
            
        else:
            raise ValueError("feature_values must be a list or dictionary")
        
        return X
    
    def _get_confidence_scores(self, model: Any, X: np.ndarray, task_type: str) -> Optional[List[float]]:
        """
        Get confidence scores for predictions.
        
        For classification: 
          - Uses maximum class probability as confidence if predict_proba available
          - For SVM without probabilities, uses normalized absolute distance from decision boundary
        
        For regression: 
          - For SVR, uses distance from hyperplane relative to epsilon tube
          - For ensemble models, uses prediction variance
          - Fallback to magnitude-based confidence for other models
        
        Returns:
            List of confidence scores (0-1), where higher values indicate more confident predictions.
        """
        try:
            # Detect if model is SVM
            is_svm = 'SVC' in str(type(model)) or 'SVR' in str(type(model)) or 'SVM' in str(type(model))
            
            if task_type == 'classification':
                print("*"*100)
                print("task_type",task_type)
                print("*"*100)
                # For classification, first try the probability approach if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    confidence_scores = np.max(probabilities, axis=1).tolist()
                    logger.info("Confidence calculated using maximum class probability")
                # SVM-specific approach when probability=False (default)
                elif is_svm and hasattr(model, 'decision_function'):
                    # Get raw decision scores
                    print("*"*100)
                    print("decision_function")
                    print("*"*100)
                    decision_scores = model.decision_function(X)
                    
                    # Handle binary vs multi-class differently
                    if decision_scores.ndim == 1:  # Binary case
                        # For binary, confidence is based on absolute distance from hyperplane
                        abs_distances = np.abs(decision_scores)
                        # Normalize to 0-1 range using sigmoid function
                        from scipy.special import expit  # sigmoid function
                        confidence_scores = expit(abs_distances).tolist()
                        logger.info("Confidence for SVM binary classification calculated using absolute distance from hyperplane")
                    else:  # Multi-class case
                        # For multiclass, confidence is the difference between highest and second highest scores
                        # Sort decision scores in descending order
                        sorted_scores = np.sort(decision_scores, axis=1)[:, ::-1]
                        # Calculate the margin between the top two classes
                        margins = sorted_scores[:, 0] - sorted_scores[:, 1]
                        # Normalize margins to 0-1 using sigmoid function
                        from scipy.special import expit
                        confidence_scores = expit(margins).tolist()
                        logger.info("Confidence for SVM multi-class calculated using margin between top classes")
                else:
                    # Fallback for other models
                    logger.warning("Model has no predict_proba or decision_function method, using fallback")
                    confidence_scores = [0.5] * X.shape[0]  # Default mid-confidence
                    logger.info("Using default mid-confidence as fallback")
            else:
                # For regression tasks
                if is_svm and hasattr(model, 'support_'):
                    # SVR-specific confidence calculation
                    try:
                        # For SVR, we can estimate confidence based on:
                        # 1. Distance from hyperplane relative to epsilon tube
                        # 2. If we can access dual_coef_ and support_vectors_
                        
                        if hasattr(model, 'dual_coef_') and hasattr(model, 'support_vectors_'):
                            from sklearn.metrics.pairwise import pairwise_kernels
                            
                            # Calculate kernel between test points and support vectors
                            if hasattr(model, 'kernel') and callable(model.kernel):
                                # Custom kernel
                                K = np.array([model.kernel(X, sv.reshape(1, -1)) for sv in model.support_vectors_])
                            else:
                                # Standard kernel
                                kernel = getattr(model, 'kernel', 'rbf')
                                K = pairwise_kernels(X, model.support_vectors_, metric=kernel,
                                                    gamma=getattr(model, 'gamma', 'scale'),
                                                    degree=getattr(model, 'degree', 3),
                                                    coef0=getattr(model, 'coef0', 0))
                            
                            # Compute decision values
                            decision_vals = np.dot(K.T, model.dual_coef_.T)
                            
                            # Estimate distance from epsilon tube
                            epsilon = getattr(model, 'epsilon', 0.1)
                            confidence_raw = np.abs(decision_vals.ravel()) / (1 + epsilon)
                            
                            # Normalize to 0-1 range using sigmoid
                            from scipy.special import expit
                            confidence_scores = expit(confidence_raw).tolist()
                            logger.info("Confidence calculated using SVR distance from epsilon tube")
                        else:
                            # Fallback for SVR when we can't access internals
                            predictions = model.predict(X)
                            # Use epsilon as baseline if available
                            epsilon = getattr(model, 'epsilon', 0.1)
                            confidence_scores = [1.0 / (1.0 + epsilon + abs(pred)) for pred in predictions]
                            logger.info("Confidence calculated using SVR prediction magnitude with epsilon")
                    except Exception as svr_error:
                        logger.warning(f"SVR-specific confidence calculation failed: {svr_error}, using fallback")
                        # Fallback to standard approach
                        predictions = model.predict(X)
                        confidence_scores = [1.0 / (1.0 + abs(pred)) for pred in predictions]
                        logger.info("Using standard confidence fallback for SVR")
                # Ensemble model approach (original implementation)
                elif hasattr(model, 'estimators_'):
                    # For ensemble models: Use variance across individual tree predictions
                    individual_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                    
                    # Handle multi-target predictions
                    if individual_predictions.ndim == 3:  # (n_trees, n_samples, n_targets)
                        # Calculate variance for each target separately
                        variances = np.var(individual_predictions, axis=0)  # (n_samples, n_targets)
                        # Convert variance to confidence (inverse relationship)
                        confidence_scores = 1.0 / (1.0 + variances)
                        # Take mean confidence across all targets for each sample
                        confidence_scores = np.mean(confidence_scores, axis=1).tolist()
                    else:  # (n_trees, n_samples) - single target
                        variances = np.var(individual_predictions, axis=0)
                        # Convert variance to confidence (inverse relationship)
                        confidence_scores = (1.0 / (1.0 + variances)).tolist()
                    
                    logger.info("Confidence calculated using ensemble tree variance")
                elif hasattr(model, 'get_booster'):
                    # XGBoost: Use leaf value variance across trees
                    try:
                        import xgboost as xgb
                        
                        # Get predictions from individual trees
                        booster = model.get_booster()
                        dmatrix = xgb.DMatrix(X)
                        
                        # Get predictions from each tree iteration
                        n_trees = model.n_estimators
                        tree_predictions = []
                        
                        for i in range(n_trees):
                            # Predict using only trees up to iteration i
                            pred_i = booster.predict(dmatrix, iteration_range=(i, i+1))
                            tree_predictions.append(pred_i)
                        
                        tree_predictions = np.array(tree_predictions)
                        
                        # Calculate variance across tree predictions for each sample
                        if tree_predictions.ndim == 3:  # (n_trees, n_samples, n_targets)
                            variances = np.var(tree_predictions, axis=0)  # (n_samples, n_targets)
                            confidence_scores = 1.0 / (1.0 + np.mean(variances, axis=1))
                        else:  # (n_trees, n_samples) - single target
                            variances = np.var(tree_predictions, axis=0)  # (n_samples,)
                            confidence_scores = 1.0 / (1.0 + variances)
                        
                        confidence_scores = confidence_scores.tolist()
                        logger.info("Confidence calculated using XGBoost tree-by-tree variance")
                        
                    except Exception as xgb_error:
                        logger.warning(f"XGBoost-specific confidence calculation failed: {xgb_error}, using fallback")
                        # Fallback to magnitude-based confidence
                        predictions = model.predict(X)
                        if predictions.ndim == 1:
                            confidence_scores = [1.0 / (1.0 + abs(pred)) for pred in predictions]
                        else:
                            confidence_scores = [1.0 / (1.0 + np.mean(np.abs(pred))) for pred in predictions]
                        logger.info("Confidence calculated using XGBoost prediction magnitude (fallback)")
                        
                else:
                    # Fallback for other models
                    predictions = model.predict(X)
                    # Use inverse of prediction magnitude as a simple confidence measure
                    if predictions.ndim == 1:
                        confidence_scores = [1.0 / (1.0 + abs(pred)) for pred in predictions]
                    else:
                        # Multi-target: average across targets
                        confidence_scores = [1.0 / (1.0 + np.mean(np.abs(pred))) for pred in predictions]
                    
                    logger.info("Confidence calculated using prediction magnitude (general fallback)")
            
            return confidence_scores
            
        except Exception as e:
            logger.warning(f"Failed to compute confidence scores: {str(e)}")
            return None
    
    
    def _create_prediction_folder(self, prediction_id: str, model_id: str, save_intermediate_files: bool) -> Optional[Path]:
        """Create folder structure for prediction results under the model's directory."""
        if not save_intermediate_files:
            return None
            
        try:
            # Create prediction folder under the model's directory
            model_folder = Path("trained_models") / model_id
            if not model_folder.exists():
                logger.error(f"Model folder not found: {model_folder}")
                return None
            
            # Create predictions subdirectory in the model folder
            predictions_base = model_folder / "predictions"
            predictions_base.mkdir(exist_ok=True)
            
            # Create specific prediction folder with unique ID
            prediction_folder = predictions_base / prediction_id
            prediction_folder.mkdir(parents=True, exist_ok=True)
            

            return prediction_folder
            
        except Exception as e:
            logger.error(f"Failed to create prediction folder: {str(e)}")
            return None
    
    def _create_prediction_archive(self, prediction_folder: Path, prediction_id: str) -> Optional[str]:
        """
        Create a ZIP archive of the prediction results folder.
        
        Args:
            prediction_folder: Path to the prediction folder to archive
            prediction_id: Prediction identifier for naming the archive
            
        Returns:
            Path to the created ZIP file, or None if failed
        """
        try:
            # Create archives directory in trained_models if it doesn't exist
            archives_dir = self.model_manager.models_dir / "archives"
            archives_dir.mkdir(exist_ok=True)
            
            # Create ZIP file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"prediction_{prediction_id}_{timestamp}.zip"
            zip_path = archives_dir / zip_filename
            
            # Create ZIP archive
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through all files in the prediction folder
                for root, dirs, files in os.walk(prediction_folder):
                    for file in files:
                        file_path = Path(root) / file
                        # Calculate relative path from prediction folder
                        arcname = file_path.relative_to(prediction_folder)
                        zipf.write(file_path, arcname)
                        
            # Calculate file size
            file_size = zip_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"Prediction archive created: {zip_path} ({file_size_mb:.2f} MB)")
            
            # Update prediction metadata with archive information if metadata file exists
            try:
                metadata_path = prediction_folder / "prediction_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Calculate relative path from project root
                    relative_path = zip_path.relative_to(self.model_manager.models_dir.parent)
                    
                    metadata['archive_info'] = {
                        'archive_path': str(zip_path),
                        'archive_filename': zip_filename,
                        'archive_size_bytes': file_size,
                        'archive_size_mb': round(file_size_mb, 2),
                        'created_at': datetime.now().isoformat(),
                        'download_url': f"/download/file/{relative_path.as_posix()}",
                        'relative_path': str(relative_path.as_posix())
                    }
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                        
                    logger.info(f"Archive information added to prediction metadata")
                    
            except Exception as e:
                logger.warning(f"Could not update prediction metadata with archive info: {e}")
            
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Failed to create prediction archive: {e}")
            return None
    
    def _save_prediction_details(
        self, 
        prediction_folder: Path, 
        df_original: pd.DataFrame,
        X_processed: np.ndarray,
        predictions_processed: np.ndarray,
        predictions_original: np.ndarray,
        confidence_scores: Optional[List[float]],
        processed_feature_names: List[str],
        original_feature_names: List[str],
        model_metadata: Optional[Dict] = None,
        label_mapping: Optional[Dict] = None,
        error_occurred: bool = False,
        error_message: str = ""
    ) -> Dict[str, str]:
        """Save detailed prediction data to separate files."""
        saved_files = {}
        try:
            # Save original data (01_original_data.csv) if not already saved
            if df_original is not None:
                original_data_path = prediction_folder / "01_original_data.csv"
                if not original_data_path.exists():
                    df_original.to_csv(original_data_path, index=False)
                    logger.info(f"Saved original data to: {original_data_path}")
                saved_files['original_data'] = str(original_data_path)
            
            # Save processed features (02_processed_features.csv) if not already saved
            if X_processed is not None and len(processed_feature_names) > 0:
                processed_data_path = prediction_folder / "02_processed_features.csv"
                if not processed_data_path.exists():
                    df_processed = pd.DataFrame(X_processed, columns=processed_feature_names)
                    df_processed.to_csv(processed_data_path, index=False)
                    logger.info(f"Saved processed features to: {processed_data_path}")
                saved_files['processed_features'] = str(processed_data_path)
            
            # Save error information if there was an error
            if error_occurred:
                error_info_path = prediction_folder / "error_info.txt"
                with open(error_info_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error occurred during prediction:\n")
                    f.write(f"Error message: {error_message}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                saved_files['error_info'] = str(error_info_path)
                logger.info(f"Saved error information to: {error_info_path}")
                
                # Return early if error occurred and we can't save predictions
                if predictions_original is None:
                    return saved_files
            # Get target names from model metadata
            target_names = []
            if model_metadata:
                target_names = model_metadata.get('target_name', [])
                if not target_names:
                    target_names = model_metadata.get('target_column', [])
            
            # Handle both single-target and multi-target predictions
            # Use predictions_original if predictions_processed is None
            pred_to_check = predictions_processed if predictions_processed is not None else predictions_original
            
            # Determine if this is a classification task
            is_classification = model_metadata and model_metadata.get('task_type') == 'classification'
            
            if pred_to_check.ndim == 1:
                # Single target
                if predictions_processed is not None:
                    pred_processed_values = predictions_processed.reshape(-1, 1)
                else:
                    pred_processed_values = predictions_original.reshape(-1, 1)
                pred_original_values = predictions_original.reshape(-1, 1)
                
                # For classification, convert numeric predictions to class labels
                if is_classification and label_mapping:
                    class_to_label = label_mapping.get('class_to_label', {})
                    
                    def safe_classification_lookup(pred_value):
                        """Safely lookup classification label, handling both string and numeric predictions."""
                        if isinstance(pred_value, str):
                            return pred_value
                        try:
                            return (class_to_label.get(str(int(pred_value))) or 
                                    class_to_label.get(int(pred_value)) or 
                                    f'Unknown_{int(pred_value)}')
                        except (ValueError, TypeError):
                            return str(pred_value) if pred_value is not None else 'Unknown'
                    
                    # Convert processed scale predictions to class labels
                    pred_processed_labels = [safe_classification_lookup(p) for p in pred_processed_values.ravel()]
                    pred_original_labels = [safe_classification_lookup(p) for p in pred_original_values.ravel()]
                    
                    df_pred_processed = pd.DataFrame(pred_processed_labels, columns=['prediction_processed_scale'])
                    df_pred_original = pd.DataFrame(pred_original_labels, columns=['prediction_original_scale'])
                else:
                    df_pred_processed = pd.DataFrame(pred_processed_values, columns=['prediction_processed_scale'])
                    df_pred_original = pd.DataFrame(pred_original_values, columns=['prediction_original_scale'])
            else:
                # Multi-target
                n_targets = pred_to_check.shape[1]
                if target_names and len(target_names) == n_targets:
                    # Use real target names
                    processed_cols = [f'prediction_processed_scale_{name}' for name in target_names]
                    original_cols = [f'prediction_original_scale_{name}' for name in target_names]
                else:
                    # Fallback to generic names
                    processed_cols = [f'prediction_processed_scale_target_{i+1}' for i in range(n_targets)]
                    original_cols = [f'prediction_original_scale_target_{i+1}' for i in range(n_targets)]
                
                if predictions_processed is not None:
                    pred_processed_values = predictions_processed
                else:
                    pred_processed_values = predictions_original
                pred_original_values = predictions_original
                
                # For classification, convert numeric predictions to class labels
                if is_classification and label_mapping:
                    class_to_label = label_mapping.get('class_to_label', {})
                    
                    def safe_classification_lookup_multi(pred_value):
                        """Safely lookup classification label for multi-target, handling both string and numeric predictions."""
                        if isinstance(pred_value, str):
                            return pred_value
                        try:
                            return (class_to_label.get(str(int(pred_value))) or 
                                    class_to_label.get(int(pred_value)) or 
                                    f'Unknown_{int(pred_value)}')
                        except (ValueError, TypeError):
                            return str(pred_value) if pred_value is not None else 'Unknown'
                    
                    # Convert processed scale predictions
                    pred_processed_labels = []
                    for i in range(pred_processed_values.shape[0]):
                        row_labels = [safe_classification_lookup_multi(pred_processed_values[i, j]) 
                                     for j in range(pred_processed_values.shape[1])]
                        pred_processed_labels.append(row_labels)
                    
                    # Convert original scale predictions
                    pred_original_labels = []
                    for i in range(pred_original_values.shape[0]):
                        row_labels = [safe_classification_lookup_multi(pred_original_values[i, j]) 
                                     for j in range(pred_original_values.shape[1])]
                        pred_original_labels.append(row_labels)
                    
                    df_pred_processed = pd.DataFrame(pred_processed_labels, columns=processed_cols)
                    df_pred_original = pd.DataFrame(pred_original_labels, columns=original_cols)
                else:
                    df_pred_processed = pd.DataFrame(pred_processed_values, columns=processed_cols)
                    df_pred_original = pd.DataFrame(pred_original_values, columns=original_cols)
            
            # Save predictions in processed scale
            pred_processed_path = prediction_folder / "03_predictions_processed_scale.csv"
            df_pred_processed.to_csv(pred_processed_path, index=False)
            saved_files['predictions_processed'] = str(pred_processed_path)
            
            # Save predictions in original scale
            pred_original_path = prediction_folder / "04_predictions_original_scale.csv"
            df_pred_original.to_csv(pred_original_path, index=False)
            saved_files['predictions_original'] = str(pred_original_path)
            
            # Save confidence scores if available
            if confidence_scores:
                # Handle confidence scores for multi-target predictions
                conf_array = np.array(confidence_scores)
                if conf_array.ndim == 1:
                    # Single target
                    df_confidence = pd.DataFrame(conf_array, columns=['confidence_score'])
                else:
                    # Multi-target
                    n_targets = conf_array.shape[1]
                    if target_names and len(target_names) == n_targets:
                        # Use real target names
                        conf_cols = [f'confidence_score_{name}' for name in target_names]
                    else:
                        # Fallback to generic names
                        conf_cols = [f'confidence_score_target_{i+1}' for i in range(n_targets)]
                    df_confidence = pd.DataFrame(conf_array, columns=conf_cols)
                
                confidence_path = prediction_folder / "05_confidence_scores.csv"
                df_confidence.to_csv(confidence_path, index=False)
                saved_files['confidence_scores'] = str(confidence_path)
            
            # Create combined results file
            combined_df = df_original[original_feature_names].copy()
            
            # Add processed features if different from original
            if len(processed_feature_names) != len(original_feature_names) or processed_feature_names != original_feature_names:
                df_processed_features = pd.DataFrame(X_processed, columns=processed_feature_names)
                for col in processed_feature_names:
                    combined_df[f"{col}_processed"] = df_processed_features[col]
            
            # Add predictions (use the same logic as above for classification conversion)
            if pred_to_check.ndim == 1:
                # Single target
                if is_classification and label_mapping:
                    class_to_label = label_mapping.get('class_to_label', {})
                    
                    def safe_combined_lookup(pred_value):
                        """Safely lookup classification label for combined results, handling both string and numeric predictions."""
                        if isinstance(pred_value, str):
                            return pred_value
                        try:
                            return (class_to_label.get(str(int(pred_value))) or 
                                    class_to_label.get(int(pred_value)) or 
                                    f'Unknown_{int(pred_value)}')
                        except (ValueError, TypeError):
                            return str(pred_value) if pred_value is not None else 'Unknown'
                    
                    if predictions_processed is not None:
                        combined_df['prediction_processed_scale'] = [safe_combined_lookup(p) for p in predictions_processed]
                    else:
                        combined_df['prediction_processed_scale'] = [safe_combined_lookup(p) for p in predictions_original]
                    combined_df['prediction_original_scale'] = [safe_combined_lookup(p) for p in predictions_original]
                else:
                    if predictions_processed is not None:
                        combined_df['prediction_processed_scale'] = predictions_processed
                    else:
                        combined_df['prediction_processed_scale'] = predictions_original
                    combined_df['prediction_original_scale'] = predictions_original
            else:
                # Multi-target - add each target as a separate column
                n_targets = pred_to_check.shape[1]
                if target_names and len(target_names) == n_targets:
                    # Use real target names
                    for i, name in enumerate(target_names):
                        if is_classification and label_mapping:
                            class_to_label = label_mapping.get('class_to_label', {})
                            
                            def safe_multi_combined_lookup(pred_value):
                                """Safely lookup classification label for multi-target combined results."""
                                if isinstance(pred_value, str):
                                    return pred_value
                                try:
                                    return (class_to_label.get(str(int(pred_value))) or 
                                            class_to_label.get(int(pred_value)) or 
                                            f'Unknown_{int(pred_value)}')
                                except (ValueError, TypeError):
                                    return str(pred_value) if pred_value is not None else 'Unknown'
                            
                            if predictions_processed is not None:
                                combined_df[f'prediction_processed_scale_{name}'] = [safe_multi_combined_lookup(predictions_processed[j, i]) for j in range(predictions_processed.shape[0])]
                            else:
                                combined_df[f'prediction_processed_scale_{name}'] = [safe_multi_combined_lookup(predictions_original[j, i]) for j in range(predictions_original.shape[0])]
                            combined_df[f'prediction_original_scale_{name}'] = [safe_multi_combined_lookup(predictions_original[j, i]) for j in range(predictions_original.shape[0])]
                        else:
                            if predictions_processed is not None:
                                combined_df[f'prediction_processed_scale_{name}'] = predictions_processed[:, i]
                            else:
                                combined_df[f'prediction_processed_scale_{name}'] = predictions_original[:, i]
                            combined_df[f'prediction_original_scale_{name}'] = predictions_original[:, i]
                else:
                    # Fallback to generic names
                    for i in range(n_targets):
                        if is_classification and label_mapping:
                            class_to_label = label_mapping.get('class_to_label', {})
                            
                            def safe_fallback_combined_lookup(pred_value):
                                """Safely lookup classification label for fallback multi-target combined results."""
                                if isinstance(pred_value, str):
                                    return pred_value
                                try:
                                    return (class_to_label.get(str(int(pred_value))) or 
                                            class_to_label.get(int(pred_value)) or 
                                            f'Unknown_{int(pred_value)}')
                                except (ValueError, TypeError):
                                    return str(pred_value) if pred_value is not None else 'Unknown'
                            
                            if predictions_processed is not None:
                                combined_df[f'prediction_processed_scale_target_{i+1}'] = [safe_fallback_combined_lookup(predictions_processed[j, i]) for j in range(predictions_processed.shape[0])]
                            else:
                                combined_df[f'prediction_processed_scale_target_{i+1}'] = [safe_fallback_combined_lookup(predictions_original[j, i]) for j in range(predictions_original.shape[0])]
                            combined_df[f'prediction_original_scale_target_{i+1}'] = [safe_fallback_combined_lookup(predictions_original[j, i]) for j in range(predictions_original.shape[0])]
                        else:
                            if predictions_processed is not None:
                                combined_df[f'prediction_processed_scale_target_{i+1}'] = predictions_processed[:, i]
                            else:
                                combined_df[f'prediction_processed_scale_target_{i+1}'] = predictions_original[:, i]
                            combined_df[f'prediction_original_scale_target_{i+1}'] = predictions_original[:, i]
            
            # Add confidence scores
            if confidence_scores:
                conf_array = np.array(confidence_scores)
                if conf_array.ndim == 1:
                    # Single target
                    combined_df['confidence_score'] = conf_array
                else:
                    # Multi-target
                    n_conf_targets = conf_array.shape[1]
                    if target_names and len(target_names) == n_conf_targets:
                        # Use real target names
                        for i, name in enumerate(target_names):
                            combined_df[f'confidence_score_{name}'] = conf_array[:, i]
                    else:
                        # Fallback to generic names
                        for i in range(n_conf_targets):
                            combined_df[f'confidence_score_target_{i+1}'] = conf_array[:, i]
            
            combined_path = prediction_folder / "06_combined_results.csv"
            combined_df.to_csv(combined_path, index=False)
            saved_files['combined_results'] = str(combined_path)
            
            logger.info(f"Saved detailed prediction files to: {prediction_folder}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save prediction details: {str(e)}")
            return {}

    def _save_prediction_form_values_results_metadata(self, metadata: Dict[str, Any], prediction_folder: Path):
            # Prepare metadata
        try:

            metadata['folder_structure'] = {
                '01_original_data.csv': 'Original input data as provided',
                '02_processed_features.csv': 'Features after preprocessing/normalization',
                '03_predictions_processed_scale.csv': 'Predictions in processed/normalized scale',
                '04_predictions_original_scale.csv': 'Predictions transformed back to original scale',
                '05_confidence_scores.csv': 'Confidence/uncertainty scores for each prediction',
                '06_combined_results.csv': 'All data combined: features + predictions + confidence',
                'prediction_summary.json': 'Complete prediction results and metadata',
                'prediction_report.html': 'Detailed HTML report'
            }

            metadata_path = prediction_folder / "prediction_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Prediction metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save prediction metadata: {str(e)}")


    def _save_predictions(self, results: Dict[str, Any], output_path: str):
        """Save prediction results to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Also save to prediction folder if available
            if results.get('prediction_folder'):
                prediction_folder = Path(results['prediction_folder'])
                summary_path = prediction_folder / "prediction_summary.json"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Prediction summary saved to: {summary_path}")
            
            logger.info(f"Prediction results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {str(e)}")
            raise
    
    def _generate_prediction_report(self, results: Dict[str, Any], input_data: pd.DataFrame, prediction_folder: Optional[Path] = None,model_info:Dict[str,Any]=None) -> str:
        """Generate detailed prediction report in both HTML and Markdown formats using specialized generators."""
        try:
            html_report_path = ""
            md_report_path = ""
            
            if prediction_folder:
                # Use specialized generators for file-based approach
                from .html_report_generator import HTMLReportGenerator
                from .academic_report_generator import AcademicReportGenerator
                
                # Generate HTML report using HTMLReportGenerator
                html_generator = HTMLReportGenerator()
                html_report_path = html_generator.generate_prediction_report_from_folder(prediction_folder,model_info)
                
                # Generate Markdown report using AcademicReportGenerator
                md_generator = AcademicReportGenerator()
                md_report_path = md_generator.generate_prediction_experiment_report_from_folder(prediction_folder,model_info)
                
                logger.info(f"Prediction reports generated via specialized generators: {html_report_path}, {md_report_path}")
                return html_report_path if html_report_path else ""
            else:
                # Fallback for cases without prediction folder
                report_dir = Path("prediction_reports")
                report_dir.mkdir(exist_ok=True)
                html_report_path = report_dir / f"prediction_report_{results['prediction_id']}.html"
                md_report_path = report_dir / f"prediction_experiment_report_{results['prediction_id']}.md"
                logger.warning("Generating reports without prediction folder - specialized generators require saved files")
                return str(html_report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate prediction report: {str(e)}")
            return ""
    
#     def _create_html_report_legacy(self, results: Dict[str, Any], input_data: pd.DataFrame, prediction_folder: Optional[Path] = None) -> str:
#         """Create enhanced HTML content for prediction report with features, predictions, and uncertainty."""
        
#         # Get metadata
#         metadata = results.get('prediction_metadata', {})
#         num_samples = metadata.get('num_samples', len(input_data))
#         if 'num_predictions' in results:
#             num_samples = results['num_predictions']
#         elif 'predictions' in results and results['predictions'] is not None:
#             if isinstance(results['predictions'], (list, np.ndarray)):
#                 num_samples = len(results['predictions'])
#             else:
#                 num_samples = 1  # Single prediction
#         feature_names = metadata.get('feature_names', input_data.columns.tolist())
        
#         html = f"""<!DOCTYPE html>
# <html>
# <head>
#     <title>Prediction Report - {results.get('model_id', 'Unknown')}</title>
#     <style>
#         body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
#         .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
#         .section {{ margin: 25px 0; background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
#         .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e3f2fd; border-radius: 6px; min-width: 150px; text-align: center; }}
#         .metric strong {{ display: block; font-size: 1.2em; color: #1976d2; }}
#         table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
#         th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
#         th {{ background-color: #f2f2f2; font-weight: bold; color: #333; }}
#         tr:nth-child(even) {{ background-color: #f9f9f9; }}
#         tr:hover {{ background-color: #f5f5f5; }}
#         .feature-value {{ font-family: monospace; background-color: #f8f9fa; padding: 2px 6px; border-radius: 3px; }}
#         .prediction-value {{ font-weight: bold; color: #28a745; }}
#         .confidence-value {{ color: #dc3545; }}
#         .high-confidence {{ color: #28a745; }}
#         .medium-confidence {{ color: #ffc107; }}
#         .low-confidence {{ color: #dc3545; }}
#         h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
#         h3 {{ color: #555; }}
#     </style>
# </head>
# <body>
#     <div class="header">
#         <h1>🔮 ML Prediction Report</h1>
#         <p><strong>Model ID:</strong> {results.get('model_id', 'Unknown')}</p>
#         <p><strong>Task Type:</strong> {results.get('task_type', 'Unknown').title()}</p>
#         <p><strong>Prediction Time:</strong> {results.get('prediction_metadata', {}).get('prediction_timestamp', 'Unknown')}</p>
#     </div>
    
#     <div class="section">
#         <h2>📊 Summary</h2>
#         <div class="metric">
#             <strong>{num_samples}</strong>
#             <span>Total Predictions</span>
#         </div>
#         <div class="metric">
#             <strong>{results.get('prediction_metadata', {}).get('prediction_time_seconds', 0.0):.3f}s</strong>
#             <span>Processing Time</span>
#         </div>
#         <div class="metric">
#             <strong>{len(feature_names)}</strong>
#             <span>Features Used</span>
#         </div>
#         <div class="metric">
#             <strong>{'Yes' if metadata.get('preprocessing_applied') else 'No'}</strong>
#             <span>Preprocessing Applied</span>
#         </div>
#     </div>"""
        
#         # Add feature information section
#         html += f"""
#     <div class="section">
#         <h2>🔧 Feature Information</h2>
#         <p><strong>Total Features:</strong> {len(feature_names)}</p>
#         <table>
#             <tr><th>Feature Name</th><th>Sample Values</th></tr>"""
        
#         # Show feature values from input data
#         for feature_name in feature_names:
#             if feature_name in input_data.columns:
#                 sample_values = input_data[feature_name].head(3).tolist()
#                 # Handle both numeric and categorical features
#                 formatted_values = []
#                 for val in sample_values:
#                     if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
#                         formatted_values.append(f"<span class='feature-value'>{val:.4f}</span>")
#                     else:
#                         formatted_values.append(f"<span class='feature-value'>{val}</span>")
#                 sample_str = ', '.join(formatted_values)
#                 if len(input_data) > 3:
#                     sample_str += ", ..."
#                 html += f"<tr><td><strong>{feature_name}</strong></td><td>{sample_str}</td></tr>"
        
#         html += "</table></div>"

#         # Add detailed predictions with features section
#         html += f"""
#     <div class="section">
#         <h2>🎯 Detailed Predictions & Features</h2>
#         <table>
#             <tr><th>Sample #</th>"""
        
#         # Add feature columns
#         for feature_name in feature_names[:5]:  # Show first 5 features to avoid too wide table
#             html += f"<th>{feature_name}</th>"
#         if len(feature_names) > 5:
#             html += "<th>... More Features</th>"
        
#         html += "<th>Prediction</th>"
        
#         if results.get('confidence_scores'):
#             html += "<th>Uncertainty/Confidence</th>"
        
#         html += "</tr>"
        
#         # Show detailed predictions with feature values
#         # Use raw predictions for label mapping in HTML
#         raw_predictions = results.get('raw_predictions', results.get('predictions', []))
#         predictions_list = raw_predictions if isinstance(raw_predictions, (list, np.ndarray)) else [raw_predictions]
        
#         max_samples_to_show = min(10, len(predictions_list))
#         for i in range(max_samples_to_show):
#             pred = predictions_list[i]
#             html += f"<tr><td><strong>#{i+1}</strong></td>"
            
#             # Add feature values
#             for j, feature_name in enumerate(feature_names[:5]):
#                 if feature_name in input_data.columns and i < len(input_data):
#                     feature_val = input_data.iloc[i][feature_name]
#                     # Handle both numeric and categorical features
#                     if isinstance(feature_val, (int, float, np.number)):
#                         html += f"<td><span class='feature-value'>{feature_val:.4f}</span></td>"
#                     else:
#                         html += f"<td><span class='feature-value'>{feature_val}</span></td>"
#                 else:
#                     html += "<td>-</td>"
            
#             if len(feature_names) > 5:
#                 html += "<td>...</td>"
            
#             # Format prediction value properly - handle classification labels
#             if results.get('task_type') == 'classification' and results.get('label_mapping'):
#                 class_to_label = results.get('label_mapping', {}).get('class_to_label', {})
#                 if isinstance(pred, (list, np.ndarray)):
#                     # Multi-target classification
#                     pred_labels = [class_to_label.get(int(p), f'Class_{int(p)}') for p in pred]
#                     pred_str = ', '.join(pred_labels)
#                 else:
#                     # Single classification
#                     pred_str = class_to_label.get(int(pred), f'Class_{int(pred)}')
#             else:
#                 # Regression or no label mapping
#                 if isinstance(pred, (list, np.ndarray)):
#                     pred_str = ', '.join([f"{float(p):.4f}" for p in pred])
#                 else:
#                     pred_str = f"{float(pred):.4f}"
#             html += f"<td><span class='prediction-value'>{pred_str}</span></td>"
            
#             # Add confidence/uncertainty
#             if results.get('confidence_scores'):
#                 conf_score = results.get('confidence_scores', [])[i] if i < len(results.get('confidence_scores', [])) else 0
#                 if isinstance(conf_score, (list, np.ndarray)):
#                     conf_str = ', '.join([f"{float(c):.4f}" for c in conf_score])
#                     conf_val = float(conf_score[0]) if len(conf_score) > 0 else 0.0
#                 else:
#                     conf_str = f"{float(conf_score):.4f}"
#                     conf_val = float(conf_score)
                
#                 # Add color coding based on confidence level
#                 conf_class = "high-confidence" if conf_val > 0.8 else "medium-confidence" if conf_val > 0.5 else "low-confidence"
#                 html += f"<td><span class='confidence-value {conf_class}'>{conf_str}</span></td>"
            
#             html += "</tr>"
        
#         html += "</table></div>"

#         # Add feature importance section if available
#         if results.get('feature_importance'):
#             html += """
#     <div class="section">
#         <h2>📈 Feature Importance</h2>
#         <p>Features ranked by their importance in making predictions:</p>
#         <table>
#             <tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Relative Importance</th></tr>"""
            
#             sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
#             max_importance = max(results['feature_importance'].values()) if results['feature_importance'] else 1.0
            
#             for rank, (feature, importance) in enumerate(sorted_features, 1):
#                 relative_importance = (importance / max_importance) * 100
#                 html += f"""<tr>
#                     <td><strong>#{rank}</strong></td>
#                     <td>{feature}</td>
#                     <td>{importance:.6f}</td>
#                     <td>{relative_importance:.1f}%</td>
#                 </tr>"""
#             html += "</table></div>"

#         # Add saved files section if available
#         if results.get('saved_files') and prediction_folder:
#             html += """
#     <div class="section">
#         <h2>💾 Saved Files</h2>
#         <table>
#             <tr><th>File</th><th>Description</th><th>Path</th></tr>"""
#             file_descriptions = {
#                 'predictions_processed': 'Predictions in processed/normalized scale',
#                 'predictions_original': 'Predictions in original scale',
#                 'confidence_scores': 'Confidence/uncertainty scores',
#                 'combined_results': 'Complete combined dataset with features and predictions',
#                 'original_data': 'Original input features',
#                 'processed_features': 'Preprocessed features'
#             }
#             for file_key, file_path in results['saved_files'].items():
#                 file_name = Path(file_path).name
#                 description = file_descriptions.get(file_key, 'Additional prediction data')
#                 html += f"<tr><td><strong>{file_name}</strong></td><td>{description}</td><td><small>{file_path}</small></td></tr>"
#             html += "</table></div>"
        
#         html += """
#     <div class="section">
#         <h2>ℹ️ Notes</h2>
#         <ul>
#             <li><strong>Predictions:</strong> Model output values for each input sample</li>
#             <li><strong>Uncertainty/Confidence:</strong> Measure of prediction reliability (higher is more confident)</li>
#             <li><strong>Feature Importance:</strong> How much each feature contributes to the model's decisions</li>
#             <li><strong>Processing Time:</strong> Total time taken for data processing and prediction</li>
#         </ul>
#     </div>
# </body>
# </html>"""
        
#         return html
    
#     def _create_markdown_experiment_report_legacy(self, results: Dict[str, Any], input_data: pd.DataFrame, prediction_folder: Optional[Path] = None) -> str:
#         """Create detailed Markdown experiment report."""
#         from datetime import datetime
        
#         # Extract key information
#         model_id = results.get('model_id', 'Unknown')
#         prediction_id = results.get('prediction_id', 'unknown_prediction')
#         task_type = results.get('task_type', 'unknown')
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
#         # Get prediction metadata
#         metadata = results.get('prediction_metadata', {})
#         num_predictions = metadata.get('num_samples', len(input_data))
#         data_shape = metadata.get('data_shape', 'N/A')
#         processing_time = metadata.get('prediction_time_seconds', 0)
#         preprocessing_applied = metadata.get('preprocessing_applied', False)
        
#         # Get feature and target information
#         feature_names = input_data.columns.tolist()
#         num_features = len(feature_names)
        
#         # Get real target names from model metadata
#         model_metadata = results.get('model_metadata', {})
#         if not model_metadata:
#             # Try to get metadata from model manager
#             try:
#                 model_info = self.model_manager.get_model_info(model_id)
#                 model_metadata = model_info
#             except:
#                 model_metadata = {}
        
#         real_target_names = model_metadata.get('target_name', [])
#         if not real_target_names:
#             real_target_names = model_metadata.get('target_column', [])
        
#         # Determine target information from predictions
#         predictions = results.get('predictions', [])
        
#         # Handle both single predictions (float/int) and batch predictions (list)
#         if isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
#             # Batch predictions
#             sample_pred = predictions[0]
#             if isinstance(sample_pred, (list, np.ndarray)):
#                 num_targets = len(sample_pred)
#                 # Use real target names if available, otherwise fallback to generic names
#                 if real_target_names and len(real_target_names) == num_targets:
#                     target_names = real_target_names
#                 else:
#                     target_names = [f"target_{i+1}" for i in range(num_targets)]
#             else:
#                 num_targets = 1
#                 target_names = real_target_names[:1] if real_target_names else ["target"]
#         elif isinstance(predictions, (int, float, np.number)):
#             # Single prediction
#             num_targets = 1
#             target_names = real_target_names[:1] if real_target_names else ["target"]
#             # Convert single prediction to list for consistency in downstream processing
#             predictions = [predictions]
#         else:
#             # Default case
#             num_targets = 1
#             target_names = real_target_names[:1] if real_target_names else ["target"]
#             if not isinstance(predictions, (list, np.ndarray)):
#                 predictions = [predictions] if predictions is not None else []
        
#         # Calculate prediction statistics
#         pred_stats = self._calculate_prediction_statistics_legacy(predictions, task_type, target_names)
        
#         # Start building the markdown report
#         md_content = f"""# Prediction Experiment Report

# **Generated on:** {timestamp}  
# **Experiment Name:** {prediction_id}  
# **Model ID:** `{model_id}`  
# **Output Directory:** `trained_models/{model_id}/predictions/{prediction_id}`

# ## Executive Summary

# This report documents a comprehensive machine learning prediction experiment conducted using a pre-trained {task_type} model. The experiment involved preprocessing input data, making predictions, and providing detailed statistical analysis of the prediction results.

# ### Key Results
# - **Number of Predictions:** {num_predictions:,}
# - **Feature Count:** {num_features}
# - **Target Count:** {num_targets}
# - **Model Type:** {task_type.title()}
# - **Processing Time:** {processing_time:.3f} seconds

# ---

# ## 1. Experiment Setup

# ### 1.1 Input Data Information

# | Parameter | Value |
# |-----------|-------|
# | Number of Samples | {num_predictions:,} |
# | Number of Features | {num_features} |
# | Number of Targets | {num_targets} |
# | Data Shape | {data_shape} |
# | Data Type | Numerical (floating-point) |
# | Preprocessing Applied | {'Yes' if preprocessing_applied else 'No'} |

# ### 1.2 Feature Information

# **Input Features ({num_features} columns):**
#         {self._format_feature_list_legacy(feature_names)}

# **Target Variables ({num_targets} column{'s' if num_targets > 1 else ''}):**
#         {self._format_feature_list_legacy(target_names)}

# ### 1.3 Model Information

# | Component | Details |
# |-----------|---------|
# | **Model Type** | {task_type.title()} Model |
# | **Model ID** | `{model_id}` |
# | **Framework** | Random Forest (scikit-learn) |
# | **Prediction Method** | {'Ensemble averaging' if task_type == 'regression' else 'Majority voting'} |

# ---

# ## 2. Prediction Results

# ### 2.1 Prediction Statistics

#         {self._format_prediction_statistics_md_legacy(pred_stats, task_type, target_names)}

# ---

# ## 3. Generated Files

# | File | Description |
# |------|-------------|"""
        
#         # Add information about saved files
#         if results.get('saved_files') and prediction_folder:
#             file_descriptions = {
#                 'original_data': 'Original input data as provided',
#                 'processed_features': 'Features after preprocessing/normalization',
#                 'predictions_processed': 'Predictions in processed/normalized scale',
#                 'predictions_original': 'Predictions transformed back to original scale',
#                 'confidence_scores': 'Confidence/uncertainty scores for each prediction',
#                 'combined_results': 'All data combined: features + predictions + confidence',
#                 'metadata': 'Complete prediction metadata and configuration',
#                 'html_report': 'Interactive HTML report with detailed analysis'
#             }
            
#             # List the files that were actually saved
#             saved_files = results.get('saved_files', {})
#             for file_key, file_path in saved_files.items():
#                 file_name = Path(file_path).name
#                 description = file_descriptions.get(file_key, 'Prediction data file')
#                 md_content += f"\n| `{file_name}` | {description} |"
            
#             # Add standard files
#             md_content += f"\n| `prediction_experiment_report.md` | This detailed experiment report |"
#             md_content += f"\n| `prediction_report.html` | Interactive HTML report |"
#         else:
#             md_content += f"\n| `prediction_results_{prediction_id}.csv` | Complete prediction results |"
#             md_content += f"\n| `prediction_experiment_report.md` | This detailed experiment report |"
        
#         # Add feature importance section if available
#         if results.get('feature_importance'):
#             md_content += "\n\n---\n\n## 4. Feature Importance Analysis\n\n"
#             md_content += "| Feature | Importance | Relative Contribution |\n"
#             md_content += "|---------|------------|----------------------|\n"
            
#             feature_importance = results['feature_importance']
#             total_importance = sum(feature_importance.values()) if feature_importance.values() else 1
            
#             for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
#                 relative_pct = (importance / total_importance) * 100 if total_importance > 0 else 0
#                 md_content += f"| {feature} | {importance:.4f} | {relative_pct:.1f}% |\n"
        
#         # Add detailed prediction results section
#         md_content += "\n\n---\n\n## 5. Detailed Prediction Results\n\n"
#         md_content += "This section provides a comprehensive view of each prediction with corresponding input features and confidence scores.\n\n"
        
#         # Create a simple prediction table - use raw predictions for label mapping
#         raw_predictions = results.get('raw_predictions', predictions)
#         predictions_list = raw_predictions if isinstance(raw_predictions, (list, np.ndarray)) else [raw_predictions]
#         confidence_scores = results.get('confidence_scores', [])
        
#         # Show up to 10 samples with key features
#         max_samples_to_show = min(10, len(predictions_list))
#         feature_cols = feature_names[:4]  # Show first 4 features for readability
        
#         md_content += "### Feature Values and Predictions\n\n"
        
#         # Create table header
#         md_content += "| Sample | "
#         for fname in feature_cols:
#             md_content += f"{fname} | "
#         if len(feature_names) > 4:
#             md_content += "... | "
#         prediction_label = "Prediction" + ("s" if num_targets > 1 else "")
#         md_content += f"{prediction_label} | Confidence |\n"
        
#         # Create separator
#         md_content += "|--------|"
#         for _ in feature_cols:
#             md_content += "--------|"
#         if len(feature_names) > 4:
#             md_content += "-----|"
#         md_content += "-----------|----------|\n"
        
#         # Add data rows
#         for i in range(max_samples_to_show):
#             pred = predictions_list[i] if i < len(predictions_list) else 0
#             conf = confidence_scores[i] if i < len(confidence_scores) else 0.0
            
#             md_content += f"| **#{i+1}** |"
            
#             # Add feature values
#             for fname in feature_cols:
#                 if fname in input_data.columns and i < len(input_data):
#                     feature_val = input_data.iloc[i][fname]
#                     if isinstance(feature_val, (int, float, np.number)) and not isinstance(feature_val, bool):
#                         md_content += f" {feature_val:.3f} |"
#                     else:
#                         md_content += f" {feature_val} |"
#                 else:
#                     md_content += " - |"
            
#             if len(feature_names) > 4:
#                 md_content += " ... |"
            
#             # Add prediction - handle classification labels
#             if task_type == 'classification' and results.get('label_mapping'):
#                 class_to_label = results['label_mapping'].get('class_to_label', {})
#                 if isinstance(pred, (list, np.ndarray)):
#                     # Multi-target classification
#                     pred_labels = [class_to_label.get(int(p), f'Class_{int(p)}') for p in pred]
#                     pred_str = ", ".join(pred_labels)
#                 else:
#                     # Single classification
#                     pred_str = class_to_label.get(int(pred), f'Class_{int(pred)}')
#             else:
#                 # Regression or no label mapping
#                 if isinstance(pred, (list, np.ndarray)):
#                     # Safely format each prediction - handle both numeric and string values
#                     formatted_preds = []
#                     for p in pred:
#                         try:
#                             formatted_preds.append(f"{float(p):.4f}")
#                         except (ValueError, TypeError):
#                             formatted_preds.append(str(p))
#                     pred_str = ", ".join(formatted_preds)
#                 else:
#                     # Safely format prediction - handle both numeric and string values
#                     try:
#                         pred_str = f"{float(pred):.4f}"
#                     except (ValueError, TypeError):
#                         pred_str = str(pred)
#             md_content += f" {pred_str} |"
            
#             # Add confidence
#             if isinstance(conf, (list, np.ndarray)):
#                 conf_str = f"{conf[0]:.3f}" if len(conf) > 0 else "N/A"
#             else:
#                 conf_str = f"{conf:.3f}"
#             md_content += f" {conf_str} |\n"
        
#         if len(predictions_list) > max_samples_to_show:
#             md_content += f"\n*Showing {max_samples_to_show} of {len(predictions_list)} total predictions.*\n"
        
#         md_content += "\n### Interpretation Guide\n\n"
#         if task_type == 'regression':
#             md_content += "- **Prediction Values**: Continuous numerical predictions representing the target variable\n"
#         else:
#             md_content += "- **Prediction Values**: Class labels or probabilities for classification\n"
        
#         md_content += "- **Confidence Scores**: Range from 0.0 (low confidence) to 1.0 (high confidence)\n"
#         md_content += "  - **≥0.8**: High confidence - Very reliable predictions\n"
#         md_content += "  - **0.5-0.8**: Medium confidence - Moderately reliable\n"
#         md_content += "  - **<0.5**: Low confidence - Review recommended\n"
        
#         # Add confidence analysis if available
#         if results.get('confidence_scores'):
#             md_content += "\n\n---\n\n## 6. Confidence Analysis\n\n"
            
#             # Add confidence calculation methodology
#             md_content += "### 5.1 Confidence Calculation Method\n\n"
            
#             task_type = results.get('task_type', 'unknown')
#             if task_type == 'classification':
#                 md_content += "**Classification Confidence Calculation:**\n"
#                 md_content += "- **Method**: Maximum class probability from model predictions\n"
#                 md_content += "- **Formula**: `confidence = max(class_probabilities)`\n"
#                 md_content += "- **Range**: 0-1, where values closer to 1 indicate high certainty in the predicted class\n"
#                 md_content += "- **Interpretation**: Higher values mean the model is more confident about the predicted class\n\n"
#             else:
#                 md_content += "**Regression Confidence Calculation:**\n"
#                 md_content += "- **Primary Method**: Prediction variance across individual trees in Random Forest ensemble\n"
#                 md_content += "- **Formula**: `confidence = 1 / (1 + variance_across_trees)`\n"
#                 md_content += "- **Range**: 0-1, where higher values indicate more confident predictions\n"
#                 md_content += "- **Multi-target Handling**: For models with multiple targets, confidence is averaged across all targets\n"
#                 md_content += "- **Fallback Method**: For non-ensemble models, uses inverse relationship with prediction magnitude\n\n"
                
#                 md_content += "**Technical Details:**\n"
#                 md_content += "- Each tree in the Random Forest makes an independent prediction\n"
#                 md_content += "- Variance across these individual predictions indicates uncertainty\n"
#                 md_content += "- Low variance (trees agree) → High confidence\n"
#                 md_content += "- High variance (trees disagree) → Low confidence\n\n"
            
#             md_content += "### 5.2 Confidence Statistics\n\n"
#             confidence_stats = self._analyze_confidence_scores_legacy(results['confidence_scores'])
#             md_content += f"**Mean Confidence:** {confidence_stats['mean']:.3f}  \n"
#             md_content += f"**Standard Deviation:** {confidence_stats['std']:.3f}  \n"
#             md_content += f"**Min Confidence:** {confidence_stats['min']:.3f}  \n"
#             md_content += f"**Max Confidence:** {confidence_stats['max']:.3f}  \n\n"
            
#             md_content += "### 5.3 Confidence Distribution\n\n"
#             md_content += "| Confidence Level | Count | Percentage | Description |\n"
#             md_content += "|------------------|-------|------------|-------------|\n"
#             md_content += f"| High (≥0.8) | {confidence_stats['high_count']} | {confidence_stats['high_pct']:.1f}% | Very reliable predictions |\n"
#             md_content += f"| Medium (0.5-0.8) | {confidence_stats['medium_count']} | {confidence_stats['medium_pct']:.1f}% | Moderately reliable predictions |\n"
#             md_content += f"| Low (<0.5) | {confidence_stats['low_count']} | {confidence_stats['low_pct']:.1f}% | Uncertain predictions - review recommended |\n\n"
            
#             md_content += "### 5.4 Confidence Interpretation Guide\n\n"
#             if task_type == 'classification':
#                 md_content += "**For Classification Models:**\n"
#                 md_content += "- **High Confidence (≥0.8)**: Strong certainty in predicted class\n"
#                 md_content += "- **Medium Confidence (0.5-0.8)**: Moderate certainty, acceptable for most applications\n"
#                 md_content += "- **Low Confidence (<0.5)**: Uncertain prediction, consider:\n"
#                 md_content += "  - Reviewing input features for anomalies\n"
#                 md_content += "  - Gathering more training data for similar cases\n"
#                 md_content += "  - Using ensemble of multiple models\n\n"
#             else:
#                 md_content += "**For Regression Models:**\n"
#                 md_content += "- **High Confidence (≥0.8)**: Trees in ensemble show strong agreement\n"
#                 md_content += "- **Medium Confidence (0.5-0.8)**: Moderate agreement, reliable for most applications\n"
#                 md_content += "- **Low Confidence (<0.5)**: High prediction variance, may indicate:\n"
#                 md_content += "  - Input data outside training distribution\n"
#                 md_content += "  - Insufficient training data for similar cases\n"
#                 md_content += "  - High inherent noise in target variable\n"
#                 md_content += "  - Model complexity mismatch with data complexity\n\n"
        
#         # Add footer
#         md_content += f"\n\n---\n\n*Report generated on {timestamp}*\n"
#         md_content += f"*MCP Random Forest Tool - Prediction Experiment Report v1.0.0*\n"
        
#         # Save markdown report to file if prediction folder is provided
#         if prediction_folder:
#             try:
#                 report_path = prediction_folder / "prediction_experiment_report.md"
#                 with open(report_path, 'w', encoding='utf-8') as f:
#                     f.write(md_content)
#                 logger.info(f"Markdown experiment report saved to: {report_path}")
#                 return str(report_path)
#             except Exception as e:
#                 logger.error(f"Failed to save markdown report: {e}")
#                 return md_content
#         else:
#             return md_content
    
    def _format_feature_list_legacy(self, features: List[str]) -> str:
        """Format feature list for markdown."""
        if len(features) <= 10:
            return "`" + "`, `".join(features) + "`"
        else:
            first_five = features[:5]
            last_two = features[-2:]
            return "`" + "`, `".join(first_five) + "`, ..., `" + "`, `".join(last_two) + "`"
    
    def _calculate_prediction_statistics_legacy(self, predictions: List, task_type: str, target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate comprehensive prediction statistics."""
        if not predictions:
            return {}
        
        predictions_array = np.array(predictions)
        
        if task_type == 'regression':
            if predictions_array.ndim == 1:
                # Single target regression
                return {
                    'mean': float(np.mean(predictions_array)),
                    'std': float(np.std(predictions_array)),
                    'min': float(np.min(predictions_array)),
                    'max': float(np.max(predictions_array)),
                    'range': float(np.max(predictions_array) - np.min(predictions_array)),
                    'median': float(np.median(predictions_array)),
                    'q25': float(np.percentile(predictions_array, 25)),
                    'q75': float(np.percentile(predictions_array, 75))
                }
            else:
                # Multi-target regression
                stats = {}
                n_targets = predictions_array.shape[1]
                for i in range(n_targets):
                    target_preds = predictions_array[:, i]
                    # Use real target name if available, otherwise fallback to generic name
                    if target_names and len(target_names) == n_targets:
                        target_key = target_names[i]
                    else:
                        target_key = f'target_{i+1}'
                    
                    stats[target_key] = {
                        'mean': float(np.mean(target_preds)),
                        'std': float(np.std(target_preds)),
                        'min': float(np.min(target_preds)),
                        'max': float(np.max(target_preds)),
                        'range': float(np.max(target_preds) - np.min(target_preds)),
                        'median': float(np.median(target_preds)),
                        'q25': float(np.percentile(target_preds, 25)),
                        'q75': float(np.percentile(target_preds, 75))
                    }
                return stats
        else:
            # Classification statistics
            from collections import Counter
            class_counts = Counter(predictions)
            total = len(predictions)
            return {
                'class_distribution': dict(class_counts),
                'unique_classes': len(class_counts),
                'most_common_class': class_counts.most_common(1)[0] if class_counts else None,
                'total_predictions': total
            }
    
    def _format_prediction_statistics_md_legacy(self, stats: Dict[str, Any], task_type: str, target_names: List[str]) -> str:
        """Format prediction statistics for markdown."""
        if not stats:
            return "No prediction statistics available."
        
        if task_type == 'regression':
            if len(target_names) == 1:
                # Single target regression
                return f"""
#### Single Target Prediction Statistics

**Target: {target_names[0]}**

| Statistic | Value |
|-----------|-------|
| Mean Prediction | {stats['mean']:.6f} |
| Standard Deviation | {stats['std']:.6f} |
| Minimum Prediction | {stats['min']:.6f} |
| Maximum Prediction | {stats['max']:.6f} |
| Prediction Range | {stats['range']:.6f} |
| Median | {stats['median']:.6f} |
| 25th Percentile | {stats['q25']:.6f} |
| 75th Percentile | {stats['q75']:.6f} |
"""
            else:
                # Multi-target regression
                md = "\n#### Multi-Target Prediction Statistics\n\n"
                for target_name in target_names:
                    if target_name in stats:
                        target_stats = stats[target_name]
                        md += f"**Target: {target_name}**\n\n"
                        md += "| Statistic | Value |\n"
                        md += "|-----------|-------|\n"
                        md += f"| Mean Prediction | {target_stats['mean']:.6f} |\n"
                        md += f"| Standard Deviation | {target_stats['std']:.6f} |\n"
                        md += f"| Minimum Prediction | {target_stats['min']:.6f} |\n"
                        md += f"| Maximum Prediction | {target_stats['max']:.6f} |\n"
                        md += f"| Prediction Range | {target_stats['range']:.6f} |\n"
                        md += f"| Median | {target_stats['median']:.6f} |\n\n"
                return md
        else:
            # Classification statistics
            md = "\n#### Classification Prediction Statistics\n\n"
            md += "| Statistic | Value |\n"
            md += "|-----------|-------|\n"
            md += f"| Total Predictions | {stats['total_predictions']:,} |\n"
            md += f"| Unique Classes | {stats['unique_classes']} |\n"
            
            if stats['most_common_class']:
                most_common = stats['most_common_class']
                md += f"| Most Common Class | {most_common[0]} ({most_common[1]} predictions) |\n"
            
            md += "\n**Class Distribution:**\n\n"
            md += "| Class | Count | Percentage |\n"
            md += "|-------|-------|------------|\n"
            
            total = stats['total_predictions']
            for class_label, count in sorted(stats['class_distribution'].items()):
                percentage = (count / total) * 100 if total > 0 else 0
                md += f"| {class_label} | {count} | {percentage:.1f}% |\n"
            
            return md
    
    def _analyze_confidence_scores_legacy(self, confidence_scores: List) -> Dict[str, Any]:
        """Analyze confidence scores and return statistics."""
        if not confidence_scores:
            return {}
        
        # Handle both single values and arrays
        if isinstance(confidence_scores[0], (list, np.ndarray)):
            # Multi-target confidence scores - take mean across targets
            flat_scores = [np.mean(score) for score in confidence_scores]
        else:
            flat_scores = confidence_scores
        
        scores_array = np.array(flat_scores)
        total_count = len(scores_array)
        
        high_count = np.sum(scores_array >= 0.8)
        medium_count = np.sum((scores_array >= 0.5) & (scores_array < 0.8))
        low_count = np.sum(scores_array < 0.5)
        
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'high_count': int(high_count),
            'medium_count': int(medium_count),
            'low_count': int(low_count),
            'high_pct': (high_count / total_count) * 100 if total_count > 0 else 0,
            'medium_pct': (medium_count / total_count) * 100 if total_count > 0 else 0,
            'low_pct': (low_count / total_count) * 100 if total_count > 0 else 0
        }
    
    def _format_single_prediction(self, prediction: np.ndarray, task_type: str) -> Union[float, int, List[float], List[int]]:
        """Format single prediction result based on task type and prediction shape."""
        try:
            if task_type == 'classification':
                # Classification: return integer class
                if prediction.ndim == 0:  # scalar
                    return int(prediction)
                elif prediction.shape == (1,):  # single element array
                    return int(prediction[0])
                else:  # multi-output classification (shouldn't happen with single prediction)
                    return [int(p) for p in prediction]
            else:
                # Regression: handle both single and multi-target
                if prediction.ndim == 0:  # scalar
                    return float(prediction)
                elif prediction.shape == (1,):  # single element array
                    return float(prediction[0])
                else:  # multi-target regression
                    return [float(p) for p in prediction]
        except Exception as e:
            logger.warning(f"Failed to format prediction: {e}, returning as-is")
            return prediction.tolist() if hasattr(prediction, 'tolist') else prediction
    
    def _is_batch_input(self, feature_values: Union[List, Dict]) -> bool:
        """Determine if input represents batch prediction."""
        if isinstance(feature_values, dict):
            return False  # Single sample as dict
        elif isinstance(feature_values, list):
            if not feature_values:
                return False
            # Check if first element is also a list/dict (batch) or scalar (single)
            first_element = feature_values[0]
            if isinstance(first_element, (list, dict)):
                return True  # Batch: [[1,2,3], [4,5,6]] or [{'a':1}, {'a':4}]
            else:
                return False  # Single: [1, 2, 3]
        return False
    
    def _prepare_dataframe_for_preprocessing(
        self, 
        feature_values: Union[List, Dict], 
        feature_names: Optional[List[str]], 
        expected_features: List[str], 
        is_batch: bool
    ) -> pd.DataFrame:
        """Prepare DataFrame for preprocessing from various input formats."""
        if is_batch:
            # Batch prediction
            if isinstance(feature_values[0], dict):
                # List of dicts: [{'feature1': 1, 'feature2': 2}, ...]
                df = pd.DataFrame(feature_values)
            else:
                # List of lists: [[1, 2, 3], [4, 5, 6], ...]
                if feature_names is None:
                    feature_names = expected_features
                if len(feature_values[0]) != len(feature_names):
                    raise ValueError(f"Number of feature values ({len(feature_values[0])}) doesn't match feature names ({len(feature_names)})")
                df = pd.DataFrame(feature_values, columns=feature_names)
        else:
            # Single prediction
            if isinstance(feature_values, dict):
                # Single dict: {'feature1': 1, 'feature2': 2}
                df = pd.DataFrame([feature_values])
            else:
                # Single list: [1, 2, 3]
                if feature_names is None:
                    feature_names = expected_features
                if len(feature_values) != len(feature_names):
                    raise ValueError(f"Number of feature values ({len(feature_values)}) doesn't match feature names ({len(feature_names)})")
                df = pd.DataFrame([dict(zip(feature_names, feature_values))])
        
        # Ensure column order matches expected features
        df = df[expected_features]
        
        # Convert columns to appropriate types to prevent preprocessing errors
        df = self._convert_column_types(df)
        
        return df
    
    def _convert_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to appropriate types to prevent preprocessing issues."""
        df_converted = df.copy()
        
        for col in df_converted.columns:
            try:
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(df_converted[col], errors='coerce')
                
                # If all values can be converted to numeric (no NaN introduced), it's numeric
                if not numeric_series.isna().any():
                    df_converted[col] = numeric_series
                else:
                    # Keep as object/string for categorical processing
                    df_converted[col] = df_converted[col].astype(str)
            except:
                # If conversion fails, keep as string
                df_converted[col] = df_converted[col].astype(str)
        
        return df_converted
    
    def _enforce_training_data_types(self, df: pd.DataFrame, preprocessor: 'DataPreprocessor') -> pd.DataFrame:
        """Enforce data types to match training-time configuration."""
        df_typed = df.copy()
        
        # Get preprocessing configuration
        config = preprocessor.preprocessing_config
        numerical_features = config.get('numerical_features', [])
        categorical_features = config.get('categorical_features', [])
        
        # Ensure numerical features are numeric
        for col in numerical_features:
            if col in df_typed.columns:
                try:
                    df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to numeric: {e}")
        
        # Ensure categorical features are strings/objects
        for col in categorical_features:
            if col in df_typed.columns:
                try:
                    df_typed[col] = df_typed[col].astype(str)
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to string: {e}")
        
        return df_typed
    
    def _prepare_feature_array_batch(
        self, 
        feature_values: Union[List, Dict], 
        feature_names: Optional[List[str]], 
        expected_features: List[str], 
        is_batch: bool
    ) -> np.ndarray:
        """Prepare feature array for batch or single predictions without preprocessing."""
        if is_batch:
            # Batch prediction
            if isinstance(feature_values[0], dict):
                # List of dicts: [{'feature1': 1, 'feature2': 2}, ...]
                X = []
                for sample in feature_values:
                    missing_features = set(expected_features) - set(sample.keys())
                    if missing_features:
                        raise ValueError(f"Missing required features: {missing_features}")
                    X.append([sample[feature] for feature in expected_features])
                return np.array(X)
            else:
                # List of lists: [[1, 2, 3], [4, 5, 6], ...]
                X = np.array(feature_values)
                if X.shape[1] != len(expected_features):
                    raise ValueError(f"Expected {len(expected_features)} features, got {X.shape[1]}")
                return X
        else:
            # Single prediction - use existing method and reshape
            X = self._prepare_feature_array(feature_values, feature_names, expected_features)
            return X.reshape(1, -1)
    

    def _format_predictions(
        self, 
        predictions: np.ndarray, 
        task_type: str, 
        is_batch: bool
    ) -> Union[float, int, List[float], List[int], List[List[float]], List[List[int]]]:
        """Format predictions based on task type and batch status."""
        try:
            if task_type == 'classification':
                # Classification predictions
                if is_batch:
                    if predictions.ndim == 1:
                        return [int(p) for p in predictions]
                    else:
                        # Multi-output classification
                        return [[int(p) for p in sample] for sample in predictions]
                else:
                    # Single prediction
                    if predictions.ndim == 0 or np.isscalar(predictions):
                        return int(predictions)
                    elif predictions.shape == (1,):
                        return int(predictions[0])
                    else:
                        return [int(p) for p in predictions]
            else:
                # Regression predictions
                if is_batch:
                    if predictions.ndim == 1:
                        # Single target batch
                        return [float(p) for p in predictions]
                    else:
                        # Multi-target batch
                        return [[float(p) for p in sample] for sample in predictions]
                else:
                    # Single prediction
                    if predictions.ndim == 0 or np.isscalar(predictions):
                        return float(predictions)
                    elif predictions.shape == (1,):
                        return float(predictions[0])
                    else:
                        # Multi-target single prediction
                        return [float(p) for p in predictions]
                        
        except Exception as e:
            logger.debug(f"Using fallback for prediction formatting due to: {e}")
            # Fallback: use tolist() which handles most numpy array formats
            return predictions.tolist() if hasattr(predictions, 'tolist') else predictions 