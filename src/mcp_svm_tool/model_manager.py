"""
Model Management System

This module provides unified model storage, version management,
and performance monitoring capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
import json
import pickle
from datetime import datetime
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages trained models with storage, versioning, and metadata tracking.
    
    Features:
    - Model serialization and deserialization
    - Automatic metadata recording
    - Version control support
    - Performance benchmarking
    - Secure file management
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create metadata file path
        self.metadata_file = self.models_dir / "models_metadata.json"
        
        # Load existing metadata or create new
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized ModelManager with directory: {models_dir}")
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}, creating new")
                return {"models": {}, "created_at": datetime.now().isoformat()}
        else:
            return {"models": {}, "created_at": datetime.now().isoformat()}
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def register_model(self, 
                      model_id: str,
                      model_name: str,
                      model_path: str,
                      task_type: str,
                      performance_metrics: Dict[str, Any],
                      feature_names: List[str],
                      hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a trained model with metadata.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            model_path: Path to the saved model file
            task_type: Type of task (classification/regression)
            performance_metrics: Model performance metrics
            feature_names: List of feature names
            hyperparameters: Model hyperparameters
            
        Returns:
            Registration status and information
        """
        try:
            model_info = {
                "model_id": model_id,
                "model_name": model_name,
                "model_path": model_path,
                "task_type": task_type,
                "performance_metrics": performance_metrics,
                "feature_names": feature_names,
                "hyperparameters": hyperparameters,
                "registered_at": datetime.now().isoformat(),
                "file_exists": Path(model_path).exists() if model_path else False
            }
            
            # Add to metadata
            self.metadata["models"][model_id] = model_info
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Registered model: {model_name} (ID: {model_id})")
            
            return {
                "success": True,
                "model_id": model_id,
                "message": f"Model {model_name} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def save_model(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Save a trained model with metadata in its own directory.
        
        Args:
            model: Trained model object
            model_id: Unique identifier for the model
            metadata: Model metadata
            
        Returns:
            Path to saved model
        """
        try:
            # Create model-specific directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Define paths for different files
            model_path = model_dir / "model.joblib"
            metadata_path = model_dir / "metadata.json"
            
            # Save model using joblib
            import joblib
            joblib.dump(model, model_path)
            
            # Save detailed metadata in the model directory
            detailed_metadata = {
                "model_id": model_id,
                "saved_at": datetime.now().isoformat(),
                **metadata
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # Register in global metadata
            self.register_model(
                model_id=model_id,
                model_name=metadata.get("model_name", f"model_{model_id}"),
                model_path=str(model_path),
                task_type=metadata.get("task_type", "unknown"),
                performance_metrics=metadata.get("performance_metrics", {}),
                feature_names=metadata.get("feature_names", []),
                hyperparameters=metadata.get("hyperparameters", {})
            )
            
            logger.info(f"Model saved to directory: {model_dir}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def get_model_directory(self, model_id: str) -> Path:
        """
        Get the directory path for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model directory
        """
        return self.models_dir / model_id
    
    def create_model_directory(self, model_id: str) -> Path:
        """
        Create and return the directory path for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model directory
        """
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        return model_dir
        
    def load_model(self, model_id: str) -> Any:
        """
        Load a trained model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model object
        """
        try:
            # Refresh metadata to ensure we have the latest models
            self.metadata = self._load_metadata()
            
            if model_id not in self.metadata["models"]:
                # Also try direct file loading as fallback
                model_dir = self.models_dir / model_id
                model_path = model_dir / "model.joblib"
                
                if model_path.exists():
                    logger.warning(f"Model {model_id} not in metadata but file exists, loading directly")
                    import joblib
                    model = joblib.load(model_path)
                    logger.info(f"Loaded model directly: {model_id}")
                    return model
                else:
                    raise ValueError(f"Model {model_id} not found")
            
            model_info = self.metadata["models"][model_id]
            model_path = model_info["model_path"]
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            import joblib
            model = joblib.load(model_path)
            
            logger.info(f"Loaded model: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information
        """
        try:
            models_list = []
            for model_id, model_info in self.metadata["models"].items():
                # Check if file still exists
                model_info_copy = model_info.copy()
                if model_info.get("model_path"):
                    model_info_copy["file_exists"] = Path(model_info["model_path"]).exists()
                models_list.append(model_info_copy)
            
            return models_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
        
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata and information
        """
        try:
            # First try to get detailed metadata from model directory
            model_dir = self.models_dir / model_id
            metadata_path = model_dir / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        detailed_metadata = json.load(f)
                    
                    # Check if model file still exists
                    model_path = model_dir / "model.joblib"
                    detailed_metadata["file_exists"] = model_path.exists()
                    
                    logger.info(f"Retrieved detailed metadata for model: {model_id}")
                    return detailed_metadata
                    
                except Exception as e:
                    logger.warning(f"Failed to read detailed metadata for {model_id}: {e}")
            
            # Fallback to global metadata if detailed metadata not available
            if model_id not in self.metadata["models"]:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.metadata["models"][model_id].copy()
            
            # Check if file still exists
            if model_info.get("model_path"):
                model_info["file_exists"] = Path(model_info["model_path"]).exists()
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
        
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """
        Delete a model and its entire directory.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Deletion status and information
        """
        try:
            if model_id not in self.metadata["models"]:
                return {
                    "success": False,
                    "error": f"Model {model_id} not found"
                }
            
            model_dir = self.models_dir / model_id
            
            # Delete entire model directory if it exists
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model directory: {model_dir}")
            
            # Remove from metadata
            del self.metadata["models"][model_id]
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Deleted model: {model_id}")
            
            return {
                "success": True,
                "model_id": model_id,
                "message": f"Model {model_id} and all associated files deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return {
                "success": False,
                "error": str(e)
            } 