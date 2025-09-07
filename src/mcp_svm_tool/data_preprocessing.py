"""
Data Preprocessing Module for XGBoost MCP Tool
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import json
import pickle

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Enhanced data preprocessing pipeline with one-hot encoding and target preprocessing."""
    
    def __init__(self, use_one_hot: bool = True):
        """
        Initialize the data preprocessor.
        
        Args:
            use_one_hot: Whether to use one-hot encoding for categorical features.
                        If False, uses label encoding.
        """
        self.use_one_hot = use_one_hot
        self.feature_preprocessor = None
        self.target_preprocessor = None
        self.target_scaler = None
        self.target_names = []  # Initialize target_names
        self.is_fitted = False
        self.target_is_fitted = False
        self.feature_names_in = []
        self.feature_names_out = []
        self.preprocessing_config = {}
        
        logger.info(f"DataPreprocessor initialized with {'one-hot' if use_one_hot else 'label'} encoding for categorical features")
    
    def _detect_categorical_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Detect and categorize features for preprocessing.
        
        Simple logic since preprocessing is only executed once:
        - Numeric types (int, float) -> numerical features  
        - Text/string types (object, category) -> categorical features
        
        Args:
            X: Input dataframe
            
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        numerical_features = []
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'float16']:
                numerical_features.append(col)
                logger.debug(f"Column '{col}' -> numerical (dtype: {X[col].dtype})")
            elif X[col].dtype in ['object', 'category']:
                categorical_features.append(col)
                unique_count = X[col].nunique()
                logger.info(f"Column '{col}' -> categorical (unique values: {unique_count})")
            else:
                # Fallback: treat unknown types as categorical
                categorical_features.append(col)
                logger.warning(f"Column '{col}' has unknown dtype {X[col].dtype}, treating as categorical")
        
        logger.info(f"Feature detection completed:")
        logger.info(f"  - Numerical features: {len(numerical_features)} columns")
        logger.info(f"  - Categorical features: {len(categorical_features)} columns")
        
        return numerical_features, categorical_features
    
    def fit_feature_preprocessing(self, 
                                X: pd.DataFrame,
                                scaling_method: str = "standard",
                                impute_missing: bool = True) -> 'DataPreprocessor':
        """
        Fit feature preprocessing pipeline.
        
        Args:
            X: Feature dataframe
            scaling_method: Scaling method for numerical features
            impute_missing: Whether to impute missing values
            
        Returns:
            Self with fitted feature preprocessing
        """
        logger.info("Fitting feature preprocessing pipeline...")
        
        self.feature_names_in = X.columns.tolist()
        
        # Detect feature types
        numerical_features, categorical_features = self._detect_categorical_features(X)
        
        # Store configuration
        self.preprocessing_config.update({
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'scaling_method': scaling_method,
            'impute_missing': impute_missing,
            'use_one_hot': self.use_one_hot
        })
        
        # Create preprocessing steps
        preprocessors = []
        
        # Numerical features pipeline
        if numerical_features:
            numerical_steps = []
            if impute_missing:
                numerical_steps.append(('imputer', SimpleImputer(strategy='mean')))
            numerical_steps.append(('scaler', self._create_scaler(scaling_method)))
            
            numerical_pipeline = Pipeline(numerical_steps)
            preprocessors.append(('numerical', numerical_pipeline, numerical_features))
            logger.info(f"Created numerical pipeline for {len(numerical_features)} features")

        # Categorical features pipeline
        if categorical_features:
            categorical_steps = []
            if impute_missing:
                categorical_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
            
            if self.use_one_hot:
                categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                pipeline_name = 'categorical_onehot'
                logger.info(f"Created one-hot encoding pipeline for {len(categorical_features)} categorical features")
            else:
                categorical_steps.append(('label_encoder', LabelEncoderWrapper()))
                pipeline_name = 'categorical_label'
                logger.info(f"Created label encoding pipeline for {len(categorical_features)} categorical features")
            
            categorical_pipeline = Pipeline(categorical_steps)
            preprocessors.append((pipeline_name, categorical_pipeline, categorical_features))
        
        # Create the column transformer
        if preprocessors:
            self.feature_preprocessor = ColumnTransformer(
                preprocessors,
                remainder='drop',  # Drop any columns not included in the transformers
                sparse_threshold=0,
                n_jobs=None
            )
            
            # Fit the preprocessing pipeline
            self.feature_preprocessor.fit(X)
            self.is_fitted = True
            
            # Generate feature names for output
            self.feature_names_out = self._get_feature_names_out()
            
            logger.info(f"Feature preprocessing pipeline fitted successfully")
            logger.info(f"Input features: {len(self.feature_names_in)}, Output features: {len(self.feature_names_out)}")
        else:
            logger.warning("No preprocessing steps created - no features detected")
            self.feature_preprocessor = None
            self.is_fitted = False
            
        return self
    
    def fit_target_preprocessing(self, 
                                y: Union[pd.Series, np.ndarray, pd.DataFrame],
                                task_type: str = None,
                                target_scaling_method: str = "standard") -> 'DataPreprocessor':
        """
        Fit target variable preprocessing.
        
        Args:
            y: Target variable(s)
            task_type: Task type ('regression', 'classification', or 'auto')
            target_scaling_method: Scaling method for regression targets
            
        Returns:
            Self with fitted target preprocessing
        """
        logger.info("Fitting target preprocessing...")
        
        # Convert to appropriate format
        if isinstance(y, pd.Series):
            y_array = y.values
            self.target_names = [y.name] if y.name else ['target']
        elif isinstance(y, pd.DataFrame):
            y_array = y.values
            self.target_names = y.columns.tolist()
        else:
            y_array = np.array(y)
            if y_array.ndim == 1:
                self.target_names = ['target']
            else:
                self.target_names = [f'target_{i}' for i in range(y_array.shape[1])]
        
        # Auto-detect task type if needed
        use_task_type = task_type if task_type is not None else "auto"
        if use_task_type == "auto":
            if y_array.ndim > 1:
                use_task_type = "regression"  # Multi-target is typically regression
            else:
                unique_values = len(np.unique(y_array))
                if unique_values <= 20 and np.all(y_array == y_array.astype(int)):
                    use_task_type = "classification"
                else:
                    use_task_type = "regression"
        
        # Store configuration
        self.preprocessing_config.update({
            'target_task_type': use_task_type,
            'target_scaling_method': target_scaling_method,
            'target_names': self.target_names,
            'target_shape': y_array.shape
        })
        
        # Fit target preprocessing based on task type
        if use_task_type == "classification":
            # Use label encoder for classification
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y_array.ravel())
            
            # Create label mapping for later use in local importance analysis
            # Convert numpy types to Python native types for JSON serialization
            classes_native = [int(cls) if isinstance(cls, (np.integer, np.int64, np.int32)) 
                             else float(cls) if isinstance(cls, (np.floating, np.float64, np.float32))
                             else str(cls) if not isinstance(cls, (int, float, str, bool))
                             else cls 
                             for cls in self.target_encoder.classes_]
            
            self.label_mapping = {
                'class_to_label': {i: classes_native[i] for i in range(len(classes_native))},
                'label_to_class': {classes_native[i]: i for i in range(len(classes_native))},
                'classes': classes_native
            }
            
            logger.info(f"Target classification preprocessing fitted: {len(self.target_encoder.classes_)} classes")
            logger.info(f"Label mapping created: {self.label_mapping['class_to_label']}")
        else:
            # Use scaler for regression
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            
            self.target_scaler = self._create_scaler(target_scaling_method)
            self.target_scaler.fit(y_array)
            logger.info(f"Target regression preprocessing fitted with {target_scaling_method} scaling")
        
        self.target_is_fitted = True
        return self
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessing."""
        if not self.is_fitted:
            raise ValueError("Feature preprocessing must be fitted before transform")
        
        X_transformed = self.feature_preprocessor.transform(X)
        return X_transformed
    
    def transform_target(self, y: Union[pd.Series, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform target using fitted preprocessing."""
        if not self.target_is_fitted:
            raise ValueError("Target preprocessing must be fitted before transform")
        
        # Convert to array
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Apply transformation based on task type
        task_type = self.preprocessing_config['target_task_type']
        
        if task_type == "classification":
            return self.target_encoder.transform(y_array.ravel())
        else:
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            return self.target_scaler.transform(y_array)
    
    def inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform target predictions back to original scale."""
        if not self.target_is_fitted:
            raise ValueError("Target preprocessing must be fitted before inverse transform")
        
        task_type = self.preprocessing_config['target_task_type']
        
        if task_type == "classification":
            return self.target_encoder.inverse_transform(y_transformed)
        else:
            if y_transformed.ndim == 1:
                y_transformed = y_transformed.reshape(-1, 1)
            result = self.target_scaler.inverse_transform(y_transformed)
            return result.ravel() if result.shape[1] == 1 else result
    
    def fit_transform_features(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Fit and transform features in one step."""
        return self.fit_feature_preprocessing(X, **kwargs).transform_features(X)
    
    def fit_transform_target(self, y: Union[pd.Series, np.ndarray, pd.DataFrame], task_type: str = None, target_scaling_method: str = "standard") -> np.ndarray:
        """
        Fit and transform target variable(s) with explicit task_type传递。
        """
        self.fit_target_preprocessing(y, task_type=task_type, target_scaling_method=target_scaling_method)
        return self.transform_target(y)
    
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray, pd.DataFrame] = None, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Fit and transform both features and target."""
        X_transformed = self.fit_transform_features(X, **kwargs)
        
        if y is not None:
            y_transformed = self.fit_transform_target(y, **kwargs)
            return X_transformed, y_transformed
        
        return X_transformed
    
    def _get_feature_names_out(self) -> List[str]:
        """Get output feature names after preprocessing."""
        feature_names = []
        
        # Get names from each transformer
        for name, transformer, columns in self.feature_preprocessor.transformers_:
            if name == 'remainder':
                continue
                
            if name == 'categorical_onehot':
                # Get one-hot encoded feature names
                onehot_encoder = transformer.named_steps['onehot']
                for i, col in enumerate(columns):
                    if hasattr(onehot_encoder, 'categories_'):
                        categories = onehot_encoder.categories_[i]
                    else:
                        categories = onehot_encoder.get_feature_names_out([col])
                    
                    for category in categories:
                        feature_names.append(f"{col}_{category}")
            elif name == 'categorical_label':
                # Label encoded features keep their original names
                feature_names.extend(columns)
            elif name == 'numerical':
                # Numerical features keep their names
                feature_names.extend(columns)
        
        return feature_names
    
    def _create_scaler(self, scaling_method: str):
        """Create scaler instance based on method."""
        if scaling_method == "standard":
            return StandardScaler()
        elif scaling_method == "minmax":
            return MinMaxScaler()
        elif scaling_method == "robust":
            return RobustScaler()
        elif scaling_method == "quantile":
            return QuantileTransformer(output_distribution='normal')
        elif scaling_method == "power":
            return PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing information."""
        info = {
            'is_fitted': self.is_fitted,
            'target_is_fitted': self.target_is_fitted,
            'config': self.preprocessing_config.copy(),
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out,
            'target_names': self.target_names,
            'input_feature_count': len(self.feature_names_in),
            'output_feature_count': len(self.feature_names_out),
            'preprocessing_steps': []
        }
        
        # Add detailed transformation info
        if self.is_fitted and self.feature_preprocessor:
            for name, transformer, columns in self.feature_preprocessor.transformers_:
                if name != 'remainder':
                    step_info = {
                        'transformer_name': name,
                        'columns': columns,
                        'column_count': len(columns)
                    }
                    
                    if name == 'categorical_onehot':
                        onehot_encoder = transformer.named_steps['onehot']
                        step_info['encoding_type'] = 'one_hot'
                        step_info['categories'] = {
                            col: list(onehot_encoder.categories_[i]) 
                            for i, col in enumerate(columns)
                        }
                    elif name == 'categorical_label':
                        step_info['encoding_type'] = 'label'
                    elif name == 'numerical':
                        step_info['scaling_method'] = self.preprocessing_config.get('scaling_method', 'unknown')
                    
                    info['preprocessing_steps'].append(step_info)
        
        return info
    
    def save_pipeline(self, filepath: str) -> str:
        """
        Save complete preprocessing pipeline to pickle file.
        
        Args:
            filepath: Path to save pipeline
            
        Returns:
            Path to saved pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive pipeline data
        pipeline_data = {
            'feature_preprocessor': self.feature_preprocessor,
            'target_preprocessor': {
                'target_scaler': self.target_scaler,
                'target_encoder': getattr(self, 'target_encoder', None)  # Use getattr to handle missing attribute
            },
            'preprocessing_config': self.preprocessing_config,
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out,
            'target_names': getattr(self, 'target_names', []),  # Use getattr for safety
            'is_fitted': self.is_fitted,
            'target_is_fitted': self.target_is_fitted,
            'use_one_hot': self.use_one_hot,
            'preprocessing_info': self.get_preprocessing_info()
        }
        
        # Save using pickle for complete sklearn object preservation
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Complete preprocessing pipeline saved to: {filepath}")
        return str(filepath)
    
    def load_pipeline(self, filepath: str) -> 'DataPreprocessor':
        """
        Load complete preprocessing pipeline from pickle file.
        
        Args:
            filepath: Path to pipeline file
            
        Returns:
            Self with loaded pipeline
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        # Load pipeline data
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Restore all attributes
        self.feature_preprocessor = pipeline_data['feature_preprocessor']
        self.target_scaler = pipeline_data['target_preprocessor']['target_scaler']
        target_encoder = pipeline_data['target_preprocessor']['target_encoder']
        if target_encoder is not None:
            self.target_encoder = target_encoder
        self.preprocessing_config = pipeline_data['preprocessing_config']
        self.feature_names_in = pipeline_data['feature_names_in']
        self.feature_names_out = pipeline_data['feature_names_out']
        self.target_names = pipeline_data.get('target_names', [])
        self.is_fitted = pipeline_data['is_fitted']
        self.target_is_fitted = pipeline_data['target_is_fitted']
        self.use_one_hot = pipeline_data['use_one_hot']
        
        logger.info(f"Complete preprocessing pipeline loaded from: {filepath}")
        logger.info(f"Pipeline info: {pipeline_data.get('preprocessing_info', {})}")
        return self

    def get_feature_names(self) -> List[str]:
        """
        Get output feature names after preprocessing in correct order.
        
        Returns:
            List of feature names in the same order as the transformed output
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        
        return self.feature_names_out


class LabelEncoderWrapper:
    """Wrapper for LabelEncoder to handle multiple columns and unknown values."""
    
    def __init__(self):
        self.encoders = {}
        self.columns = []
    
    def fit(self, X):
        self.columns = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))
        
        for i, col in enumerate(self.columns):
            encoder = LabelEncoder()
            column_data = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            
            # 处理category类型 - 使用原始字符串值而不是category代码
            if hasattr(column_data, 'dtype') and column_data.dtype.name == 'category':
                # 对于category类型，使用原始字符串值
                column_data_str = column_data.astype(str)
            else:
                column_data_str = column_data.astype(str)
            
            encoder.fit(column_data_str)
            self.encoders[col] = encoder
        
        return self
    
    def transform(self, X):
        # 创建输出数组，使用float类型以支持可能的-1值
        if hasattr(X, 'shape'):
            X_transformed = np.zeros(X.shape, dtype=float)
        else:
            X_transformed = np.zeros_like(X, dtype=float)
        
        for i, col in enumerate(self.columns):
            column_data = X.iloc[:, i] if hasattr(X, 'iloc') else X[:, i]
            
            # 处理category类型 - 使用原始字符串值
            if hasattr(column_data, 'dtype') and column_data.dtype.name == 'category':
                column_data_str = column_data.astype(str)
            else:
                column_data_str = column_data.astype(str)
            
            # Handle unknown values
            encoder = self.encoders[col]
            mask = np.isin(column_data_str, encoder.classes_)
            
            # Transform known values
            X_transformed[mask, i] = encoder.transform(column_data_str[mask])
            # Unknown values get encoded as -1 (or could be set to a default class)
            X_transformed[~mask, i] = -1
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# Backward compatibility - keep the old interface
class DataPreprocessorLegacy(DataPreprocessor):
    """Legacy interface for backward compatibility."""
    
    def __init__(self):
        super().__init__(use_one_hot=False)  # Use label encoding by default for backward compatibility
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def fit_preprocessing_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        """Legacy method - redirects to new interface."""
        return self.fit_feature_preprocessing(X, **kwargs)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Legacy method - returns DataFrame instead of array."""
        X_transformed = self.transform_features(X)
        
        # Convert back to DataFrame with appropriate column names
        if len(self.feature_names_out) == X_transformed.shape[1]:
            return pd.DataFrame(X_transformed, columns=self.feature_names_out, index=X.index)
        else:
            # Fallback if column names don't match
            columns = [f'feature_{i}' for i in range(X_transformed.shape[1])]
            return pd.DataFrame(X_transformed, columns=columns, index=X.index) 