"""
Data Validation and Integrity Checks

This module provides comprehensive data validation mechanisms to ensure
data integrity and quality before machine learning model training.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import chi2_contingency, pearsonr
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation and integrity checking system.
    
    Features:
    - Data dimension consistency checks
    - Target variable validity verification
    - Feature name normalization and validation
    - Data leakage detection
    - Sample balance analysis
    - Data quality scoring
    - Comprehensive integrity reporting
    - Feature analysis visualizations
    """
    
    def __init__(self, enable_visualizations: bool = True):
        """
        Initialize DataValidator.
        
        Args:
            enable_visualizations: Whether to generate validation visualizations
        """
        self.validation_rules = {
            'min_samples': 50,
            'max_missing_ratio': 0.5,
            'min_unique_ratio': 0.01,
            'max_cardinality': 1000,
            'min_class_ratio': 0.01,
            'max_correlation_threshold': 0.95,
            'feature_name_pattern': r'^[a-zA-Z][a-zA-Z0-9_]*$'
        }
        self.validation_results = {}
        self.enable_visualizations = enable_visualizations
        
        # Initialize visualization generator if enabled
        self.visualizer = None
        if self.enable_visualizations:
            try:
                from .visualization_generator import DataValidationVisualizer
                self.visualizer = DataValidationVisualizer()
                logger.info("Data validation visualizations enabled")
            except ImportError as e:
                logger.warning(f"Could not import DataValidationVisualizer: {e}")
                self.enable_visualizations = False
            except Exception as e:
                logger.warning(f"Could not initialize DataValidationVisualizer: {e}")
                self.enable_visualizations = False
        else:
            logger.info("Data validation visualizations disabled")
            
        logger.info("Initialized DataValidator")
    
    def set_validation_rules(self, rules: Dict[str, Any]) -> None:
        """
        Update validation rules.
        
        Args:
            rules: Dictionary of validation rules to update
        """
        self.validation_rules.update(rules)
        logger.info(f"Updated validation rules: {list(rules.keys())}")
    
    def check_data_dimensions(
        self,
        train_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Check dimension consistency across datasets.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
            validation_data: Validation dataset (optional)
            
        Returns:
            Dimension consistency report
        """
        dimension_report = {
            'check_name': 'data_dimensions',
            'passed': True,
            'issues': [],
            'details': {}
        }
        
        # Basic dimension checks
        train_shape = train_data.shape
        dimension_report['details']['train_shape'] = train_shape
        
        # Minimum sample size check
        if train_shape[0] < self.validation_rules['min_samples']:
            dimension_report['passed'] = False
            dimension_report['issues'].append(
                f"Training data has insufficient samples: {train_shape[0]} < {self.validation_rules['min_samples']}"
            )
        
        # Feature consistency checks
        train_features = set(train_data.columns)
        
        if test_data is not None:
            test_shape = test_data.shape
            test_features = set(test_data.columns)
            dimension_report['details']['test_shape'] = test_shape
            
            # Check feature consistency
            missing_in_test = train_features - test_features
            extra_in_test = test_features - train_features
            
            if missing_in_test:
                dimension_report['passed'] = False
                dimension_report['issues'].append(
                    f"Features missing in test data: {list(missing_in_test)}"
                )
            
            if extra_in_test:
                dimension_report['passed'] = False
                dimension_report['issues'].append(
                    f"Extra features in test data: {list(extra_in_test)}"
                )
        
        if validation_data is not None:
            val_shape = validation_data.shape
            val_features = set(validation_data.columns)
            dimension_report['details']['validation_shape'] = val_shape
            
            # Check feature consistency with validation data
            missing_in_val = train_features - val_features
            extra_in_val = val_features - train_features
            
            if missing_in_val:
                dimension_report['passed'] = False
                dimension_report['issues'].append(
                    f"Features missing in validation data: {list(missing_in_val)}"
                )
            
            if extra_in_val:
                dimension_report['passed'] = False
                dimension_report['issues'].append(
                    f"Extra features in validation data: {list(extra_in_val)}"
                )
        
        logger.info(f"Dimension consistency check completed: {'PASSED' if dimension_report['passed'] else 'FAILED'}")
        return dimension_report
    
    def validate_target_variable(
        self,
        data: pd.DataFrame,
        target_column: Union[str, List[str]],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Validate target variable(s) for ML task.
        
        Args:
            data: Input dataset
            target_column: Name of target column or list of target columns (for multi-target)
            task_type: Type of ML task ('classification', 'regression', 'auto')
            
        Returns:
            Target variable validation report
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"🎯 Starting target variable validation...")
            logger.info(f"  Target column: {target_column}")
            logger.info(f"  Target column type: {type(target_column)}")
            logger.info(f"  Task type: {task_type}")
            logger.info(f"  DataFrame shape: {data.shape}")
            logger.info(f"  DataFrame columns: {list(data.columns)}")
            
            target_report = {
                'check_name': 'target_variable',
                'passed': True,
                'issues': [],
                'details': {}
            }
            
            # Handle multi-target case
            if isinstance(target_column, list):
                logger.info("📊 Multi-target detected, delegating to multi-target validation")
                return self._validate_multi_target_variables(data, target_column, task_type)
            
            # Check if target column exists
            logger.info(f"🔍 Checking if target column '{target_column}' exists in data...")
            if target_column not in data.columns:
                logger.error(f"❌ Target column '{target_column}' not found in data columns: {list(data.columns)}")
                target_report['passed'] = False
                target_report['issues'].append(f"Target column '{target_column}' not found in data")
                return target_report
            
            logger.info(f"✅ Target column '{target_column}' found, accessing data...")
            target_series = data[target_column]
            logger.info(f"✅ Target series successfully accessed, shape: {target_series.shape}")
        
        except Exception as e:
            logger.error(f"❌ Error in validate_target_variable: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(f"   Exception args: {e.args}")
            import traceback
            logger.error(f"   Full traceback:\n{traceback.format_exc()}")
            raise
        
        # Basic target statistics
        target_stats = {
            'total_samples': len(target_series),
            'missing_values': target_series.isnull().sum(),
            'unique_values': target_series.nunique(),
            'data_type': str(target_series.dtype)
        }
        
        # Missing values check
        missing_ratio = target_stats['missing_values'] / target_stats['total_samples']
        if missing_ratio > 0:
            if missing_ratio > 0.1:  # More than 10% missing
                target_report['passed'] = False
                target_report['issues'].append(
                    f"Target variable has high missing value ratio: {missing_ratio:.2%}"
                )
            else:
                target_report['issues'].append(
                    f"Target variable has missing values: {missing_ratio:.2%}"
                )
        
        # Infer task type if auto
        if task_type == 'auto':
            if pd.api.types.is_numeric_dtype(target_series):
                if target_stats['unique_values'] <= 10:
                    inferred_task = 'classification'
                else:
                    inferred_task = 'regression'
            else:
                inferred_task = 'classification'
            task_type = inferred_task
        
        target_stats['inferred_task_type'] = task_type
        
        # Task-specific validation
        if task_type == 'classification':
            # Class distribution analysis
            class_counts = target_series.value_counts()
            target_stats['class_distribution'] = class_counts.to_dict()
            target_stats['num_classes'] = len(class_counts)
            
            # Check for class imbalance
            min_class_ratio = class_counts.min() / class_counts.sum()
            if min_class_ratio < self.validation_rules['min_class_ratio']:
                target_report['issues'].append(
                    f"Severe class imbalance detected: minimum class ratio {min_class_ratio:.3f}"
                )
            
            # Check for single class
            if target_stats['num_classes'] == 1:
                target_report['passed'] = False
                target_report['issues'].append("Target variable has only one class")
            
            # Binary vs multiclass
            target_stats['is_binary'] = target_stats['num_classes'] == 2
            
        elif task_type == 'regression':
            # Regression-specific checks
            target_clean = target_series.dropna()
            target_stats.update({
                'mean': float(target_clean.mean()),
                'std': float(target_clean.std()),
                'min': float(target_clean.min()),
                'max': float(target_clean.max()),
                'skewness': float(target_clean.skew()),
                'kurtosis': float(target_clean.kurtosis())
            })
            
            # Check for constant target
            if target_clean.std() == 0:
                target_report['passed'] = False
                target_report['issues'].append("Target variable is constant (zero variance)")
            
            # Check for extreme skewness
            if abs(target_stats['skewness']) > 3:
                target_report['issues'].append(
                    f"Target variable is heavily skewed: skewness = {target_stats['skewness']:.2f}"
                )
        
        target_report['details'] = target_stats
        logger.info(f"Target variable validation completed: {'PASSED' if target_report['passed'] else 'FAILED'}")
        return target_report
    
    def _validate_multi_target_variables(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Validate multiple target variables for multi-target ML task.
        
        Args:
            data: Input dataset
            target_columns: List of target column names
            task_type: Type of ML task ('classification', 'regression', 'auto')
            
        Returns:
            Multi-target validation report
        """
        multi_target_report = {
            'check_name': 'multi_target_variables',
            'passed': True,
            'issues': [],
            'details': {
                'num_targets': len(target_columns),
                'target_columns': target_columns,
                'individual_reports': {}
            }
        }
        
        # Check if all target columns exist
        missing_columns = []
        for col in target_columns:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            multi_target_report['passed'] = False
            multi_target_report['issues'].append(
                f"Target columns not found in data: {missing_columns}"
            )
            return multi_target_report
        
        # Validate each target column individually
        for col in target_columns:
            individual_report = self._validate_single_target(data, col, task_type)
            multi_target_report['details']['individual_reports'][col] = individual_report
            
            # If any individual target fails, the overall validation fails
            if not individual_report['passed']:
                multi_target_report['passed'] = False
                multi_target_report['issues'].append(
                    f"Target '{col}' validation failed: {individual_report['issues']}"
                )
        
        # Multi-target specific checks
        target_data = data[target_columns]
        
        # Check for high correlations between targets
        if len(target_columns) > 1:
            target_corr = target_data.corr()
            high_corr_pairs = []
            
            for i in range(len(target_columns)):
                for j in range(i + 1, len(target_columns)):
                    corr_val = abs(target_corr.iloc[i, j])
                    if corr_val > 0.95:  # Very high correlation
                        high_corr_pairs.append((target_columns[i], target_columns[j], corr_val))
            
            if high_corr_pairs:
                multi_target_report['issues'].append(
                    f"High correlations between targets: {high_corr_pairs}"
                )
            
            multi_target_report['details']['target_correlations'] = target_corr.to_dict()
        
        # Overall target statistics
        multi_target_report['details']['overall_stats'] = {
            'total_samples': len(target_data),
            'missing_values_per_target': target_data.isnull().sum().to_dict(),
            'data_types': target_data.dtypes.astype(str).to_dict()
        }
        
        logger.info(f"Multi-target validation completed: {'PASSED' if multi_target_report['passed'] else 'FAILED'}")
        return multi_target_report
    
    def _validate_single_target(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Validate a single target variable (helper method for multi-target validation).
        
        Args:
            data: Input dataset
            target_column: Name of target column
            task_type: Type of ML task ('classification', 'regression', 'auto')
            
        Returns:
            Single target validation report
        """
        target_series = data[target_column]
        
        target_report = {
            'check_name': f'target_variable_{target_column}',
            'passed': True,
            'issues': [],
            'details': {}
        }
        
        # Basic target statistics
        target_stats = {
            'total_samples': len(target_series),
            'missing_values': target_series.isnull().sum(),
            'unique_values': target_series.nunique(),
            'data_type': str(target_series.dtype)
        }
        
        # Missing values check
        missing_ratio = target_stats['missing_values'] / target_stats['total_samples']
        if missing_ratio > 0:
            if missing_ratio > 0.1:  # More than 10% missing
                target_report['passed'] = False
                target_report['issues'].append(
                    f"Target variable has high missing value ratio: {missing_ratio:.2%}"
                )
            else:
                target_report['issues'].append(
                    f"Target variable has missing values: {missing_ratio:.2%}"
                )
        
        # Infer task type if auto
        if task_type == 'auto':
            if pd.api.types.is_numeric_dtype(target_series):
                if target_stats['unique_values'] <= 10:
                    inferred_task = 'classification'
                else:
                    inferred_task = 'regression'
            else:
                inferred_task = 'classification'
            task_type = inferred_task
        
        target_stats['inferred_task_type'] = task_type
        
        # Task-specific validation
        if task_type == 'classification':
            # Class distribution analysis
            class_counts = target_series.value_counts()
            target_stats['class_distribution'] = class_counts.to_dict()
            target_stats['num_classes'] = len(class_counts)
            
            # Check for class imbalance
            min_class_ratio = class_counts.min() / class_counts.sum()
            if min_class_ratio < self.validation_rules['min_class_ratio']:
                target_report['issues'].append(
                    f"Severe class imbalance detected: minimum class ratio {min_class_ratio:.3f}"
                )
            
            # Check for single class
            if target_stats['num_classes'] == 1:
                target_report['passed'] = False
                target_report['issues'].append("Target variable has only one class")
            
            # Binary vs multiclass
            target_stats['is_binary'] = target_stats['num_classes'] == 2
            
        elif task_type == 'regression':
            # Regression-specific checks
            target_clean = target_series.dropna()
            target_stats.update({
                'mean': float(target_clean.mean()),
                'std': float(target_clean.std()),
                'min': float(target_clean.min()),
                'max': float(target_clean.max()),
                'skewness': float(target_clean.skew()),
                'kurtosis': float(target_clean.kurtosis())
            })
            
            # Check for constant target
            if target_clean.std() == 0:
                target_report['passed'] = False
                target_report['issues'].append("Target variable is constant (zero variance)")
            
            # Check for extreme skewness
            if abs(target_stats['skewness']) > 3:
                target_report['issues'].append(
                    f"Target variable is heavily skewed: skewness = {target_stats['skewness']:.2f}"
                )
        
        target_report['details'] = target_stats
        return target_report
    
    def normalize_feature_names(
        self,
        data: pd.DataFrame,
        inplace: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize and validate feature names.
        
        Args:
            data: Input dataframe
            inplace: Whether to modify dataframe in place
            
        Returns:
            Tuple of (normalized dataframe, normalization report)
        """
        normalization_report = {
            'check_name': 'feature_names',
            'passed': True,
            'issues': [],
            'details': {}
        }
        
        if not inplace:
            data = data.copy()
        
        original_columns = list(data.columns)
        normalized_columns = []
        column_mapping = {}
        
        for col in original_columns:
            original_col = col
            normalized_col = str(col)
            
            # Remove leading/trailing whitespace
            normalized_col = normalized_col.strip()
            
            # Replace spaces and special characters with underscores
            normalized_col = re.sub(r'[^\w]', '_', normalized_col)
            
            # Remove multiple consecutive underscores
            normalized_col = re.sub(r'_+', '_', normalized_col)
            
            # Remove leading/trailing underscores
            normalized_col = normalized_col.strip('_')
            
            # Ensure it starts with a letter
            if not normalized_col[0].isalpha():
                normalized_col = 'feature_' + normalized_col
            
            # Handle empty names
            if not normalized_col:
                normalized_col = f'feature_{len(normalized_columns)}'
            
            # Handle duplicates
            if normalized_col in normalized_columns:
                counter = 1
                base_name = normalized_col
                while f"{base_name}_{counter}" in normalized_columns:
                    counter += 1
                normalized_col = f"{base_name}_{counter}"
            
            normalized_columns.append(normalized_col)
            column_mapping[original_col] = normalized_col
            
            # Check if normalization was needed
            if original_col != normalized_col:
                normalization_report['issues'].append(
                    f"Column renamed: '{original_col}' -> '{normalized_col}'"
                )
        
        # Apply normalization
        data.columns = normalized_columns
        
        # Validate against pattern
        invalid_names = []
        pattern = re.compile(self.validation_rules['feature_name_pattern'])
        
        for col in normalized_columns:
            if not pattern.match(col):
                invalid_names.append(col)
        
        if invalid_names:
            normalization_report['passed'] = False
            normalization_report['issues'].extend([
                f"Invalid feature name after normalization: '{name}'" for name in invalid_names
            ])
        
        normalization_report['details'] = {
            'original_columns': original_columns,
            'normalized_columns': normalized_columns,
            'column_mapping': column_mapping,
            'changes_made': len([k for k, v in column_mapping.items() if k != v])
        }
        
        logger.info(f"Feature name normalization completed: {normalization_report['details']['changes_made']} changes made")
        return data, normalization_report
    
    def detect_data_leakage(
        self,
        data: pd.DataFrame,
        target_column: str,
        time_column: Optional[str] = None,
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Detect potential data leakage.
        
        Args:
            data: Input dataset
            target_column: Name of target column
            time_column: Name of time column (optional)
            threshold: Correlation threshold for leakage detection
            
        Returns:
            Data leakage detection report
        """
        leakage_report = {
            'check_name': 'data_leakage',
            'passed': True,
            'issues': [],
            'details': {}
        }
        
        if target_column not in data.columns:
            leakage_report['passed'] = False
            leakage_report['issues'].append(f"Target column '{target_column}' not found")
            return leakage_report
        
        # Remove target and time columns from feature analysis
        feature_columns = [col for col in data.columns if col not in [target_column, time_column]]
        
        if not feature_columns:
            leakage_report['issues'].append("No features available for leakage detection")
            return leakage_report
        
        target_series = data[target_column]
        suspicious_features = []
        
        # Correlation-based leakage detection
        for feature in feature_columns:
            feature_series = data[feature]
            
            # Skip non-numeric features for correlation
            if not pd.api.types.is_numeric_dtype(feature_series):
                continue
            
            # Calculate correlation
            try:
                # Remove missing values for correlation calculation
                clean_data = pd.concat([feature_series, target_series], axis=1).dropna()
                if len(clean_data) < 10:  # Too few samples
                    continue
                
                correlation = clean_data.corr().iloc[0, 1]
                
                if abs(correlation) > threshold:
                    suspicious_features.append({
                        'feature': feature,
                        'correlation': correlation,
                        'suspicion_level': 'high' if abs(correlation) > 0.98 else 'medium'
                    })
                    
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {feature}: {e}")
        
        # Perfect prediction check (for categorical features)
        for feature in feature_columns:
            feature_series = data[feature]
            
            # Check if feature perfectly predicts target
            try:
                contingency_table = pd.crosstab(feature_series, target_series)
                
                # Check if each feature value maps to only one target value
                perfect_mapping = True
                for idx in contingency_table.index:
                    non_zero_counts = (contingency_table.loc[idx] > 0).sum()
                    if non_zero_counts > 1:
                        perfect_mapping = False
                        break
                
                if perfect_mapping and len(contingency_table) > 1:
                    suspicious_features.append({
                        'feature': feature,
                        'correlation': 1.0,
                        'suspicion_level': 'high',
                        'reason': 'perfect_prediction'
                    })
                    
            except Exception:
                pass  # Skip features that can't be analyzed
        
        # Time-based leakage detection
        if time_column and time_column in data.columns:
            try:
                time_series = pd.to_datetime(data[time_column])
                
                # Check for future information
                future_features = []
                for feature in feature_columns:
                    # Simple heuristic: check if feature name suggests future information
                    if any(keyword in feature.lower() for keyword in ['future', 'next', 'after', 'post', 'later']):
                        future_features.append(feature)
                
                if future_features:
                    leakage_report['issues'].extend([
                        f"Feature name suggests future information: '{feature}'" for feature in future_features
                    ])
                    
            except Exception as e:
                logger.warning(f"Could not analyze time-based leakage: {e}")
        
        # Report findings
        if suspicious_features:
            high_suspicion = [f for f in suspicious_features if f['suspicion_level'] == 'high']
            medium_suspicion = [f for f in suspicious_features if f['suspicion_level'] == 'medium']
            
            if high_suspicion:
                leakage_report['passed'] = False
                leakage_report['issues'].extend([
                    f"High suspicion data leakage: '{f['feature']}' (correlation: {f['correlation']:.3f})"
                    for f in high_suspicion
                ])
            
            if medium_suspicion:
                leakage_report['issues'].extend([
                    f"Potential data leakage: '{f['feature']}' (correlation: {f['correlation']:.3f})"
                    for f in medium_suspicion
                ])
        
        leakage_report['details'] = {
            'suspicious_features': suspicious_features,
            'total_features_analyzed': len(feature_columns),
            'correlation_threshold': threshold
        }
        
        logger.info(f"Data leakage detection completed: {len(suspicious_features)} suspicious features found")
        return leakage_report
    
    def analyze_sample_balance(
        self,
        data: pd.DataFrame,
        target_column: str,
        categorical_features: Optional[List[str]] = None,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Analyze sample balance and distribution.
        
        Args:
            data: Input dataset
            target_column: Name of target column
            categorical_features: List of categorical features to analyze
            task_type: Type of ML task ('classification', 'regression', 'auto')
            
        Returns:
            Sample balance analysis report
        """
        balance_report = {
            'check_name': 'sample_balance',
            'passed': True,
            'issues': [],
            'details': {}
        }
        
        if target_column not in data.columns:
            balance_report['passed'] = False
            balance_report['issues'].append(f"Target column '{target_column}' not found")
            return balance_report
        
        target_series = data[target_column]
        
        # Infer task type if auto
        if task_type == 'auto':
            if pd.api.types.is_numeric_dtype(target_series):
                if target_series.nunique() <= 10:
                    inferred_task = 'classification'
                else:
                    inferred_task = 'regression'
            else:
                inferred_task = 'classification'
            task_type = inferred_task
        
        balance_report['details']['task_type'] = task_type
        
        # Task-specific analysis
        if task_type == 'classification':
            # Classification: analyze class distribution
            target_distribution = target_series.value_counts(normalize=True).to_dict()
            balance_report['details']['target_distribution'] = target_distribution
            
            # Identify minority classes
            min_class_ratio = min(target_distribution.values())
            minority_classes = [k for k, v in target_distribution.items() if v == min_class_ratio]
            
            balance_report['details']['minority_class_ratio'] = min_class_ratio
            balance_report['details']['minority_classes'] = minority_classes
            
            # Check for severe imbalance
            if min_class_ratio < self.validation_rules['min_class_ratio']:
                balance_report['passed'] = False
                balance_report['issues'].append(
                    f"Severe class imbalance: minority class ratio {min_class_ratio:.3f}"
                )
            elif min_class_ratio < 0.1:
                balance_report['issues'].append(
                    f"Moderate class imbalance: minority class ratio {min_class_ratio:.3f}"
                )
        
        elif task_type == 'regression':
            # Regression: analyze target distribution statistics
            target_clean = target_series.dropna()
            balance_report['details']['target_statistics'] = {
                'count': len(target_clean),
                'mean': float(target_clean.mean()),
                'std': float(target_clean.std()),
                'min': float(target_clean.min()),
                'max': float(target_clean.max()),
                '25th_percentile': float(target_clean.quantile(0.25)),
                '50th_percentile': float(target_clean.quantile(0.50)),
                '75th_percentile': float(target_clean.quantile(0.75)),
                'skewness': float(target_clean.skew()),
                'kurtosis': float(target_clean.kurtosis())
            }
            
            # Check for outliers (values beyond 3 standard deviations)
            mean_val = target_clean.mean()
            std_val = target_clean.std()
            outliers = target_clean[(target_clean < mean_val - 3*std_val) | (target_clean > mean_val + 3*std_val)]
            outlier_ratio = len(outliers) / len(target_clean)
            
            balance_report['details']['outlier_ratio'] = outlier_ratio
            
            if outlier_ratio > 0.05:  # More than 5% outliers
                balance_report['issues'].append(
                    f"High outlier ratio in target variable: {outlier_ratio:.1%}"
                )
            
            # No minority class concept for regression
            balance_report['details']['minority_class_ratio'] = None
            balance_report['details']['minority_classes'] = None
        
        # Analyze balance within categorical features
        if categorical_features:
            categorical_balance = {}
            
            for cat_feature in categorical_features:
                if cat_feature not in data.columns:
                    continue
                
                # Cross-tabulation
                try:
                    crosstab = pd.crosstab(data[cat_feature], target_series, normalize='index')
                    
                    # Check for categories with severe imbalance
                    imbalanced_categories = []
                    for category in crosstab.index:
                        category_distribution = crosstab.loc[category]
                        min_ratio = category_distribution.min()
                        
                        if min_ratio < self.validation_rules['min_class_ratio']:
                            imbalanced_categories.append({
                                'category': category,
                                'min_ratio': min_ratio,
                                'distribution': category_distribution.to_dict()
                            })
                    
                    categorical_balance[cat_feature] = {
                        'total_categories': len(crosstab),
                        'imbalanced_categories': imbalanced_categories
                    }
                    
                    if imbalanced_categories:
                        balance_report['issues'].append(
                            f"Imbalanced categories in '{cat_feature}': {len(imbalanced_categories)} categories"
                        )
                        
                except Exception as e:
                    logger.warning(f"Could not analyze balance for {cat_feature}: {e}")
            
            balance_report['details']['categorical_balance'] = categorical_balance
        
        # Sample size recommendations (only for classification)
        total_samples = len(data)
        balance_report['details']['total_samples'] = total_samples
        
        if task_type == 'classification':
            min_samples_per_class = total_samples * min_class_ratio
            
            if min_samples_per_class < 30:
                balance_report['issues'].append(
                    f"Insufficient samples for minority class: {min_samples_per_class:.0f} samples"
                )
            
            balance_report['details']['min_samples_per_class'] = min_samples_per_class
        else:
            # For regression, just note the total sample size
            if total_samples < 100:
                balance_report['issues'].append(
                    f"Small sample size for regression: {total_samples} samples"
                )
            balance_report['details']['min_samples_per_class'] = None
        
        if task_type == 'classification':
            logger.info(f"Sample balance analysis completed: minority class ratio {min_class_ratio:.3f}")
        else:
            logger.info(f"Sample balance analysis completed for regression: {total_samples} samples analyzed")
        return balance_report
    
    def calculate_data_quality_score(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall data quality score based on validation results.
        
        Args:
            validation_results: List of validation check results
            
        Returns:
            Data quality score and breakdown
        """
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results if result['passed'])
        
        # Base score from passed checks
        base_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Penalty for critical issues
        critical_penalties = 0
        warning_count = 0
        
        for result in validation_results:
            for issue in result['issues']:
                issue_lower = issue.lower()
                
                # Critical issues (major penalties)
                if any(keyword in issue_lower for keyword in 
                       ['severe', 'critical', 'high suspicion', 'constant', 'single class']):
                    critical_penalties += 10
                
                # Warning issues (minor penalties)
                elif any(keyword in issue_lower for keyword in 
                        ['moderate', 'potential', 'missing', 'imbalance']):
                    warning_count += 1
        
        # Apply penalties
        penalty_score = critical_penalties + (warning_count * 2)
        final_score = max(0, base_score - penalty_score)
        
        quality_assessment = {
            'overall_score': final_score,
            'score_breakdown': {
                'base_score': base_score,
                'critical_penalties': critical_penalties,
                'warning_penalties': warning_count * 2,
                'final_score': final_score
            },
            'quality_level': self._get_quality_level(final_score),
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'critical_issues': critical_penalties // 10,
            'warning_issues': warning_count
        }
        
        return quality_assessment
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 90:
            return 'Excellent'
        elif score >= 80:
            return 'Good'
        elif score >= 60:
            return 'Fair'
        else:
            return 'Poor'
    
    def validate_dataset(
        self,
        data: pd.DataFrame,
        target_column: Union[str, List[str]],
        task_type: str,
        test_data: Optional[pd.DataFrame] = None,
        validation_data: Optional[pd.DataFrame] = None,
        time_column: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        normalize_names: bool = True,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive dataset validation.
        
        Args:
            data: Training dataset
            target_column: Name of target column
            task_type: Type of ML task ('classification', 'regression')
            test_data: Test dataset (optional)
            validation_data: Validation dataset (optional)
            time_column: Name of time column (optional)
            categorical_features: List of categorical features (optional)
            normalize_names: Whether to normalize feature names
            model_directory: Directory to save visualizations (optional)
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive dataset validation")
        
        validation_results = []
        # Use deep copy to avoid categorical modification issues
        normalized_data = data.copy(deep=True)
        
        # Convert categorical columns to object type to avoid modification restrictions
        for col in normalized_data.columns:
            if hasattr(normalized_data[col], 'cat'):
                normalized_data[col] = normalized_data[col].astype(str)
        
        # 1. Feature name normalization
        if normalize_names:
            normalized_data, name_report = self.normalize_feature_names(normalized_data)
            validation_results.append(name_report)
        
        # 2. Dimension consistency checks
        dimension_report = self.check_data_dimensions(normalized_data, test_data, validation_data)
        validation_results.append(dimension_report)
        
        # 3. Target variable validation
        target_report = self.validate_target_variable(normalized_data, target_column, task_type)
        validation_results.append(target_report)
        
        # 4. Data leakage detection
        if isinstance(target_column, list):
            # For multi-target, check leakage for each target
            for idx, target in enumerate(target_column):
                leakage_report = self.detect_data_leakage(
                    normalized_data, target, time_column
                )
                leakage_report['check_name'] = f'data_leakage_{target}'
                validation_results.append(leakage_report)
        else:
            leakage_report = self.detect_data_leakage(
                normalized_data, target_column, time_column
            )
            validation_results.append(leakage_report)
        
        # 5. Sample balance analysis
        if isinstance(target_column, list):
            # For multi-target, analyze balance for each target
            for idx, target in enumerate(target_column):
                balance_report = self.analyze_sample_balance(
                    normalized_data, target, categorical_features, task_type
                )
                balance_report['check_name'] = f'sample_balance_{target}'
                validation_results.append(balance_report)
        else:
            balance_report = self.analyze_sample_balance(
                normalized_data, target_column, categorical_features, task_type
            )
            validation_results.append(balance_report)
        
        # 6. Feature correlation analysis
        # Use different thresholds based on correlation method (matching academic report standards)
        correlation_report = self.analyze_feature_correlations(
            normalized_data, target_column, task_type, categorical_features,
            correlation_threshold=0.7,  # Use 0.7 for Pearson/Spearman to match academic report High threshold
            model_directory=model_directory
        )
        validation_results.append(correlation_report)
        
        # 7. Multicollinearity detection (VIF analysis)
        multicollinearity_report = self.detect_multicollinearity(
            normalized_data, target_column, task_type, categorical_features,
            vif_threshold=5.0,  # Standard threshold for multicollinearity detection
            model_directory=model_directory
        )
        validation_results.append(multicollinearity_report)
        
        # 8. Feature distribution analysis
        distribution_report = self.analyze_feature_distributions(
            normalized_data, target_column, categorical_features, task_type, model_directory
        )
        validation_results.append(distribution_report)
        
        
        # 9. Calculate overall quality score
        quality_score = self.calculate_data_quality_score(validation_results)
        
        # Compile comprehensive report
        comprehensive_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': normalized_data.shape,
                'target_column': target_column,
                'is_multi_target': isinstance(target_column, list),
                'task_type': target_report['details'].get('inferred_task_type', task_type),
                'features_analyzed': len([col for col in normalized_data.columns 
                                         if col not in (target_column if isinstance(target_column, list) else [target_column])])
            },
            'validation_results': validation_results,
            'quality_assessment': quality_score,
            'recommendations': self._generate_recommendations(validation_results),
            'data_ready_for_training': quality_score['overall_score'] >= 60 and all(
                result['passed'] for result in validation_results 
                if result['check_name'] in ['data_dimensions'] or 
                result['check_name'].startswith('target_variable') or
                result['check_name'] == 'multi_target_variables'
            )
        }
        
        # Store results for later use
        self.validation_results = comprehensive_report
        
        logger.info(f"Dataset validation completed: Quality Score = {quality_score['overall_score']:.1f}/100")
        return comprehensive_report
    
    def _generate_recommendations(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        for result in validation_results:
            check_name = result['check_name']
            
            if not result['passed']:
                if check_name == 'data_dimensions':
                    recommendations.append("Ensure feature consistency across train/test/validation datasets")
                
                elif check_name == 'target_variable':
                    recommendations.append("Review target variable quality and consider data cleaning")
                
                elif check_name == 'data_leakage':
                    recommendations.append("Remove or investigate highly correlated features for potential data leakage")
                
                elif check_name == 'sample_balance':
                    recommendations.append("Consider techniques to handle class imbalance (sampling, weighting)")
                
                elif check_name == 'feature_correlations':
                    recommendations.append("Address multicollinearity through feature selection or regularization")
                
                elif check_name == 'multicollinearity_detection':
                    recommendations.append("Resolve multicollinearity using VIF-guided feature selection or regularization")
                
                elif check_name == 'feature_distributions':
                    recommendations.append("Address distribution issues through transformation or preprocessing")
            
            # Specific recommendations based on issues
            for issue in result['issues']:
                issue_lower = issue.lower()
                
                if 'missing' in issue_lower:
                    recommendations.append("Implement missing value handling strategy")
                
                if 'imbalance' in issue_lower or 'balance' in issue_lower:
                    recommendations.append("Apply balancing techniques (SMOTE, undersampling, class weights)")
                
                if 'correlation' in issue_lower:
                    recommendations.append("Investigate high correlations and consider feature selection")
                
                if 'skew' in issue_lower:
                    recommendations.append("Consider target transformation for heavily skewed targets")
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    def save_validation_report(
        self,
        report: Dict[str, Any],
        output_path: str = "validation_report.json"
    ) -> str:
        """
        Save validation report to file.
        
        Args:
            report: Validation report to save
            output_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"📄 Starting validation report save process...")
            logger.info(f"  Output path: {output_path}")
            logger.info(f"  Report type: {type(report)}")
            logger.info(f"  Report keys: {list(report.keys()) if isinstance(report, dict) else 'Not a dict'}")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, bool):
                    return obj
                return obj
            
            # Deep convert the report
            def deep_convert(data):
                try:
                    if isinstance(data, dict):
                        return {k: deep_convert(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [deep_convert(item) for item in data]
                    elif isinstance(data, Path):
                        return str(data)
                    else:
                        return convert_types(data)
                except Exception as e:
                    logger.error(f"Error converting data: {data}, Error: {e}")
                    raise
            
            logger.info("🔄 Converting report data for JSON serialization...")
            converted_report = deep_convert(report)
            logger.info("✅ Data conversion completed successfully")
            
            logger.info(f"💾 Writing to file: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Validation report saved successfully to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error in save_validation_report: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(f"   Exception args: {e.args}")
            import traceback
            logger.error(f"   Full traceback:\n{traceback.format_exc()}")
            raise
    
    def analyze_feature_correlations(
        self,
        data: pd.DataFrame,
        target_column: Union[str, List[str]],
        task_type: str,
        categorical_features: Optional[List[str]] = None,
        correlation_threshold: float = 0.95,
        include_target_correlation: bool = True,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feature correlations and potential multicollinearity.
        
        Args:
            data: Input dataset
            target_column: Name of target column or list of target columns
            task_type: Type of ML task
            categorical_features: List of categorical features
            correlation_threshold: Threshold for high correlation detection
            include_target_correlation: Whether to include target correlations
            model_directory: Directory to save visualizations
            
        Returns:
            Feature correlation analysis report
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"📊 Starting feature correlation analysis...")
            logger.info(f"  Target column: {target_column}")
            logger.info(f"  Target column type: {type(target_column)}")
            logger.info(f"  DataFrame shape: {data.shape}")
            logger.info(f"  DataFrame columns: {list(data.columns)}")
            logger.info(f"  Correlation threshold: {correlation_threshold}")
            
            # Verify target column exists
            if isinstance(target_column, list):
                missing_targets = [col for col in target_column if col not in data.columns]
                if missing_targets:
                    logger.error(f"❌ Missing target columns: {missing_targets}")
                    raise KeyError(f"Target columns not found: {missing_targets}")
            else:
                if target_column not in data.columns:
                    logger.error(f"❌ Target column '{target_column}' not found in columns: {list(data.columns)}")
                    raise KeyError(f"Target column '{target_column}' not found in data")
        
        except Exception as e:
            logger.error(f"❌ Error in analyze_feature_correlations: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(f"   Exception args: {e.args}")
            import traceback
            logger.error(f"   Full traceback:\n{traceback.format_exc()}")
            raise
        
        # Comprehensive feature correlation analysis considering task type and feature types.
        # This method analyzes correlations between:
        # 1. Continuous features (Pearson/Spearman correlation)
        # 2. Categorical features (Cramér's V, Chi-square test)
        # 3. Mixed features (Point-biserial correlation, ANOVA F-statistic)
        # 4. Features with target (appropriate method based on target and feature types)
        
        logger.info("Starting comprehensive feature correlation analysis")
        
        # Handle multi-target case - for correlation analysis, treat each target separately
        if isinstance(target_column, list):
            if len(target_column) > 1:
                logger.info(f"Multi-target correlation analysis: analyzing each of {len(target_column)} targets separately")
                # For now, use the first target for feature-feature correlations, 
                # but analyze all targets for feature-target correlations
                primary_target = target_column[0]
                all_targets = target_column
            else:
                primary_target = target_column[0]
                all_targets = target_column
        else:
            primary_target = target_column
            all_targets = [target_column]
        
        # Auto-detect categorical features if not provided
        if categorical_features is None:
            categorical_features = self._detect_categorical_features(data, exclude_target=all_targets)
        
        # Ensure targets are not in categorical features
        for target in all_targets:
            if target in categorical_features:
                categorical_features.remove(target)
        
        # Classify features
        feature_classification = self._classify_features(data, categorical_features, primary_target)
        
        correlation_report = {
            'check_name': 'feature_correlations',
            'passed': True,
            'issues': [],
            'details': {
                'task_type': task_type,
                'correlation_threshold': correlation_threshold,
                'feature_classification': feature_classification,
                'correlation_matrices': {},
                'high_correlations': [],
                'target_correlations': {},
                'correlation_summary': {}
            }
        }
        
        try:
            # 1. Continuous-Continuous correlations (Pearson/Spearman)
            if len(feature_classification['continuous']) > 1:
                cont_corr = self._analyze_continuous_correlations(
                    data, feature_classification['continuous'], correlation_threshold, model_directory
                )
                correlation_report['details']['correlation_matrices']['continuous_continuous'] = cont_corr
                correlation_report['details']['high_correlations'].extend(cont_corr['high_correlations'])
            
            # 2. Categorical-Categorical correlations (Cramér's V)
            if len(feature_classification['categorical']) > 1:
                cat_corr = self._analyze_categorical_correlations(
                    data, feature_classification['categorical'], correlation_threshold, model_directory
                )
                correlation_report['details']['correlation_matrices']['categorical_categorical'] = cat_corr
                correlation_report['details']['high_correlations'].extend(cat_corr['high_correlations'])
            
            # 3. Mixed correlations (Continuous-Categorical)
            if len(feature_classification['continuous']) > 0 and len(feature_classification['categorical']) > 0:
                mixed_corr = self._analyze_mixed_correlations(
                    data, feature_classification['continuous'], feature_classification['categorical'], correlation_threshold, model_directory
                )
                correlation_report['details']['correlation_matrices']['continuous_categorical'] = mixed_corr
                correlation_report['details']['high_correlations'].extend(mixed_corr['high_correlations'])
            
            # 4. Feature-Target correlations
            if include_target_correlation:
                if len(all_targets) == 1:
                    # Single target case
                    target_corr = self._analyze_target_correlations(
                        data, primary_target, feature_classification, task_type, model_directory
                    )
                    correlation_report['details']['target_correlations'] = target_corr
                else:
                    # Multi-target case: analyze each target separately
                    multi_target_corr = {}
                    for target in all_targets:
                        target_corr = self._analyze_target_correlations(
                            data, target, feature_classification, task_type, model_directory
                        )
                        multi_target_corr[target] = target_corr
                    correlation_report['details']['target_correlations'] = multi_target_corr
                    correlation_report['details']['is_multi_target'] = True
            
            # 5. Generate correlation summary
            correlation_report['details']['correlation_summary'] = self._generate_correlation_summary(
                correlation_report['details']
            )
            
            # 6. Check for issues
            total_high_correlations = len(correlation_report['details']['high_correlations'])
            if total_high_correlations > 0:
                correlation_report['issues'].append(
                    f"Found {total_high_correlations} high feature correlations (threshold: {correlation_threshold})"
                )
                if total_high_correlations > 5:
                    correlation_report['passed'] = False
                    correlation_report['issues'].append(
                        "Excessive multicollinearity detected - consider feature selection"
                    )
            
            logger.info(f"Feature correlation analysis completed: {total_high_correlations} high correlations found")
            
        except Exception as e:
            logger.error(f"Feature correlation analysis failed: {e}")
            correlation_report['passed'] = False
            correlation_report['issues'].append(f"Analysis failed: {str(e)}")
        
        return correlation_report
    
    
    def _detect_categorical_features(self, data: pd.DataFrame, exclude_target: Union[str, List[str]]) -> List[str]:
        """
        Auto-detect categorical features following the same logic as data preprocessing.
        
        Detection logic:
        - Non-numeric types (object, category) -> categorical
        - Numeric types with unique count <= 5 -> categorical 
        - All other numeric types -> continuous
        
        Args:
            data: Input dataframe
            exclude_target: Target column(s) to exclude from detection
            
        Returns:
            List of categorical feature names
        """
        categorical_features = []
        
        # Handle both single target and multi-target cases
        if isinstance(exclude_target, str):
            exclude_columns = {exclude_target}
        else:
            exclude_columns = set(exclude_target)
        
        for col in data.columns:
            if col in exclude_columns:
                continue
                
            # Following data preprocessing logic exactly
            if data[col].dtype in ['object', 'category']:
                categorical_features.append(col)
                logger.debug(f"Column '{col}' -> categorical (dtype: {data[col].dtype})")
            elif data[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32', 'float16']:
                # Additional check for low cardinality numeric features
                unique_count = data[col].nunique()
                if unique_count <= 5:
                    categorical_features.append(col)
                    logger.debug(f"Column '{col}' -> categorical (low cardinality: {unique_count})")
                else:
                    logger.debug(f"Column '{col}' -> continuous (dtype: {data[col].dtype}, unique: {unique_count})")
            else:
                # Fallback: treat unknown types as categorical (same as preprocessing)
                categorical_features.append(col)
                logger.warning(f"Column '{col}' has unknown dtype {data[col].dtype}, treating as categorical")
        
        return categorical_features
    
    def _classify_features(self, data: pd.DataFrame, categorical_features: List[str], target_column: Union[str, List[str]]) -> Dict[str, List[str]]:
        """Classify features into continuous and categorical, supporting multi-target."""
        # Handle both single target and multi-target cases
        if isinstance(target_column, str):
            target_columns = [target_column]
        else:
            target_columns = target_column
        
        all_features = [col for col in data.columns if col not in target_columns]
        continuous_features = [col for col in all_features if col not in categorical_features]
        
        return {
            'continuous': continuous_features,
            'categorical': categorical_features,
            'target': target_column  # Keep original format for backward compatibility
        }
    
    def _analyze_continuous_correlations(
        self, 
        data: pd.DataFrame, 
        continuous_features: List[str], 
        threshold: float,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between continuous features."""
        if len(continuous_features) < 2:
            return {'correlation_matrix': None, 'high_correlations': []}
        
        continuous_data = data[continuous_features]
        
        # Calculate both Pearson and Spearman correlations
        pearson_corr = continuous_data.corr(method='pearson')
        spearman_corr = continuous_data.corr(method='spearman')
        
        # Find high correlations using adaptive thresholds matching academic report
        high_correlations = []
        
        for i in range(len(continuous_features)):
            for j in range(i + 1, len(continuous_features)):
                feature1 = continuous_features[i]
                feature2 = continuous_features[j]
                
                pearson_val = abs(pearson_corr.iloc[i, j])
                spearman_val = abs(spearman_corr.iloc[i, j])
                
                # Use threshold=0.7 for "High" correlations (matching academic report)
                if pearson_val >= 0.7 or spearman_val >= 0.7:
                    high_correlations.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation_type': 'continuous-continuous',
                        'pearson_correlation': float(pearson_corr.iloc[i, j]),
                        'spearman_correlation': float(spearman_corr.iloc[i, j]),
                        'max_correlation': float(max(pearson_val, spearman_val))
                    })
        
        # Generate visualizations if enabled
        visualization_info = {}
        if self.enable_visualizations and self.visualizer is not None and model_directory is not None:
            try:
                visualization_info = self.visualizer.generate_continuous_correlation_plots(
                    pearson_corr, spearman_corr, continuous_features, model_directory, threshold
                )
            except Exception as e:
                logger.warning(f"Failed to generate continuous correlation plots: {e}")
                visualization_info = {'error': str(e)}
        
        return {
            'correlation_matrix': {
                'pearson': pearson_corr.to_dict(),
                'spearman': spearman_corr.to_dict()
            },
            'high_correlations': high_correlations,
            'method': 'pearson_spearman',
            'visualizations': visualization_info
        }
    
    def _analyze_categorical_correlations(
        self, 
        data: pd.DataFrame, 
        categorical_features: List[str], 
        threshold: float,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between categorical features using Cramér's V."""
        if len(categorical_features) < 2:
            return {'correlation_matrix': None, 'high_correlations': []}
        
        # Calculate Cramér's V matrix
        cramers_v_matrix = pd.DataFrame(
            index=categorical_features, 
            columns=categorical_features, 
            dtype=float
        )
        
        high_correlations = []
        
        for i, feature1 in enumerate(categorical_features):
            for j, feature2 in enumerate(categorical_features):
                if i == j:
                    cramers_v_matrix.iloc[i, j] = 1.0
                elif i < j:
                    # Calculate Cramér's V
                    cramers_v = self._calculate_cramers_v(data[feature1], data[feature2])
                    cramers_v_matrix.iloc[i, j] = cramers_v
                    cramers_v_matrix.iloc[j, i] = cramers_v
                    
                    # Use threshold=0.5 for "High" correlations (matching academic report)
                    if cramers_v >= 0.5:
                        high_correlations.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation_type': 'categorical-categorical',
                            'cramers_v': float(cramers_v),
                            'max_correlation': float(cramers_v)
                        })
        
        # Generate visualizations if enabled
        visualization_info = {}
        if self.enable_visualizations and self.visualizer is not None and model_directory is not None:
            try:
                visualization_info = self.visualizer.generate_categorical_correlation_plots(
                    cramers_v_matrix, categorical_features, model_directory, threshold
                )
            except Exception as e:
                logger.warning(f"Failed to generate categorical correlation plots: {e}")
                visualization_info = {'error': str(e)}
        
        return {
            'correlation_matrix': cramers_v_matrix.to_dict(),
            'high_correlations': high_correlations,
            'method': 'cramers_v',
            'visualizations': visualization_info
        }
    
    def _analyze_mixed_correlations(
        self, 
        data: pd.DataFrame, 
        continuous_features: List[str], 
        categorical_features: List[str], 
        threshold: float,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between continuous and categorical features."""
        correlation_matrix = pd.DataFrame(
            index=continuous_features, 
            columns=categorical_features, 
            dtype=float
        )
        
        high_correlations = []
        
        for cont_feature in continuous_features:
            for cat_feature in categorical_features:
                # Use correlation ratio (eta-squared) for continuous-categorical
                correlation_value = self._calculate_correlation_ratio(
                    data[cont_feature], data[cat_feature]
                )
                correlation_matrix.loc[cont_feature, cat_feature] = correlation_value
                
                # Use threshold=0.14 for "Large Effect" (matching academic report)
                if correlation_value >= 0.14:
                    high_correlations.append({
                        'feature1': cont_feature,
                        'feature2': cat_feature,
                        'correlation_type': 'continuous-categorical',
                        'correlation_ratio': float(correlation_value),
                        'max_correlation': float(correlation_value)
                    })
        
        # Generate visualizations if enabled
        visualization_info = {}
        if self.enable_visualizations and self.visualizer is not None and model_directory is not None:
            try:
                visualization_info = self.visualizer.generate_mixed_correlation_plots(
                    correlation_matrix, continuous_features, categorical_features, model_directory, threshold
                )
            except Exception as e:
                logger.warning(f"Failed to generate mixed correlation plots: {e}")
                visualization_info = {'error': str(e)}
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'method': 'correlation_ratio',
            'visualizations': visualization_info
        }
    
    def _analyze_target_correlations(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_classification: Dict[str, List[str]], 
        task_type: str,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between features and target variable."""
        target_correlations = {
            'continuous_features': {},
            'categorical_features': {},
            'task_type': task_type
        }
        
        target_series = data[target_column]
        target_is_categorical = target_column in feature_classification['categorical'] or task_type == 'classification'
        
        # Continuous features with target
        for feature in feature_classification['continuous']:
            if target_is_categorical:
                # Continuous feature vs categorical target (Point-biserial or ANOVA F-statistic)
                correlation_value = self._calculate_correlation_ratio(data[feature], target_series)
                target_correlations['continuous_features'][feature] = {
                    'method': 'correlation_ratio',
                    'value': float(correlation_value)
                }
            else:
                # Continuous feature vs continuous target (Pearson/Spearman)
                pearson_corr, _ = pearsonr(data[feature].dropna(), target_series.dropna())
                target_correlations['continuous_features'][feature] = {
                    'method': 'pearson',
                    'value': float(pearson_corr)
                }
        
        # Categorical features with target
        for feature in feature_classification['categorical']:
            if target_is_categorical:
                # Categorical feature vs categorical target (Cramér's V)
                cramers_v = self._calculate_cramers_v(data[feature], target_series)
                target_correlations['categorical_features'][feature] = {
                    'method': 'cramers_v',
                    'value': float(cramers_v)
                }
            else:
                # Categorical feature vs continuous target (ANOVA F-statistic)
                correlation_value = self._calculate_correlation_ratio(target_series, data[feature])
                target_correlations['categorical_features'][feature] = {
                    'method': 'correlation_ratio',
                    'value': float(correlation_value)
                }
        
        # Generate visualizations if enabled
        visualization_info = {}
        if self.enable_visualizations and self.visualizer is not None and model_directory is not None:
            try:
                visualization_info = self.visualizer.generate_feature_target_correlation_plots(
                    target_correlations, target_column, model_directory, task_type
                )
            except Exception as e:
                logger.warning(f"Failed to generate feature-target correlation plots: {e}")
                visualization_info = {'error': str(e)}
        
        target_correlations['visualizations'] = visualization_info
        return target_correlations
    
    def _calculate_cramers_v(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate Cramér's V statistic for categorical variables."""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(series1, series2)
            
            # Perform chi-square test
            chi2, _, _, _ = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            
            if min_dim == 0:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return min(cramers_v, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def _calculate_correlation_ratio(self, continuous_var: pd.Series, categorical_var: pd.Series) -> float:
        """Calculate correlation ratio (eta-squared) between continuous and categorical variables."""
        try:
            # Remove missing values
            mask = ~(continuous_var.isna() | categorical_var.isna())
            cont_clean = continuous_var[mask]
            cat_clean = categorical_var[mask]
            
            if len(cont_clean) < 2:
                return 0.0
            
            # Calculate overall mean
            overall_mean = cont_clean.mean()
            overall_variance = cont_clean.var()
            
            if overall_variance == 0:
                return 0.0
            
            # Calculate between-group variance
            grouped = cont_clean.groupby(cat_clean, observed=False)
            group_means = grouped.mean()
            group_sizes = grouped.size()
            
            between_variance = sum(
                size * (mean - overall_mean) ** 2 
                for mean, size in zip(group_means, group_sizes)
            ) / len(cont_clean)
            
            # Calculate eta-squared (correlation ratio)
            eta_squared = between_variance / overall_variance
            return min(eta_squared, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0
    
    def _generate_correlation_summary(self, correlation_details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of correlation analysis."""
        summary = {
            'total_high_correlations': len(correlation_details['high_correlations']),
            'correlation_types': {},
            'strongest_correlations': [],
            'recommendations': []
        }
        
        # Count by correlation type
        for corr in correlation_details['high_correlations']:
            corr_type = corr['correlation_type']
            summary['correlation_types'][corr_type] = summary['correlation_types'].get(corr_type, 0) + 1
        
        # Find strongest correlations (top 5)
        sorted_correlations = sorted(
            correlation_details['high_correlations'],
            key=lambda x: x['max_correlation'],
            reverse=True
        )
        summary['strongest_correlations'] = sorted_correlations[:5]
        
        # Generate recommendations
        if summary['total_high_correlations'] > 0:
            summary['recommendations'].append("Consider feature selection to reduce multicollinearity")
            
            if summary['correlation_types'].get('continuous-continuous', 0) > 0:
                summary['recommendations'].append("Use PCA or remove redundant continuous features")
                
            if summary['correlation_types'].get('categorical-categorical', 0) > 0:
                summary['recommendations'].append("Combine or remove redundant categorical features")
                
            if summary['correlation_types'].get('continuous-categorical', 0) > 0:
                summary['recommendations'].append("Consider feature engineering to reduce mixed correlations")
        
        return summary 

    def detect_multicollinearity(
        self,
        data: pd.DataFrame,
        target_column: Union[str, List[str]],
        task_type: str,
        categorical_features: Optional[List[str]] = None,
        vif_threshold: float = 5.0,
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect multicollinearity using Variance Inflation Factor (VIF) analysis.
        
        Args:
            data: DataFrame to analyze
            target_column: Target column(s) to exclude
            task_type: Type of ML task ('classification', 'regression')
            categorical_features: List of categorical features (optional)
            vif_threshold: VIF threshold for multicollinearity detection (default: 5.0)
            model_directory: Directory to save VIF visualizations (optional)
            
        Returns:
            Multicollinearity detection report
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import warnings
        warnings.filterwarnings('ignore')
        
        # Handle multi-target case
        if isinstance(target_column, list):
            all_targets = target_column
            primary_target = target_column[0]
        else:
            all_targets = [target_column]
            primary_target = target_column
        
        # Auto-detect categorical features if not provided
        if categorical_features is None:
            categorical_features = self._detect_categorical_features(data, exclude_target=all_targets)
        
        # Get feature columns (exclude targets)
        feature_columns = [col for col in data.columns if col not in all_targets]
        
        if len(feature_columns) < 2:
            return {
                'check_name': 'multicollinearity_detection',
                'passed': True,
                'status': 'PASSED',
                'issues': [],
                'details': {
                    'message': 'Insufficient features for multicollinearity analysis',
                    'vif_scores': {},
                    'high_vif_features': [],
                    'multicollinear_pairs': []
                }
            }
        
        # Prepare data for VIF calculation
        analysis_data = data[feature_columns].copy()
        
        # Check for high cardinality categorical features and exclude them
        excluded_features = []
        high_cardinality_features = []
        
        for col in categorical_features:
            if col in analysis_data.columns:
                unique_count = analysis_data[col].nunique()
                total_count = len(analysis_data[col].dropna())
                cardinality_ratio = unique_count / total_count if total_count > 0 else 0
                
                # Exclude features with very high cardinality (>80% unique values or >50 unique categories)
                if cardinality_ratio > 0.8 or unique_count > 50:
                    excluded_features.append(col)
                    high_cardinality_features.append({
                        'feature': col,
                        'unique_count': unique_count,
                        'total_count': total_count,
                        'cardinality_ratio': cardinality_ratio
                    })
                    analysis_data = analysis_data.drop(columns=[col])
        
        # Update feature columns list
        feature_columns = [col for col in feature_columns if col not in excluded_features]
        
        if len(feature_columns) < 2:
            return {
                'check_name': 'multicollinearity_detection',
                'passed': True,
                'status': 'PASSED',
                'issues': [],
                'details': {
                    'message': 'Insufficient features for multicollinearity analysis after excluding high cardinality features',
                    'vif_scores': {},
                    'high_vif_features': [],
                    'multicollinear_pairs': [],
                    'excluded_features': excluded_features,
                    'high_cardinality_features': high_cardinality_features
                }
            }
        
        # Encode categorical features using different strategies based on cardinality
        encoding_details = {}
        low_cardinality_threshold = 10  # Configurable threshold for one-hot vs target encoding
        
        for col in categorical_features:
            if col in analysis_data.columns:
                unique_count = analysis_data[col].nunique()
                
                try:
                    if unique_count <= low_cardinality_threshold:
                        # Use One-hot encoding for low cardinality features
                        encoding_details[col] = {
                            'method': 'one_hot',
                            'unique_count': unique_count,
                            'dummy_columns': []
                        }
                        
                        # Create dummy variables, drop first to avoid multicollinearity
                        dummies = pd.get_dummies(analysis_data[col], prefix=col, drop_first=True)
                        encoding_details[col]['dummy_columns'] = list(dummies.columns)
                        
                        # Drop original column and add dummy columns
                        analysis_data = analysis_data.drop(columns=[col])
                        analysis_data = pd.concat([analysis_data, dummies], axis=1)
                        
                        # Update feature columns list
                        feature_columns = [f for f in feature_columns if f != col] + list(dummies.columns)
                        
                    else:
                        # Use Target encoding for medium cardinality features
                        encoding_details[col] = {
                            'method': 'target_encoding',
                            'unique_count': unique_count
                        }
                        
                        target_mean = data.groupby(col)[primary_target].mean()
                        
                        # Handle missing values in categorical column
                        non_null_mask = analysis_data[col].notna()
                        if non_null_mask.sum() > 0:
                            # Map categories to target means
                            analysis_data.loc[non_null_mask, col] = analysis_data.loc[non_null_mask, col].map(target_mean)
                            
                            # Fill remaining missing values with overall target mean
                            overall_mean = data[primary_target].mean()
                            analysis_data[col] = analysis_data[col].fillna(overall_mean)
                        else:
                            # If all values are missing, use overall target mean
                            overall_mean = data[primary_target].mean()
                            analysis_data[col] = overall_mean
                        
                except Exception as e:
                    # Fallback: use simple label encoding if other methods fail
                    logger.warning(f"Encoding failed for {col}, using label encoding: {e}")
                    encoding_details[col] = {
                        'method': 'label_encoding_fallback',
                        'unique_count': unique_count,
                        'error': str(e)
                    }
                    
                    # Ensure column still exists before applying label encoding
                    if col in analysis_data.columns:
                        le = LabelEncoder()
                        non_null_mask = analysis_data[col].notna()
                        if non_null_mask.sum() > 0:
                            if hasattr(analysis_data[col], 'cat'):
                                analysis_data[col] = analysis_data[col].astype(str)
                            analysis_data.loc[non_null_mask, col] = le.fit_transform(analysis_data.loc[non_null_mask, col])
                        else:
                            analysis_data[col] = 0
                    else:
                        # Column was already processed/deleted, just mark as processed
                        logger.warning(f"Column {col} was already processed or removed during encoding")
        
        # Fill remaining missing values
        analysis_data = analysis_data.fillna(analysis_data.mean())
        
        # Calculate VIF scores with improved handling for one-hot encoded features
        vif_scores = {}
        categorical_vif_scores = {}  # Store VIF info for original categorical features
        high_vif_features = []
        multicollinear_pairs = []
        
        try:
            # Calculate VIF for all individual features first
            individual_vifs = {}
            for feature in feature_columns:
                if feature not in analysis_data.columns:
                    continue
                    
                # Get other features as predictors
                other_features = [f for f in feature_columns if f != feature and f in analysis_data.columns]
                
                if len(other_features) == 0:
                    individual_vifs[feature] = 1.0
                    continue
                
                X = analysis_data[other_features]
                y = analysis_data[feature]
                
                # Fit linear regression
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    # Calculate R-squared
                    r2 = r2_score(y, y_pred)
                    
                    # Calculate VIF
                    if r2 >= 0.999:  # Very high R-squared, set high VIF
                        vif = 1000.0
                    else:
                        vif = 1 / (1 - r2)
                    
                    individual_vifs[feature] = float(vif)
                        
                except Exception as e:
                    # If regression fails, set VIF to 1 (no multicollinearity)
                    individual_vifs[feature] = 1.0
            
            # Process VIF scores: aggregate one-hot encoded features
            for original_col, encoding_info in encoding_details.items():
                if encoding_info['method'] == 'one_hot':
                    # For one-hot encoded features, calculate aggregated VIF
                    dummy_columns = encoding_info['dummy_columns']
                    dummy_vifs = [individual_vifs.get(col, 1.0) for col in dummy_columns if col in individual_vifs]
                    
                    if dummy_vifs:
                        # Use multiple aggregation methods for better representation
                        max_vif = max(dummy_vifs)
                        mean_vif = sum(dummy_vifs) / len(dummy_vifs)
                        
                        # Store detailed information
                        categorical_vif_scores[original_col] = {
                            'method': 'one_hot_aggregated',
                            'max_vif': max_vif,
                            'mean_vif': mean_vif,
                            'individual_vifs': dict(zip(dummy_columns, dummy_vifs)),
                            'dummy_count': len(dummy_columns)
                        }
                        
                        # Use max VIF as representative (conservative approach)
                        vif_scores[original_col] = max_vif
                        
                        # Check if any dummy variable exceeds threshold
                        if max_vif >= vif_threshold:
                            high_vif_features.append({
                                'feature': original_col,
                                'vif_score': float(max_vif),
                                'r_squared': float(1 - 1/max_vif) if max_vif > 1 else 0.0,
                                'encoding_method': 'one_hot',
                                'dummy_details': categorical_vif_scores[original_col]
                            })
                    else:
                        vif_scores[original_col] = 1.0
                else:
                    # For target encoded features, use individual VIF
                    if original_col in individual_vifs:
                        vif_scores[original_col] = individual_vifs[original_col]
                        
                        if individual_vifs[original_col] >= vif_threshold:
                            high_vif_features.append({
                                'feature': original_col,
                                'vif_score': float(individual_vifs[original_col]),
                                'r_squared': float(1 - 1/individual_vifs[original_col]) if individual_vifs[original_col] > 1 else 0.0,
                                'encoding_method': encoding_info['method']
                            })
            
            # Add continuous features
            for feature in feature_columns:
                # Check if this is a continuous feature (not a dummy variable from one-hot encoding)
                is_dummy = any(feature in info.get('dummy_columns', []) for info in encoding_details.values())
                is_original_categorical = feature in encoding_details
                
                if not is_dummy and not is_original_categorical and feature in individual_vifs:
                    vif_scores[feature] = individual_vifs[feature]
                    
                    if individual_vifs[feature] >= vif_threshold:
                        high_vif_features.append({
                            'feature': feature,
                            'vif_score': float(individual_vifs[feature]),
                            'r_squared': float(1 - 1/individual_vifs[feature]) if individual_vifs[feature] > 1 else 0.0,
                            'encoding_method': 'continuous'
                        })
                    
        except Exception as e:
            return {
                'check_name': 'multicollinearity_detection',
                'passed': False,
                'status': 'ERROR',
                'issues': [f'VIF calculation failed: {str(e)}'],
                'details': {
                    'error': str(e),
                    'vif_scores': {},
                    'high_vif_features': [],
                    'multicollinear_pairs': [],
                    'excluded_features': excluded_features,
                    'high_cardinality_features': high_cardinality_features
                }
            }
        
        # Use existing correlation analysis to find multicollinear pairs
        try:
            correlation_result = self.analyze_feature_correlations(
                data, target_column, task_type=task_type, 
                categorical_features=categorical_features,
                correlation_threshold=0.7,  # Use 0.7 for Pearson/Spearman to match academic report High threshold
                include_target_correlation=False
            )
            multicollinear_pairs = correlation_result['details']['high_correlations']
        except:
            multicollinear_pairs = []
        
        # Determine if multicollinearity is detected
        issues = []
        passed = True
        
        if len(high_vif_features) > 0:
            passed = False
            issues.append(f"Detected {len(high_vif_features)} features with high VIF (>= {vif_threshold})")
        
        if len(multicollinear_pairs) > 0:
            issues.append(f"Found {len(multicollinear_pairs)} highly correlated feature pairs")
        
        # Add issues for excluded high cardinality features
        if len(excluded_features) > 0:
            issues.append(f"Excluded {len(excluded_features)} high cardinality features from VIF analysis")
        
        # Generate detailed report
        details = {
            'vif_threshold': vif_threshold,
            'low_cardinality_threshold': low_cardinality_threshold,
            'total_features_analyzed': len([f for f in feature_columns if not any(f in info.get('dummy_columns', []) for info in encoding_details.values())]),
            'excluded_features': excluded_features,
            'high_cardinality_features': high_cardinality_features,
            'encoding_details': encoding_details,
            'categorical_vif_details': categorical_vif_scores,
            'vif_scores': vif_scores,
            'high_vif_features': sorted(high_vif_features, key=lambda x: x['vif_score'], reverse=True),
            'multicollinear_pairs': multicollinear_pairs,
            'average_vif': float(np.mean(list(vif_scores.values()))) if vif_scores else 0.0,
            'max_vif': float(max(vif_scores.values())) if vif_scores else 0.0,
            'recommendations': []
        }
        
        # Generate recommendations for high cardinality features
        if len(high_cardinality_features) > 0:
            details['recommendations'].extend([
                f"Consider removing high cardinality features: {', '.join(excluded_features)}",
                "High cardinality features (>80% unique values or >50 categories) can cause overfitting",
                "For ID-like features with unique values per row, removal is strongly recommended"
            ])
        
        # Generate encoding-specific recommendations
        one_hot_features = [col for col, info in encoding_details.items() if info['method'] == 'one_hot']
        target_encoded_features = [col for col, info in encoding_details.items() if info['method'] == 'target_encoding']
        
        if one_hot_features:
            details['recommendations'].append(
                f"One-hot encoded features (≤{low_cardinality_threshold} categories): {', '.join(one_hot_features)}"
            )
        
        if target_encoded_features:
            details['recommendations'].append(
                f"Target encoded features (>{low_cardinality_threshold} categories): {', '.join(target_encoded_features)}"
            )
        
        # Generate recommendations for multicollinearity
        if len(high_vif_features) > 0:
            # Separate recommendations for different encoding types
            one_hot_high_vif = [f['feature'] for f in high_vif_features if f.get('encoding_method') == 'one_hot']
            continuous_high_vif = [f['feature'] for f in high_vif_features if f.get('encoding_method') == 'continuous']
            
            if one_hot_high_vif:
                details['recommendations'].append(
                    f"One-hot encoded features with high VIF (consider merging categories): {', '.join(one_hot_high_vif)}"
                )
            
            if continuous_high_vif:
                details['recommendations'].append(
                    f"Continuous features with high VIF (consider removal): {', '.join(continuous_high_vif)}"
                )
            
            details['recommendations'].extend([
                "Apply dimensionality reduction techniques (PCA, ICA) for high VIF features",
                "Use regularized models (Ridge, Lasso, Elastic Net) to handle multicollinearity"
            ])
        
        if len(multicollinear_pairs) > 0:
            details['recommendations'].extend([
                "Remove one feature from highly correlated pairs",
                "Combine correlated features through feature engineering"
            ])
        
        # Generate VIF visualizations if model_directory is provided and visualizations are enabled
        vif_visualization_result = None
        if model_directory and self.enable_visualizations:
            try:
                # Prepare VIF results for visualization
                vif_plot_data = {
                    'vif_scores': vif_scores,
                    'high_vif_features': high_vif_features,
                    'categorical_vif_details': categorical_vif_scores,
                    'encoding_details': encoding_details,
                    'vif_threshold': vif_threshold
                }
                
                # Generate VIF plots
                from .visualization_generator import DataValidationVisualizer
                visualizer = DataValidationVisualizer()
                vif_visualization_result = visualizer.generate_vif_plots(
                    vif_plot_data, model_directory
                )
                
                # Add visualization info to details
                details['vif_visualization'] = vif_visualization_result
                
            except Exception as e:
                logger.warning(f"Failed to generate VIF visualizations: {e}")
                details['vif_visualization'] = {
                    'error': str(e),
                    'plots_generated': []
                }
        
        return {
            'check_name': 'multicollinearity_detection',
            'passed': passed,
            'status': 'PASSED' if passed else 'WARNING',
            'issues': issues,
            'details': details
        }

    def analyze_feature_distributions(
        self,
        data: pd.DataFrame,
        target_column: Union[str, List[str]],
        categorical_features: Optional[List[str]] = None,
        task_type: str = 'classification',
        model_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feature distributions to detect potential data quality issues.
        
        Args:
            data: DataFrame to analyze
            target_column: Target column(s) to exclude
            categorical_features: List of categorical features (optional)
            
        Returns:
            Feature distribution analysis report
        """
        from scipy import stats
        from scipy.stats import jarque_bera, shapiro
        import warnings
        warnings.filterwarnings('ignore')
        
        # Handle multi-target case
        if isinstance(target_column, list):
            all_targets = target_column
        else:
            all_targets = [target_column]
        
        # Auto-detect categorical features if not provided (reuse existing function)
        if categorical_features is None:
            categorical_features = self._detect_categorical_features(data, exclude_target=all_targets)
        
        # Use existing feature classification function
        feature_classification = self._classify_features(data, categorical_features, target_column)
        continuous_features = feature_classification['continuous']
        categorical_features = feature_classification['categorical']
        
        # Initialize report
        distribution_report = {
            'check_name': 'feature_distributions',
            'passed': True,
            'status': 'PASSED',
            'issues': [],
            'details': {
                'continuous_distributions': {},
                'categorical_distributions': {},
                'distribution_issues': [],
                'outlier_detection': {},
                'summary_statistics': {}
            }
        }
        
        # Analyze continuous features
        if continuous_features:
            distribution_report['details']['continuous_distributions'] = self._analyze_continuous_distributions(
                data, continuous_features
            )
            
            # Extract issues from continuous analysis
            cont_analysis = distribution_report['details']['continuous_distributions']
            if cont_analysis.get('issues'):
                distribution_report['issues'].extend(cont_analysis['issues'])
                if any('severe' in issue.lower() or 'extreme' in issue.lower() 
                      for issue in cont_analysis['issues']):
                    distribution_report['passed'] = False
                    distribution_report['status'] = 'WARNING'
        
        # Analyze categorical features
        if categorical_features:
            distribution_report['details']['categorical_distributions'] = self._analyze_categorical_distributions(
                data, categorical_features
            )
            
            # Extract issues from categorical analysis
            cat_analysis = distribution_report['details']['categorical_distributions']
            if cat_analysis.get('issues'):
                distribution_report['issues'].extend(cat_analysis['issues'])
                if any('severe' in issue.lower() or 'extreme' in issue.lower() 
                      for issue in cat_analysis['issues']):
                    distribution_report['passed'] = False
                    distribution_report['status'] = 'WARNING'
        
        # Generate overall summary
        distribution_report['details']['analysis_summary'] = self._generate_distribution_summary(
            distribution_report['details']
        )
        
        # Generate visualizations if enabled and model directory is provided
        if self.visualizer and model_directory:
            try:
                continuous_analysis = distribution_report['details'].get('continuous_distributions', {})
                categorical_analysis = distribution_report['details'].get('categorical_distributions', {})
                
                viz_result = self.visualizer.generate_feature_distribution_plots(
                    data, continuous_analysis, categorical_analysis,
                    target_column, task_type, model_directory
                )
                
                if 'error' not in viz_result:
                    distribution_report['details']['visualizations'] = viz_result
                    logger.info(f"Generated {len(viz_result.get('plots_generated', []))} feature distribution visualizations")
                else:
                    logger.warning(f"Failed to generate feature distribution visualizations: {viz_result['error']}")
                    
            except Exception as e:
                logger.error(f"Error generating feature distribution visualizations: {e}")
        
        return distribution_report
    
    def _analyze_continuous_distributions(
        self,
        data: pd.DataFrame,
        continuous_features: List[str]
    ) -> Dict[str, Any]:
        """Analyze distributions of continuous features."""
        from scipy import stats
        from scipy.stats import jarque_bera, shapiro, normaltest
        
        analysis = {
            'features_analyzed': len(continuous_features),
            'normality_tests': {},
            'skewness_analysis': {},
            'outlier_detection': {},
            'distribution_statistics': {},
            'issues': [],
            'recommendations': []
        }
        
        for feature in continuous_features:
            feature_data = data[feature].dropna()
            
            if len(feature_data) < 8:  # Insufficient data for analysis
                continue
                
            # Basic statistics
            feature_stats = {
                'count': len(feature_data),
                'mean': float(feature_data.mean()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'q25': float(feature_data.quantile(0.25)),
                'median': float(feature_data.median()),
                'q75': float(feature_data.quantile(0.75)),
                'skewness': float(stats.skew(feature_data)),
                'kurtosis': float(stats.kurtosis(feature_data))
            }
            analysis['distribution_statistics'][feature] = feature_stats
            
            # Normality tests
            normality_results = {}
            
            try:
                # Shapiro-Wilk test (best for small samples)
                if len(feature_data) <= 5000:
                    shapiro_stat, shapiro_p = shapiro(feature_data)
                    normality_results['shapiro'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                
                # Jarque-Bera test (good for larger samples)
                if len(feature_data) >= 50:
                    jb_stat, jb_p = jarque_bera(feature_data)
                    normality_results['jarque_bera'] = {
                        'statistic': float(jb_stat),
                        'p_value': float(jb_p),
                        'is_normal': jb_p > 0.05
                    }
                
                # D'Agostino's normality test
                if len(feature_data) >= 20:
                    da_stat, da_p = normaltest(feature_data)
                    normality_results['dagostino'] = {
                        'statistic': float(da_stat),
                        'p_value': float(da_p),
                        'is_normal': da_p > 0.05
                    }
                    
            except Exception as e:
                normality_results['error'] = str(e)
            
            analysis['normality_tests'][feature] = normality_results
            
            # Skewness analysis
            skewness = feature_stats['skewness']
            skewness_assessment = {
                'value': skewness,
                'interpretation': self._interpret_skewness(skewness),
                'severity': self._assess_skewness_severity(skewness)
            }
            analysis['skewness_analysis'][feature] = skewness_assessment
            
            # Outlier detection using IQR method
            q1, q3 = feature_stats['q25'], feature_stats['q75']
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
            outlier_ratio = len(outliers) / len(feature_data)
            
            outlier_info = {
                'count': len(outliers),
                'ratio': float(outlier_ratio),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'severity': 'high' if outlier_ratio > 0.1 else 'moderate' if outlier_ratio > 0.05 else 'low'
            }
            analysis['outlier_detection'][feature] = outlier_info
            
            # Generate issues and recommendations
            self._assess_continuous_feature_issues(feature, feature_stats, skewness_assessment, 
                                                 outlier_info, normality_results, analysis)
        
        return analysis
    
    def _analyze_categorical_distributions(
        self,
        data: pd.DataFrame,
        categorical_features: List[str]
    ) -> Dict[str, Any]:
        """Analyze distributions of categorical features."""
        analysis = {
            'features_analyzed': len(categorical_features),
            'distribution_statistics': {},
            'imbalance_analysis': {},
            'cardinality_analysis': {},
            'issues': [],
            'recommendations': []
        }
        
        for feature in categorical_features:
            feature_data = data[feature].dropna()
            
            if len(feature_data) == 0:
                continue
            
            # Value counts and proportions
            value_counts = feature_data.value_counts()
            value_props = feature_data.value_counts(normalize=True)
            
            # Basic statistics
            feature_stats = {
                'count': len(feature_data),
                'unique_count': len(value_counts),
                'most_common_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'most_common_ratio': float(value_props.iloc[0]) if len(value_counts) > 0 else 0.0,
                'least_common_value': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_common_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'least_common_ratio': float(value_props.iloc[-1]) if len(value_counts) > 0 else 0.0
            }
            analysis['distribution_statistics'][feature] = feature_stats
            
            # Imbalance analysis
            imbalance_metrics = self._calculate_imbalance_metrics(value_counts, value_props)
            analysis['imbalance_analysis'][feature] = imbalance_metrics
            
            # Cardinality analysis
            cardinality_info = {
                'cardinality': len(value_counts),
                'cardinality_ratio': len(value_counts) / len(feature_data),
                'assessment': self._assess_cardinality(len(value_counts), len(feature_data))
            }
            analysis['cardinality_analysis'][feature] = cardinality_info
            
            # Generate issues
            self._assess_categorical_feature_issues(feature, feature_stats, imbalance_metrics, 
                                                  cardinality_info, analysis)
        
        return analysis
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        abs_skew = abs(skewness)
        if abs_skew < 0.5:
            return "approximately symmetric"
        elif abs_skew < 1.0:
            return "moderately skewed"
        elif abs_skew < 2.0:
            return "highly skewed"
        else:
            return "extremely skewed"
    
    def _assess_skewness_severity(self, skewness: float) -> str:
        """Assess skewness severity."""
        abs_skew = abs(skewness)
        if abs_skew < 0.5:
            return "low"
        elif abs_skew < 1.0:
            return "moderate"
        elif abs_skew < 2.0:
            return "high"
        else:
            return "severe"
    
    def _calculate_imbalance_metrics(self, value_counts: pd.Series, value_props: pd.Series) -> Dict[str, Any]:
        """Calculate imbalance metrics for categorical features."""
        # Gini coefficient for class imbalance
        props_sorted = value_props.sort_values()
        n = len(props_sorted)
        cumsum = props_sorted.cumsum()
        # Fix Gini coefficient calculation
        if n > 1:
            gini_numerator = (cumsum.sum() - cumsum).sum()
            gini = (n + 1 - 2 * gini_numerator / cumsum.iloc[-1]) / n
        else:
            gini = 0
        
        # Imbalance ratio (majority/minority class ratio)
        max_prop = value_props.max()
        min_prop = value_props.min()
        imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        return {
            'gini_coefficient': float(gini),
            'imbalance_ratio': float(imbalance_ratio),
            'max_class_proportion': float(max_prop),
            'min_class_proportion': float(min_prop),
            'entropy': float(-sum(p * np.log2(p) for p in value_props if p > 0)),
            'severity': self._assess_imbalance_severity(imbalance_ratio, max_prop)
        }
    
    def _assess_imbalance_severity(self, imbalance_ratio: float, max_prop: float) -> str:
        """Assess class imbalance severity."""
        if imbalance_ratio < 2:
            return "balanced"
        elif imbalance_ratio < 5:
            return "mild_imbalance"
        elif imbalance_ratio < 10:
            return "moderate_imbalance"
        elif max_prop < 0.99:
            return "severe_imbalance"
        else:
            return "extreme_imbalance"
    
    def _assess_cardinality(self, unique_count: int, total_count: int) -> str:
        """Assess cardinality appropriateness."""
        ratio = unique_count / total_count
        if ratio > 0.9:
            return "too_high"  # Almost unique values
        elif ratio > 0.5:
            return "high"
        elif ratio > 0.1:
            return "moderate"
        else:
            return "appropriate"
    
    def _assess_continuous_feature_issues(
        self, 
        feature: str, 
        stats: Dict[str, Any], 
        skewness_info: Dict[str, Any], 
        outlier_info: Dict[str, Any], 
        normality_results: Dict[str, Any],
        analysis: Dict[str, Any]
    ):
        """Assess and record issues for continuous features."""
        issues = []
        recommendations = []
        
        # Skewness issues
        if skewness_info['severity'] == 'severe':
            issues.append(f"Feature '{feature}' has extreme skewness ({stats['skewness']:.2f})")
            recommendations.append(f"Apply log transformation or Box-Cox transformation to '{feature}'")
        elif skewness_info['severity'] == 'high':
            issues.append(f"Feature '{feature}' is highly skewed ({stats['skewness']:.2f})")
            recommendations.append(f"Consider transformation for '{feature}' to reduce skewness")
        
        # Outlier issues
        if outlier_info['severity'] == 'high':
            issues.append(f"Feature '{feature}' has high outlier ratio ({outlier_info['ratio']:.1%})")
            recommendations.append(f"Investigate and handle outliers in '{feature}'")
        elif outlier_info['severity'] == 'moderate':
            issues.append(f"Feature '{feature}' has moderate outlier ratio ({outlier_info['ratio']:.1%})")
        
        # Normality issues
        non_normal_tests = [test for test, result in normality_results.items() 
                           if isinstance(result, dict) and not result.get('is_normal', True)]
        if len(non_normal_tests) >= 2:  # Multiple tests suggest non-normality
            issues.append(f"Feature '{feature}' significantly deviates from normal distribution")
            recommendations.append(f"Consider transformation or non-parametric methods for '{feature}'")
        
        # Constant or near-constant features
        if stats['std'] < 1e-8:
            issues.append(f"Feature '{feature}' is constant or near-constant (std={stats['std']:.2e})")
            recommendations.append(f"Consider removing constant feature '{feature}'")
        
        analysis['issues'].extend(issues)
        analysis['recommendations'].extend(recommendations)
    
    def _assess_categorical_feature_issues(
        self,
        feature: str,
        stats: Dict[str, Any],
        imbalance_info: Dict[str, Any],
        cardinality_info: Dict[str, Any],
        analysis: Dict[str, Any]
    ):
        """Assess and record issues for categorical features."""
        issues = []
        recommendations = []
        
        # Imbalance issues
        if imbalance_info['severity'] == 'extreme_imbalance':
            issues.append(f"Feature '{feature}' has extreme class imbalance (ratio: {imbalance_info['imbalance_ratio']:.1f})")
            recommendations.append(f"Apply sampling techniques or class weights for '{feature}'")
        elif imbalance_info['severity'] == 'severe_imbalance':
            issues.append(f"Feature '{feature}' has severe class imbalance (ratio: {imbalance_info['imbalance_ratio']:.1f})")
            recommendations.append(f"Consider balancing techniques for '{feature}'")
        
        # Cardinality issues
        if cardinality_info['assessment'] == 'too_high':
            issues.append(f"Feature '{feature}' has too many unique values ({cardinality_info['cardinality']} out of {stats['count']})")
            recommendations.append(f"Consider grouping rare categories or using feature hashing for '{feature}'")
        elif cardinality_info['assessment'] == 'high':
            issues.append(f"Feature '{feature}' has high cardinality ({cardinality_info['cardinality']} unique values)")
            recommendations.append(f"Monitor encoding impact of high-cardinality feature '{feature}'")
        
        # Single dominant class
        if stats['most_common_ratio'] > 0.95:
            issues.append(f"Feature '{feature}' is dominated by single class ({stats['most_common_ratio']:.1%})")
            recommendations.append(f"Consider removing low-variance feature '{feature}'")
        
        analysis['issues'].extend(issues)
        analysis['recommendations'].extend(recommendations)
    
    def _generate_distribution_summary(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall distribution analysis summary."""
        continuous_analysis = details.get('continuous_distributions', {})
        categorical_analysis = details.get('categorical_distributions', {})
        
        summary = {
            'total_features_analyzed': (
                continuous_analysis.get('features_analyzed', 0) + 
                categorical_analysis.get('features_analyzed', 0)
            ),
            'continuous_features_count': continuous_analysis.get('features_analyzed', 0),
            'categorical_features_count': categorical_analysis.get('features_analyzed', 0),
            'total_issues_found': (
                len(continuous_analysis.get('issues', [])) + 
                len(categorical_analysis.get('issues', []))
            ),
            'distribution_quality': 'good',
            'recommendations': []
        }
        
        # Collect all recommendations
        all_recommendations = (
            continuous_analysis.get('recommendations', []) + 
            categorical_analysis.get('recommendations', [])
        )
        summary['recommendations'] = list(set(all_recommendations))  # Remove duplicates
        
        # Assess overall distribution quality
        total_issues = summary['total_issues_found']
        if total_issues == 0:
            summary['distribution_quality'] = 'excellent'
        elif total_issues <= 2:
            summary['distribution_quality'] = 'good'
        elif total_issues <= 5:
            summary['distribution_quality'] = 'fair'
        else:
            summary['distribution_quality'] = 'poor'
        
        return summary
    

