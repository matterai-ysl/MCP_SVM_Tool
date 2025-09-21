"""
Data Processing and Validation Utilities

This module provides intelligent data preprocessing, quality checks,
and validation mechanisms for machine learning workflows.
"""

import logging
import chardet
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from scipy import stats

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data loading, preprocessing, and validation.
    
    Features:
    - Automatic file format detection (CSV/Excel/TSV)
    - Encoding detection (UTF-8/GBK/Latin-1)  
    - Data type inference and conversion
    - Missing value detection and handling
    - Outlier identification
    - Data quality assessment
    """
    
    def __init__(self):
        """Initialize DataProcessor."""
        self.supported_formats = ['.csv', '.tsv', '.txt', '.xlsx', '.xls']
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        logger.info("Initialized DataProcessor")
        
    def detect_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """
        Detect file encoding using chardet.
        
        Args:
            file_path: Path to the file
            sample_size: Number of bytes to sample for detection
            
        Returns:
            Detected encoding
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback to common encodings if confidence is low
            if confidence < 0.7:
                for fallback_encoding in self.supported_encodings:
                    try:
                        with open(file_path, 'r', encoding=fallback_encoding) as f:
                            f.read(1000)  # Try to read a small sample
                        logger.info(f"Fallback encoding selected: {fallback_encoding}")
                        return fallback_encoding
                    except UnicodeDecodeError:
                        continue
                        
            return encoding or 'utf-8'
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def detect_delimiter(self, file_path: str, encoding: str) -> str:
        """
        Detect CSV delimiter.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            Detected delimiter
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(10000)
                
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            logger.info(f"Detected delimiter: '{delimiter}'")
            return delimiter
            
        except Exception as e:
            # Common delimiters to try
            delimiters = [',', '\t', ';', '|']
            sample_lines = []
            
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    sample_lines = f.readlines()[:5]
            except Exception:
                logger.warning(f"Delimiter detection failed: {e}, using comma")
                return ','
                
            delimiter_counts = {}
            for delim in delimiters:
                count = sum(line.count(delim) for line in sample_lines)
                delimiter_counts[delim] = count
                
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            logger.info(f"Fallback delimiter selected: '{best_delimiter}'")
            return best_delimiter
    
    def detect_file_format(self, file_path: str) -> str:
        """
        Detect file format based on extension.

        Args:
            file_path: Path to the file

        Returns:
            File format
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension in ['.xlsx', '.xls']:
            return 'excel'
        elif extension in ['.tsv']:
            return 'tsv'
        elif extension in ['.csv', '.txt']:
            return 'csv'
        else:
            logger.warning(f"Unknown file format: {extension}, treating as CSV")
            return 'csv'

    def _detect_url_format(self, url: str) -> str:
        """
        Detect file format from URL path.

        Args:
            url: URL to analyze

        Returns:
            Detected file format
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.lower()

        if path.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif path.endswith('.tsv'):
            return 'tsv'
        elif path.endswith(('.csv', '.txt')):
            return 'csv'
        else:
            # Default to CSV for URLs without clear extension
            logger.info(f"No clear file extension in URL, defaulting to CSV format")
            return 'csv'
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file or URL with automatic format detection.

        Args:
            file_path: Path to the data file or URL
            **kwargs: Additional arguments for pandas read functions

        Returns:
            Loaded dataframe
        """
        # Check if it's a URL
        is_url = file_path.startswith(('http://', 'https://'))

        if is_url:
            # URL data source
            file_format = self._detect_url_format(file_path)
            logger.info(f"Loading URL: {file_path} (format: {file_format})")
        else:
            # Local file data source
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            file_format = self.detect_file_format(file_path_obj)
            logger.info(f"Loading file: {file_path} (format: {file_format})")

        try:
            if file_format == 'excel':
                # Try loading Excel with better error handling
                try:
                    df = pd.read_excel(file_path, **kwargs)
                except Exception as excel_error:
                    logger.warning(f"Failed to read as Excel: {excel_error}")
                    # Try different Excel engines
                    for engine in ['openpyxl', 'xlrd']:
                        try:
                            df = pd.read_excel(file_path, engine=engine, **kwargs)
                            logger.info(f"Successfully loaded Excel file using {engine} engine")
                            break
                        except Exception:
                            continue
                    else:
                        raise excel_error
            else:
                # For CSV/TSV files, handle encoding and delimiter differently for URLs vs files
                if is_url:
                    # For URLs, let pandas handle encoding automatically
                    encoding = None
                    if file_format == 'tsv':
                        delimiter = '\t'
                    else:
                        delimiter = ','  # Default for URLs
                else:
                    # For local files, detect encoding and delimiter
                    encoding = self.detect_encoding(str(file_path))
                    if file_format == 'tsv':
                        delimiter = '\t'
                    else:
                        delimiter = self.detect_delimiter(str(file_path), encoding)

                # Update kwargs with detected parameters
                read_kwargs = {}
                if encoding is not None:
                    read_kwargs['encoding'] = encoding
                read_kwargs['sep'] = delimiter
                read_kwargs.update(kwargs)

                df = pd.read_csv(file_path, **read_kwargs)
            
            # Post-loading data cleaning and validation
            original_shape = df.shape
            logger.info(f"Raw data loaded: {original_shape[0]} rows, {original_shape[1]} columns")
            
            # Clean the dataframe
            df = self._clean_dataframe(df)
            cleaned_shape = df.shape
            
            if cleaned_shape != original_shape:
                logger.info(f"Data cleaned: {cleaned_shape[0]} rows, {cleaned_shape[1]} columns (removed {original_shape[0] - cleaned_shape[0]} rows)")
            
            logger.info(f"Successfully loaded and cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess dataframe after loading.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning dataframe...")
        
        # Remove completely empty rows and columns
        original_shape = df.shape
        df = df.dropna(how='all')  # Remove rows where all values are NaN
        df = df.loc[:, ~df.isnull().all()]  # Remove columns where all values are NaN
        
        if df.shape != original_shape:
            logger.info(f"Removed empty rows/columns: {original_shape} -> {df.shape}")
        
        # Convert object columns to numeric where possible, but preserve categorical data
        for col in df.columns:
            if df[col].dtype == 'object':
                # First check if it could be categorical
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 and df[col].nunique() < 50:  # Likely categorical
                    logger.debug(f"Column '{col}' appears to be categorical ({df[col].nunique()} unique values), preserving as object")
                    # Keep as object/categorical - don't convert to numeric
                    continue
                
                # For non-categorical object columns, try numeric conversion
                try:
                    # Check if all non-null values can be converted to numeric
                    test_series = pd.to_numeric(df[col].dropna(), errors='coerce')
                    if not test_series.isna().any():  # All values converted successfully
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.debug(f"Converted column '{col}' to numeric")
                except:
                    pass
        
        # Handle text columns that might be intended as categorical
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 and df[col].nunique() < 50:  # Likely categorical
                    logger.debug(f"Column '{col}' appears to be categorical ({df[col].nunique()} unique values)")
                    # Convert to category for better memory usage
                    df[col] = df[col].astype('category')
        
        # Only remove columns that were originally all NaN (not those that became NaN after failed conversion)
        original_null_columns = df.columns[df.isnull().all()]
        if len(original_null_columns) > 0:
            logger.info(f"Removing {len(original_null_columns)} columns that are entirely null: {list(original_null_columns)}")
            df = df.drop(columns=original_null_columns)
        
        # Ensure we have at least some data
        if df.empty:
            raise ValueError("Dataframe is empty after cleaning")
        
        if df.shape[1] < 2:
            raise ValueError("Dataframe must have at least 2 columns (features + target)")
        
        logger.info(f"Dataframe cleaning completed: {df.shape}")
        return df
    
    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Intelligently infer data types for columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary mapping column names to inferred types
        """
        type_mapping = {}
        
        for column in df.columns:
            series = df[column].dropna()
            
            if len(series) == 0:
                type_mapping[column] = 'unknown'
                continue
            
            # Check if it's numeric
            try:
                pd.to_numeric(series)
                # Check if it's integer or float
                if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                    type_mapping[column] = 'integer'
                else:
                    # Check if all values are integers (even if stored as float)
                    numeric_series = pd.to_numeric(series)
                    if (numeric_series % 1 == 0).all():
                        type_mapping[column] = 'integer'
                    else:
                        type_mapping[column] = 'float'
            except (ValueError, TypeError):
                # Check if it's datetime
                try:
                    pd.to_datetime(series)
                    type_mapping[column] = 'datetime'
                except (ValueError, TypeError):
                    # Check if it's boolean
                    unique_values = set(str(v).lower() for v in series.unique())
                    bool_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
                    if unique_values.issubset(bool_values):
                        type_mapping[column] = 'boolean'
                    else:
                        # Check if it's categorical (low cardinality)
                        if len(series.unique()) / len(series) < 0.1 and len(series.unique()) < 50:
                            type_mapping[column] = 'categorical'
                        else:
                            type_mapping[column] = 'text'
        
        logger.info(f"Inferred data types: {type_mapping}")
        return type_mapping
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and analyze missing values.
        
        Args:
            df: Input dataframe
            
        Returns:
            Missing value analysis
        """
        missing_info = {}
        
        for column in df.columns:
            # Standard missing values
            null_count = df[column].isnull().sum()
            
            # Custom missing value patterns
            if df[column].dtype == 'object':
                # Common missing value representations
                missing_patterns = ['', ' ', 'null', 'NULL', 'None', 'n/a', 'N/A', 
                                  'na', 'NA', '-', '?', 'missing', 'unknown']
                
                custom_missing = df[column].isin(missing_patterns).sum()
                total_missing = null_count + custom_missing
            else:
                custom_missing = 0
                total_missing = null_count
            
            missing_percentage = (total_missing / len(df)) * 100
            
            missing_info[column] = {
                'null_count': int(null_count),
                'custom_missing': int(custom_missing),
                'total_missing': int(total_missing),
                'missing_percentage': round(missing_percentage, 2)
            }
        
        logger.info(f"Missing value analysis completed")
        return missing_info
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: Input dataframe
            method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            
        Returns:
            Dictionary mapping column names to outlier indices
        """
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) < 4:  # Not enough data for outlier detection
                outliers[column] = []
                continue
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(series))
                outlier_mask = z_scores > 3
                # Map back to original dataframe indices
                outlier_mask = df[column].index.isin(series.index[outlier_mask])
                
            else:  # isolation forest would require sklearn
                logger.warning(f"Method {method} not implemented, using IQR")
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            outlier_indices = df.index[outlier_mask].tolist()
            outliers[column] = outlier_indices
        
        logger.info(f"Outlier detection completed using {method} method")
        return outliers
        
    def validate_data(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate data integrity and quality.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            Validation report
        """
        logger.info("Starting data validation")
        
        validation_report = {
            'basic_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'data_types': self.infer_data_types(df),
            'missing_values': self.detect_missing_values(df),
            'outliers': self.detect_outliers(df),
            'duplicates': df.duplicated().sum(),
            'data_quality_score': 0.0
        }
        
        # Calculate data quality score
        total_cells = df.shape[0] * df.shape[1]
        total_missing = sum(info['total_missing'] for info in validation_report['missing_values'].values())
        missing_penalty = (total_missing / total_cells) * 30
        
        duplicate_penalty = (validation_report['duplicates'] / len(df)) * 20
        
        # Outlier penalty (simplified)
        total_outliers = sum(len(outliers) for outliers in validation_report['outliers'].values())
        outlier_penalty = (total_outliers / total_cells) * 10
        
        data_quality_score = max(0, 100 - missing_penalty - duplicate_penalty - outlier_penalty)
        validation_report['data_quality_score'] = round(data_quality_score, 2)
        
        # Target column validation
        if target_column:
            if target_column not in df.columns:
                validation_report['target_validation'] = {
                    'exists': False,
                    'error': f"Target column '{target_column}' not found"
                }
            else:
                target_series = df[target_column]
                validation_report['target_validation'] = {
                    'exists': True,
                    'unique_values': len(target_series.unique()),
                    'missing_count': target_series.isnull().sum(),
                    'data_type': str(target_series.dtype)
                }
        
        logger.info(f"Data validation completed. Quality score: {data_quality_score:.2f}")
        return validation_report
        
    def preprocess_data(self, df: pd.DataFrame, target_column: Optional[str] = None,
                       handle_missing: str = 'auto', handle_outliers: bool = False) -> pd.DataFrame:
        """
        Preprocess data for machine learning.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            handle_missing: Strategy for missing values ('auto', 'drop', 'fill')
            handle_outliers: Whether to handle outliers
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("Starting data preprocessing")
        df_processed = df.copy()
        
        # Handle missing values
        if handle_missing == 'auto':
            missing_info = self.detect_missing_values(df_processed)
            
            for column, info in missing_info.items():
                if info['missing_percentage'] > 50:
                    # Drop columns with >50% missing
                    logger.info(f"Dropping column '{column}' (>{info['missing_percentage']:.1f}% missing)")
                    df_processed = df_processed.drop(columns=[column])
                elif info['total_missing'] > 0:
                    # Fill missing values based on data type
                    if df_processed[column].dtype in ['int64', 'float64']:
                        # Fill numeric with median
                        df_processed[column].fillna(df_processed[column].median(), inplace=True)
                    else:
                        # Fill categorical with mode
                        mode_value = df_processed[column].mode()
                        if len(mode_value) > 0:
                            df_processed[column].fillna(mode_value[0], inplace=True)
                        else:
                            df_processed[column].fillna('unknown', inplace=True)
                            
        elif handle_missing == 'drop':
            df_processed = df_processed.dropna()
            
        # Handle outliers if requested
        if handle_outliers:
            outliers = self.detect_outliers(df_processed)
            outlier_indices = set()
            
            for column, indices in outliers.items():
                outlier_indices.update(indices)
            
            if outlier_indices:
                logger.info(f"Removing {len(outlier_indices)} outlier rows")
                df_processed = df_processed.drop(index=list(outlier_indices))
        
        # Remove duplicates
        duplicates_before = df_processed.duplicated().sum()
        if duplicates_before > 0:
            df_processed = df_processed.drop_duplicates()
            logger.info(f"Removed {duplicates_before} duplicate rows")
        
        logger.info(f"Preprocessing completed. Shape: {df.shape} -> {df_processed.shape}")
        return df_processed 