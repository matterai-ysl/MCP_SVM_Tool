"""
HTML Report Generator

This module provides comprehensive HTML report generation for
training processes and prediction results.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)

class HTMLReportGenerator:
    """
    Generates detailed reports for training and prediction processes.
    
    Features:
    - Training process reports
    - Prediction result reports
    - Performance analysis
    - Visualization integration
    - Multiple output formats (HTML, JSON)
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize HTMLReportGenerator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized HTMLReportGenerator with directory: {output_dir}")
        
    def generate_training_report(
        self,
        training_results: Dict[str, Any],
        format_type: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a comprehensive training report.
        
        Args:
            training_results: Training process results from TrainingEngine
            format_type: Report format ("html" or "json")
            include_visualizations: Whether to include visualization references
            
        Returns:
            Path to generated report file
        """
        try:
            logger.info(f"Generating {format_type.upper()} training report...")
            
            if format_type.lower() == "html":
                # Create temporary directory to use the unified template system
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_model_dir = Path(temp_dir)
                    
                    # Add model_directory to training_results for unified template
                    training_results_with_dir = training_results.copy()
                    training_results_with_dir['model_directory'] = str(temp_model_dir)
                    
                    # Generate HTML using unified template system
                    html_content = self._create_html_training_report(training_results_with_dir)
                    
                    # Save to output directory
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"training_report_{training_results.get('model_name', 'model')}_{timestamp}.html"
                    filepath = self.output_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    logger.info(f"HTML training report saved to: {filepath}")
                    return str(filepath)
                    
            elif format_type.lower() == "json":
                return self._generate_json_training_report_to_file(training_results)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            raise
    

    
    def _get_training_report_styles(self) -> str:
        """Get CSS styles for training report."""
        return """
    <style>
        /* Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        /* Header Styles */
        .report-header {
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        /* Section Styles */
        .section {
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 5px solid #2a5298;
        }
        
        .section h2 {
            color: #2a5298;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .section h3 {
            color: #495057;
            margin: 20px 0 15px 0;
            font-size: 1.3em;
        }
        
        /* Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .metric:hover {
            border-color: #2a5298;
            transform: translateY(-2px);
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2a5298;
        }
        
        /* Table Styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        th {
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        /* Recommendation Styles */
        .recommendation {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 5px solid #2196f3;
        }
        
        .recommendation strong {
            color: #1976d2;
        }
        
        /* Status and Alert Styles */
        .status-good {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border-left-color: #4caf50;
            color: #2e7d32;
        }
        
        .status-warning {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left-color: #ff9800;
            color: #ef6c00;
        }
        
        .status-error {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left-color: #f44336;
            color: #c62828;
        }
        
        /* Footer */
        .report-footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
            margin-top: 30px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            
            .metric-grid {
                grid-template-columns: 1fr;
            }
            
            .report-header h1 {
                font-size: 2em;
            }
        }
    </style>"""
    
    def _generate_json_training_report(self, training_results: Dict[str, Any]) -> str:
        """Generate JSON format training report content."""
        # Prepare report data
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'training_report',
                'model_name': training_results.get('model_name', 'Unknown'),
                'model_id': training_results.get('model_id', 'Unknown'),
                'report_version': '1.0.0'
            },
            'training_summary': {
                'task_type': training_results.get('task_type'),
                'training_time_seconds': training_results.get('training_time_seconds'),
                'model_path': training_results.get('model_path'),
                'training_completed_at': training_results.get('training_completed_at')
            },
            'dataset_overview': training_results.get('data_shape', {}),
            'feature_names': training_results.get('feature_names', []),
            'model_configuration': {
                'hyperparameters': training_results.get('hyperparameters', {}),
                'model_info': training_results.get('model_info', {})
            },
            'hyperparameter_optimization': training_results.get('optimization_results', {}),
            'cross_validation_results': training_results.get('cross_validation_results', {}),
            'feature_importance': training_results.get('feature_importance', {}),
            'recommendations': self._generate_recommendations(training_results),
            'raw_training_results': training_results
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'to_dict'):  # Handle pandas DataFrame/Series
                return obj.to_dict()
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
                # Handle complex objects by converting to string representation
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy_types(item) for item in obj]
            elif callable(obj):
                # Handle functions/methods by converting to string
                return str(obj)
            return obj
        
        converted_report = convert_numpy_types(report_data)
        
        # Return JSON string instead of saving to file
        return json.dumps(converted_report, indent=2, ensure_ascii=False)

    def _generate_json_training_report_to_file(self, training_results: Dict[str, Any]) -> str:
        """Generate JSON format training report and save to file (for backward compatibility)."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_report_{training_results.get('model_name', 'model')}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        json_content = self._generate_json_training_report(training_results)
        
        # Save JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        logger.info(f"JSON training report saved to: {filepath}")
        return str(filepath)
    
    def generate_prediction_report(
        self,
        prediction_results: Dict[str, Any],
        model_info: Dict[str, Any],
        format_type: str = "html"
    ) -> str:
        """
        Generate a detailed prediction report.
        
        Args:
            prediction_results: Prediction results from PredictionEngine
            model_info: Model information and metadata
            format_type: Report format ("html" or "json")
            
        Returns:
            Path to generated report file
        """
        try:
            if format_type.lower() == "html":
                return self._generate_html_prediction_report(prediction_results, model_info)
            elif format_type.lower() == "json":
                return self._generate_json_prediction_report(prediction_results, model_info)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
        except Exception as e:
            logger.error(f"Error generating prediction report: {str(e)}")
            raise
    
    def _generate_html_prediction_report(
        self, 
        prediction_results: Dict[str, Any], 
        model_info: Dict[str, Any]
    ) -> str:
        """Generate HTML format prediction report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = prediction_results.get('model_id', 'unknown_model')
        filename = f"prediction_report_{model_id}_{timestamp}.html"
        filepath = self.output_dir / filename
        
        # Generate CSS styles
        css_styles = self._get_prediction_report_styles()
        
        # Generate HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Report - {model_id}</title>
    {css_styles}
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>🔮 Prediction Report</h1>
            <p class="subtitle">Comprehensive Analysis of Model Predictions</p>
        </header>
        
        {self._add_prediction_summary_section(prediction_results, model_info)}
        {self._add_result_statistics_section(prediction_results)}
        {self._add_confidence_distribution_section(prediction_results)}
        {self._add_feature_contribution_section(prediction_results)}
        {self._add_prediction_details_section(prediction_results)}
        {self._add_prediction_recommendations_section(prediction_results, model_info)}
        
        <footer class="report-footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                            <p>MCP SVM Tool - Prediction Analysis Report v1.0.0</p>
        </footer>
    </div>
</body>
</html>"""
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML prediction report saved to: {filepath}")
        return str(filepath)
    
    def _get_prediction_report_styles(self) -> str:
        """Get CSS styles for prediction report."""
        return """
    <style>
        /* Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        /* Header Styles */
        .report-header {
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        /* Section Styles */
        .section {
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 5px solid #2a5298;
        }
        
        .section h2 {
            color: #2a5298;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .section h3 {
            color: #495057;
            margin: 20px 0 15px 0;
            font-size: 1.3em;
        }
        
        /* Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .metric:hover {
            border-color: #2a5298;
            transform: translateY(-2px);
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2a5298;
        }
        
        /* Table Styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        th {
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        /* Recommendation Styles */
        .recommendation {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 5px solid #2196f3;
        }
        
        .recommendation strong {
            color: #1976d2;
        }
        
        /* Status and Alert Styles */
        .status-good {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border-left-color: #4caf50;
            color: #2e7d32;
        }
        
        .status-warning {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left-color: #ff9800;
            color: #ef6c00;
        }
        
        .status-error {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left-color: #f44336;
            color: #c62828;
        }
        
        /* Footer */
        .report-footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
            margin-top: 30px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            
            .metric-grid {
                grid-template-columns: 1fr;
            }
            
            .report-header h1 {
                font-size: 2em;
            }
        }
    </style>"""
    
    def _add_prediction_summary_section(self, prediction_results: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Add prediction summary section to HTML report."""
        predictions = prediction_results.get('predictions', [])
        model_name = prediction_results.get('model_name', 'Unknown')
        prediction_time = prediction_results.get('prediction_time_seconds', 0)
        
        return f"""
        <div class="section">
            <h2>📊 Prediction Summary</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Model Name</div>
                    <div class="metric-value">{model_name}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Predictions</div>
                    <div class="metric-value">{len(predictions)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Prediction Time</div>
                    <div class="metric-value">{self._safe_format_numeric(prediction_time, '.3f')} seconds</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Time per Sample</div>
                    <div class="metric-value">
                        {self._safe_format_numeric((prediction_time / max(len(predictions), 1) * 1000), '.2f')} ms
                    </div>
                </div>
            </div>
        </div>"""
    
    def _add_result_statistics_section(self, prediction_results: Dict[str, Any]) -> str:
        """Add result statistics section to HTML report."""
        predictions = prediction_results.get('predictions', [])
        
        if not predictions:
            return """
        <div class="section">
            <h2>📈 Result Statistics</h2>
            <p>No predictions available for statistical analysis.</p>
        </div>"""
        
        # Calculate statistics
        try:
            import numpy as np
            pred_array = np.array(predictions)
            
            stats = {
                'mean': np.mean(pred_array),
                'median': np.median(pred_array),
                'std': np.std(pred_array),
                'min': np.min(pred_array),
                'max': np.max(pred_array),
                'q25': np.percentile(pred_array, 25),
                'q75': np.percentile(pred_array, 75)
            }
        except Exception:
            # Fallback for non-numeric predictions
            unique_values = list(set(predictions))
            value_counts = {val: predictions.count(val) for val in unique_values}
            most_common = max(value_counts, key=value_counts.get)
            
            return f"""
        <div class="section">
            <h2>📈 Result Statistics</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Unique Values</div>
                    <div class="metric-value">{len(unique_values)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Most Common</div>
                    <div class="metric-value">{most_common}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Frequency</div>
                    <div class="metric-value">{value_counts[most_common]}</div>
                </div>
            </div>
            <h3>Value Distribution</h3>
            <table>
                <tr><th>Value</th><th>Count</th><th>Percentage</th></tr>
                {self._generate_value_distribution_table(value_counts, len(predictions))}
            </table>
        </div>"""
        
        return f"""
        <div class="section">
            <h2>📈 Result Statistics</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Mean</div>
                    <div class="metric-value">{self._safe_format_numeric(stats['mean'], '.4f')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Median</div>
                    <div class="metric-value">{self._safe_format_numeric(stats['median'], '.4f')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">{self._safe_format_numeric(stats['std'], '.4f')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Range</div>
                    <div class="metric-value">{self._safe_format_numeric(stats['min'], '.4f')} - {self._safe_format_numeric(stats['max'], '.4f')}</div>
                </div>
            </div>
            <h3>Distribution Details</h3>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Minimum</td><td>{self._safe_format_numeric(stats['min'], '.6f')}</td></tr>
                <tr><td>25th Percentile</td><td>{self._safe_format_numeric(stats['q25'], '.6f')}</td></tr>
                <tr><td>Median (50th)</td><td>{self._safe_format_numeric(stats['median'], '.6f')}</td></tr>
                <tr><td>75th Percentile</td><td>{self._safe_format_numeric(stats['q75'], '.6f')}</td></tr>
                <tr><td>Maximum</td><td>{self._safe_format_numeric(stats['max'], '.6f')}</td></tr>
                <tr><td>Standard Deviation</td><td>{self._safe_format_numeric(stats['std'], '.6f')}</td></tr>
            </table>
        </div>"""
    
    def _generate_value_distribution_table(self, value_counts: Dict, total_count: int) -> str:
        """Generate value distribution table for categorical predictions."""
        table_rows = ""
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        for value, count in sorted_items:
            percentage = (count / total_count) * 100
            table_rows += f"""
            <tr>
                <td>{value}</td>
                <td>{count}</td>
                <td>{self._safe_format_numeric(percentage, '.1f')}%</td>
            </tr>"""
        
        return table_rows
    
    def _add_confidence_distribution_section(self, prediction_results: Dict[str, Any]) -> str:
        """Add confidence distribution section to HTML report."""
        confidence_scores = prediction_results.get('confidence_scores', [])
        
        if not confidence_scores:
            return """
        <div class="section">
            <h2>🎯 Confidence Analysis</h2>
            <p>Confidence scores are not available for this prediction session.</p>
        </div>"""
        
        try:
            import numpy as np
            conf_array = np.array(confidence_scores)
            
            # Calculate confidence statistics
            conf_stats = {
                'mean': np.mean(conf_array),
                'median': np.median(conf_array),
                'std': np.std(conf_array),
                'min': np.min(conf_array),
                'max': np.max(conf_array)
            }
            
            # Categorize confidence levels
            high_conf = np.sum(conf_array >= 0.8)
            medium_conf = np.sum((conf_array >= 0.6) & (conf_array < 0.8))
            low_conf = np.sum(conf_array < 0.6)
            
            return f"""
        <div class="section">
            <h2>🎯 Confidence Analysis</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Average Confidence</div>
                    <div class="metric-value">{self._safe_format_numeric(conf_stats['mean'], '.3f')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">High Confidence (≥0.8)</div>
                    <div class="metric-value">{high_conf} ({self._safe_format_numeric((high_conf/len(confidence_scores)*100), '.1f')}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Medium Confidence (0.6-0.8)</div>
                    <div class="metric-value">{medium_conf} ({self._safe_format_numeric((medium_conf/len(confidence_scores)*100), '.1f')}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Low Confidence (<0.6)</div>
                    <div class="metric-value">{low_conf} ({self._safe_format_numeric((low_conf/len(confidence_scores)*100), '.1f')}%)</div>
                </div>
            </div>
            <h3>Confidence Statistics</h3>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Mean Confidence</td><td>{self._safe_format_numeric(conf_stats['mean'], '.6f')}</td></tr>
                <tr><td>Median Confidence</td><td>{self._safe_format_numeric(conf_stats['median'], '.6f')}</td></tr>
                <tr><td>Standard Deviation</td><td>{self._safe_format_numeric(conf_stats['std'], '.6f')}</td></tr>
                <tr><td>Minimum Confidence</td><td>{self._safe_format_numeric(conf_stats['min'], '.6f')}</td></tr>
                <tr><td>Maximum Confidence</td><td>{self._safe_format_numeric(conf_stats['max'], '.6f')}</td></tr>
            </table>
        </div>"""
        except Exception as e:
            return f"""
        <div class="section">
            <h2>🎯 Confidence Analysis</h2>
            <p>Error analyzing confidence scores: {str(e)}</p>
        </div>"""
    
    def _add_feature_contribution_section(self, prediction_results: Dict[str, Any]) -> str:
        """Add feature contribution section to HTML report."""
        feature_contributions = prediction_results.get('feature_contributions', {})
        
        if not feature_contributions:
            return """
        <div class="section">
            <h2>🔍 Feature Contributions</h2>
            <p>Feature contribution analysis is not available for this prediction session.</p>
        </div>"""
        
        # Get average contributions across all predictions
        avg_contributions = {}
        for feature, contributions in feature_contributions.items():
            if isinstance(contributions, list) and contributions:
                avg_contributions[feature] = sum(contributions) / len(contributions)
        
        # Sort by absolute contribution
        sorted_features = sorted(avg_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # Create contribution table
        contrib_table = "<table><tr><th>Rank</th><th>Feature</th><th>Avg Contribution</th><th>Impact</th></tr>"
        for rank, (feature, contrib) in enumerate(sorted_features, 1):
            impact = "Positive" if contrib > 0 else "Negative" if contrib < 0 else "Neutral"
            contrib_table += f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{feature}</strong></td>
                <td>{self._safe_format_numeric(contrib, '.6f')}</td>
                <td>{impact}</td>
            </tr>"""
        contrib_table += "</table>"
        
        return f"""
        <div class="section">
            <h2>🔍 Feature Contributions</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Features Analyzed</div>
                    <div class="metric-value">{len(feature_contributions)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Top Contributor</div>
                    <div class="metric-value">{sorted_features[0][0] if sorted_features else 'N/A'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Top Contribution</div>
                    <div class="metric-value">{self._safe_format_numeric(sorted_features[0][1], '.4f') if sorted_features else 'N/A'}</div>
                </div>
            </div>
            <h3>Top 10 Feature Contributions</h3>
            {contrib_table}
            <div class="recommendation">
                <strong>💡 Feature Insight:</strong> 
                Feature contributions show how each input variable influences the final predictions. 
                Positive contributions increase the predicted value, while negative contributions decrease it.
            </div>
        </div>"""
    
    def _add_prediction_details_section(self, prediction_results: Dict[str, Any]) -> str:
        """Add prediction details section to HTML report."""
        predictions = prediction_results.get('predictions', [])
        confidence_scores = prediction_results.get('confidence_scores', [])
        input_data = prediction_results.get('input_data', [])
        
        if not predictions:
            return """
        <div class="section">
            <h2>📋 Prediction Details</h2>
            <p>No prediction details available.</p>
        </div>"""
        
        # Show first 20 predictions in detail
        max_display = min(20, len(predictions))
        
        details_table = "<table><tr><th>Index</th><th>Prediction</th>"
        if confidence_scores:
            details_table += "<th>Confidence</th>"
        if input_data and len(input_data) > 0:
            # Show first few input features
            sample_input = input_data[0] if input_data else {}
            if isinstance(sample_input, dict):
                feature_names = list(sample_input.keys())[:3]  # Show first 3 features
                for feature in feature_names:
                    details_table += f"<th>{feature}</th>"
        details_table += "</tr>"
        
        for i in range(max_display):
            details_table += f"<tr><td>{i+1}</td><td>{self._safe_format_numeric(predictions[i], '.4f')}</td>"
            
            if confidence_scores and i < len(confidence_scores):
                details_table += f"<td>{self._safe_format_numeric(confidence_scores[i], '.3f')}</td>"
            elif confidence_scores:
                details_table += "<td>N/A</td>"
            
            if input_data and i < len(input_data):
                sample = input_data[i]
                if isinstance(sample, dict):
                    for feature in feature_names:
                        value = sample.get(feature, 'N/A')
                        details_table += f"<td>{self._safe_format_numeric(value, '.3f') if isinstance(value, (int, float)) else value}</td>"
                elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
                    for j in range(min(3, len(sample))):
                        details_table += f"<td>{self._safe_format_numeric(sample[j], '.3f')}</td>"
            
            details_table += "</tr>"
        
        details_table += "</table>"
        
        return f"""
        <div class="section">
            <h2>📋 Prediction Details</h2>
            <p>Showing first {max_display} predictions out of {len(predictions)} total predictions.</p>
            {details_table}
            {f'<p><em>Note: {len(predictions) - max_display} additional predictions not shown.</em></p>' if len(predictions) > max_display else ''}
        </div>"""
    
    def _add_prediction_recommendations_section(self, prediction_results: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Add prediction recommendations section to HTML report."""
        recommendations = self._generate_prediction_recommendations(prediction_results, model_info)
        
        recommendations_html = ""
        for i, rec in enumerate(recommendations, 1):
            recommendations_html += f"""
            <div class="recommendation">
                <strong>💡 Recommendation {i}:</strong> {rec}
            </div>"""
        
        return f"""
        <div class="section">
            <h2>💡 Prediction Recommendations</h2>
            {recommendations_html}
        </div>"""
    
    def _generate_prediction_recommendations(self, prediction_results: Dict[str, Any], model_info: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on prediction results."""
        recommendations = []
        
        predictions = prediction_results.get('predictions', [])
        confidence_scores = prediction_results.get('confidence_scores', [])
        
        # Check prediction confidence
        if confidence_scores:
            import numpy as np
            avg_confidence = np.mean(confidence_scores)
            low_confidence_count = sum(1 for conf in confidence_scores if conf < 0.6)
            
            if avg_confidence < 0.7:
                recommendations.append(
                    f"Average prediction confidence is {avg_confidence:.3f}. "
                    "Consider retraining the model with more data or feature engineering."
                )
            
            if low_confidence_count > len(confidence_scores) * 0.2:  # More than 20% low confidence
                recommendations.append(
                    f"{low_confidence_count} predictions have low confidence (<0.6). "
                    "Review these predictions carefully and consider manual validation."
                )
        
        # Check prediction variance
        if predictions and len(predictions) > 1:
            try:
                import numpy as np
                pred_std = np.std(predictions)
                pred_range = np.max(predictions) - np.min(predictions)
                
                if pred_range > pred_std * 10:  # High variance
                    recommendations.append(
                        "High variance detected in predictions. "
                        "Ensure input data is properly normalized and within training distribution."
                    )
            except Exception:
                pass
        
        # Check prediction time
        prediction_time = prediction_results.get('prediction_time_seconds', 0)
        if prediction_time > 1.0 and len(predictions) > 0:  # More than 1 second
            avg_time_per_sample = prediction_time / len(predictions) * 1000  # ms
            if avg_time_per_sample > 100:  # More than 100ms per sample
                recommendations.append(
                    f"Prediction time is {avg_time_per_sample:.1f}ms per sample. "
                    "Consider model optimization for production deployment."
                )
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Predictions completed successfully. Monitor prediction quality over time.",
                "Consider implementing prediction monitoring and alerting for production use.",
                "Document prediction assumptions and limitations for stakeholders."
            ])
        
        return recommendations
    
    def _generate_json_prediction_report(self, prediction_results: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Generate JSON format prediction report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = prediction_results.get('model_name', 'unknown_model')
        filename = f"prediction_report_{model_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare report data
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'prediction_report',
                'model_name': model_name,
                'report_version': '1.0.0'
            },
            'prediction_summary': {
                'total_predictions': len(prediction_results.get('predictions', [])),
                'prediction_time_seconds': prediction_results.get('prediction_time_seconds'),
                'model_info': model_info
            },
            'predictions': prediction_results.get('predictions', []),
            'confidence_scores': prediction_results.get('confidence_scores', []),
            'feature_contributions': prediction_results.get('feature_contributions', {}),
            'input_data': prediction_results.get('input_data', []),
            'recommendations': self._generate_prediction_recommendations(prediction_results, model_info),
            'raw_prediction_results': prediction_results
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        converted_report = convert_numpy_types(report_data)
        
        # Save JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON prediction report saved to: {filepath}")
        return str(filepath)

    def _add_training_summary_section(self, training_results: Dict[str, Any]) -> str:
        """Add training summary section to HTML report."""
        task_type = training_results.get('task_type', 'Unknown')
        training_time = training_results.get('training_time_seconds', 0)
        model_path = training_results.get('model_path', 'Not saved')
        
        status_class = "status-success" if model_path != 'Not saved' else "status-info"
        
        return f"""
        <div class="section">
            <h2>📊 Training Summary</h2>
            <div class="metric-grid">
                <div class="metric success">
                    <div class="metric-label">Model ID</div>
                    <div class="metric-value">{training_results.get('model_id', 'N/A')}</div>
                </div>
                <div class="metric info">
                    <div class="metric-label">Task Type</div>
                    <div class="metric-value">
                        <span class="status-badge {status_class}">{task_type.upper()}</span>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Training Time</div>
                    <div class="metric-value">{self._safe_format_numeric(training_time, '.2f')} seconds</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Model Status</div>
                    <div class="metric-value">
                        <span class="status-badge {status_class}">
                            {"Saved" if model_path != 'Not saved' else "Not Saved"}
                        </span>
                    </div>
                </div>
            </div>
            <div class="metric">
                <div class="metric-label">Model Path</div>
                <div class="metric-value">{model_path}</div>
            </div>
        </div>"""
    
    def _add_data_validation_section(self, training_results: Dict[str, Any]) -> str:
        """Add comprehensive data validation section to HTML report by loading data_validation_report.json."""
        import os
        from pathlib import Path
        import json
        
        # 尝试自动检测模型目录
        model_dir = training_results.get('model_directory') or training_results.get('model_dir')
        if not model_dir:
            return ''
        
        reports_dir = Path(model_dir) / "reports"
        validation_path = reports_dir / "data_validation_report.json"
        
        if not validation_path.exists():
            return '''
            <div class="section">
                <h2>🧪 Data Validation & Quality Assessment</h2>
                <p style="color:#888; font-style:italic;">Data validation report not available. Run training with validate_data=True to generate comprehensive data analysis.</p>
            </div>'''
        
        try:
            with open(validation_path, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
        except Exception as e:
            return f'<div class="section"><h2>🧪 Data Validation</h2><p style="color:red;">Failed to load data validation report: {e}</p></div>'
        
        # 构建详细的数据验证报告
        html = [
            '<div class="section">',
            '<h2>🧪 Data Validation & Quality Assessment</h2>'
        ]
        
        # 总体质量评估
        qa = validation_data.get('quality_assessment', {})
        if qa:
            score = qa.get("overall_score", 0)
            level = qa.get("quality_level", "Unknown")
            score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
            
            html.append(f'''
            <div class="metric-grid">
                <div class="metric success">
                    <div class="metric-label">Overall Data Quality Score</div>
                    <div class="metric-value" style="color:{score_color};">{score}/100</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Quality Level</div>
                    <div class="metric-value">{level}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Data Ready for Training</div>
                    <div class="metric-value">{"✅ Yes" if validation_data.get("data_ready_for_training", False) else "❌ No"}</div>
                </div>
            </div>''')
        
        # 详细的数据分析结果
        html.append(self._generate_detailed_validation_analysis(validation_data))
        
        # 生成建议
        if 'recommendations' in validation_data and validation_data['recommendations']:
            html.append('<div class="recommendation"><strong>💡 Improvement Recommendations:</strong><ul>')
            for rec in validation_data['recommendations'][:8]:
                html.append(f'<li>{rec}</li>')
            html.append('</ul></div>')
        
        # 关键问题
        critical_issues = []
        for check in validation_data.get('validation_results', []):
            if not check.get('passed', True):
                critical_issues.extend(check.get('issues', []))
        
        if critical_issues:
            html.append('<div class="recommendation" style="background:#ffeaea; border-left-color:#dc3545;"><strong>🚨 Critical Issues Detected:</strong><ul>')
            for issue in critical_issues[:6]:
                html.append(f'<li style="color:#dc3545;">{issue}</li>')
            html.append('</ul></div>')
        
        # 验证检查摘要表
        html.append('<h3>📋 Validation Checks Summary</h3>')
        html.append('<table style="width:100%;"><tr><th>Check Type</th><th>Status</th><th>Issues Found</th><th>Details</th></tr>')
        
        for check in validation_data.get('validation_results', []):
            name = check.get('check_name', 'Unknown').replace('_', ' ').title()
            status = '<span style="color:green; font-weight:bold;">✅ PASSED</span>' if check.get('passed', True) else '<span style="color:red; font-weight:bold;">❌ FAILED</span>'
            issues_count = len(check.get('issues', []))
            details = '; '.join(check.get('issues', [])[:2]) if check.get('issues') else 'No issues detected'
            if len(details) > 100:
                details = details[:100] + '...'
            
            html.append(f'<tr><td>{name}</td><td>{status}</td><td>{issues_count}</td><td>{details}</td></tr>')
        
        html.append('</table>')
        html.append('</div>')
        
        return '\n'.join(html)
    
    def _generate_detailed_validation_analysis(self, validation_data: Dict[str, Any]) -> str:
        """Generate detailed analysis sections for each validation check."""
        html = []
        
        # 分析每个验证结果的详细信息
        for check in validation_data.get('validation_results', []):
            check_name = check.get('check_name', '')
            details = check.get('details', {})
            
            if check_name == 'feature_correlations':
                html.append(self._format_correlation_analysis(details))
            elif check_name == 'multicollinearity_detection':
                html.append(self._format_multicollinearity_analysis(details))
            elif check_name == 'feature_distributions':
                html.append(self._format_distribution_analysis(details))
            elif check_name == 'sample_balance':
                html.append(self._format_balance_analysis(details))
        
        return '\n'.join(html)
    
    def _format_correlation_analysis(self, details: Dict[str, Any]) -> str:
        """Format feature correlation analysis for HTML report."""
        if not details:
            return ""
        
        high_corr = details.get('high_correlations', [])
        correlation_summary = details.get('correlation_summary', {})
        correlation_types = correlation_summary.get('correlation_types', {})
        
        html = ['<h3>🔗 Feature Correlation Analysis (Adaptive Thresholds)</h3>']
        
        # Display adaptive thresholds info
        html.append('<div class="recommendation" style="background:#e7f3ff; border-left-color:#007bff; margin:10px 0;">')
        html.append('<p><strong>📏 Adaptive Thresholds Used:</strong></p>')
        html.append('<ul style="margin:5px 0; font-size:12px;">')
        html.append('<li><strong>Pearson/Spearman</strong> (continuous ↔ continuous): ≥ 0.7 (High)</li>')
        html.append('<li><strong>Cramér\'s V</strong> (categorical ↔ categorical): ≥ 0.5 (High)</li>')
        html.append('<li><strong>Correlation Ratio (η²)</strong> (continuous ↔ categorical): ≥ 0.14 (Large Effect)</li>')
        html.append('</ul>')
        html.append('</div>')
        
        # Summary metrics
        if correlation_types:
            html.append('<div class="metric-grid">')
            for corr_type, count in correlation_types.items():
                formatted_type = corr_type.replace('-', ' ↔ ')
                html.append(f'''<div class="metric">
                    <div class="metric-label">{formatted_type}</div>
                    <div class="metric-value">{count}</div>
                </div>''')
            html.append('</div>')
        
        if high_corr:
            html.append(f'<p><strong>Found {len(high_corr)} highly correlated feature pairs:</strong></p>')
            html.append('<div style="max-height:250px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:5px;">')
            html.append('<table style="width:100%; font-size:12px;"><tr><th>Feature 1</th><th>Feature 2</th><th>Type</th><th>Method</th><th>Value</th><th>Significance</th></tr>')
            
            for corr in high_corr[:15]:  # 显示前15个
                corr_type = corr.get('correlation_type', 'N/A')
                max_corr = corr.get('max_correlation', 0)
                
                # Determine method and significance based on correlation type and value
                if 'continuous-continuous' in corr_type:
                    method = 'Pearson/Spearman'
                    if max_corr >= 0.9:
                        significance = 'Very High'
                        color = 'red'
                    elif max_corr >= 0.7:
                        significance = 'High'
                        color = 'orange'
                    else:
                        significance = 'Moderate'
                        color = 'green'
                elif 'categorical-categorical' in corr_type:
                    method = "Cramér's V"
                    if max_corr >= 0.7:
                        significance = 'Very High'
                        color = 'red'
                    elif max_corr >= 0.5:
                        significance = 'High'
                        color = 'orange'
                    else:
                        significance = 'Moderate'
                        color = 'green'
                elif 'continuous-categorical' in corr_type:
                    method = 'Correlation Ratio (η²)'
                    if max_corr >= 0.14:
                        significance = 'Large Effect'
                        color = 'red'
                    elif max_corr >= 0.06:
                        significance = 'Medium Effect'
                        color = 'orange'
                    else:
                        significance = 'Small Effect'
                        color = 'green'
                else:
                    method = 'Mixed'
                    significance = 'Unknown'
                    color = 'gray'
                
                html.append(f'''<tr>
                    <td>{corr.get('feature1', 'N/A')}</td>
                    <td>{corr.get('feature2', 'N/A')}</td>
                    <td>{corr_type.replace('-', ' ↔ ')}</td>
                    <td>{method}</td>
                    <td style="color:{color}; font-weight:bold;">{max_corr:.3f}</td>
                    <td style="color:{color};">{significance}</td>
                </tr>''')
            
            html.append('</table></div>')
            
            # Add actionable recommendations
            if len(high_corr) > 0:
                html.append('<div class="recommendation" style="background:#fff3cd; border-left-color:#ffc107; margin:10px 0;">')
                html.append('<p><strong>💡 Recommendations:</strong></p>')
                html.append('<ul style="font-size:12px;">')
                
                continuous_pairs = [c for c in high_corr if 'continuous-continuous' in c.get('correlation_type', '')]
                categorical_pairs = [c for c in high_corr if 'categorical-categorical' in c.get('correlation_type', '')]
                mixed_pairs = [c for c in high_corr if 'continuous-categorical' in c.get('correlation_type', '')]
                
                if continuous_pairs:
                    html.append(f'<li><strong>Continuous Features</strong>: Consider PCA or removing one feature from {len(continuous_pairs)} highly correlated pairs</li>')
                if categorical_pairs:
                    html.append(f'<li><strong>Categorical Features</strong>: Consider merging categories or feature engineering for {len(categorical_pairs)} pairs</li>')
                if mixed_pairs:
                    html.append(f'<li><strong>Mixed Correlations</strong>: Review {len(mixed_pairs)} continuous-categorical associations for feature engineering opportunities</li>')
                
                html.append('<li><strong>General</strong>: Use regularized models (Ridge, Lasso) to handle multicollinearity</li>')
                html.append('</ul>')
                html.append('</div>')
        else:
            html.append('<p style="color:green;">✅ No highly correlated features detected using adaptive thresholds.</p>')
        
        return '\n'.join(html)
    
    def _format_multicollinearity_analysis(self, details: Dict[str, Any]) -> str:
        """Format multicollinearity analysis for HTML report."""
        if not details:
            return ""
        
        vif_threshold = details.get('vif_threshold', 5.0)
        high_vif = details.get('high_vif_features', [])
        avg_vif = details.get('average_vif', 0)
        max_vif = details.get('max_vif', 0)
        excluded_features = details.get('excluded_features', [])
        high_cardinality_features = details.get('high_cardinality_features', [])
        multicollinear_pairs = details.get('multicollinear_pairs', [])
        
        html = [f'<h3>📊 Multicollinearity Analysis (VIF threshold: {vif_threshold})</h3>']
        
        html.append(f'''
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Average VIF</div>
                <div class="metric-value">{avg_vif:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Maximum VIF</div>
                <div class="metric-value">{max_vif:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">High VIF Features</div>
                <div class="metric-value">{len(high_vif)}</div>
            </div>
        </div>''')
        
        # Display excluded high cardinality features
        if excluded_features:
            html.append('<div class="recommendation" style="background:#fff3cd; border-left-color:#ffc107; margin:10px 0;">')
            html.append(f'<p><strong>⚠️ Excluded High Cardinality Features ({len(excluded_features)}):</strong></p>')
            html.append('<div style="max-height:150px; overflow-y:auto; border:1px solid #ddd; padding:8px; border-radius:3px; background:#f8f9fa;">')
            
            for feature_info in high_cardinality_features:
                feature_name = feature_info.get('feature', 'N/A')
                unique_count = feature_info.get('unique_count', 0)
                total_count = feature_info.get('total_count', 0)
                cardinality_ratio = feature_info.get('cardinality_ratio', 0)
                
                html.append(f'''<div style="margin-bottom:5px; font-size:12px;">
                    <strong>{feature_name}</strong>: {unique_count:,} unique values / {total_count:,} total ({cardinality_ratio:.1%} unique)
                </div>''')
            
            html.append('</div>')
            html.append('<p style="font-size:12px; color:#856404; margin-top:8px;">Rationale: Features with >80% unique values or >50 categories excluded to prevent overfitting and computational issues.</p>')
            html.append('</div>')
        
        # Display highly correlated feature pairs
        if multicollinear_pairs:
            html.append(f'<p><strong>Highly Correlated Feature Pairs ({len(multicollinear_pairs)}):</strong></p>')
            html.append('<div style="max-height:200px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:5px;">')
            html.append('<table style="width:100%; font-size:12px;"><tr><th>Feature 1</th><th>Feature 2</th><th>Type</th><th>Method</th><th>Correlation</th></tr>')
            
            for pair in multicollinear_pairs[:10]:  # Show top 10 pairs
                corr_type = pair.get('correlation_type', 'N/A')
                max_corr = pair.get('max_correlation', 0)
                
                # Determine method based on correlation type
                if 'continuous-continuous' in corr_type:
                    method = 'Pearson/Spearman'
                elif 'categorical-categorical' in corr_type:
                    method = "Cramér's V"
                elif 'continuous-categorical' in corr_type:
                    method = 'Correlation Ratio (η²)'
                else:
                    method = 'Mixed'
                
                color = "red" if max_corr >= 0.9 else "orange" if max_corr >= 0.7 else "green"
                
                html.append(f'''<tr>
                    <td>{pair.get('feature1', 'N/A')}</td>
                    <td>{pair.get('feature2', 'N/A')}</td>
                    <td>{corr_type.replace('-', ' ↔ ')}</td>
                    <td>{method}</td>
                    <td style="color:{color}; font-weight:bold;">{max_corr:.3f}</td>
                </tr>''')
            
            html.append('</table></div>')
        
        if high_vif:
            html.append('<p><strong>Features with high VIF scores:</strong></p>')
            html.append('<div style="max-height:200px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:5px;">')
            html.append('<table style="width:100%; font-size:12px;"><tr><th>Feature</th><th>VIF Score</th><th>R²</th><th>Encoding</th></tr>')
            
            for feature in high_vif[:8]:
                vif_score = feature.get('vif_score', 0)
                r_squared = feature.get('r_squared', 0)
                encoding_method = feature.get('encoding_method', 'N/A')
                color = "red" if vif_score >= 10 else "orange"
                html.append(f'''<tr>
                    <td>{feature.get('feature', 'N/A')}</td>
                    <td style="color:{color}; font-weight:bold;">{vif_score:.2f}</td>
                    <td>{r_squared:.3f}</td>
                    <td>{encoding_method}</td>
                </tr>''')
            
            html.append('</table></div>')
        
        return '\n'.join(html)
    
    def _format_distribution_analysis(self, details: Dict[str, Any]) -> str:
        """Format feature distribution analysis for HTML report."""
        if not details:
            return ""
        
        cont_dist = details.get('continuous_distributions', {})
        cat_dist = details.get('categorical_distributions', {})
        issues = details.get('distribution_issues', [])
        
        html = ['<h3>📈 Feature Distribution Analysis</h3>']
        
        # 分布问题概览
        if issues:
            html.append(f'<p><strong>Found {len(issues)} distribution issues:</strong></p>')
            html.append('<div class="recommendation" style="background:#fff3cd; border-left-color:#ffc107; margin:10px 0;">')
            html.append('<ul>')
            for issue in issues[:5]:
                html.append(f'<li>{issue}</li>')
            html.append('</ul></div>')
        
        # 连续特征分析
        if cont_dist:
            html.append('<h4>Continuous Features Analysis</h4>')
            html.append('<div style="max-height:300px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:5px;">')
            html.append('<table style="width:100%; font-size:11px;"><tr><th>Feature</th><th>Skewness</th><th>Outliers (%)</th><th>Normality</th><th>Issues</th></tr>')
            
            for feature, data in list(cont_dist.items())[:10]:
                # Ensure data is a dictionary before calling .get()
                if not isinstance(data, dict):
                    continue
                    
                skewness = data.get('skewness_analysis', {}).get('skewness', 0)
                outlier_ratio = data.get('outlier_detection', {}).get('outlier_ratio', 0) * 100
                normality = "Yes" if data.get('normality_tests', {}).get('is_normal', False) else "No"
                issues = len(data.get('identified_issues', []))
                
                skew_color = "red" if abs(skewness) > 2 else "orange" if abs(skewness) > 1 else "green"
                outlier_color = "red" if outlier_ratio > 10 else "orange" if outlier_ratio > 5 else "green"
                
                html.append(f'''<tr>
                    <td>{feature}</td>
                    <td style="color:{skew_color};">{skewness:.2f}</td>
                    <td style="color:{outlier_color};">{outlier_ratio:.1f}%</td>
                    <td>{normality}</td>
                    <td>{issues}</td>
                </tr>''')
            
            html.append('</table></div>')
        
        # 分类特征分析
        if cat_dist:
            html.append('<h4>Categorical Features Analysis</h4>')
            html.append('<div style="max-height:300px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:5px;">')
            html.append('<table style="width:100%; font-size:11px;"><tr><th>Feature</th><th>Classes</th><th>Imbalance Ratio</th><th>Dominant Class</th><th>Issues</th></tr>')
            
            for feature, data in list(cat_dist.items())[:10]:
                # Ensure data is a dictionary before calling .get()
                if not isinstance(data, dict):
                    continue
                    
                num_classes = data.get('basic_stats', {}).get('unique_count', 0)
                imbalance_ratio = data.get('imbalance_analysis', {}).get('imbalance_ratio', 1)
                dominant_class = data.get('imbalance_analysis', {}).get('most_common_class', 'N/A')
                issues = len(data.get('identified_issues', []))
                
                ratio_color = "red" if imbalance_ratio > 10 else "orange" if imbalance_ratio > 5 else "green"
                
                html.append(f'''<tr>
                    <td>{feature}</td>
                    <td>{num_classes}</td>
                    <td style="color:{ratio_color};">{imbalance_ratio:.1f}:1</td>
                    <td>{dominant_class}</td>
                    <td>{issues}</td>
                </tr>''')
            
            html.append('</table></div>')
        
        return '\n'.join(html)
    
    def _format_balance_analysis(self, details: Dict[str, Any]) -> str:
        """Format sample balance analysis for HTML report."""
        if not details:
            return ""
        
        html = ['<h3>⚖️ Sample Balance Analysis</h3>']
        
        minority_ratio = details.get('minority_class_ratio', 0)
        is_balanced = details.get('is_balanced', True)
        target_dist = details.get('target_distribution', {})
        
        balance_color = "green" if is_balanced else "red"
        
        html.append(f'''
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Minority Class Ratio</div>
                <div class="metric-value" style="color:{balance_color};">{minority_ratio:.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Dataset Balance</div>
                <div class="metric-value">{"✅ Balanced" if is_balanced else "❌ Imbalanced"}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Number of Classes</div>
                <div class="metric-value">{len(target_dist)}</div>
            </div>
        </div>''')
        
        if target_dist:
            html.append('<h4>Class Distribution</h4>')
            html.append('<table style="width:100%;"><tr><th>Class</th><th>Count</th><th>Percentage</th></tr>')
            
            total = sum(target_dist.values())
            for class_val, count in sorted(target_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                html.append(f'<tr><td>{class_val}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>')
            
            html.append('</table>')
        
        return '\n'.join(html)
    
    def _add_dataset_overview_section(self, training_results: Dict[str, Any]) -> str:
        """Add dataset overview section to HTML report."""
        data_shape = training_results.get('data_shape', {})
        feature_names = training_results.get('feature_names', [])
        
        return f"""
        <div class="section">
            <h2>📈 Dataset Overview</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Number of Samples</div>
                    <div class="metric-value">{self._safe_format_numeric(data_shape.get('n_samples'), ':,d')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Number of Features</div>
                    <div class="metric-value">{data_shape.get('n_features', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Features per Sample Ratio</div>
                    <div class="metric-value">
                        {self._safe_format_numeric(data_shape.get('n_features', 0) / max(data_shape.get('n_samples', 1), 1), '.4f')}
                    </div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h3>Feature Names</h3>
                <div class="json-container">
                    {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}
                    {f' (showing 10 of {len(feature_names)} features)' if len(feature_names) > 10 else f' (total: {len(feature_names)} features)'}
                </div>
            </div>
        </div>"""
    
    def _add_model_configuration_section(self, training_results: Dict[str, Any]) -> str:
        """Add model configuration section to HTML report."""
        hyperparameters = training_results.get('hyperparameters', {})
        model_info = training_results.get('model_info', {})
        
        # Create hyperparameters table
        hyperparams_table = "<table><tr><th>Parameter</th><th>Value</th></tr>"
        for param, value in hyperparameters.items():
            hyperparams_table += f"<tr><td>{param}</td><td>{value}</td></tr>"
        hyperparams_table += "</table>"
        
        return f"""
        <div class="section">
            <h2>⚙️ Model Configuration</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value">{model_info.get('task_type', 'SVM').title()}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Kernel</div>
                    <div class="metric-value">{hyperparameters.get('kernel', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Depth</div>
                    <div class="metric-value">{hyperparameters.get('max_depth', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Features</div>
                    <div class="metric-value">{hyperparameters.get('max_features', 'N/A')}</div>
                </div>
            </div>
            <h3>All Hyperparameters</h3>
            {hyperparams_table}
        </div>"""
    
    def _add_hyperparameter_optimization_section(self, training_results: Dict[str, Any]) -> str:
        """Add hyperparameter optimization section to HTML report."""
        opt_results = training_results.get('optimization_results', {})
        
        if not opt_results:
            return """
        <div class="section">
            <h2>🎯 Hyperparameter Optimization</h2>
            <p>Hyperparameter optimization was not performed for this training session.</p>
        </div>"""
        
        best_score = self._safe_format_numeric(opt_results.get('best_score'), '.6f')
        n_trials = opt_results.get('n_trials', 'N/A')
        sampler_type = opt_results.get('sampler_type', 'N/A')
        best_trial_number = opt_results.get('best_trial_number', 'N/A')
        
        # Generate hyperparameter history table
        history_table = self._generate_hyperparameter_history_table(opt_results)
        
        # Generate parameter exploration summary
        param_summary = self._generate_parameter_exploration_summary(opt_results)
        
        # Safe formatting for recommendation text
        best_score_rec = self._safe_format_numeric(opt_results.get('best_score', 0), '.6f', '0.000000')
        
        return f"""
        <div class="section">
            <h2>🎯 Hyperparameter Optimization</h2>
            <div class="metric-grid">
                <div class="metric success">
                    <div class="metric-label">Best Score</div>
                    <div class="metric-value">{best_score}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Trial</div>
                    <div class="metric-value">#{best_trial_number}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Trials</div>
                    <div class="metric-value">{n_trials}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Algorithm</div>
                    <div class="metric-value">{sampler_type.upper() if isinstance(sampler_type, str) else sampler_type}</div>
                </div>
            </div>
            
            <h3>📊 Parameter Exploration Summary</h3>
            {param_summary}
            
            <h3>📋 Hyperparameter Trial History</h3>
            <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px;">
                {history_table}
            </div>
            
            <div class="recommendation">
                <strong>💡 Optimization Result:</strong> 
                The hyperparameter optimization completed {n_trials} trials 
                and found the best score of {best_score_rec} using the 
                {sampler_type} algorithm. The best configuration was discovered in trial #{best_trial_number}.
            </div>
        </div>"""
    
    def _generate_hyperparameter_history_table(self, opt_results: Dict[str, Any]) -> str:
        """Generate detailed hyperparameter history table."""
        history = opt_results.get('optimization_history', [])
        
        if not history:
            return "<p>No trial history available.</p>"
        
        # Create table header
        table_html = """
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="border: 1px solid #ddd; padding: 8px;">Trial</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">State</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">n_estimators</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">max_depth</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">min_samples_split</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">min_samples_leaf</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">max_features</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">bootstrap</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Sort trials by trial number
        sorted_history = sorted(history, key=lambda x: x.get('trial_number', 0))
        best_score = opt_results.get('best_score')
        
        for trial in sorted_history:
            trial_num = trial.get('trial_number', 'N/A')
            score = self._safe_format_numeric(trial.get('value'), '.6f')
            state = trial.get('state', 'UNKNOWN')
            params = trial.get('params', {})
            
            # Highlight best trial
            row_style = ""
            if trial.get('value') == best_score:
                row_style = 'style="background-color: #d4edda; font-weight: bold;"'
            elif state != 'COMPLETE':
                row_style = 'style="background-color: #f8d7da;"'
            
            table_html += f"""
                <tr {row_style}>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{trial_num}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{score}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{state}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{params.get('n_estimators', 'N/A')}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{params.get('max_depth', 'N/A')}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{params.get('min_samples_split', 'N/A')}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{params.get('min_samples_leaf', 'N/A')}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{params.get('max_features', 'N/A')}</td>
                    <td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{params.get('bootstrap', 'N/A')}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 10px;">
            <span style="background-color: #d4edda; padding: 2px 6px; border-radius: 3px;">Green</span> = Best trial &nbsp;
            <span style="background-color: #f8d7da; padding: 2px 6px; border-radius: 3px;">Red</span> = Failed trial
        </p>
        """
        
        return table_html
    
    def _generate_parameter_exploration_summary(self, opt_results: Dict[str, Any]) -> str:
        """Generate parameter exploration summary statistics."""
        history = opt_results.get('optimization_history', [])
        
        if not history:
            return "<p>No parameter exploration data available.</p>"
        
        # Collect parameter statistics
        param_stats = {}
        completed_trials = [t for t in history if t.get('state') == 'COMPLETE']
        
        if not completed_trials:
            return "<p>No completed trials available for analysis.</p>"
        
        # Analyze each parameter
        for trial in completed_trials:
            params = trial.get('params', {})
            for param_name, param_value in params.items():
                if param_name not in param_stats:
                    param_stats[param_name] = []
                
                # Handle different parameter types
                if isinstance(param_value, (int, float)):
                    param_stats[param_name].append(param_value)
                else:
                    param_stats[param_name].append(str(param_value))
        
        # Generate summary table
        summary_html = """
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="border: 1px solid #ddd; padding: 8px;">Parameter</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Explored Range</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Best Value</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Unique Values</th>
                </tr>
            </thead>
            <tbody>
        """
        
        best_params = opt_results.get('best_params', {})
        
        for param_name, values in param_stats.items():
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric parameter
                min_val = min(values)
                max_val = max(values)
                explored_range = f"{min_val} - {max_val}"
                unique_count = len(set(values))
            else:
                # Categorical parameter
                unique_vals = list(set(values))
                explored_range = ", ".join(map(str, unique_vals[:3]))
                if len(unique_vals) > 3:
                    explored_range += "..."
                unique_count = len(unique_vals)
            
            best_value = best_params.get(param_name, 'N/A')
            
            summary_html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">{param_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{explored_range}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{best_value}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{unique_count}</td>
                </tr>
            """
        
        summary_html += """
            </tbody>
        </table>
        """
        
        # Add exploration statistics
        total_trials = len(history)
        completed_trials_count = len(completed_trials)
        failed_trials = total_trials - completed_trials_count
        success_rate = (completed_trials_count / total_trials * 100) if total_trials > 0 else 0
        
        summary_html += f"""
        <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <strong>Exploration Statistics:</strong><br>
            • Total trials: {total_trials}<br>
            • Completed: {completed_trials_count}<br>
            • Failed: {failed_trials}<br>
            • Success rate: {self._safe_format_numeric(success_rate, '.1f')}%
        </div>
        """
        
        return summary_html
    
    def _safe_format_numeric(self, value: Any, format_spec: str = "", default: str = "N/A") -> str:
        """Safely format numeric values, handling None, strings, and missing values."""
        if value is None:
            return default
        
        try:
            if isinstance(value, (int, float)):
                if format_spec:
                    return f"{value:{format_spec}}"
                return str(value)
            elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                # Try to convert string numbers
                num_val = float(value)
                if format_spec:
                    return f"{num_val:{format_spec}}"
                return str(num_val)
            else:
                return str(value) if value != 'N/A' else default
        except (ValueError, TypeError):
            return default 
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name with proper capitalization and direction indicators."""
        # Handle regression metrics
        if metric == 'R2':
            return "R² (higher is better)"
        elif metric == 'MAE':
            return "MAE (lower is better)"
        elif metric == 'MSE':
            return "MSE (lower is better)"
        elif metric == 'RMSE':
            return "RMSE (lower is better)"
        elif metric == 'MAPE':
            return "MAPE (lower is better)"
        elif metric.upper() in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            return f"{metric.upper()} (lower is better)"
        elif metric.upper() == 'R2':
            return "R² (higher is better)"
        
        # Handle classification metrics
        elif metric == 'accuracy':
            return "Accuracy (higher is better)"
        elif metric == 'precision':
            return "Precision (higher is better)"
        elif metric == 'recall':
            return "Recall (higher is better)"
        elif metric == 'f1':
            return "F1-Score (higher is better)"
        elif metric == 'f1_score':
            return "F1-Score (higher is better)"
        elif metric == 'roc_auc':
            return "ROC-AUC (higher is better)"
        elif metric == 'auc':
            return "AUC (higher is better)"
        elif metric.upper() in ['ACCURACY', 'PRECISION', 'RECALL', 'F1', 'F1_SCORE', 'ROC_AUC', 'AUC']:
            return f"{metric.replace('_', '-').title()} (higher is better)"
        
        else:
            # Default formatting for other metrics
            return metric.replace('_', ' ').title()

    def _add_cross_validation_section(self, training_results: Dict[str, Any]) -> str:
        """Add enhanced cross-validation section to HTML report with detailed metrics and visualizations."""
        cv_results = training_results.get('cross_validation_results', {})
        test_scores = training_results.get('test_scores', cv_results.get('test_scores', {}))
        train_scores = cv_results.get('train_scores', {})
        
        # Create detailed CV results table with enhanced metrics
        cv_table = "<table><tr><th>Metric</th><th>Mean (Test)</th><th>Std (Test)</th><th>Min</th><th>Max</th><th>95% CI</th>"
        if train_scores:
            cv_table += "<th>Mean (Train)</th><th>Std (Train)</th>"
        cv_table += "</tr>"
        
        for metric, scores in test_scores.items():
            if isinstance(scores, dict):
                mean_score = self._safe_format_numeric(scores.get('mean', 0), '.6f')
                std_score = self._safe_format_numeric(scores.get('std', 0), '.6f')
                min_score = self._safe_format_numeric(scores.get('min', 0), '.6f')
                max_score = self._safe_format_numeric(scores.get('max', 0), '.6f')
                
                # Calculate 95% confidence interval if we have individual scores
                ci_text = "N/A"
                if 'scores' in scores and scores['scores']:
                    import numpy as np
                    scores_array = np.array(scores['scores'])
                    n = len(scores_array)
                    if n > 1:
                        mean_val = np.mean(scores_array)
                        std_val = np.std(scores_array)
                        ci_margin = 1.96 * std_val / np.sqrt(n)
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                        ci_text = f"[{ci_lower:.6f}, {ci_upper:.6f}]"
                
                metric_name = str(metric or 'Unknown').replace('_', ' ').upper()
                cv_table += f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td>{mean_score}</td>
                    <td>{std_score}</td>
                    <td>{min_score}</td>
                    <td>{max_score}</td>
                    <td>{ci_text}</td>"""
                
                if train_scores and metric in train_scores:
                    train_score = train_scores[metric]
                    if isinstance(train_score, dict) and 'mean' in train_score:
                        train_mean = self._safe_format_numeric(train_score['mean'], '.6f')
                        train_std = self._safe_format_numeric(train_score['std'], '.6f')
                        cv_table += f"""
                        <td>{train_mean}</td>
                        <td>{train_std}</td>"""
                    else:
                        cv_table += "<td>N/A</td><td>N/A</td>"
                
                cv_table += "</tr>"
        
        cv_table += "</table>"
        
        # Add fold-wise results table if detailed scores are available
        fold_table = ""
        if test_scores:
            # Check if we have individual fold scores
            first_metric = next(iter(test_scores.values()))
            if isinstance(first_metric, dict) and 'scores' in first_metric:
                fold_table = self._generate_fold_wise_table(test_scores)
        
        # Add visualization section
        visualization_section = self._add_cross_validation_visualizations(training_results)
        
        return f"""
        <div class="section">
            <h2>📋 Cross-Validation Performance</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">CV Folds</div>
                    <div class="metric-value">{cv_results.get('cv_folds', training_results.get('cv_folds', 'N/A'))}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">CV Strategy</div>
                    <div class="metric-value">{cv_results.get('cv_strategy', 'Standard K-Fold')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Scoring Metrics</div>
                    <div class="metric-value">{len(test_scores)} metrics</div>
                </div>
            </div>
            
            <h3>📊 Performance Summary</h3>
            {cv_table}
            
            {fold_table}
            
            {visualization_section}
            
            <div class="recommendation">
                <strong>📈 Performance Insight:</strong> 
                Cross-validation provides robust performance estimates across {cv_results.get('cv_folds', training_results.get('cv_folds', 5))} folds. 
                Multiple metrics (MAE, MSE, R²) offer comprehensive model evaluation. 
                Low standard deviation indicates consistent performance across different data splits.
            </div>
        </div>"""

    def _generate_fold_wise_table(self, test_scores: Dict[str, Any]) -> str:
        """Generate detailed fold-wise results table."""
        # Get number of folds from first metric
        first_metric = next(iter(test_scores.values()))
        if not isinstance(first_metric, dict) or 'scores' not in first_metric:
            return ""
        
        n_folds = len(first_metric['scores'])
        if n_folds == 0:
            return ""
        
        # Create fold-wise table
        fold_table = """
            <h3>🎯 Fold-wise Performance Details</h3>
            <table>
                <tr>
                    <th>Fold</th>"""
        
        for metric in test_scores.keys():
            metric_name = metric.replace('_', ' ').upper()
            fold_table += f"<th>{metric_name}</th>"
        
        fold_table += "</tr>"
        
        # Add rows for each fold
        for fold_idx in range(n_folds):
            fold_table += f"<tr><td><strong>Fold {fold_idx + 1}</strong></td>"
            
            for metric_name, metric_data in test_scores.items():
                if isinstance(metric_data, dict) and 'scores' in metric_data:
                    score = metric_data['scores'][fold_idx]
                    fold_table += f"<td>{self._safe_format_numeric(score, '.6f')}</td>"
                else:
                    fold_table += "<td>N/A</td>"
            
            fold_table += "</tr>"
        
        fold_table += "</table>"
        return fold_table

    def _add_cross_validation_visualizations(self, training_results: Dict[str, Any]) -> str:
        """Add cross-validation visualization section with image support."""
        # Initialize cv_plots list
        cv_plots = []
        
        # Check for visualization files in model directory
        model_path = training_results.get('model_path', '')
        if model_path:
            from pathlib import Path
            model_dir = Path(model_path).parent
            
            # Look for common CV visualization files
            plot_extensions = ['.png', '.jpg', '.jpeg', '.svg']
            
            if model_dir.exists():
                for plot_file in model_dir.glob('*'):
                    if (plot_file.suffix.lower() in plot_extensions and 
                        any(keyword in plot_file.name.lower() for keyword in 
                            ['cross_validation', 'cv_', 'fold', 'performance', 'metrics'])):
                        cv_plots.append(plot_file.name)
        
        # Generate visualization section
        viz_section = ""
        if cv_plots:
            viz_section = """
            <h3>📈 Cross-Validation Visualizations</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0;">"""
            
            for plot_file in cv_plots:
                # Determine plot description based on filename
                if 'roc' in plot_file.lower():
                    description = "ROC curves showing classification performance across different classes"
                elif 'scatter' in plot_file.lower():
                    description = "Cross-validation predictions vs actual values scatter plot"
                elif 'fold' in plot_file.lower():
                    description = "Performance across CV folds"
                elif 'metrics' in plot_file.lower():
                    description = "Multiple metrics comparison"
                elif 'classification' in plot_file.lower():
                    description = "Classification performance visualization"
                else:
                    description = "Cross-validation performance visualization"
                
                viz_section += f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #fafafa;">
                    <img src="{plot_file}" alt="{description}" style="width: 100%; height: auto; border-radius: 4px; margin-bottom: 10px;">
                    <p style="margin: 0; font-size: 0.9em; color: #666; text-align: center;">
                        <strong>{plot_file}</strong><br>
                        {description}
                    </p>
                </div>"""
            
            viz_section += """
            </div>"""
        else:
            # Fallback message when no visualizations are found
            viz_section = """
            <h3>📈 Visualizations</h3>
            <div class="recommendation">
                <strong>📊 Visualization Note:</strong> 
                Cross-validation visualizations (scatter plots for regression, ROC curves for classification, fold performance charts) would be displayed here when available.
                These plots help visualize prediction accuracy and model consistency across different data splits.
            </div>"""
        
        return viz_section

    def _add_feature_importance_section(self, training_results: Dict[str, Any]) -> str:
        """Add enhanced feature importance section to HTML report with visualization support."""
        feature_importance = training_results.get('feature_importance', {})
        
        if not feature_importance:
            return """
        <div class="section">
            <h2>🔍 Feature Importance Analysis</h2>
            <p>Feature importance analysis was not available for this training session.</p>
        </div>"""
        
        # Get top features
        importance_scores = feature_importance.get('importance_scores', {})
        top_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Create importance table
        importance_table = "<table><tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Percentage</th></tr>"
        total_importance = sum(importance_scores.values()) if importance_scores else 1
        
        for rank, (feature, score) in enumerate(top_features, 1):
            formatted_score = self._safe_format_numeric(score, '.6f')
            percentage = (score / total_importance) * 100 if total_importance > 0 else 0
            formatted_percentage = self._safe_format_numeric(percentage, '.2f')
            importance_table += f"<tr><td>{rank}</td><td><strong>{feature}</strong></td><td>{formatted_score}</td><td>{formatted_percentage}%</td></tr>"
        importance_table += "</table>"
        
        # Add feature importance visualizations
        visualization_section = self._add_feature_importance_visualizations(training_results)
        
        return f"""
        <div class="section">
            <h2>🔍 Feature Importance Analysis</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Analysis Type</div>
                    <div class="metric-value">{feature_importance.get('analysis_type', 'Tree-based Importance')}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Top Feature</div>
                    <div class="metric-value">{top_features[0][0] if top_features else 'N/A'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Top Score</div>
                    <div class="metric-value">{self._safe_format_numeric(top_features[0][1], '.6f') if top_features else 'N/A'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Features</div>
                    <div class="metric-value">{len(importance_scores)}</div>
                </div>
            </div>
            
            <h3>🏆 Top 15 Most Important Features</h3>
            {importance_table}
            
            {visualization_section}
            
            <div class="recommendation">
                <strong>🎯 Feature Insight:</strong> 
                Feature importance analysis reveals which variables contribute most to model predictions. 
                SVM provides comprehensive feature importance metrics including gain, weight, and cover. 
                Top features with high importance scores should be prioritized for feature engineering and domain analysis.
            </div>
        </div>"""

    def _add_feature_importance_visualizations(self, training_results: Dict[str, Any]) -> str:
        """Add feature importance visualization section with image support."""
        # Initialize importance_plots list
        importance_plots = []
        
        # Check for visualization files in model directory
        model_path = training_results.get('model_path', '')
        if model_path:
            from pathlib import Path
            model_dir = Path(model_path).parent
            
            # Look for feature importance visualization files
            plot_extensions = ['.png', '.jpg', '.jpeg', '.svg']
            
            if model_dir.exists():
                for plot_file in model_dir.glob('*'):
                    if (plot_file.suffix.lower() in plot_extensions and 
                        any(keyword in plot_file.name.lower() for keyword in 
                            ['feature_importance', 'importance', 'features', 'permutation'])):
                        importance_plots.append(plot_file.name)
        
        # Generate visualization section
        viz_section = ""
        if importance_plots:
            viz_section = """
            <h3>📊 Feature Importance Visualizations</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0;">"""
            
            for plot_file in importance_plots:
                # Determine plot description based on filename
                if 'permutation' in plot_file.lower():
                    description = "Permutation importance analysis showing feature contribution to model performance"
                elif 'tree' in plot_file.lower() or 'basic' in plot_file.lower():
                    description = "Tree-based feature importance (Mean Decrease in Impurity)"
                elif 'comparison' in plot_file.lower():
                    description = "Comparison of different feature importance methods"
                elif 'top' in plot_file.lower():
                    description = "Top features ranked by importance score"
                else:
                    description = "Feature importance visualization"
                
                viz_section += f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #fafafa;">
                    <img src="{plot_file}" alt="{description}" style="width: 100%; height: auto; border-radius: 4px; margin-bottom: 10px;">
                    <p style="margin: 0; font-size: 0.9em; color: #666; text-align: center;">
                        <strong>{plot_file}</strong><br>
                        {description}
                    </p>
                </div>"""
            
            viz_section += """
            </div>"""
        else:
            # Fallback message when no visualizations are found
            viz_section = """
            <h3>📊 Visualizations</h3>
            <div class="recommendation">
                <strong>📈 Visualization Note:</strong> 
                Feature importance visualizations (bar charts, permutation plots) would be displayed here when available.
                These plots help identify the most influential features and compare different importance calculation methods.
            </div>"""
        
        return viz_section
    
    def _add_training_timeline_section(self, training_results: Dict[str, Any]) -> str:
        """Add training timeline section to HTML report."""
        completed_at = training_results.get('training_completed_at', 'Unknown')
        training_time = training_results.get('training_time_seconds', 0)
        
        return f"""
        <div class="section">
            <h2>⏱️ Training Timeline</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Completed At</div>
                    <div class="metric-value">{completed_at}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Training Time</div>
                    <div class="metric-value">{self._safe_format_numeric(training_time, '.2f')} seconds</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Average Time per Sample</div>
                    <div class="metric-value">
                        {self._safe_format_numeric((training_time / max(training_results.get('data_shape', {}).get('n_samples', 1), 1) * 1000), '.3f')} ms
                    </div>
                </div>
            </div>
        </div>"""
    
    def _add_recommendations_section(self, training_results: Dict[str, Any]) -> str:
        """Add recommendations section to HTML report."""
        recommendations = self._generate_recommendations(training_results)
        
        recommendations_html = ""
        for i, rec in enumerate(recommendations, 1):
            recommendations_html += f"""
            <div class="recommendation">
                <strong>💡 Recommendation {i}:</strong> {rec}
            </div>"""
        
        return f"""
        <div class="section">
            <h2>💡 Recommendations & Next Steps</h2>
            {recommendations_html}
        </div>"""
    
    def _generate_recommendations(self, training_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on training results, distinguishing between classification and regression."""
        recommendations = []
        
        try:
            # Determine task type
            task_type = training_results.get('task_type', 'unknown')
            logger.debug(f"Generating recommendations for task_type: {task_type}")
            
            # Performance-based recommendations based on task type
            cv_results = training_results.get('cross_validation_results', {})
            
            # Handle case where cv_results might be a string (due to previous serialization issues)
            if isinstance(cv_results, str):
                try:
                    import json
                    cv_results = json.loads(cv_results)
                    logger.debug("Successfully parsed cv_results from string to dict")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse cv_results string: {e}")
                    cv_results = {}
            
            test_scores = cv_results.get('test_scores', {})
            logger.debug(f"Found test_scores keys: {list(test_scores.keys()) if test_scores else 'None'}")
            
            if test_scores:
                if task_type == 'classification':
                    # For classification models, use accuracy for performance evaluation
                    # Try both 'accuracy' and 'ACCURACY' keys for backward compatibility
                    accuracy_data = test_scores.get('accuracy') or test_scores.get('ACCURACY')
                    if accuracy_data and isinstance(accuracy_data, dict):
                        acc_mean = accuracy_data.get('mean', 0)
                        acc_std = accuracy_data.get('std', 0)
                        
                        logger.debug(f"Classification accuracy: mean={acc_mean:.3f}, std={acc_std:.3f}")
                        
                        if acc_mean > 0.90:
                            recommendations.append("🎯 Excellent classification performance (accuracy > 90%)! Consider this model ready for production.")
                        elif acc_mean > 0.85:
                            recommendations.append("✅ Good classification performance (accuracy > 85%). Consider fine-tuning hyperparameters for potential improvement.")
                        elif acc_mean > 0.70:
                            recommendations.append("📊 Moderate classification performance (accuracy > 70%). Consider feature engineering or additional data.")
                        else:
                            recommendations.append("⚠️ Low classification performance (accuracy < 70%). Consider reviewing data quality, feature selection, or trying different algorithms.")
                        
                        if acc_std > 0.1:
                            recommendations.append("⚠️ High variance detected in accuracy scores. Consider increasing training data or regularization.")
                    
                    # Additional classification-specific recommendations
                    f1_data = test_scores.get('f1') or test_scores.get('F1')
                    if f1_data and isinstance(f1_data, dict):
                        f1_mean = f1_data.get('mean', 0)
                        logger.debug(f"F1 score: mean={f1_mean:.3f}")
                        if f1_mean < 0.7:
                            recommendations.append("📊 F1-score could be improved. Consider addressing class imbalance or adjusting classification threshold.")
                    
                    precision_data = test_scores.get('precision') or test_scores.get('PRECISION')
                    recall_data = test_scores.get('recall') or test_scores.get('RECALL')
                    if precision_data and recall_data and isinstance(precision_data, dict) and isinstance(recall_data, dict):
                        precision_mean = precision_data.get('mean', 0)
                        recall_mean = recall_data.get('mean', 0)
                        
                        logger.debug(f"Precision: {precision_mean:.3f}, Recall: {recall_mean:.3f}")
                        
                        if precision_mean > 0.9 and recall_mean < 0.7:
                            recommendations.append("⚖️ High precision but low recall detected. Consider adjusting classification threshold to improve recall.")
                        elif recall_mean > 0.9 and precision_mean < 0.7:
                            recommendations.append("⚖️ High recall but low precision detected. Consider adjusting classification threshold to improve precision.")
                
                elif task_type == 'regression':
                    # For regression models, use R² for performance evaluation
                    r2_score_data = test_scores.get('R2') or test_scores.get('r2')
                    if r2_score_data and isinstance(r2_score_data, dict):
                        r2_mean = r2_score_data.get('mean', 0)
                        r2_std = r2_score_data.get('std', 0)
                        
                        logger.debug(f"Regression R²: mean={r2_mean:.3f}, std={r2_std:.3f}")
                        
                        if r2_mean > 0.90:
                            recommendations.append("🎯 Excellent regression performance (R² > 0.90)! Consider this model ready for production.")
                        elif r2_mean > 0.85:
                            recommendations.append("✅ Good regression performance (R² > 0.85). Consider fine-tuning hyperparameters for potential improvement.")
                        elif r2_mean > 0.70:
                            recommendations.append("📊 Moderate regression performance (R² > 0.70). Consider feature engineering or additional data.")
                        else:
                            recommendations.append("⚠️ Low regression performance (R² < 0.70). Consider reviewing data quality, feature selection, or trying different algorithms.")
                        
                        if r2_std > 0.1:
                            recommendations.append("⚠️ High variance detected in R² scores. Consider increasing training data or regularization.")
                    
                    # Additional regression-specific recommendations
                    mae_data = test_scores.get('MAE') or test_scores.get('mae')
                    if mae_data and isinstance(mae_data, dict):
                        mae_mean = mae_data.get('mean', 0)
                        mae_mean = abs(mae_mean) if mae_mean < 0 else mae_mean  # Handle negative MAE
                        logger.debug(f"MAE: {mae_mean:.3f}")
                        if mae_mean > 0.5:
                            recommendations.append("📊 Mean Absolute Error is relatively high. Consider feature engineering or model complexity adjustment.")
                
                else:
                    # Unknown task type - use generic approach
                    primary_metric = next(iter(test_scores.keys()), None)
                    logger.debug(f"Unknown task type, using primary metric: {primary_metric}")
                    if primary_metric:
                        score_data = test_scores[primary_metric]
                        if isinstance(score_data, dict):
                            mean_score = score_data.get('mean', 0)
                            std_score = score_data.get('std', 0)
                            
                            # For negative metrics (MAE, MSE), use absolute values for comparison
                            if primary_metric in ['MAE', 'MSE'] and mean_score < 0:
                                mean_score = abs(mean_score)
                                if mean_score < 0.1:
                                    recommendations.append(f"🎯 Excellent model performance (low {primary_metric})! Consider this model ready for production.")
                                elif mean_score < 0.3:
                                    recommendations.append(f"✅ Good model performance (low {primary_metric}). Consider fine-tuning hyperparameters for potential improvement.")
                                else:
                                    recommendations.append(f"📊 Model performance could be improved (high {primary_metric}). Consider feature engineering or different algorithms.")
                            elif primary_metric.upper() in ['ACCURACY', 'F1', 'PRECISION', 'RECALL']:
                                if mean_score > 0.85:
                                    recommendations.append(f"🎯 Excellent model performance ({primary_metric} > 0.85)! Consider this model ready for production.")
                                elif mean_score > 0.70:
                                    recommendations.append(f"✅ Good model performance ({primary_metric} > 0.70). Consider fine-tuning hyperparameters for potential improvement.")
                                else:
                                    recommendations.append(f"📊 Model performance could be improved ({primary_metric} < 0.70). Consider feature engineering or different algorithms.")
                            
                            if std_score > 0.1:
                                recommendations.append("⚠️ High variance detected. Consider increasing training data or regularization.")
            
            # Feature importance recommendations
            feature_importance = training_results.get('feature_importance', {})
            print('cccccccccccccccccccccccccccccccccccccccc')
            print(feature_importance)
            if feature_importance and isinstance(feature_importance, dict):
                logger.debug(f"Found feature importance data for {len(feature_importance)} features")
                # Find features with very low importance
                low_importance_features = []
                importance_values = []
                
                for feature, data in feature_importance.items():
                    # Ensure data is a dictionary before calling .get()
                    if not isinstance(data, dict):
                        continue
                        
                    # For XGBoost, use 'gain' as the primary importance metric
                    # (equivalent to what was called 'tree_importance' in random forest)
                    importance_value = data.get('gain', 0)
                    if isinstance(importance_value, (int, float)) and importance_value < 0.01:
                        low_importance_features.append(feature)
                    importance_values.append(importance_value if isinstance(importance_value, (int, float)) else 0)
                
                if low_importance_features:
                    recommendations.append(f"🔍 Consider removing low-importance features: {', '.join(low_importance_features[:3])}")
                
                # Check for dominant features
                max_importance = max(importance_values) if importance_values else 0
                
                if max_importance > 0.5:
                    recommendations.append("⚖️ One feature dominates importance. Verify data quality and consider feature engineering.")
            
            # Hyperparameter optimization recommendations
            opt_results = training_results.get('optimization_results', {})
            
            # Handle case where opt_results might be a string (due to previous serialization issues)
            if isinstance(opt_results, str):
                try:
                    import json
                    opt_results = json.loads(opt_results)
                    logger.debug("Successfully parsed opt_results from string to dict")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse opt_results string: {e}")
                    opt_results = {}
            
            if opt_results and isinstance(opt_results, dict):
                logger.debug(f"Found optimization results with keys: {list(opt_results.keys())}")
                
                # Check best score and n_trials for recommendations
                best_score = opt_results.get('best_score', 0)
                n_trials = opt_results.get('n_trials', 0)
                
                logger.debug(f"Optimization: best_score={best_score}, n_trials={n_trials}")
                
                # For classification, best_score should be positive and high
                # For regression, it depends on the scoring metric used
                if task_type == 'classification' and best_score > 0:
                    if best_score < 0.8:
                        recommendations.append("📊 Hyperparameter optimization score could be improved. Consider feature engineering or different algorithms.")
                elif task_type == 'regression':
                    # For regression, best_score from optimization is usually R² or negative error
                    if best_score > 0 and best_score < 0.8:  # Assuming R² score
                        recommendations.append("📊 Model R² could be improved. Consider feature engineering or different algorithms.")
                    elif best_score < 0 and best_score < -0.3:  # Assuming negative error score
                        recommendations.append("📊 Model error is relatively high. Consider feature engineering or different algorithms.")
                
                if n_trials < 50:
                    recommendations.append("🔄 Consider running more optimization trials for better hyperparameter exploration.")
            
            # Data quality recommendations
            data_validation = training_results.get('data_validation')
            if data_validation and isinstance(data_validation, dict):
                quality_score = data_validation.get('overall_quality_score', 100)
                logger.debug(f"Data quality score: {quality_score}")
                
                if quality_score < 80:
                    recommendations.append("🧹 Data quality issues detected. Review data preprocessing and cleaning steps.")
            
            # Training time recommendations
            training_time = training_results.get('training_time_seconds', 0)
            if training_time > 300:  # More than 5 minutes
                recommendations.append("⏱️ Long training time detected. Consider feature selection or model simplification for faster iterations.")
            
            # Model complexity recommendations based on hyperparameters
            hyperparams = training_results.get('hyperparameters', {})
            if hyperparams:
                n_estimators = hyperparams.get('n_estimators', 0)
                max_depth = hyperparams.get('max_depth', 0)
                
                if n_estimators > 500:
                    recommendations.append("🌲 High number of trees detected. Consider reducing n_estimators for faster prediction times.")
                if max_depth and max_depth > 25:
                    recommendations.append("📏 Very deep trees detected. Consider reducing max_depth to prevent overfitting.")
            
            # Default recommendation if none generated
            if not recommendations:
                logger.debug("No specific recommendations generated, using default")
                recommendations.append("📈 Model training completed successfully. Monitor performance on new data.")
                
            logger.debug(f"Generated {len(recommendations)} recommendations")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            recommendations.append("📋 Review training results and consider model validation on test data.")
        
        return recommendations

    def generate_training_report_for_model_directory(self, model_directory: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate comprehensive training report and save to model directory.
        This method is compatible with the TrainingEngine interface.
        
        Args:
            model_directory: Path to model directory
            metadata: Model metadata and training results
            
        Returns:
            str: Path to the generated HTML report
        """
        try:
            # Ensure model_directory is a Path object
            if isinstance(model_directory, str):
                model_directory = Path(model_directory)
            
            # Add model directory to metadata for image detection
            metadata['model_directory'] = str(model_directory)
            
            # Generate HTML report
            html_report = self._create_html_training_report(metadata)
            html_path = model_directory / "training_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Generate JSON report (filter out non-serializable objects)
            json_metadata = {}
            for key, value in metadata.items():
                try:
                    # Skip objects that are not JSON serializable
                    if key in ['model', 'X', 'y', 'y_true', 'y_pred']:
                        continue
                    # Test if the value is JSON serializable
                    json.dumps(value)
                    json_metadata[key] = value
                except (TypeError, ValueError):
                    # Special handling for important structured data - preserve structure
                    if key in ['cross_validation_results', 'optimization_results']:
                        # Handle complex objects by filtering out non-serializable parts
                        if isinstance(value, dict):
                            filtered_value = {}
                            for sub_key, sub_value in value.items():
                                try:
                                    # Skip DataFrame objects
                                    if hasattr(sub_value, 'to_csv') and hasattr(sub_value, 'columns'):
                                        # Skip pandas DataFrame - it's saved separately as CSV
                                        continue
                                    json.dumps(sub_value)
                                    filtered_value[sub_key] = sub_value
                                except (TypeError, ValueError):
                                    # Convert problematic objects to string representation
                                    filtered_value[sub_key] = str(sub_value)
                            json_metadata[key] = filtered_value
                        else:
                            json_metadata[key] = value
                    elif hasattr(value, 'to_dict'):
                        json_metadata[key] = value.to_dict()
                    else:
                        json_metadata[key] = str(value)
            
            json_content = self._generate_json_training_report(json_metadata)
            json_path = model_directory / "training_report.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            
            # Save feature importance as CSV
            if 'feature_importance' in metadata:
                import pandas as pd
                feature_importance_df = pd.DataFrame([
                    {
                        'feature': feature,
                        'gain': data.get('gain', 0) if isinstance(data, dict) else 0,
                        'permutation_mean': data.get('permutation_mean', 0) if isinstance(data, dict) else 0,
                        'permutation_std': data.get('permutation_std', 0) if isinstance(data, dict) else 0
                    }
                    for feature, data in metadata['feature_importance'].items()
                ])
                feature_importance_df.to_csv(model_directory / "feature_importance.csv", index=False)
            
            # Save cross-validation results as JSON
            if 'cross_validation_results' in metadata:
                cv_results_path = model_directory / "cross_validation_results.json"
                with open(cv_results_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata['cross_validation_results'], f, indent=2, default=str)
            
            # Save optimization history if available
            if metadata.get('optimization_results'):
                opt_results = metadata['optimization_results']
                
                # Save optimization history CSV from trials_dataframe
                if 'trials_dataframe' in opt_results and opt_results['trials_dataframe'] is not None:
                    trials_df = opt_results['trials_dataframe']
                    # Check if trials_df is actually a DataFrame
                    if hasattr(trials_df, 'to_csv'):
                        opt_history_path = model_directory / "optimization_history.csv"
                        trials_df.to_csv(opt_history_path, index=False)
                        logger.info(f"Optimization history saved to: {opt_history_path}")
                    else:
                        logger.warning(f"trials_dataframe is not a DataFrame, type: {type(trials_df)}")
                elif 'optimization_history' in opt_results:
                    # Fallback: create CSV from optimization_history list
                    opt_history = opt_results['optimization_history']
                    if opt_history:
                        import pandas as pd
                        opt_history_df = pd.DataFrame(opt_history)
                        opt_history_path = model_directory / "optimization_history.csv"
                        opt_history_df.to_csv(opt_history_path, index=False)
                        logger.info(f"Optimization history (fallback) saved to: {opt_history_path}")
            
            logger.info(f"HTML training report generated in {model_directory}")
            return str(html_path)
            
        except Exception as e:
            logger.warning(f"Failed to generate HTML training report: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            return None

    def _create_html_training_report(self, metadata: Dict[str, Any]) -> str:
        """Create enhanced HTML training report with image support."""
        
        # Get model directory for image detection
        model_directory = Path(metadata.get('model_directory', ''))
        
        # Generate visualization sections
        cv_visualization_section = self._generate_cv_visualization_section(model_directory)
        feature_importance_visualization_section = self._generate_feature_importance_visualization_section(model_directory)
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {metadata.get('model_name', 'Unknown Model')}</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white; 
            border-radius: 10px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            overflow: hidden; 
        }}
        .header {{ 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
            color: white; 
            padding: 30px; 
            text-align: center; 
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
        .header p {{ margin: 5px 0 0 0; opacity: 0.9; font-size: 1.2em; }}
        .section {{ 
            margin: 0; 
            padding: 25px; 
            border-bottom: 1px solid #eee; 
        }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{ 
            color: #2a5298; 
            border-bottom: 2px solid #e9ecef; 
            padding-bottom: 10px; 
            margin-top: 0; 
            font-size: 1.8em;
        }}
        .metric-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 20px 0; 
        }}
        .metric {{ 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #2a5298; 
            transition: transform 0.2s ease;
        }}
        .metric:hover {{ transform: translateY(-2px); }}
        .metric-label {{ font-weight: bold; color: #495057; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #2a5298; font-weight: bold; }}
        /* Table Container - Add horizontal scrolling for wide tables */
        .table-container {{
            overflow-x: auto;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            min-width: 600px; /* Ensure minimum width for readability */
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ 
            border: 1px solid #dee2e6; 
            padding: 12px; 
            text-align: left; 
            vertical-align: top; /* Align content to top for long text */
        }}
        th {{ 
            background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%); 
            font-weight: bold; 
            color: white; 
            white-space: nowrap; /* Prevent header text wrapping */
        }}
        td {{
            word-wrap: break-word; /* Allow long words to break */
            word-break: break-word; /* Handle very long strings */
            max-width: 300px; /* Limit cell width for readability */
        }}
        /* Special handling for Details column */
        .details-cell {{
            max-width: 400px;
            max-height: 150px;
            overflow-y: auto;
            line-height: 1.4;
            font-size: 0.9em;
        }}
        /* Scrollbar styling for details cells */
        .details-cell::-webkit-scrollbar {{
            width: 6px;
        }}
        .details-cell::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 3px;
        }}
        .details-cell::-webkit-scrollbar-thumb {{
            background: #2a5298;
            border-radius: 3px;
        }}
        .details-cell::-webkit-scrollbar-thumb:hover {{
            background: #1e3c72;
        }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e3f2fd; }}
        .visualization-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px; 
            margin: 20px 0; 
        }}
        .visualization-item {{ 
            border: 1px solid #dee2e6; 
            border-radius: 8px; 
            padding: 15px; 
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .visualization-item img {{ 
            width: 100%; 
            height: auto; 
            border-radius: 4px; 
            margin-bottom: 10px; 
        }}
        .visualization-description {{ 
            margin: 0; 
            font-size: 0.9em; 
            color: #6c757d; 
            text-align: center; 
        }}
        .status-badge {{ 
            display: inline-block; 
            padding: 4px 8px; 
            border-radius: 12px; 
            font-size: 0.8em; 
            font-weight: bold; 
            text-transform: uppercase; 
            background-color: #d4edda; 
            color: #155724; 
        }}
        .recommendation {{ 
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
            border-left: 4px solid #2196F3; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 4px; 
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        /* Scrollable content areas for long lists */
        .scrollable-content {{
            max-height: 200px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            background: #f8f9fa;
        }}
        .scrollable-content::-webkit-scrollbar {{
            width: 8px;
        }}
        .scrollable-content::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}
        .scrollable-content::-webkit-scrollbar-thumb {{
            background: #2a5298;
            border-radius: 4px;
        }}
        .scrollable-content::-webkit-scrollbar-thumb:hover {{
            background: #1e3c72;
        }}
        /* Long text content areas */
        .expandable-text {{
            max-height: 100px;
            overflow-y: auto;
            padding: 8px;
            background: rgba(255,255,255,0.5);
            border-radius: 4px;
            border: 1px solid rgba(0,0,0,0.1);
            font-size: 0.9em;
            line-height: 1.4;
        }}
        .expandable-text::-webkit-scrollbar {{
            width: 6px;
        }}
        .expandable-text::-webkit-scrollbar-track {{
            background: rgba(241,241,241,0.5);
            border-radius: 3px;
        }}
        .expandable-text::-webkit-scrollbar-thumb {{
            background: rgba(42,82,152,0.7);
            border-radius: 3px;
        }}
        .footer {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 SVM Training Report</h1>
            <p>Comprehensive Analysis of Model Training Process</p>
        </div>
        
        {self._format_training_summary_simple(metadata)}
        {self._format_data_validation_simple(metadata)}
        {self._format_dataset_overview_simple(metadata)}
        {self._format_model_configuration_simple(metadata)}
        {self._format_optimization_results_simple(metadata)}
        {self._format_cross_validation_simple(metadata)}
        {cv_visualization_section}
        {self._format_feature_importance_simple(metadata)}
        {feature_importance_visualization_section}
        {self._format_training_timeline_simple(metadata)}
        {self._format_recommendations_simple(metadata)}
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                            <p>MCP SVM Tool - Training Analysis Report v1.0.0</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_template

    def _generate_cv_visualization_section(self, model_directory: Path) -> str:
        """Generate cross-validation visualization section with image detection."""
        try:
            # Look for cross-validation related images
            cv_images = []
            
            # Check for ROC curves
            roc_pattern = model_directory / "cross_validation_data" / "*roc_curves.png"
            import glob
            roc_files = glob.glob(str(roc_pattern))
            if roc_files:
                cv_images.append({
                    'path': roc_files[0],
                    'title': 'ROC Curves',
                    'description': 'Receiver Operating Characteristic curves for each class and fold'
                })
            
            # Check for other CV plots
            cv_data_dir = model_directory / "cross_validation_data"
            if cv_data_dir.exists():
                for img_file in cv_data_dir.glob("*.png"):
                    if "roc" not in img_file.name.lower():
                        cv_images.append({
                            'path': str(img_file),
                            'title': img_file.stem.replace('_', ' ').title(),
                            'description': f'Cross-validation visualization: {img_file.stem}'
                        })
            
            if not cv_images:
                return ""
            
            # Generate HTML for visualizations
            visualization_html = '<div class="section"><h2>📊 Cross-Validation Visualizations</h2><div class="visualization-grid">'
            
            for img in cv_images:
                # Convert absolute path to relative for HTML
                img_path = Path(img['path']).name
                visualization_html += f'''
                <div class="visualization-item">
                    <img src="cross_validation_data/{img_path}" alt="{img['title']}">
                    <p class="visualization-description"><strong>{img['title']}</strong><br>{img['description']}</p>
                </div>
                '''
            
            visualization_html += '</div></div>'
            return visualization_html
            
        except Exception as e:
            logger.warning(f"Error generating CV visualization section: {e}")
            return ""

    def _generate_feature_importance_visualization_section(self, model_directory: Path) -> str:
        """Generate feature importance visualization section with image detection."""
        try:
            # Look for feature importance images
            fi_images = []
            
            # Check common feature importance plot names
            for pattern in ["*feature_importance*.png", "*importance*.png", "*features*.png"]:
                import glob
                files = glob.glob(str(model_directory / pattern))
                for file_path in files:
                    fi_images.append({
                        'path': file_path,
                        'title': 'Feature Importance',
                        'description': 'Relative importance of features in the trained model'
                    })
            
            if not fi_images:
                return ""
            
            # Generate HTML for visualizations
            visualization_html = '<div class="section"><h2>🎯 Feature Importance Visualizations</h2><div class="visualization-grid">'
            
            for img in fi_images:
                # Convert absolute path to relative for HTML
                img_path = Path(img['path']).name
                visualization_html += f'''
                <div class="visualization-item">
                    <img src="{img_path}" alt="{img['title']}">
                    <p class="visualization-description"><strong>{img['title']}</strong><br>{img['description']}</p>
                </div>
                '''
            
            visualization_html += '</div></div>'
            return visualization_html
            
        except Exception as e:
            logger.warning(f"Error generating feature importance visualization section: {e}")
            return ""

    def _format_training_summary_simple(self, metadata: Dict[str, Any]) -> str:
        """Format training summary section."""
        model_name = metadata.get('model_name', 'Unknown Model')
        task_type = metadata.get('task_type', 'unknown')
        training_time = metadata.get('training_time_seconds', 0)
        
        # Get performance metrics
        cv_results = metadata.get('cross_validation_results', {})
        test_scores = cv_results.get('test_scores', {})
        
        performance_html = ""
        if test_scores:
            performance_html = '<div class="metric-grid">'
            for metric, score_data in test_scores.items():
                if score_data and isinstance(score_data, dict):
                    # score_data is a dict with 'mean', 'std', 'scores', etc.
                    mean_score = score_data.get('mean', 0)
                    std_score = score_data.get('std', 0)
                    
                    # Format metric name and values
                    metric_display = self._format_metric_name(metric)
                    mean_display = mean_score
                    std_display = std_score
                    
                    # For MAE and MSE, display as positive values if they're negative
                    if metric in ['MAE', 'MSE'] and mean_score < 0:
                        mean_display = abs(mean_score)
                        std_display = abs(std_score)
                    
                    performance_html += f'''
                    <div class="metric">
                        <div class="metric-label">{metric_display}</div>
                        <div class="metric-value">{mean_display:.3f} ± {std_display:.3f}</div>
                    </div>
                    '''
                elif score_data and isinstance(score_data, (list, np.ndarray)):
                    # Fallback: score_data is a list/array of scores
                    mean_score = np.mean(score_data)
                    std_score = np.std(score_data)
                    
                    # Format metric name and values
                    metric_display = self._format_metric_name(metric)
                    mean_display = mean_score
                    std_display = std_score
                    
                    # For MAE and MSE, display as positive values if they're negative
                    if metric in ['MAE', 'MSE'] and mean_score < 0:
                        mean_display = abs(mean_score)
                        std_display = abs(std_score)
                    
                    performance_html += f'''
                    <div class="metric">
                        <div class="metric-label">{metric_display}</div>
                        <div class="metric-value">{mean_display:.3f} ± {std_display:.3f}</div>
                    </div>
                    '''
            performance_html += '</div>'
        
        return f'''
        <div class="section">
            <h2>📋 Training Summary</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Model Name</div>
                    <div class="metric-value">{model_name}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Task Type</div>
                    <div class="metric-value">{task_type.title()}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Training Time</div>
                    <div class="metric-value">{training_time:.2f}s</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value"><span class="status-badge">Completed</span></div>
                </div>
            </div>
            {performance_html}
        </div>
        '''

    def _format_data_validation_simple(self, metadata: Dict[str, Any]) -> str:
        """Format data validation section with comprehensive reporting."""
        import os
        from pathlib import Path
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        
        # 首先尝试从metadata中获取数据验证信息
        data_validation = metadata.get('data_validation', {})
        
        # 如果metadata中没有，尝试从模型目录中读取数据验证报告
        if not data_validation:
            model_dir = metadata.get('model_directory')
            if model_dir:
                reports_dir = Path(model_dir) / "reports"
                validation_path = reports_dir / "data_validation_report.json"
                
                if validation_path.exists():
                    try:
                        with open(validation_path, 'r', encoding='utf-8') as f:
                            validation_data = json.load(f)
                        
                        # 构建全面的数据验证报告
                        return self._build_comprehensive_data_validation_html(validation_data)
                    except Exception as e:
                        logger.warning(f"Failed to load data validation report: {e}")
                        return ""
        
        # 如果有简单的数据验证信息，显示基础版本
        if data_validation:
            quality_score = data_validation.get('overall_quality_score', 0)
            return f'''
        <div class="section">
            <h2>🔍 Data Validation</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Overall Quality Score</div>
                    <div class="metric-value">{quality_score:.1f}/100</div>
                </div>
            </div>
        </div>
        '''
        
        return ""
    
    def _build_comprehensive_data_validation_html(self, validation_data: Dict[str, Any]) -> str:
        """构建全面的数据验证HTML报告"""
        html = ['<div class="section">', '<h2>🧪 Data Validation & Quality Assessment</h2>']
        
        # 总体质量评估
        qa = validation_data.get('quality_assessment', {})
        if qa:
            score = qa.get("overall_score", 0)
            level = qa.get("quality_level", "Unknown")
            score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
            
            html.append(f'''
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Overall Data Quality Score</div>
                    <div class="metric-value" style="color:{score_color};">{score:.1f}/100</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Quality Level</div>
                    <div class="metric-value">{level}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Data Ready for Training</div>
                    <div class="metric-value">{"✅ Yes" if validation_data.get("data_ready_for_training", False) else "❌ No"}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Validation Checks</div>
                    <div class="metric-value">{len(validation_data.get("validation_results", []))}</div>
                </div>
            </div>''')
        
        # 关键数据统计
        dataset_info = validation_data.get('dataset_info', {})
        if dataset_info:
            shape = dataset_info.get('shape', [0, 0])
            task_type = dataset_info.get('task_type', 'unknown')
            features_analyzed = dataset_info.get('features_analyzed', 0)
            
            html.append(f'''
            <h3>📊 Dataset Information</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Samples</div>
                    <div class="metric-value">{shape[0]:,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Features</div>
                    <div class="metric-value">{features_analyzed}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Task Type</div>
                    <div class="metric-value">{task_type.title()}</div>
                </div>
            </div>''')
        
        # 验证检查摘要表
        validation_results = validation_data.get('validation_results', [])
        if validation_results:
            html.append('<h3>📋 Validation Checks Summary</h3>')
            html.append('<div style="background:#e3f2fd; border-left:4px solid #2196f3; padding:10px; margin:10px 0; border-radius:4px; font-style:italic; color:#1565c0;">')
            html.append('💡 <strong>Note:</strong> These validation checks are for data quality assessment and reference only. They do not affect model training capabilities or usage.')
            html.append('</div>')
            html.append('<div class="table-container">')
            html.append('<table><tr><th>Check Type</th><th>Status</th><th>Issues Found</th><th>Details</th></tr>')
            
            for check in validation_results:
                name = check.get('check_name', 'Unknown').replace('_', ' ').title()
                status = '<span style="color:green; font-weight:bold;">✅ PASSED</span>' if check.get('passed', True) else '<span style="color:red; font-weight:bold;">❌ FAILED</span>'
                issues_count = len(check.get('issues', []))
                # Create scrollable details cell for long text
                full_details = '; '.join(check.get('issues', [])) if check.get('issues') else 'No issues detected'
                details_class = 'details-cell' if len(full_details) > 80 else ''
                
                html.append(f'<tr><td>{name}</td><td>{status}</td><td>{issues_count}</td><td class="{details_class}">{full_details}</td></tr>')
            
            html.append('</table>')
            html.append('</div>')
        
        # 关键问题和建议
        critical_issues = []
        for check in validation_results:
            if not check.get('passed', True):
                critical_issues.extend(check.get('issues', []))
        
        if critical_issues:
            html.append('<h3>🚨 Critical Issues</h3>')
            html.append('<div style="background:#ffeaea; border-left:4px solid #dc3545; padding:15px; margin:15px 0; border-radius:4px;">')
            # Use scrollable content for long lists
            if len(critical_issues) > 5:
                html.append('<div class="scrollable-content">')
            html.append('<ul>')
            for issue in critical_issues:  # 显示所有关键问题
                html.append(f'<li style="color:#dc3545; margin:5px 0;">{issue}</li>')
            html.append('</ul>')
            if len(critical_issues) > 5:
                html.append('</div>')
            html.append('</div>')
        
        # 建议
        recommendations = validation_data.get('recommendations', [])
        if recommendations:
            html.append('<h3>💡 Improvement Recommendations</h3>')
            html.append('<div style="background:#e3f2fd; border-left:4px solid #2196f3; padding:15px; margin:15px 0; border-radius:4px;">')
            # Use scrollable content for long lists
            if len(recommendations) > 6:
                html.append('<div class="scrollable-content">')
            html.append('<ul>')
            for rec in recommendations:  # 显示所有建议
                html.append(f'<li style="margin:5px 0;">{rec}</li>')
            html.append('</ul>')
            if len(recommendations) > 6:
                html.append('</div>')
            html.append('</div>')
        
        html.append('</div>')
        return '\n'.join(html)

    def _format_dataset_overview_simple(self, metadata: Dict[str, Any]) -> str:
        """Format dataset overview section."""
        # Get data shape information
        data_shape = metadata.get('data_shape', {})
        n_samples = data_shape.get('n_samples', metadata.get('n_samples', 0))
        n_features = data_shape.get('n_features', metadata.get('n_features', 0))
        feature_names = metadata.get('feature_names', [])
        
        # Get target information
        target_name = metadata.get('target_column', 'Unknown')
        if isinstance(target_name, list):
            target_name = ', '.join(target_name)
        
        # Get task type
        task_type = metadata.get('task_type', 'Unknown').title()
        
        return f'''
        <div class="section">
            <h2>📊 Dataset Overview</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Number of Samples</div>
                    <div class="metric-value">{n_samples:,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Number of Features</div>
                    <div class="metric-value">{n_features}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Task Type</div>
                    <div class="metric-value">{task_type}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Target Variable</div>
                    <div class="metric-value">{target_name}</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <h3>Features ({len(feature_names)} total):</h3>
                <div class="expandable-text" style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; font-family: monospace; font-size: 0.9em;">
                    {', '.join(feature_names) if feature_names else 'N/A'}
                </div>
            </div>
        </div>
        '''

    def _format_model_configuration_simple(self, metadata: Dict[str, Any]) -> str:
        """Format model configuration section."""
        model_params = metadata.get('model_params', {})
        if not model_params:
            return ""
        
        params_html = '<div class="metric-grid">'
        for param, value in model_params.items():
            params_html += f'''
            <div class="metric">
                <div class="metric-label">{param.replace('_', ' ').title()}</div>
                <div class="metric-value">{value}</div>
            </div>
            '''
        params_html += '</div>'
        
        return f'''
        <div class="section">
            <h2>⚙️ Model Configuration</h2>
            {params_html}
        </div>
        '''

    def _format_optimization_results_simple(self, metadata: Dict[str, Any]) -> str:
        """Format optimization results section."""
        opt_results = metadata.get('optimization_results', {})
        if not opt_results:
            return ""
        
        best_score = opt_results.get('best_score', 0)
        n_trials = opt_results.get('n_trials', 0)
        best_params = opt_results.get('best_params', {})
        
        # Get scoring metric from tool input details
        scoring_metric = metadata.get('tool_input_details', {}).get('scoring_metric', 'unknown')
        
        # Format best score display with metric information
        score_display = f"{best_score:.4f}"
        metric_info = ""
        
        # For negative metrics (MAE, MSE), display as positive values
        if scoring_metric in ['neg_mean_absolute_error', 'neg_mean_squared_error']:
            score_display = f"{abs(best_score):.4f}"
            if scoring_metric == 'neg_mean_absolute_error':
                metric_info = " (MAE - lower is better)"
            elif scoring_metric == 'neg_mean_squared_error':
                metric_info = " (MSE - lower is better)"
        elif scoring_metric == 'r2' or 'r2' in scoring_metric.lower():
            metric_info = " (R² - higher is better)"
        elif scoring_metric != 'unknown':
            metric_info = f" ({scoring_metric})"
        
        params_html = ""
        if best_params:
            params_html = '<h3>Best Parameters:</h3><div class="metric-grid">'
            for param, value in best_params.items():
                params_html += f'''
                <div class="metric">
                    <div class="metric-label">{param.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value}</div>
                </div>
                '''
            params_html += '</div>'
        
        return f'''
        <div class="section">
            <h2>🎯 Hyperparameter Optimization</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Best Score{metric_info}</div>
                    <div class="metric-value">{score_display}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Number of Trials</div>
                    <div class="metric-value">{n_trials}</div>
                </div>
            </div>
            {params_html}
        </div>
        '''

    def _format_cross_validation_simple(self, metadata: Dict[str, Any]) -> str:
        """Format cross-validation section."""
        cv_results = metadata.get('cross_validation_results', {})
        if not cv_results:
            return ""
        
        test_scores = cv_results.get('test_scores', {})
        n_folds = cv_results.get('cv_folds', 5)
        
        scores_html = ""
        if test_scores:
            scores_html = '<div class="metric-grid">'
            for metric, score_data in test_scores.items():
                if score_data and isinstance(score_data, dict):
                    # score_data is a dict with 'mean', 'std', 'scores', etc.
                    mean_score = score_data.get('mean', 0)
                    std_score = score_data.get('std', 0)
                    
                    # Format metric name and values
                    metric_display = self._format_metric_name(metric)
                    mean_display = mean_score
                    std_display = std_score
                    
                    # For MAE and MSE, display as positive values if they're negative
                    if metric in ['MAE', 'MSE'] and mean_score < 0:
                        mean_display = abs(mean_score)
                        std_display = abs(std_score)
                    
                    scores_html += f'''
                    <div class="metric">
                        <div class="metric-label">{metric_display}</div>
                        <div class="metric-value">{mean_display:.3f} ± {std_display:.3f}</div>
                    </div>
                    '''
                elif score_data and isinstance(score_data, (list, np.ndarray)):
                    # Fallback: score_data is a list/array of scores
                    mean_score = np.mean(score_data)
                    std_score = np.std(score_data)
                    
                    # Format metric name and values
                    metric_display = self._format_metric_name(metric)
                    mean_display = mean_score
                    std_display = std_score
                    
                    # For MAE and MSE, display as positive values if they're negative
                    if metric in ['MAE', 'MSE'] and mean_score < 0:
                        mean_display = abs(mean_score)
                        std_display = abs(std_score)
                    
                    scores_html += f'''
                    <div class="metric">
                        <div class="metric-label">{metric_display}</div>
                        <div class="metric-value">{mean_display:.3f} ± {std_display:.3f}</div>
                    </div>
                    '''
            scores_html += '</div>'
        
        return f'''
        <div class="section">
            <h2>🔄 Cross-Validation Results</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Number of Folds</div>
                    <div class="metric-value">{n_folds}</div>
                </div>
            </div>
            {scores_html}
        </div>
        '''

    def _format_feature_importance_simple(self, metadata: Dict[str, Any]) -> str:
        """Format feature importance section."""
        feature_importance = metadata.get('feature_importance', {})
        if not feature_importance:
            return ""
        
        # Create comprehensive table of feature importance with scrollable container
        table_html = '''
        <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Gain</th>
                    <th>Weight</th>
                    <th>Cover</th>
                    <th>Permutation Mean</th>
                    <th>Permutation Std</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        # Sort features by gain (most commonly used importance metric)
        def get_gain_importance(data):
            if not isinstance(data, dict):
                return 0
            return data.get('gain', 0)
        
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: get_gain_importance(x[1]),
            reverse=True
        )
        
        for feature, data in sorted_features:
            # Ensure data is a dictionary before calling .get()
            if not isinstance(data, dict):
                continue
                
            # Extract all available importance metrics
            gain = data.get('gain', 0)
            weight = data.get('weight', 0)
            cover = data.get('cover', 0)
            perm_mean = data.get('permutation_mean', 0)
            perm_std = data.get('permutation_std', 0)
            
            table_html += f'''
            <tr>
                <td><strong>{feature}</strong></td>
                <td>{gain:.4f}</td>
                <td>{weight:.0f}</td>
                <td>{cover:.2f}</td>
                <td>{perm_mean:.4f}</td>
                <td>{perm_std:.4f}</td>
            </tr>
            '''
        
        table_html += '''</tbody></table></div>
        
        <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <h4>📖 Metric Description:</h4>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">
                <li><strong>Gain</strong>: The total information gain brought by the feature when splitting (primary importance metric)</li>
                <li><strong>Weight</strong>: The number of times the feature is used for splitting across all trees</li>
                <li><strong>Cover</strong>: The total number of samples covered when the feature is used in a split</li>
                <li><strong>Permutation Mean</strong>: Mean permutation importance (the extent to which performance decreases after shuffling the feature)</li>
                <li><strong>Permutation Std</strong>: Standard deviation of permutation importance (stability of the importance)</li>
            </ul>
        </div>'''
        
        return f'''
        <div class="section">
            <h2>🎯 Feature Importance</h2>
            {table_html}
        </div>
        '''

    def _format_training_timeline_simple(self, metadata: Dict[str, Any]) -> str:
        """Format training timeline section."""
        training_time = metadata.get('training_time', 0)
        if training_time <= 0:
            return ""
        
        return f'''
        <div class="section">
            <h2>⏱️ Training Timeline</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Total Training Time</div>
                    <div class="metric-value">{training_time:.2f} seconds</div>
                </div>
            </div>
        </div>
        '''

    def _format_recommendations_simple(self, metadata: Dict[str, Any]) -> str:
        """Format recommendations for simple HTML generation"""
        # Generate intelligent recommendations based on training data
        recommendations = self._generate_recommendations(metadata)
        
        # Build HTML for recommendations
        recommendations_html = ""
        for i, rec in enumerate(recommendations, 1):
            recommendations_html += f"""
            <div class="recommendation">
                <strong>{i}.</strong> {rec}
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>💡 Recommendations & Next Steps</h2>
            {recommendations_html}
        </div>
        """

    # === Prediction Report Generation ===
    
    def generate_prediction_report_from_folder(self, prediction_folder: Union[str, Path],model_info:Dict[str,Any]=None) -> Optional[str]:
        """Generate HTML prediction report from prediction folder containing saved files.
        
        Args:
            prediction_folder: Path to prediction folder containing results
            
        Returns:
            Path to generated HTML report or None if failed
        """
        try:
            prediction_folder = Path(prediction_folder)
            
            # Load prediction results from metadata
            metadata_path = prediction_folder / "prediction_metadata.json"
            if not metadata_path.exists():
                logger.error(f"Prediction metadata not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Load input data
            input_data_path = prediction_folder / "01_original_data.csv"
            if not input_data_path.exists():
                logger.error(f"Original data not found: {input_data_path}")
                return None
            
            input_data = pd.read_csv(input_data_path)
            
            # Generate HTML report
            html_content = self._create_prediction_html_report(results, input_data, model_info, prediction_folder)
            
            # Save HTML report
            html_report_path = prediction_folder / "prediction_report.html"
            with open(html_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML prediction report generated: {html_report_path}")
            return str(html_report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate HTML prediction report: {e}")
            return None

    def _create_prediction_html_report(self, results: Dict[str, Any], input_data: pd.DataFrame, model_info:Dict[str,Any]=None, prediction_folder: Optional[Path] = None) -> str:
        """Create enhanced HTML content for prediction report with features, predictions, and uncertainty."""
        
        # Get metadata
        metadata = results.get('prediction_metadata', {})
        num_samples = metadata.get('num_samples', len(input_data))
        if 'num_predictions' in results:
            num_samples = results['num_predictions']
        elif 'predictions' in results and results['predictions'] is not None:
            if isinstance(results['predictions'], (list, np.ndarray)):
                num_samples = len(results['predictions'])
            else:
                num_samples = 1  # Single prediction
        feature_names = metadata.get('feature_names', input_data.columns.tolist())
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Report - {results.get('model_id', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ margin: 25px 0; background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e3f2fd; border-radius: 6px; min-width: 150px; text-align: center; }}
        .metric strong {{ display: block; font-size: 1.2em; color: #1976d2; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; color: #333; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .feature-value {{ font-family: monospace; background-color: #f8f9fa; padding: 2px 6px; border-radius: 3px; }}
        .prediction-value {{ font-weight: bold; color: #28a745; }}
        .confidence-value {{ color: #dc3545; }}
        .high-confidence {{ color: #28a745; }}
        .medium-confidence {{ color: #ffc107; }}
        .low-confidence {{ color: #dc3545; }}
        h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
        h3 {{ color: #555; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔮 SVM Prediction Report</h1>
        <p><strong>Model ID:</strong> {results.get('model_id', 'Unknown')}</p>
        <p><strong>Task Type:</strong> {model_info.get('task_type', 'Unknown').title()}</p>
        <p><strong>Prediction Time:</strong> {results.get('prediction_metadata', {}).get('prediction_timestamp', 'Unknown')}</p>
    </div>
    
        <div class="section">
        <h2>📊 Summary</h2>
        <div class="metric">
            <strong>{num_samples}</strong>
            <span>Total Predictions</span>
        </div>
        <div class="metric">
            <strong>{results.get('prediction_metadata', {}).get('prediction_time_seconds', 0.0):.3f}s</strong>
            <span>Processing Time</span>
        </div>
        <div class="metric">
            <strong>{len(feature_names)}</strong>
            <span>Features Used</span>
        </div>
        <div class="metric">
            <strong>{'Yes' if metadata.get('preprocessing_applied') else 'No'}</strong>
            <span>Preprocessing Applied</span>
        </div>
    </div>"""
        
        # Add feature information section
        html += f"""
    <div class="section">
        <h2>🔧 Feature Information</h2>
        <p><strong>Total Features:</strong> {len(feature_names)}</p>
        <table>
            <tr><th>Feature Name</th><th>Sample Values</th></tr>"""
        
        # Show feature values from input data
        for feature_name in feature_names:
            if feature_name in input_data.columns:
                sample_values = input_data[feature_name].head(5).tolist()
                # Handle both numeric and categorical features
                formatted_values = []
                for val in sample_values:
                    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                        formatted_values.append(f"<span class='feature-value'>{val:.4f}</span>")
                    else:
                        formatted_values.append(f"<span class='feature-value'>{val}</span>")
                sample_str = ', '.join(formatted_values)
                if len(input_data) > 5:
                    sample_str += ", ..."
                html += f"<tr><td><strong>{feature_name}</strong></td><td>{sample_str}</td></tr>"
        
        html += "</table></div>"

        # Add detailed predictions with features section
        html += f"""
    <div class="section">
        <h2>🎯 Detailed Predictions & Features</h2>
        <table>
            <tr><th>Sample #</th>"""
        
        # Add feature columns
        for feature_name in feature_names[:5]:  # Show first 5 features to avoid too wide table
            html += f"<th>{feature_name}</th>"
        if len(feature_names) > 5:
            html += "<th>... More Features</th>"
        
        html += "<th>Prediction</th>"
        
        if results.get('confidence_scores'):
            html += "<th>Uncertainty/Confidence</th>"
        
        html += "</tr>"
        
        # Show detailed predictions with feature values
        # Use raw predictions for label mapping in HTML
        raw_predictions = results.get('raw_predictions', results.get('predictions', []))

        predictions_list = raw_predictions if isinstance(raw_predictions, (list, np.ndarray)) else [raw_predictions]
        
        max_samples_to_show = min(10, len(predictions_list))
        for i in range(max_samples_to_show):
            pred = predictions_list[i]
            html += f"<tr><td><strong>#{i+1}</strong></td>"
            
            # Add feature values
            for j, feature_name in enumerate(feature_names[:5]):
                if feature_name in input_data.columns and i < len(input_data):
                    feature_val = input_data.iloc[i][feature_name]
                    # Handle both numeric and categorical features
                    if isinstance(feature_val, (int, float, np.number)):
                        html += f"<td><span class='feature-value'>{feature_val:.4f}</span></td>"
                    else:
                        html += f"<td><span class='feature-value'>{feature_val}</span></td>"
                else:
                    html += "<td>-</td>"
            
            if len(feature_names) > 5:
                html += "<td>...</td>"
            
            # Format prediction value properly - handle classification labels
            if results.get('task_type') == 'classification' and results.get('label_mapping'):
                class_to_label = results['label_mapping'].get('class_to_label', {})
                if isinstance(class_to_label, dict):
                    class_to_label = {k: str(v) for k, v in class_to_label.items()}

                def safe_html_label_lookup(pred_value):
                    """Safely lookup classification label for HTML report, handling both string and numeric predictions."""
                    if isinstance(pred_value, str):
                        return pred_value
                    try:
                        return (class_to_label.get(str(int(pred_value))) or 
                                class_to_label.get(int(pred_value)) or 
                                f'Class_{int(pred_value)}')
                    except (ValueError, TypeError):
                        return str(pred_value) if pred_value is not None else 'Unknown'
                
                if isinstance(pred, (list, np.ndarray)):
                    # Multi-target classification
                    pred_labels = [safe_html_label_lookup(p) for p in pred]
                    pred_str = ', '.join(pred_labels)
                else:
                    # Single classification
                    pred_str = safe_html_label_lookup(pred)
                    print(">>> pred_str", pred_str)
            else:
                # Regression or no label mapping
                if isinstance(pred, (list, np.ndarray)):
                    pred_str = ', '.join([f"{float(p):.4f}" for p in pred])
                else:
                    pred_str = f"{float(pred):.4f}"
            html += f"<td><span class='prediction-value'>{pred_str}</span></td>"
            
            # Add confidence/uncertainty
            if results.get('confidence_scores'):
                conf_score = results['confidence_scores'][i]
                if isinstance(conf_score, (list, np.ndarray)):
                    conf_str = ', '.join([f"{float(c):.4f}" for c in conf_score])
                    conf_val = float(conf_score[0]) if len(conf_score) > 0 else 0.0
                else:
                    conf_str = f"{float(conf_score):.4f}"
                    conf_val = float(conf_score)
                
                # Add color coding based on confidence level
                conf_class = "high-confidence" if conf_val > 0.8 else "medium-confidence" if conf_val > 0.5 else "low-confidence"
                html += f"<td><span class='confidence-value {conf_class}'>{conf_str}</span></td>"
            
            html += "</tr>"
        
        html += "</table></div>"

        # Add feature importance section if available
        if results.get('feature_importance'):
            html += """
    <div class="section">
        <h2>📈 Feature Importance</h2>
        <p>Features ranked by their importance in making predictions:</p>
        <table>
            <tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Relative Importance</th></tr>"""
            
            sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            max_importance = max(results['feature_importance'].values()) if results['feature_importance'] else 1.0
            
            for rank, (feature, importance) in enumerate(sorted_features, 1):
                relative_importance = (importance / max_importance) * 100
                html += f"""<tr>
                    <td><strong>#{rank}</strong></td>
                    <td>{feature}</td>
                    <td>{importance:.6f}</td>
                    <td>{relative_importance:.1f}%</td>
                </tr>"""
            html += "</table></div>"

        # Add saved files section if available
        if results.get('saved_files') and prediction_folder:
            html += """
    <div class="section">
        <h2>💾 Saved Files</h2>
        <table>
            <tr><th>File</th><th>Description</th><th>Path</th></tr>"""
            file_descriptions = {
                'predictions_processed': 'Predictions in processed/normalized scale',
                'predictions_original': 'Predictions in original scale',
                'confidence_scores': 'Confidence/uncertainty scores',
                'combined_results': 'Complete combined dataset with features and predictions',
                'original_data': 'Original input features',
                'processed_features': 'Preprocessed features'
            }
            for file_key, file_path in results['saved_files'].items():
                file_name = Path(file_path).name
                description = file_descriptions.get(file_key, 'Additional prediction data')
                html += f"<tr><td><strong>{file_name}</strong></td><td>{description}</td><td><small>{file_path}</small></td></tr>"
            html += "</table></div>"
        
        html += """
    <div class="section">
        <h2>ℹ️ Notes</h2>
        <ul>
            <li><strong>Predictions:</strong> Model output values for each input sample</li>
            <li><strong>Uncertainty/Confidence:</strong> Measure of prediction reliability (higher is more confident)</li>
            <li><strong>Feature Importance:</strong> How much each feature contributes to the model's decisions</li>
            <li><strong>Processing Time:</strong> Total time taken for data processing and prediction</li>
        </ul>
    </div>
</body>
</html>"""
        
        return html