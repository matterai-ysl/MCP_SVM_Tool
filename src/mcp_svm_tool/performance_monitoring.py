"""
Performance Monitoring Module

This module provides advanced performance monitoring capabilities for XGBoost models:
- Historical performance tracking with time-series data
- Model performance comparison across versions and time
- Performance degradation detection with trend analysis
- Automated retraining recommendations
- Integration with existing PerformanceAnalyzer
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict

from .performance_analysis import (
    PerformanceAnalyzer, 
    PredictionPerformance, 
    TrainingPerformance, 
    PerformanceRegression
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric with timestamp."""
    metric_name: str
    value: float
    unit: str
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class ModelPerformanceHistory:
    """Complete performance history for a model."""
    model_id: str
    creation_date: str
    last_updated: str
    prediction_metrics: List[PredictionPerformance]
    training_metrics: List[TrainingPerformance]
    custom_metrics: List[PerformanceMetric]
    versions: List[str]
    total_evaluations: int


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""
    model_id: str
    metric_name: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0-1 scale
    data_points: int
    time_span_days: float
    slope: float
    r_squared: float
    confidence_level: float


@dataclass
class RetrainingRecommendation:
    """Model retraining recommendation."""
    model_id: str
    recommendation_type: str  # 'urgent', 'recommended', 'optional'
    reasons: List[str]
    performance_degradation: float
    confidence_score: float
    suggested_actions: List[str]
    timeline: str
    timestamp: str


class PerformanceMonitor:
    """Advanced performance monitoring system with historical tracking."""
    
    def __init__(self, monitoring_dir: str = "performance_monitoring"):
        """
        Initialize PerformanceMonitor.
        
        Args:
            monitoring_dir: Directory to store monitoring data
        """
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.history_dir = self.monitoring_dir / "history"
        self.trends_dir = self.monitoring_dir / "trends"
        self.recommendations_dir = self.monitoring_dir / "recommendations"
        self.alerts_dir = self.monitoring_dir / "alerts"
        
        for dir_path in [self.history_dir, self.trends_dir, 
                        self.recommendations_dir, self.alerts_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize base analyzer
        self.performance_analyzer = PerformanceAnalyzer(
            results_dir=str(self.monitoring_dir / "analysis")
        )
        
        # Initialize monitoring data
        self.model_histories: Dict[str, ModelPerformanceHistory] = {}
        self.performance_trends: Dict[str, List[PerformanceTrend]] = {}
        self.retraining_recommendations: List[RetrainingRecommendation] = []
        
        # Configuration
        self.degradation_thresholds = {
            'prediction_time_ms': 0.20,  # 20% increase
            'throughput_predictions_per_sec': -0.15,  # 15% decrease
            'memory_usage_mb': 0.30,  # 30% increase
            'training_time_seconds': 0.25,  # 25% increase
            'cv_scores': -0.05  # 5% decrease in accuracy
        }
        
        self.trend_analysis_window_days = 30
        self.min_data_points_for_trend = 5
        
        # Load existing data
        self._load_monitoring_data()
        
        logger.info(f"Initialized PerformanceMonitor at: {monitoring_dir}")
    
    def record_prediction_performance(
        self,
        model,
        X_test: np.ndarray,
        model_id: str,
        version: str = "1.0.0",
        batch_sizes: List[int] = None,
        custom_metrics: Dict[str, float] = None
    ) -> List[PredictionPerformance]:
        """
        Record prediction performance and add to historical tracking.
        
        Args:
            model: Trained model object
            X_test: Test data for predictions
            model_id: Model identifier
            version: Model version
            batch_sizes: List of batch sizes to test
            custom_metrics: Additional custom metrics to record
            
        Returns:
            List of prediction performance results
        """
        # Use base analyzer to benchmark performance
        performance_results = self.performance_analyzer.benchmark_prediction_performance(
            model=model,
            X_test=X_test,
            model_id=model_id,
            version=version,
            batch_sizes=batch_sizes
        )
        
        # Add to historical tracking
        if model_id not in self.model_histories:
            self.model_histories[model_id] = ModelPerformanceHistory(
                model_id=model_id,
                creation_date=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                prediction_metrics=[],
                training_metrics=[],
                custom_metrics=[],
                versions=[],
                total_evaluations=0
            )
        
        history = self.model_histories[model_id]
        history.prediction_metrics.extend(performance_results)
        history.last_updated = datetime.now().isoformat()
        history.total_evaluations += len(performance_results)
        
        if version not in history.versions:
            history.versions.append(version)
        
        # Record custom metrics if provided
        if custom_metrics:
            timestamp = datetime.now().isoformat()
            for metric_name, value in custom_metrics.items():
                custom_metric = PerformanceMetric(
                    metric_name=metric_name,
                    value=value,
                    unit="custom",
                    timestamp=timestamp,
                    metadata={"model_id": model_id, "version": version}
                )
                history.custom_metrics.append(custom_metric)
        
        # Save updated history
        self._save_model_history(model_id)
        
        # Trigger performance analysis
        self._analyze_performance_trends(model_id)
        
        logger.info(f"Recorded prediction performance for model {model_id} v{version}")
        return performance_results
    
    def record_training_performance(
        self,
        model_params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_id: str,
        version: str = "1.0.0",
        cv_folds: int = 5,
        hyperopt_trials: int = 0,
        custom_metrics: Dict[str, float] = None
    ) -> TrainingPerformance:
        """
        Record training performance and add to historical tracking.
        
        Args:
            model_params: Model parameters used for training
            X_train: Training features
            y_train: Training labels
            model_id: Model identifier
            version: Model version
            cv_folds: Number of cross-validation folds
            hyperopt_trials: Number of hyperparameter optimization trials
            custom_metrics: Additional custom metrics to record
            
        Returns:
            Training performance result
        """
        # Use base analyzer to benchmark training performance
        training_result = self.performance_analyzer.benchmark_training_performance(
            model_params=model_params,
            X_train=X_train,
            y_train=y_train,
            model_id=model_id,
            version=version,
            cv_folds=cv_folds,
            hyperopt_trials=hyperopt_trials
        )
        
        # Add to historical tracking
        if model_id not in self.model_histories:
            self.model_histories[model_id] = ModelPerformanceHistory(
                model_id=model_id,
                creation_date=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                prediction_metrics=[],
                training_metrics=[],
                custom_metrics=[],
                versions=[],
                total_evaluations=0
            )
        
        history = self.model_histories[model_id]
        history.training_metrics.append(training_result)
        history.last_updated = datetime.now().isoformat()
        history.total_evaluations += 1
        
        if version not in history.versions:
            history.versions.append(version)
        
        # Record custom metrics if provided
        if custom_metrics:
            timestamp = datetime.now().isoformat()
            for metric_name, value in custom_metrics.items():
                custom_metric = PerformanceMetric(
                    metric_name=metric_name,
                    value=value,
                    unit="custom",
                    timestamp=timestamp,
                    metadata={"model_id": model_id, "version": version}
                )
                history.custom_metrics.append(custom_metric)
        
        # Save updated history
        self._save_model_history(model_id)
        
        # Trigger performance analysis
        self._analyze_performance_trends(model_id)
        
        logger.info(f"Recorded training performance for model {model_id} v{version}")
        return training_result
    
    def get_performance_history(
        self,
        model_id: str,
        metric_names: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        version: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve performance history for a model.
        
        Args:
            model_id: Model identifier
            metric_names: Specific metrics to retrieve
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            version: Specific version filter
            
        Returns:
            Performance history data
        """
        if model_id not in self.model_histories:
            return {"error": f"No history found for model {model_id}"}
        
        history = self.model_histories[model_id]
        result = {
            "model_id": model_id,
            "creation_date": history.creation_date,
            "last_updated": history.last_updated,
            "versions": history.versions,
            "total_evaluations": history.total_evaluations
        }
        
        # Filter and process prediction metrics
        prediction_metrics = history.prediction_metrics
        if start_date:
            prediction_metrics = [m for m in prediction_metrics if m.timestamp >= start_date]
        if end_date:
            prediction_metrics = [m for m in prediction_metrics if m.timestamp <= end_date]
        if version:
            prediction_metrics = [m for m in prediction_metrics if m.version == version]
        
        result["prediction_metrics"] = [asdict(m) for m in prediction_metrics]
        
        # Filter and process training metrics
        training_metrics = history.training_metrics
        if start_date:
            training_metrics = [m for m in training_metrics if m.timestamp >= start_date]
        if end_date:
            training_metrics = [m for m in training_metrics if m.timestamp <= end_date]
        if version:
            training_metrics = [m for m in training_metrics if m.version == version]
        
        result["training_metrics"] = [asdict(m) for m in training_metrics]
        
        # Filter and process custom metrics
        custom_metrics = history.custom_metrics
        if start_date:
            custom_metrics = [m for m in custom_metrics if m.timestamp >= start_date]
        if end_date:
            custom_metrics = [m for m in custom_metrics if m.timestamp <= end_date]
        if metric_names:
            custom_metrics = [m for m in custom_metrics if m.metric_name in metric_names]
        
        result["custom_metrics"] = [asdict(m) for m in custom_metrics]
        
        return result
    
    def compare_model_performance(
        self,
        model_ids: List[str],
        metric_names: List[str] = None,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare performance across multiple models.
        
        Args:
            model_ids: List of model identifiers to compare
            metric_names: Specific metrics to compare
            time_window_days: Time window for comparison (days)
            
        Returns:
            Comparison analysis results
        """
        if metric_names is None:
            metric_names = [
                'prediction_time_ms', 
                'throughput_predictions_per_sec',
                'memory_usage_mb',
                'training_time_seconds'
            ]
        
        cutoff_date = (datetime.now() - timedelta(days=time_window_days)).isoformat()
        comparison_data = {}
        
        for model_id in model_ids:
            if model_id not in self.model_histories:
                continue
            
            history = self.model_histories[model_id]
            model_data = {}
            
            # Aggregate prediction metrics
            recent_predictions = [
                m for m in history.prediction_metrics 
                if m.timestamp >= cutoff_date
            ]
            
            if recent_predictions:
                for metric_name in metric_names:
                    if hasattr(recent_predictions[0], metric_name):
                        values = [getattr(m, metric_name) for m in recent_predictions]
                        model_data[metric_name] = {
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0,
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
            
            # Aggregate training metrics
            recent_training = [
                m for m in history.training_metrics 
                if m.timestamp >= cutoff_date
            ]
            
            if recent_training:
                for metric_name in metric_names:
                    if hasattr(recent_training[0], metric_name):
                        values = [getattr(m, metric_name) for m in recent_training]
                        if metric_name == 'cv_scores':
                            # Special handling for CV scores (list of values)
                            all_scores = []
                            for cv_list in values:
                                all_scores.extend(cv_list)
                            if all_scores:
                                model_data[metric_name] = {
                                    'mean': statistics.mean(all_scores),
                                    'median': statistics.median(all_scores),
                                    'std': statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                                    'min': min(all_scores),
                                    'max': max(all_scores),
                                    'count': len(all_scores)
                                }
                        elif values:
                            model_data[metric_name] = {
                                'mean': statistics.mean(values),
                                'median': statistics.median(values),
                                'std': statistics.stdev(values) if len(values) > 1 else 0,
                                'min': min(values),
                                'max': max(values),
                                'count': len(values)
                            }
            
            comparison_data[model_id] = model_data
        
        # Generate comparison insights
        insights = self._generate_comparison_insights(comparison_data, metric_names)
        
        return {
            'comparison_data': comparison_data,
            'insights': insights,
            'time_window_days': time_window_days,
            'models_compared': len(comparison_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_performance_trends(self, model_id: str) -> List[PerformanceTrend]:
        """
        Analyze performance trends for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of performance trends
        """
        if model_id not in self.model_histories:
            return []
        
        history = self.model_histories[model_id]
        trends = []
        
        # Analyze prediction performance trends
        if len(history.prediction_metrics) >= self.min_data_points_for_trend:
            cutoff_date = (datetime.now() - timedelta(days=self.trend_analysis_window_days)).isoformat()
            recent_metrics = [
                m for m in history.prediction_metrics 
                if m.timestamp >= cutoff_date
            ]
            
            if len(recent_metrics) >= self.min_data_points_for_trend:
                trends.extend(self._calculate_metric_trends(model_id, recent_metrics, 'prediction'))
        
        # Analyze training performance trends
        if len(history.training_metrics) >= self.min_data_points_for_trend:
            cutoff_date = (datetime.now() - timedelta(days=self.trend_analysis_window_days)).isoformat()
            recent_metrics = [
                m for m in history.training_metrics 
                if m.timestamp >= cutoff_date
            ]
            
            if len(recent_metrics) >= self.min_data_points_for_trend:
                trends.extend(self._calculate_metric_trends(model_id, recent_metrics, 'training'))
        
        # Store trends
        self.performance_trends[model_id] = trends
        self._save_performance_trends(model_id)
        
        return trends
    
    def _calculate_metric_trends(
        self, 
        model_id: str, 
        metrics: List[Union[PredictionPerformance, TrainingPerformance]], 
        metric_type: str
    ) -> List[PerformanceTrend]:
        """Calculate trends for specific metrics."""
        trends = []
        
        # Define metrics to analyze based on type
        if metric_type == 'prediction':
            metric_fields = ['prediction_time_ms', 'throughput_predictions_per_sec', 'memory_usage_mb']
        else:  # training
            metric_fields = ['training_time_seconds', 'memory_peak_mb']
        
        for field in metric_fields:
            if not hasattr(metrics[0], field):
                continue
            
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for metric in metrics:
                if hasattr(metric, field):
                    values.append(getattr(metric, field))
                    timestamps.append(datetime.fromisoformat(metric.timestamp))
            
            if len(values) < 2:
                continue
            
            # Calculate trend
            trend = self._calculate_linear_trend(values, timestamps)
            
            if trend:
                trends.append(PerformanceTrend(
                    model_id=model_id,
                    metric_name=field,
                    trend_direction=trend['direction'],
                    trend_strength=trend['strength'],
                    data_points=len(values),
                    time_span_days=trend['time_span_days'],
                    slope=trend['slope'],
                    r_squared=trend['r_squared'],
                    confidence_level=trend['confidence']
                ))
        
        return trends
    
    def _calculate_linear_trend(self, values: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
        """Calculate linear trend for a metric over time."""
        if len(values) < 2:
            return None
        
        try:
            # Convert timestamps to numeric values (days from first timestamp)
            start_time = timestamps[0]
            x_values = [(t - start_time).total_seconds() / 86400 for t in timestamps]  # Days
            y_values = values
            
            # Calculate linear regression
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            sum_y2 = sum(y * y for y in y_values)
            
            # Slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # R-squared
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for y in y_values)
            ss_res = sum((y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2 
                        for x, y in zip(x_values, y_values))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction and strength
            if abs(slope) < 0.01:
                direction = 'stable'
                strength = 0.0
            elif slope > 0:
                direction = 'improving' if sum_y > 0 else 'degrading'
                strength = min(abs(slope) / max(y_values), 1.0)
            else:
                direction = 'degrading' if sum_y > 0 else 'improving'
                strength = min(abs(slope) / max(y_values), 1.0)
            
            time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
            
            return {
                'direction': direction,
                'strength': strength,
                'slope': slope,
                'r_squared': r_squared,
                'confidence': r_squared,  # Using R² as confidence measure
                'time_span_days': time_span
            }
            
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f"Error calculating trend: {e}")
            return None
    
    def _generate_comparison_insights(
        self, 
        comparison_data: Dict[str, Dict[str, Dict[str, float]]], 
        metric_names: List[str]
    ) -> List[str]:
        """Generate insights from model comparison data."""
        insights = []
        
        if len(comparison_data) < 2:
            insights.append("Need at least 2 models for meaningful comparison")
            return insights
        
        # Find best and worst performers for each metric
        for metric_name in metric_names:
            metric_data = {}
            for model_id, model_metrics in comparison_data.items():
                if metric_name in model_metrics:
                    metric_data[model_id] = model_metrics[metric_name]['mean']
            
            if len(metric_data) < 2:
                continue
            
            # Determine if higher or lower is better for this metric
            lower_is_better = metric_name in ['prediction_time_ms', 'memory_usage_mb', 'training_time_seconds']
            
            if lower_is_better:
                best_model = min(metric_data.items(), key=lambda x: x[1])
                worst_model = max(metric_data.items(), key=lambda x: x[1])
            else:
                best_model = max(metric_data.items(), key=lambda x: x[1])
                worst_model = min(metric_data.items(), key=lambda x: x[1])
            
            if worst_model[1] != 0:
                improvement = abs(best_model[1] - worst_model[1]) / abs(worst_model[1]) * 100
            else:
                improvement = 0.0
            
            insights.append(
                f"For {metric_name}: {best_model[0]} performs {improvement:.1f}% better than {worst_model[0]}"
            )
        
        return insights
    
    def _save_model_history(self, model_id: str):
        """Save model performance history to disk."""
        if model_id not in self.model_histories:
            return
        
        history_file = self.history_dir / f"{model_id}_history.json"
        history_data = asdict(self.model_histories[model_id])
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _save_performance_trends(self, model_id: str):
        """Save performance trends to disk."""
        if model_id not in self.performance_trends:
            return
        
        trends_file = self.trends_dir / f"{model_id}_trends.json"
        trends_data = [asdict(trend) for trend in self.performance_trends[model_id]]
        
        with open(trends_file, 'w') as f:
            json.dump(trends_data, f, indent=2)
    
    def _load_monitoring_data(self):
        """Load existing monitoring data from disk."""
        # Load model histories
        for history_file in self.history_dir.glob("*_history.json"):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                model_id = history_data['model_id']
                
                # Reconstruct objects from dictionaries
                prediction_metrics = [
                    PredictionPerformance(**pm) for pm in history_data.get('prediction_metrics', [])
                ]
                training_metrics = [
                    TrainingPerformance(**tm) for tm in history_data.get('training_metrics', [])
                ]
                custom_metrics = [
                    PerformanceMetric(**cm) for cm in history_data.get('custom_metrics', [])
                ]
                
                self.model_histories[model_id] = ModelPerformanceHistory(
                    model_id=model_id,
                    creation_date=history_data['creation_date'],
                    last_updated=history_data['last_updated'],
                    prediction_metrics=prediction_metrics,
                    training_metrics=training_metrics,
                    custom_metrics=custom_metrics,
                    versions=history_data.get('versions', []),
                    total_evaluations=history_data.get('total_evaluations', 0)
                )
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading history from {history_file}: {e}")
        
        # Load performance trends
        for trends_file in self.trends_dir.glob("*_trends.json"):
            try:
                with open(trends_file, 'r') as f:
                    trends_data = json.load(f)
                
                model_id = trends_file.stem.replace('_trends', '')
                self.performance_trends[model_id] = [
                    PerformanceTrend(**trend) for trend in trends_data
                ]
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading trends from {trends_file}: {e}")
    
    def detect_performance_degradation(
        self,
        model_id: str,
        degradation_threshold: float = None,
        time_window_days: int = 14
    ) -> List[PerformanceRegression]:
        """
        Detect performance degradation for a model.
        
        Args:
            model_id: Model identifier
            degradation_threshold: Custom degradation threshold
            time_window_days: Time window for degradation analysis
            
        Returns:
            List of detected performance regressions
        """
        if model_id not in self.model_histories:
            return []
        
        history = self.model_histories[model_id]
        regressions = []
        
        # Get recent performance data
        cutoff_date = (datetime.now() - timedelta(days=time_window_days)).isoformat()
        recent_predictions = [
            m for m in history.prediction_metrics 
            if m.timestamp >= cutoff_date
        ]
        recent_training = [
            m for m in history.training_metrics 
            if m.timestamp >= cutoff_date
        ]
        
        # Get baseline performance data (older data)
        baseline_cutoff = (datetime.now() - timedelta(days=time_window_days * 2)).isoformat()
        baseline_predictions = [
            m for m in history.prediction_metrics 
            if m.timestamp < cutoff_date and m.timestamp >= baseline_cutoff
        ]
        baseline_training = [
            m for m in history.training_metrics 
            if m.timestamp < cutoff_date and m.timestamp >= baseline_cutoff
        ]
        
        # Analyze prediction performance degradation
        for metric_name, threshold in self.degradation_thresholds.items():
            if metric_name in ['prediction_time_ms', 'throughput_predictions_per_sec', 'memory_usage_mb']:
                recent_values = []
                baseline_values = []
                
                for m in recent_predictions:
                    if hasattr(m, metric_name):
                        recent_values.append(getattr(m, metric_name))
                
                for m in baseline_predictions:
                    if hasattr(m, metric_name):
                        baseline_values.append(getattr(m, metric_name))
                
                if recent_values and baseline_values:
                    recent_avg = statistics.mean(recent_values)
                    baseline_avg = statistics.mean(baseline_values)
                    
                    if baseline_avg != 0:
                        change_percent = (recent_avg - baseline_avg) / baseline_avg
                        
                        # Check if degradation exceeds threshold
                        is_degradation = False
                        if metric_name == 'throughput_predictions_per_sec':
                            # Lower throughput is worse
                            is_degradation = change_percent <= threshold
                        else:
                            # Higher time/memory is worse
                            is_degradation = change_percent >= abs(threshold)
                        
                        if is_degradation:
                            regression = PerformanceRegression(
                                model_id=model_id,
                                metric_name=metric_name,
                                baseline_value=baseline_avg,
                                current_value=recent_avg,
                                regression_percent=abs(change_percent) * 100,
                                is_regression=True,
                                threshold=abs(threshold) * 100,
                                timestamp=datetime.now().isoformat()
                            )
                            regressions.append(regression)
        
        # Analyze training performance degradation
        for metric_name, threshold in self.degradation_thresholds.items():
            if metric_name in ['training_time_seconds', 'cv_scores']:
                recent_values = []
                baseline_values = []
                
                for m in recent_training:
                    if hasattr(m, metric_name):
                        value = getattr(m, metric_name)
                        if metric_name == 'cv_scores':
                            # CV scores is a list, take the mean
                            if value:
                                recent_values.append(statistics.mean(value))
                        else:
                            recent_values.append(value)
                
                for m in baseline_training:
                    if hasattr(m, metric_name):
                        value = getattr(m, metric_name)
                        if metric_name == 'cv_scores':
                            # CV scores is a list, take the mean
                            if value:
                                baseline_values.append(statistics.mean(value))
                        else:
                            baseline_values.append(value)
                
                if recent_values and baseline_values:
                    recent_avg = statistics.mean(recent_values)
                    baseline_avg = statistics.mean(baseline_values)
                    
                    if baseline_avg != 0:
                        change_percent = (recent_avg - baseline_avg) / baseline_avg
                        
                        # Check if degradation exceeds threshold
                        is_degradation = False
                        if metric_name == 'cv_scores':
                            # Lower CV scores is worse
                            is_degradation = change_percent <= threshold
                        else:
                            # Higher training time is worse
                            is_degradation = change_percent >= abs(threshold)
                        
                        if is_degradation:
                            regression = PerformanceRegression(
                                model_id=model_id,
                                metric_name=metric_name,
                                baseline_value=baseline_avg,
                                current_value=recent_avg,
                                regression_percent=abs(change_percent) * 100,
                                is_regression=True,
                                threshold=abs(threshold) * 100,
                                timestamp=datetime.now().isoformat()
                            )
                            regressions.append(regression)
        
        return regressions
    
    def generate_retraining_recommendations(
        self,
        model_id: str,
        performance_regressions: List[PerformanceRegression] = None
    ) -> RetrainingRecommendation:
        """
        Generate retraining recommendations based on performance analysis.
        
        Args:
            model_id: Model identifier
            performance_regressions: Pre-computed performance regressions
            
        Returns:
            Retraining recommendation
        """
        if performance_regressions is None:
            performance_regressions = self.detect_performance_degradation(model_id)
        
        if not performance_regressions:
            return RetrainingRecommendation(
                model_id=model_id,
                recommendation_type="optional",
                reasons=["No significant performance degradation detected"],
                performance_degradation=0.0,
                confidence_score=0.95,
                suggested_actions=["Continue monitoring model performance"],
                timeline="Monitor for next 30 days",
                timestamp=datetime.now().isoformat()
            )
        
        # Analyze severity of regressions
        reasons = []
        suggested_actions = []
        max_degradation = 0.0
        urgent_regressions = 0
        
        for regression in performance_regressions:
            reasons.append(
                f"{regression.metric_name} degraded by {regression.regression_percent:.1f}% "
                f"(threshold: {regression.threshold:.1f}%)"
            )
            max_degradation = max(max_degradation, regression.regression_percent)
            
            if regression.regression_percent > 25:  # Severe degradation
                urgent_regressions += 1
        
        # Determine recommendation type
        if urgent_regressions > 0 or max_degradation > 30:
            recommendation_type = "urgent"
            timeline = "Within 1-2 days"
            suggested_actions.extend([
                "Collect fresh training data immediately",
                "Review recent data distribution changes",
                "Consider emergency model rollback if available",
                "Perform comprehensive hyperparameter re-optimization"
            ])
        elif max_degradation > 15:
            recommendation_type = "recommended"
            timeline = "Within 1 week"
            suggested_actions.extend([
                "Collect updated training data",
                "Analyze data drift patterns",
                "Re-train with expanded dataset",
                "Review and update feature engineering"
            ])
        else:
            recommendation_type = "optional"
            timeline = "Within 2-4 weeks"
            suggested_actions.extend([
                "Monitor performance trends closely",
                "Plan data collection for next training cycle",
                "Consider minor hyperparameter adjustments"
            ])
        
        # Calculate confidence score based on regression consistency
        confidence_score = min(0.95, 0.6 + (len(performance_regressions) * 0.1))
        
        return RetrainingRecommendation(
            model_id=model_id,
            recommendation_type=recommendation_type,
            reasons=reasons,
            performance_degradation=max_degradation,
            confidence_score=confidence_score,
            suggested_actions=suggested_actions,
            timeline=timeline,
            timestamp=datetime.now().isoformat()
        )
    
    def get_monitoring_dashboard_data(self, model_id: str = None) -> Dict[str, Any]:
        """
        Get comprehensive monitoring dashboard data.
        
        Args:
            model_id: Specific model ID or None for all models
            
        Returns:
            Dashboard data
        """
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "models": [],
            "summary": {
                "total_models": len(self.model_histories),
                "models_with_degradation": 0,
                "urgent_recommendations": 0,
                "total_evaluations": 0
            }
        }
        
        model_ids = [model_id] if model_id else list(self.model_histories.keys())
        
        for mid in model_ids:
            if mid not in self.model_histories:
                continue
            
            history = self.model_histories[mid]
            
            # Get recent performance
            recent_predictions = [
                m for m in history.prediction_metrics 
                if (datetime.now() - datetime.fromisoformat(m.timestamp)).days <= 7
            ]
            
            recent_training = [
                m for m in history.training_metrics 
                if (datetime.now() - datetime.fromisoformat(m.timestamp)).days <= 7
            ]
            
            # Detect regressions
            regressions = self.detect_performance_degradation(mid)
            
            # Generate recommendation
            recommendation = self.generate_retraining_recommendations(mid, regressions)
            
            # Get trends
            trends = self.performance_trends.get(mid, [])
            
            model_data = {
                "model_id": mid,
                "last_updated": history.last_updated,
                "versions": history.versions,
                "recent_predictions_count": len(recent_predictions),
                "recent_training_count": len(recent_training),
                "performance_regressions": len(regressions),
                "retraining_recommendation": asdict(recommendation),
                "trends": [asdict(trend) for trend in trends[-5:]],  # Last 5 trends
                "health_status": self._calculate_model_health(mid, regressions, trends)
            }
            
            dashboard_data["models"].append(model_data)
            dashboard_data["summary"]["total_evaluations"] += history.total_evaluations
            
            if regressions:
                dashboard_data["summary"]["models_with_degradation"] += 1
            
            if recommendation.recommendation_type == "urgent":
                dashboard_data["summary"]["urgent_recommendations"] += 1
        
        return dashboard_data
    
    def _calculate_model_health(
        self, 
        model_id: str, 
        regressions: List[PerformanceRegression], 
        trends: List[PerformanceTrend]
    ) -> str:
        """Calculate overall model health status."""
        if not regressions and not trends:
            return "unknown"
        
        # Check for urgent issues
        urgent_regressions = [r for r in regressions if r.regression_percent > 25]
        if urgent_regressions:
            return "critical"
        
        # Check for moderate issues
        moderate_regressions = [r for r in regressions if r.regression_percent > 10]
        degrading_trends = [t for t in trends if t.trend_direction == "degrading" and t.trend_strength > 0.5]
        
        if moderate_regressions or degrading_trends:
            return "warning"
        
        # Check for minor issues
        minor_regressions = [r for r in regressions if r.regression_percent > 5]
        if minor_regressions:
            return "attention"
        
        return "healthy" 