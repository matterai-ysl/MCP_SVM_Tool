"""
Advanced Performance Analysis Module

This module provides comprehensive performance analysis capabilities for XGBoost models:
- Model prediction performance benchmarking
- Training performance analysis
- Cross-validation performance tracking
- Performance regression detection
- Visualization and reporting
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PredictionPerformance:
    """Prediction performance metrics."""
    model_id: str
    version: str
    dataset_size: int
    feature_count: int
    prediction_time_ms: float
    throughput_predictions_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    batch_size: int
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class TrainingPerformance:
    """Training performance metrics."""
    model_id: str
    version: str
    dataset_size: int
    feature_count: int
    n_estimators: int
    training_time_seconds: float
    memory_peak_mb: float
    cpu_usage_percent: float
    cv_scores: List[float]
    hyperopt_trials: int
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class PerformanceRegression:
    """Performance regression detection result."""
    model_id: str
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    is_regression: bool
    threshold: float
    timestamp: str


class PerformanceAnalyzer:
    """Advanced performance analysis and monitoring."""
    
    def __init__(self, results_dir: str = "performance_analysis"):
        """
        Initialize PerformanceAnalyzer.
        
        Args:
            results_dir: Directory to store analysis results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.predictions_dir = self.results_dir / "predictions"
        self.training_dir = self.results_dir / "training"
        self.regressions_dir = self.results_dir / "regressions"
        self.visualizations_dir = self.results_dir / "visualizations"
        
        for dir_path in [self.predictions_dir, self.training_dir, 
                        self.regressions_dir, self.visualizations_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize performance history
        self.prediction_history: List[PredictionPerformance] = []
        self.training_history: List[TrainingPerformance] = []
        self.regression_history: List[PerformanceRegression] = []
        
        # Load existing history
        self._load_performance_history()
        
        logger.info(f"Initialized PerformanceAnalyzer at: {results_dir}")
    
    def benchmark_prediction_performance(
        self,
        model,
        X_test: np.ndarray,
        model_id: str,
        version: str = "1.0.0",
        batch_sizes: List[int] = None,
        iterations: int = 5
    ) -> List[PredictionPerformance]:
        """
        Benchmark prediction performance across different batch sizes.
        
        Args:
            model: Trained model object
            X_test: Test data for predictions
            model_id: Model identifier
            version: Model version
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per batch size
            
        Returns:
            List of prediction performance results
        """
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100, 500, 1000]
        
        results = []
        dataset_size = len(X_test)
        feature_count = X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
        
        for batch_size in batch_sizes:
            if batch_size > dataset_size:
                continue
            
            batch_times = []
            memory_usage = []
            cpu_usage = []
            
            for iteration in range(iterations):
                # Select batch
                batch_indices = np.random.choice(dataset_size, min(batch_size, dataset_size), replace=False)
                X_batch = X_test[batch_indices] if hasattr(X_test, '__getitem__') else [X_test[i] for i in batch_indices]
                
                # Measure performance
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024
                    start_cpu_times = process.cpu_times()
                
                start_time = time.time()
                
                # Perform prediction
                predictions = model.predict(X_batch)
                
                end_time = time.time()
                prediction_time = (end_time - start_time) * 1000  # Convert to ms
                
                if PSUTIL_AVAILABLE:
                    final_memory = process.memory_info().rss / 1024 / 1024
                    end_cpu_times = process.cpu_times()
                    
                    memory_used = final_memory - initial_memory
                    cpu_time = (end_cpu_times.user - start_cpu_times.user + 
                               end_cpu_times.system - start_cpu_times.system)
                    cpu_percent = (cpu_time / (prediction_time / 1000) * 100) if prediction_time > 0 else 0
                else:
                    memory_used = 0
                    cpu_percent = 0
                
                batch_times.append(prediction_time)
                memory_usage.append(memory_used)
                cpu_usage.append(cpu_percent)
            
            # Calculate statistics
            avg_time = statistics.mean(batch_times)
            throughput = (batch_size / (avg_time / 1000)) if avg_time > 0 else 0
            avg_memory = statistics.mean(memory_usage) if memory_usage else 0
            avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0
            
            # Create performance record
            performance = PredictionPerformance(
                model_id=model_id,
                version=version,
                dataset_size=dataset_size,
                feature_count=feature_count,
                prediction_time_ms=avg_time,
                throughput_predictions_per_sec=throughput,
                memory_usage_mb=avg_memory,
                cpu_usage_percent=avg_cpu,
                batch_size=batch_size,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'iterations': iterations,
                    'time_std': statistics.stdev(batch_times) if len(batch_times) > 1 else 0,
                    'memory_std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    'predictions_count': len(predictions) if hasattr(predictions, '__len__') else 1
                }
            )
            
            results.append(performance)
            self.prediction_history.append(performance)
            
            logger.info(f"Batch {batch_size}: {avg_time:.2f}ms, {throughput:.1f} pred/sec")
        
        # Save results
        self._save_performance_history()
        return results
    
    def benchmark_training_performance(
        self,
        model_params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_id: str,
        version: str = "1.0.0",
        cv_folds: int = 5,
        hyperopt_trials: int = 0
    ) -> TrainingPerformance:
        """
        Benchmark training performance.
        
        Args:
            model_params: Model parameters
            X_train: Training features
            y_train: Training targets
            model_id: Model identifier
            version: Model version
            cv_folds: Number of cross-validation folds
            hyperopt_trials: Number of hyperparameter optimization trials
            
        Returns:
            Training performance result
        """
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        
        dataset_size = len(X_train)
        feature_count = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        n_estimators = model_params.get('n_estimators', 100)
        
        # Determine task type
        unique_targets = len(np.unique(y_train))
        is_classification = unique_targets < 20 and np.all(y_train == y_train.astype(int))
        
        if is_classification:
            model_class = xgb.XGBClassifier
            scoring = 'accuracy'
        else:
            model_class = xgb.XGBRegressor
            scoring = 'r2'
        
        # Measure training performance
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            start_cpu_times = process.cpu_times()
        
        start_time = time.time()
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if PSUTIL_AVAILABLE:
            final_memory = process.memory_info().rss / 1024 / 1024
            end_cpu_times = process.cpu_times()
            
            memory_peak = final_memory - initial_memory
            cpu_time = (end_cpu_times.user - start_cpu_times.user + 
                       end_cpu_times.system - start_cpu_times.system)
            cpu_percent = (cpu_time / training_time * 100) if training_time > 0 else 0
        else:
            memory_peak = 0
            cpu_percent = 0
        
        # Create performance record
        performance = TrainingPerformance(
            model_id=model_id,
            version=version,
            dataset_size=dataset_size,
            feature_count=feature_count,
            n_estimators=n_estimators,
            training_time_seconds=training_time,
            memory_peak_mb=memory_peak,
            cpu_usage_percent=cpu_percent,
            cv_scores=cv_scores.tolist(),
            hyperopt_trials=hyperopt_trials,
            timestamp=datetime.now().isoformat(),
            metadata={
                'scoring_method': scoring,
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'model_params': model_params,
                'task_type': 'classification' if is_classification else 'regression'
            }
        )
        
        self.training_history.append(performance)
        self._save_performance_history()
        
        logger.info(f"Training completed: {training_time:.2f}s, CV score: {np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
        
        return performance
    
    def detect_performance_regression(
        self,
        model_id: str,
        baseline_version: str,
        current_version: str,
        metrics: List[str] = None,
        threshold: float = 0.05
    ) -> List[PerformanceRegression]:
        """
        Detect performance regressions between model versions.
        
        Args:
            model_id: Model identifier
            baseline_version: Baseline version for comparison
            current_version: Current version to check
            metrics: List of metrics to check
            threshold: Regression threshold (5% by default)
            
        Returns:
            List of detected regressions
        """
        if metrics is None:
            metrics = ['throughput_predictions_per_sec', 'training_time_seconds', 'cv_mean']
        
        regressions = []
        
        # Get baseline and current performance data
        baseline_predictions = [p for p in self.prediction_history 
                              if p.model_id == model_id and p.version == baseline_version]
        current_predictions = [p for p in self.prediction_history 
                             if p.model_id == model_id and p.version == current_version]
        
        baseline_training = [t for t in self.training_history 
                           if t.model_id == model_id and t.version == baseline_version]
        current_training = [t for t in self.training_history 
                          if t.model_id == model_id and t.version == current_version]
        
        # Check prediction metrics
        for metric in metrics:
            if metric == 'throughput_predictions_per_sec' and baseline_predictions and current_predictions:
                baseline_value = statistics.mean([p.throughput_predictions_per_sec for p in baseline_predictions])
                current_value = statistics.mean([p.throughput_predictions_per_sec for p in current_predictions])
                
                regression_percent = (baseline_value - current_value) / baseline_value if baseline_value > 0 else 0
                is_regression = regression_percent > threshold
                
                if is_regression:
                    regression = PerformanceRegression(
                        model_id=model_id,
                        metric_name=metric,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        regression_percent=regression_percent,
                        is_regression=is_regression,
                        threshold=threshold,
                        timestamp=datetime.now().isoformat()
                    )
                    regressions.append(regression)
                    self.regression_history.append(regression)
        
        # Check training metrics
        for metric in metrics:
            if metric == 'training_time_seconds' and baseline_training and current_training:
                baseline_value = statistics.mean([t.training_time_seconds for t in baseline_training])
                current_value = statistics.mean([t.training_time_seconds for t in current_training])
                
                # For training time, increase is regression
                regression_percent = (current_value - baseline_value) / baseline_value if baseline_value > 0 else 0
                is_regression = regression_percent > threshold
                
                if is_regression:
                    regression = PerformanceRegression(
                        model_id=model_id,
                        metric_name=metric,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        regression_percent=regression_percent,
                        is_regression=is_regression,
                        threshold=threshold,
                        timestamp=datetime.now().isoformat()
                    )
                    regressions.append(regression)
                    self.regression_history.append(regression)
            
            elif metric == 'cv_mean' and baseline_training and current_training:
                baseline_value = statistics.mean([t.metadata['cv_mean'] for t in baseline_training])
                current_value = statistics.mean([t.metadata['cv_mean'] for t in current_training])
                
                # For CV score, decrease is regression
                regression_percent = (baseline_value - current_value) / baseline_value if baseline_value > 0 else 0
                is_regression = regression_percent > threshold
                
                if is_regression:
                    regression = PerformanceRegression(
                        model_id=model_id,
                        metric_name=metric,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        regression_percent=regression_percent,
                        is_regression=is_regression,
                        threshold=threshold,
                        timestamp=datetime.now().isoformat()
                    )
                    regressions.append(regression)
                    self.regression_history.append(regression)
        
        # Save regression history
        self._save_performance_history()
        
        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions for {model_id}")
        else:
            logger.info(f"No performance regressions detected for {model_id}")
        
        return regressions
    
    def generate_performance_visualization(
        self,
        model_id: Optional[str] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate performance visualization plots.
        
        Args:
            model_id: Filter by model ID (all models if None)
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary containing plot information
        """
        if save_plots:
            plt.style.use('default')  # Use default style
        
        plots = {}
        
        # Filter data
        prediction_data = self.prediction_history
        
        if model_id:
            prediction_data = [p for p in prediction_data if p.model_id == model_id]
        
        # 1. Prediction Throughput vs Batch Size
        if prediction_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batch_sizes = [p.batch_size for p in prediction_data]
            throughputs = [p.throughput_predictions_per_sec for p in prediction_data]
            
            ax.scatter(batch_sizes, throughputs, alpha=0.7)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (predictions/sec)')
            ax.set_title('Prediction Throughput vs Batch Size')
            ax.grid(True, alpha=0.3)
            
            if save_plots:
                plot_path = self.visualizations_dir / 'throughput_vs_batch_size.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots['throughput_plot'] = str(plot_path)
            
            plt.close()
        
        logger.info(f"Generated {len(plots)} performance visualization plots")
        return plots
    
    def get_comprehensive_report(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            model_id: Filter by model ID (all models if None)
            
        Returns:
            Comprehensive performance report
        """
        # Filter data
        prediction_data = [p for p in self.prediction_history if not model_id or p.model_id == model_id]
        
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'model_filter': model_id,
            'summary': {
                'total_prediction_benchmarks': len(prediction_data)
            }
        }
        
        # Prediction performance summary
        if prediction_data:
            throughputs = [p.throughput_predictions_per_sec for p in prediction_data]
            prediction_times = [p.prediction_time_ms for p in prediction_data]
            
            report['prediction_performance'] = {
                'average_throughput': statistics.mean(throughputs),
                'max_throughput': max(throughputs),
                'min_throughput': min(throughputs),
                'average_prediction_time_ms': statistics.mean(prediction_times),
                'fastest_prediction_ms': min(prediction_times),
                'slowest_prediction_ms': max(prediction_times)
            }
        
        return report
    
    def _load_performance_history(self):
        """Load performance history from files."""
        # Load prediction history
        prediction_file = self.predictions_dir / "prediction_history.json"
        if prediction_file.exists():
            try:
                with open(prediction_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        performance = PredictionPerformance(**item)
                        self.prediction_history.append(performance)
            except Exception as e:
                logger.warning(f"Failed to load prediction history: {e}")
    
    def _save_performance_history(self):
        """Save performance history to files."""
        try:
            # Save prediction history
            prediction_file = self.predictions_dir / "prediction_history.json"
            with open(prediction_file, 'w', encoding='utf-8') as f:
                data = []
                for p in self.prediction_history:
                    data.append({
                        'model_id': p.model_id,
                        'version': p.version,
                        'dataset_size': p.dataset_size,
                        'feature_count': p.feature_count,
                        'prediction_time_ms': p.prediction_time_ms,
                        'throughput_predictions_per_sec': p.throughput_predictions_per_sec,
                        'memory_usage_mb': p.memory_usage_mb,
                        'cpu_usage_percent': p.cpu_usage_percent,
                        'batch_size': p.batch_size,
                        'timestamp': p.timestamp,
                        'metadata': p.metadata
                    })
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}") 