# -*- coding: utf-8 -*-
"""
Hyperparameter Optimization Module for SVM

This module provides hyperparameter optimization functionality using Optuna
for SVM models, supporting TPE and GP algorithms.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.svm import SVC, SVR
import optuna
from optuna.samplers import TPESampler
import warnings

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Hyperparameter optimization for SVM models using Optuna.
    Supports both classification (SVC) and regression (SVR) tasks.
    """
    
    def __init__(self, 
                 sampler_type: str = "TPE",
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: Optional[int] = 42):
        """Initialize SVM HyperparameterOptimizer."""
        self.sampler_type = sampler_type.upper()
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Optimization state
        self.study = None
        self.best_params = None
        self.best_score = None
        self.task_type = None
        
        # Store actual parameters used for each trial
        self.actual_params_list = []
        
        # Create sampler
        self.sampler = self._create_sampler()
        
    def _create_sampler(self):
        """Create Optuna sampler based on configuration."""
        if self.sampler_type == "TPE":
            return TPESampler(
                n_startup_trials=10, 
                n_ei_candidates=24,
                seed=self.random_state
            )
        elif self.sampler_type == "GP":
            try:
                from optuna.integration import SkoptSampler
                return SkoptSampler(
                    skopt_kwargs={
                        'base_estimator': 'GP',
                        'n_initial_points': 10,
                        'acq_func': 'EI'
                    }
                )
            except ImportError:
                logger.warning("Scikit-optimize not available, falling back to TPE")
                return TPESampler(seed=self.random_state)
        else:
            logger.warning(f"Unknown sampler type {self.sampler_type}, using TPE")
            return TPESampler(seed=self.random_state)
            
    def _suggest_svm_hyperparameters(self, trial, kernel: str, task_type: str, svm_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suggest hyperparameters for SVM optimization based on kernel type."""
        # Start with base parameters
        params = {
            'kernel': kernel
        }
        
        # Add any base parameters provided
        if svm_params:
            # Filter out parameters not supported by SVR for regression tasks
            if task_type == "regression":
                filtered_params = {k: v for k, v in svm_params.items() 
                                 if k not in ['random_state', 'class_weight']}
                params.update(filtered_params)
            else:
                params.update(svm_params)
        
        # Kernel-specific parameter suggestions
        if kernel == 'linear':
            params.update({
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            })
            
        elif kernel == 'rbf':
            params.update({
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            })
            # For RBF kernel, suggest gamma
            gamma_type = trial.suggest_categorical('gamma_type', ['auto', 'manual'])
            if gamma_type == 'auto':
                params['gamma'] = trial.suggest_categorical('gamma_mode', ['scale', 'auto'])
            else:
                params['gamma'] = trial.suggest_float('gamma_value', 1e-5, 1e1, log=True)
            
        elif kernel == 'poly':
            params.update({
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'degree': trial.suggest_int('degree', 2, 5),
                'coef0': trial.suggest_float('coef0', 0.0, 10.0),
            })
            # For poly kernel, suggest gamma
            gamma_type = trial.suggest_categorical('gamma_type', ['auto', 'manual'])
            if gamma_type == 'auto':
                params['gamma'] = trial.suggest_categorical('gamma_mode', ['scale', 'auto'])
            else:
                params['gamma'] = trial.suggest_float('gamma_value', 1e-5, 1e1, log=True)
            
        elif kernel == 'sigmoid':
            params.update({
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'coef0': trial.suggest_float('coef0', 0.0, 10.0),
            })
            # For sigmoid kernel, suggest gamma
            gamma_type = trial.suggest_categorical('gamma_type', ['auto', 'manual'])
            if gamma_type == 'auto':
                params['gamma'] = trial.suggest_categorical('gamma_mode', ['scale', 'auto'])
            else:
                params['gamma'] = trial.suggest_float('gamma_value ', 1e-5, 1e1, log=True)
        
        # Task-specific parameters
        if task_type == "classification":
            # Add class weight options for classification
            if 'class_weight' not in params:  # Don't override if already set in base_params
                params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None])
        else:
            # Add epsilon for regression
            if 'epsilon' not in params:  # Don't override if already set in base_params
                params['epsilon'] = trial.suggest_float('epsilon', 1e-4, 1.0, log=True)
        
        print(f"》》》Suggested parameters: {params}")
            
        return params
        
    def _create_objective_function(self, X: np.ndarray, y: np.ndarray, 
                                 kernel: str, task_type: str, scoring_metric: str, svm_params: Optional[Dict[str, Any]] = None) -> Callable:
        """Create the objective function for SVM optimization."""
            
        # Create cross-validation splitter
        if task_type == "classification":
            cv_splitter = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            cv_splitter = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
        def objective(trial):
            """Optuna objective function for SVM."""
            try:
                # Get suggested hyperparameters
                params = self._suggest_svm_hyperparameters(trial, kernel, task_type, svm_params)
                
                # Record actual parameters used for this trial
                self.actual_params_list.append(params.copy())
                
                # Create SVM model
                if task_type == "classification":
                    model = SVC(**params)
                else:
                    model = SVR(**params)
                
                # Perform cross-validation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        model, X, y,
                        cv=cv_splitter,
                        scoring=scoring_metric,
                        n_jobs=1
                    )
                
                mean_score = np.mean(scores)
                
                # Validate the score
                if np.isnan(mean_score) or np.isinf(mean_score):
                    logger.warning(f"Trial {trial.number} produced invalid score: {mean_score}")
                    raise optuna.exceptions.TrialPruned()
                
                # Store additional trial information
                trial.set_user_attr("cv_scores", scores.tolist())
                trial.set_user_attr("cv_std", float(np.std(scores)))
                trial.set_user_attr("params", params)
                
                return mean_score
                
            except Exception as e:
                logger.warning(f"SVM trial {trial.number} failed: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective

    def optimize(self, 
                X: Union[np.ndarray, pd.DataFrame], 
                y: np.ndarray,
                kernel: str = "rbf",
                task_type: Optional[str] = None,
                svm_params: Optional[Dict[str, Any]] = None,
                scoring_metric: Optional[str] = None,
                save_dir: Optional[str] = None,
                ) -> Tuple[Dict[str, Any], float, Optional[pd.DataFrame]]:
        """
        Optimize SVM hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target variable
            kernel: SVM kernel ('linear', 'rbf', 'poly', 'sigmoid')
            task_type: 'classification' or 'regression' (auto-detected if None)
            scoring_metric: Scoring metric for evaluation
            save_dir: Directory to save optimization history CSV
            svm_params: Additional SVM parameters (will not be optimized)
            
        Returns:
            Tuple of (best_params, best_score, trials_dataframe)
        """
        
        # Clear previous optimization results
        self.actual_params_list = []
        
        # Input validation
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
        if y is None or len(y) == 0:
            raise ValueError("y cannot be None or empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
            
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Check for NaN values
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values. Please clean your data first.")
        
        # Handle NaN values in target
        try:
            if np.any(np.isnan(y)):
                raise ValueError("y contains NaN values. Please clean your data first.")
        except TypeError:
            # If np.isnan fails (e.g., for categorical/string targets), use pandas isna
            if pd.Series(y).isna().any():
                raise ValueError("y contains NaN values. Please clean your data first.")
            
        # Auto-detect task type if not provided
        if task_type is None:
            # Simple heuristic: if y has few unique values, it's likely classification
            unique_values = len(np.unique(y))
            if unique_values <= 20 and (y.dtype == 'object' or np.issubdtype(y.dtype, np.integer)):
                task_type = "classification"
            else:
                task_type = "regression"
            
        self.task_type = task_type
        
        # Set default scoring metric
        if scoring_metric is None:
            scoring_metric = "f1_weighted" if task_type == "classification" else "r2"
            
        logger.info(f"Starting SVM optimization for {task_type} task")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Kernel: {kernel}")
        logger.info(f"Using {self.sampler_type} algorithm with {self.n_trials} trials")
        logger.info(f"Scoring metric: {scoring_metric}")
        
        # Create study - determine optimization direction based on scoring metric
        maximize_metrics = [
            "accuracy", "f1", "f1_weighted", "f1_macro", "f1_micro", 
            "r2", "roc_auc", "roc_auc_ovr", "roc_auc_ovo",
            "neg_mean_absolute_error", "neg_mean_squared_error", 
            "neg_root_mean_squared_error", "neg_median_absolute_error"
        ]
        direction = "maximize" if scoring_metric in maximize_metrics else "minimize"
        self.study = optuna.create_study(direction=direction, sampler=self.sampler)
        
        # Create objective function
        objective = self._create_objective_function(X, y, kernel, task_type, scoring_metric, svm_params)
        
        # Run optimization with progress bar
        self.study.optimize(
            objective, 
            n_trials=self.n_trials, 
            show_progress_bar=True
        )
        
        # Check if any trials completed successfully
        completed_trials = [trial for trial in self.study.trials 
                          if trial.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            raise ValueError(
                f"No trials completed successfully out of {len(self.study.trials)} trials. "
                "This may be due to data issues or invalid parameters."
            )
        
        # Store results - get actual parameters used for best trial
        best_trial_index = self.study.best_trial.number
        if best_trial_index < len(self.actual_params_list):
            self.best_params = self.actual_params_list[best_trial_index].copy()
        else:
            # Fallback to study.best_params if index not found
            self.best_params = self.study.best_params.copy()
        self.best_score = self.study.best_value
        
        # Get trials dataframe
        trials_df = self.study.trials_dataframe()
        
        # Save trials results if save_dir provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, 'svm_optimization_history.csv')
            trials_df.to_csv(csv_path, index=False)
            logger.info(f"SVM optimization history saved to {csv_path}")
        
        logger.info(f"SVM optimization completed:")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score}")
        
        return self.best_params, self.best_score, trials_df

    def get_optimization_results(self) -> Dict[str, Any]:
        """Get comprehensive optimization results."""
        if self.study is None:
            return {}
            
        best_trial = self.study.best_trial
        
        # Create optimization history
        optimization_history = [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete and trial.datetime_start else None
            }
            for trial in self.study.trials
        ]
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'n_completed_trials': len([t for t in self.study.trials 
                                     if t.state == optuna.trial.TrialState.COMPLETE]),
            'task_type': self.task_type,
            'sampler_type': self.sampler_type,
            'optimization_history': optimization_history
        }
        
        # Add cross-validation details from best trial
        if hasattr(best_trial, 'user_attrs'):
            if 'cv_scores' in best_trial.user_attrs:
                results['best_cv_scores'] = best_trial.user_attrs['cv_scores']
                results['best_cv_std'] = best_trial.user_attrs['cv_std']
                
        return results
        
    def create_optimized_model(self, 
                             base_params: Optional[Dict[str, Any]] = None,
                             svm_params: Optional[Dict[str, Any]] = None):
        """Create optimized SVM model with best hyperparameters using SVMWrapper."""
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")
            
        from .svm_wrapper import SVMWrapper
        # Merge optimized, base, and additional params
        optimized_params = self.best_params.copy()
        if base_params:
            for key, value in base_params.items():
                if key not in optimized_params:
                    optimized_params[key] = value
        if svm_params:
            optimized_params.update(svm_params)
        # Use SVMWrapper for unified interface
        return SVMWrapper(task_type=self.task_type,probability=True, **optimized_params)
    
    def optimize_svm(self, 
                    X: Union[np.ndarray, pd.DataFrame], 
                    y: np.ndarray,
                    kernel: str = None,
                    task_type: Optional[str] = None,
                    scoring_metric: Optional[str] = None,
                    svm_params: Optional[Dict[str, Any]] = None,
                    save_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize SVM hyperparameters and return comprehensive results.
        
        This is a convenience method that combines optimize() and get_optimization_results().
        When kernel is None, all kernels ('linear', 'rbf', 'poly', 'sigmoid') will be tried
        and the globally best one will be returned.
        
        Args:
            X: Feature matrix
            y: Target variable
            kernel: SVM kernel ('linear', 'rbf', 'poly', 'sigmoid') or None for auto-selection
            task_type: 'classification' or 'regression' (auto-detected if None)
            scoring_metric: Scoring metric for evaluation
            svm_params: Additional SVM parameters (will not be optimized)
            save_dir: Directory to save optimization history CSV
            
        Returns:
            Tuple of (best_params, optimization_results)
        """
        print(f"Running optimization with svm_params: {svm_params}")
        
        if kernel is None:
            # Try all kernels and find the best one
            return self._optimize_all_kernels(
                X=X, y=y, task_type=task_type, 
                scoring_metric=scoring_metric, svm_params=svm_params, 
                save_dir=save_dir
            )
        else:
            # Single kernel optimization
            best_params, best_score, trials_df = self.optimize(
                X=X, y=y, kernel=kernel, task_type=task_type, 
                scoring_metric=scoring_metric, save_dir=save_dir,
                svm_params=svm_params
            )
            if 'kernel' not in best_params.keys():
                best_params['kernel'] = kernel
            # Get comprehensive results
            optimization_results = self.get_optimization_results()
            optimization_results['trials_dataframe'] = trials_df
            optimization_results['best_score'] = best_score
            
            return best_params, optimization_results

    def _optimize_all_kernels(self, 
                             X: Union[np.ndarray, pd.DataFrame], 
                             y: np.ndarray,
                             task_type: Optional[str] = None,
                             scoring_metric: Optional[str] = None,
                             svm_params: Optional[Dict[str, Any]] = None,
                             save_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize all kernel types and return the globally best result.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression' (auto-detected if None)
            scoring_metric: Scoring metric for evaluation
            svm_params: Additional SVM parameters (will not be optimized)
            save_dir: Directory to save optimization history CSV
            
        Returns:
            Tuple of (best_params, optimization_results)
        """
        available_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        
        logger.info("Starting multi-kernel optimization...")
        logger.info(f"Will try kernels: {available_kernels}")
        
        global_best_params = None
        global_best_score = None
        global_best_kernel = None
        all_kernel_results = {}
        all_trials_dfs = []
        
        # Determine optimization direction based on scoring metric
        maximize_metrics = [
            "accuracy", "f1", "f1_weighted", "f1_macro", "f1_micro", 
            "r2", "roc_auc", "roc_auc_ovr", "roc_auc_ovo",
            "neg_mean_absolute_error", "neg_mean_squared_error", 
            "neg_root_mean_squared_error", "neg_median_absolute_error"
        ]
        
        # Auto-detect task type if not provided
        if task_type is None:
            unique_values = len(np.unique(y))
            if unique_values <= 20 and (y.dtype == 'object' or np.issubdtype(y.dtype, np.integer)):
                task_type = "classification"
            else:
                task_type = "regression"
        
        # Set default scoring metric
        if scoring_metric is None:
            scoring_metric = "f1_weighted" if task_type == "classification" else "r2"
            
        is_maximizing = scoring_metric in maximize_metrics
        
        for kernel in available_kernels:
            logger.info(f"\n🔍 Optimizing kernel: {kernel}")
            
            try:
                # Run optimization for this kernel
                best_params, best_score, trials_df = self.optimize(
                    X=X, y=y, kernel=kernel, task_type=task_type, 
                    scoring_metric=scoring_metric, save_dir=None,  # Don't save individual results
                    svm_params=svm_params
                )
                
                # Add kernel to trials_df for tracking
                if trials_df is not None:
                    trials_df = trials_df.copy()
                    trials_df['kernel'] = kernel
                    all_trials_dfs.append(trials_df)
                
                # Store results for this kernel
                optimization_results = self.get_optimization_results()
                all_kernel_results[kernel] = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'optimization_results': optimization_results
                }
                
                logger.info(f"✅ {kernel.upper()} kernel: score = {best_score:.4f}")
                
                # Check if this is the global best
                if global_best_score is None:
                    global_best_score = best_score
                    global_best_params = best_params.copy()
                    global_best_kernel = kernel
                else:
                    is_better = (is_maximizing and best_score > global_best_score) or \
                               (not is_maximizing and best_score < global_best_score)
                    
                    if is_better:
                        global_best_score = best_score
                        global_best_params = best_params.copy()
                        global_best_kernel = kernel
                        
            except Exception as e:
                logger.warning(f"⚠️  Failed to optimize {kernel} kernel: {e}")
                all_kernel_results[kernel] = {
                    'best_params': None,
                    'best_score': None,
                    'optimization_results': None,
                    'error': str(e)
                }
        
        if global_best_params is None:
            raise ValueError("All kernel optimizations failed. Please check your data and parameters.")
        
        # Ensure kernel is in the best params
        if 'kernel' not in global_best_params:
            global_best_params['kernel'] = global_best_kernel
            
        # Update the optimizer state with the best results
        self.best_params = global_best_params
        self.best_score = global_best_score
        
        # Combine all trials dataframes
        combined_trials_df = None
        if all_trials_dfs:
            combined_trials_df = pd.concat(all_trials_dfs, ignore_index=True)
        
        # Save combined results if save_dir provided
        if save_dir and combined_trials_df is not None:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, 'svm_multi_kernel_optimization_history.csv')
            combined_trials_df.to_csv(csv_path, index=False)
            logger.info(f"Multi-kernel optimization history saved to {csv_path}")
        
        # Create comprehensive results
        comprehensive_results = {
            'best_params': global_best_params,
            'best_score': global_best_score,
            'best_kernel': global_best_kernel,
            'task_type': task_type,
            'sampler_type': self.sampler_type,
            'scoring_metric': scoring_metric,
            'kernel_results': all_kernel_results,
            'trials_dataframe': combined_trials_df,
            'optimization_summary': {
                kernel: {
                    'best_score': result.get('best_score'),
                    'success': result.get('best_score') is not None
                }
                for kernel, result in all_kernel_results.items()
            }
        }
        
        logger.info(f"\n🎉 Multi-kernel optimization completed!")
        logger.info(f"Global best kernel: {global_best_kernel}")
        logger.info(f"Global best score: {global_best_score:.4f}")
        logger.info(f"Global best params: {global_best_params}")
        
        # Print summary of all kernels
        logger.info("\n📊 Kernel comparison summary:")
        for kernel, result in all_kernel_results.items():
            if result.get('best_score') is not None:
                logger.info(f"  {kernel:>8}: {result['best_score']:.4f}")
            else:
                logger.info(f"  {kernel:>8}: FAILED")
        
        return global_best_params, comprehensive_results

