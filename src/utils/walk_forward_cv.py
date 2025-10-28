"""
Walk-Forward Cross-Validation for Time Series
Implements expanding and sliding window validation for robust evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single fold of walk-forward validation"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray


class WalkForwardValidator:
    """
    Implements walk-forward (rolling window) cross-validation
    
    This is the GOLD STANDARD for time series model evaluation because:
    1. Respects temporal ordering
    2. Tests on unseen future data
    3. Simulates real deployment scenario
    4. Provides robust performance estimates
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def expanding_window_cv(
        self,
        data: pd.DataFrame,
        date_column: str,
        initial_train_months: int = 24,
        test_months: int = 3,
        step_months: int = 3,
        max_folds: int = None
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate expanding window splits
        
        Expanding window: training set grows over time
        Example with initial=24, test=3, step=3:
          Fold 1: Train on months 1-24,  test on months 25-27
          Fold 2: Train on months 1-27,  test on months 28-30
          Fold 3: Train on months 1-30,  test on months 31-33
        
        Args:
            data: DataFrame with temporal data
            date_column: Column containing dates
            initial_train_months: Initial training window size
            test_months: Test window size
            step_months: How far to move forward each fold
            max_folds: Maximum number of folds (None = all possible)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        data = data.sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(data[date_column])
        
        min_date = dates.min()
        max_date = dates.max()
        
        # Initial training end
        initial_train_end = min_date + pd.DateOffset(months=initial_train_months)
        
        splits = []
        current_train_end = initial_train_end
        fold_num = 0
        
        while True:
            # Test period
            test_start = current_train_end + timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > max_date:
                break
            
            # Train period (from beginning to current_train_end)
            train_mask = dates <= current_train_end
            test_mask = (dates > test_start) & (dates <= test_end)
            
            train_idx = data[train_mask].index
            test_idx = data[test_mask].index
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                fold_num += 1
                
                self.logger.info(
                    f"Fold {fold_num}: Train {len(train_idx)} samples "
                    f"({dates[train_idx].min().date()} to {dates[train_idx].max().date()}), "
                    f"Test {len(test_idx)} samples "
                    f"({dates[test_idx].min().date()} to {dates[test_idx].max().date()})"
                )
            
            if max_folds and fold_num >= max_folds:
                break
            
            # Move forward
            current_train_end += pd.DateOffset(months=step_months)
        
        self.logger.info(f"Generated {len(splits)} expanding window folds")
        return splits
    
    def sliding_window_cv(
        self,
        data: pd.DataFrame,
        date_column: str,
        train_months: int = 24,
        test_months: int = 3,
        step_months: int = 3,
        max_folds: int = None
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Generate sliding window splits
        
        Sliding window: training set size stays constant, slides forward
        Example with train=24, test=3, step=3:
          Fold 1: Train on months 1-24,  test on months 25-27
          Fold 2: Train on months 4-27,  test on months 28-30
          Fold 3: Train on months 7-30,  test on months 31-33
        
        Args:
            data: DataFrame with temporal data
            date_column: Column containing dates
            train_months: Training window size (fixed)
            test_months: Test window size
            step_months: How far to move forward each fold
            max_folds: Maximum number of folds
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        data = data.sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(data[date_column])
        
        min_date = dates.min()
        max_date = dates.max()
        
        splits = []
        fold_num = 0
        current_start = min_date
        
        while True:
            # Training period
            train_start = current_start
            train_end = train_start + pd.DateOffset(months=train_months)
            
            # Test period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > max_date:
                break
            
            train_mask = (dates >= train_start) & (dates <= train_end)
            test_mask = (dates > test_start) & (dates <= test_end)
            
            train_idx = data[train_mask].index
            test_idx = data[test_mask].index
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                fold_num += 1
                
                self.logger.info(
                    f"Fold {fold_num}: Train {len(train_idx)} samples "
                    f"({dates[train_idx].min().date()} to {dates[train_idx].max().date()}), "
                    f"Test {len(test_idx)} samples "
                    f"({dates[test_idx].min().date()} to {dates[test_idx].max().date()})"
                )
            
            if max_folds and fold_num >= max_folds:
                break
            
            # Slide forward
            current_start += pd.DateOffset(months=step_months)
        
        self.logger.info(f"Generated {len(splits)} sliding window folds")
        return splits
    
    def walk_forward_evaluate(
        self,
        data: pd.DataFrame,
        date_column: str,
        train_fn: Callable,
        evaluate_fn: Callable,
        window_type: str = 'expanding',
        **kwargs
    ) -> Dict:
        """
        Execute walk-forward cross-validation
        
        Args:
            data: Complete dataset
            date_column: Date column name
            train_fn: Function(train_data) -> model
            evaluate_fn: Function(model, test_data) -> metrics_dict
            window_type: 'expanding' or 'sliding'
            **kwargs: Arguments for window generation
            
        Returns:
            Dictionary with aggregate results and per-fold details
        """
        self.logger.info("="*70)
        self.logger.info(f"WALK-FORWARD CROSS-VALIDATION ({window_type.upper()} WINDOW)")
        self.logger.info("="*70)
        
        # Generate splits
        if window_type == 'expanding':
            splits = self.expanding_window_cv(data, date_column, **kwargs)
        elif window_type == 'sliding':
            splits = self.sliding_window_cv(data, date_column, **kwargs)
        else:
            raise ValueError(f"Unknown window_type: {window_type}")
        
        # Evaluate each fold
        fold_results = []
        all_metrics = []
        
        dates = pd.to_datetime(data[date_column])
        
        for fold_id, (train_idx, test_idx) in enumerate(splits, 1):
            self.logger.info(f"\n--- Processing Fold {fold_id}/{len(splits)} ---")
            
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Train model
            self.logger.info("Training model...")
            model = train_fn(train_data)
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            metrics, predictions, actuals = evaluate_fn(model, test_data)
            
            # Store results
            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=dates[train_idx].min(),
                train_end=dates[train_idx].max(),
                test_start=dates[test_idx].min(),
                test_end=dates[test_idx].max(),
                metrics=metrics,
                predictions=predictions,
                actuals=actuals
            )
            
            fold_results.append(fold_result)
            all_metrics.append(metrics)
            
            # Log fold results
            self.logger.info(f"Fold {fold_id} Results:")
            for metric_name, value in metrics.items():
                self.logger.info(f"  {metric_name}: {value:.4f}")
        
        # Aggregate results
        aggregate_metrics = self._aggregate_fold_metrics(all_metrics)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("AGGREGATE RESULTS ACROSS ALL FOLDS")
        self.logger.info("="*70)
        for metric_name, stats in aggregate_metrics.items():
            self.logger.info(
                f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])"
            )
        
        return {
            'fold_results': fold_results,
            'aggregate_metrics': aggregate_metrics,
            'num_folds': len(splits),
            'window_type': window_type
        }
    
    def _aggregate_fold_metrics(
        self,
        fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across folds with confidence intervals
        
        Returns:
            Dict mapping metric_name to {mean, std, ci_lower, ci_upper}
        """
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(fold_metrics)
        
        aggregated = {}
        
        for metric_name in df.columns:
            values = df[metric_name].values
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            
            # 95% confidence interval
            n = len(values)
            se = std / np.sqrt(n)
            ci_margin = 1.96 * se  # For 95% CI with normal approximation
            
            aggregated[metric_name] = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(mean - ci_margin),
                'ci_upper': float(mean + ci_margin),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return aggregated
    
    def save_results(
        self,
        results: Dict,
        output_path: str = 'results/walk_forward_results.json'
    ):
        """Save walk-forward results to JSON"""
        # Convert FoldResult objects to dicts for JSON serialization
        serializable_results = {
            'num_folds': results['num_folds'],
            'window_type': results['window_type'],
            'aggregate_metrics': results['aggregate_metrics'],
            'fold_details': [
                {
                    'fold_id': fr.fold_id,
                    'train_start': fr.train_start.isoformat(),
                    'train_end': fr.train_end.isoformat(),
                    'test_start': fr.test_start.isoformat(),
                    'test_end': fr.test_end.isoformat(),
                    'metrics': fr.metrics
                }
                for fr in results['fold_results']
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")


class TimeSeriesValidator:
    """
    Complete time series validation framework
    Combines temporal splitting with walk-forward CV
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.walk_forward = WalkForwardValidator(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_model(
        self,
        data: pd.DataFrame,
        train_fn: Callable,
        evaluate_fn: Callable,
        validation_strategy: str = 'walk_forward',
        **kwargs
    ) -> Dict:
        """
        Validate model using specified strategy
        
        Args:
            data: Complete dataset with temporal ordering
            train_fn: Training function
            evaluate_fn: Evaluation function
            validation_strategy: 'walk_forward', 'holdout', or 'both'
            **kwargs: Strategy-specific arguments
            
        Returns:
            Validation results
        """
        if validation_strategy == 'walk_forward':
            return self.walk_forward.walk_forward_evaluate(
                data=data,
                date_column='Date',
                train_fn=train_fn,
                evaluate_fn=evaluate_fn,
                **kwargs
            )
        elif validation_strategy == 'holdout':
            # Simple temporal holdout (train on past, test on future)
            return self._holdout_validate(data, train_fn, evaluate_fn, **kwargs)
        elif validation_strategy == 'both':
            wf_results = self.walk_forward.walk_forward_evaluate(
                data, 'Date', train_fn, evaluate_fn, **kwargs
            )
            holdout_results = self._holdout_validate(data, train_fn, evaluate_fn)
            
            return {
                'walk_forward': wf_results,
                'holdout': holdout_results
            }
        else:
            raise ValueError(f"Unknown validation strategy: {validation_strategy}")
    
    def _holdout_validate(
        self,
        data: pd.DataFrame,
        train_fn: Callable,
        evaluate_fn: Callable,
        train_end: str = '2021-12-31',
        val_end: str = '2022-12-31'
    ) -> Dict:
        """Simple temporal holdout validation"""
        dates = pd.to_datetime(data['Date'])
        
        train_mask = dates <= pd.to_datetime(train_end)
        val_mask = (dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(val_end))
        test_mask = dates > pd.to_datetime(val_end)
        
        train_data = data[train_mask]
        val_data = data[val_mask]
        test_data = data[test_mask]
        
        # Train on train set
        model = train_fn(train_data)
        
        # Evaluate on val and test
        val_metrics, val_preds, val_actuals = evaluate_fn(model, val_data)
        test_metrics, test_preds, test_actuals = evaluate_fn(model, test_data)
        
        return {
            'validation': val_metrics,
            'test': test_metrics,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample time series data
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'feature': np.random.randn(len(dates)),
        'target': np.random.randint(0, 3, len(dates))
    })
    
    # Simple mock training and evaluation functions
    def mock_train_fn(train_data):
        return {'dummy_model': 'trained'}
    
    def mock_evaluate_fn(model, test_data):
        predictions = np.random.randint(0, 3, len(test_data))
        actuals = test_data['target'].values
        accuracy = (predictions == actuals).mean()
        
        return {
            'accuracy': accuracy,
            'f1': accuracy * 0.95
        }, predictions, actuals
    
    # Run walk-forward validation
    validator = WalkForwardValidator({})
    
    results = validator.walk_forward_evaluate(
        data=data,
        date_column='Date',
        train_fn=mock_train_fn,
        evaluate_fn=mock_evaluate_fn,
        window_type='expanding',
        initial_train_months=24,
        test_months=3,
        step_months=3,
        max_folds=5
    )
    
    # Save results
    validator.save_results(results, 'results/walk_forward_example.json')
