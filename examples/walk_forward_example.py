"""
Example: Walk-Forward Cross-Validation
Demonstrates robust time-series evaluation with realistic backtesting
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.walk_forward_cv import WalkForwardValidator
from src.evaluation.walk_forward_evaluator import WalkForwardEvaluator, EnhancedBacktester

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic stock and news data for demonstration
    
    Returns:
        DataFrame with dates, features, targets, and returns
    """
    logger.info("Generating synthetic data...")
    
    # Generate dates
    dates = pd.date_range('2019-01-01', periods=n_samples, freq='D')
    
    # Generate features
    data = pd.DataFrame({
        'Date': dates,
        # Text embeddings (simulated FinBERT output)
        'text_embeddings': [np.random.randn(768) for _ in range(n_samples)],
        # Numerical features (simulated technical indicators)
        'numerical_features': [np.random.randn(50) for _ in range(n_samples)],
        # Target: 0=down, 1=neutral, 2=up
        'target': np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3]),
        # Actual returns
        'returns': np.random.randn(n_samples) * 0.02  # ~2% daily volatility
    })
    
    logger.info(f"Generated {n_samples} samples from {dates[0].date()} to {dates[-1].date()}")
    
    return data


def example_basic_walk_forward():
    """
    Example 1: Basic walk-forward validation with mock model
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Basic Walk-Forward Cross-Validation")
    logger.info("="*70 + "\n")
    
    # Generate data
    data = generate_synthetic_data(n_samples=1500)
    
    # Configuration
    config = {
        'features': {'lookback_window': 30},
        'training': {'epochs': 10, 'batch_size': 32}
    }
    
    # Initialize validator
    validator = WalkForwardValidator(config)
    
    # Define simple mock training and evaluation functions
    def mock_train_fn(train_data):
        """Mock training function"""
        logger.info(f"  Training on {len(train_data)} samples")
        return {'model': 'mock_trained_model'}
    
    def mock_evaluate_fn(model, test_data):
        """Mock evaluation function"""
        logger.info(f"  Evaluating on {len(test_data)} samples")
        
        # Generate random predictions
        predictions = np.random.choice([0, 1, 2], size=len(test_data))
        actuals = test_data['target'].values
        
        # Calculate metrics
        accuracy = (predictions == actuals).mean()
        f1 = accuracy * 0.95  # Mock F1
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'sharpe': np.random.randn() * 0.5 + 1.0
        }, predictions, actuals
    
    # Run walk-forward validation
    results = validator.walk_forward_evaluate(
        data=data,
        date_column='Date',
        train_fn=mock_train_fn,
        evaluate_fn=mock_evaluate_fn,
        window_type='expanding',
        initial_train_months=24,
        test_months=3,
        step_months=3,
        max_folds=4
    )
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("AGGREGATE RESULTS")
    logger.info("="*70)
    
    for metric_name, stats in results['aggregate_metrics'].items():
        logger.info(
            f"{metric_name:20s}: {stats['mean']:7.4f} ± {stats['std']:6.4f} "
            f"(95% CI: [{stats['ci_lower']:7.4f}, {stats['ci_upper']:7.4f}])"
        )
    
    logger.info(f"\nTotal folds: {results['num_folds']}")
    logger.info(f"Window type: {results['window_type']}")
    
    # Save results
    validator.save_results(results, 'results/example_walk_forward_basic.json')


def example_enhanced_backtesting():
    """
    Example 2: Enhanced backtesting with realistic trading costs
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Enhanced Backtesting with Transaction Costs")
    logger.info("="*70 + "\n")
    
    # Configuration with backtesting parameters
    config = {
        'evaluation': {
            'backtesting': {
                'initial_capital': 100000,
                'transaction_cost': 0.002,  # 0.2% per trade
                'slippage': 0.001,           # 0.1% slippage
                'max_position_pct': 0.10,    # 10% max position
                'stop_loss_pct': 0.05,       # 5% stop loss
                'confidence_threshold': 0.6   # 60% confidence to trade
            }
        }
    }
    
    # Initialize backtester
    backtester = EnhancedBacktester(config)
    
    # Generate sample predictions and returns
    n_periods = 252  # 1 year of trading days
    predictions = np.random.choice([0, 1, 2], size=n_periods, p=[0.3, 0.3, 0.4])
    
    # Prediction probabilities (more confident predictions)
    prediction_probs = np.zeros((n_periods, 3))
    for i in range(n_periods):
        probs = np.random.dirichlet([0.5, 0.5, 0.5])  # Random probabilities
        probs[predictions[i]] = max(probs[predictions[i]], 0.7)  # Boost correct class
        probs = probs / probs.sum()  # Renormalize
        prediction_probs[i] = probs
    
    # Actual returns (slightly correlated with predictions for realism)
    actual_returns = np.random.randn(n_periods) * 0.015
    for i in range(n_periods):
        if predictions[i] == 2:  # Predicted up
            actual_returns[i] += 0.005  # Slight positive bias
        elif predictions[i] == 0:  # Predicted down
            actual_returns[i] -= 0.005  # Slight negative bias
    
    # Run backtest
    metrics = backtester.backtest_with_costs(
        predictions, prediction_probs, actual_returns
    )
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("BACKTEST RESULTS (with costs)")
    logger.info("="*70)
    
    logger.info(f"Total Return:        {metrics['total_return']:>8.2%}")
    logger.info(f"CAGR:               {metrics['cagr']:>8.2%}")
    logger.info(f"Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}")
    logger.info(f"Sortino Ratio:      {metrics['sortino_ratio']:>8.2f}")
    logger.info(f"Max Drawdown:       {metrics['max_drawdown']:>8.2%}")
    logger.info(f"Win Rate:           {metrics['win_rate']:>8.2%}")
    logger.info(f"Total Trades:       {metrics['total_trades']:>8d}")
    logger.info(f"Winning Trades:     {metrics['winning_trades']:>8d}")
    logger.info(f"Final Capital:      ${metrics['final_capital']:>11,.2f}")
    logger.info(f"Total Costs:        ${metrics['total_costs']:>11,.2f}")
    logger.info(f"Cost Drag:          {metrics['cost_drag']:>8.2%}")
    logger.info(f"Annual Turnover:    {metrics['turnover']:>8.2f}x")
    
    logger.info("\nNote: These results include realistic transaction costs and slippage")


def example_expanding_vs_sliding():
    """
    Example 3: Compare expanding vs sliding window strategies
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Expanding vs Sliding Window Comparison")
    logger.info("="*70 + "\n")
    
    # Generate data
    data = generate_synthetic_data(n_samples=1500)
    
    config = {'features': {'lookback_window': 30}}
    validator = WalkForwardValidator(config)
    
    # Mock functions
    def mock_train_fn(train_data):
        return {'model': 'trained'}
    
    def mock_evaluate_fn(model, test_data):
        predictions = np.random.choice([0, 1, 2], size=len(test_data))
        actuals = test_data['target'].values
        accuracy = (predictions == actuals).mean()
        
        return {
            'accuracy': accuracy,
            'f1': accuracy * 0.95
        }, predictions, actuals
    
    # Test both strategies
    strategies = ['expanding', 'sliding']
    comparison = {}
    
    for strategy in strategies:
        logger.info(f"\nTesting {strategy.upper()} window strategy...")
        
        results = validator.walk_forward_evaluate(
            data=data,
            date_column='Date',
            train_fn=mock_train_fn,
            evaluate_fn=mock_evaluate_fn,
            window_type=strategy,
            initial_train_months=24,
            test_months=3,
            step_months=3,
            max_folds=3
        )
        
        comparison[strategy] = results['aggregate_metrics']
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*70)
    
    logger.info(f"\n{'Metric':<15} {'Expanding Mean':<15} {'Sliding Mean':<15} {'Difference':<15}")
    logger.info("-" * 70)
    
    for metric in comparison['expanding'].keys():
        exp_mean = comparison['expanding'][metric]['mean']
        sli_mean = comparison['sliding'][metric]['mean']
        diff = exp_mean - sli_mean
        
        logger.info(f"{metric:<15} {exp_mean:<15.4f} {sli_mean:<15.4f} {diff:<+15.4f}")
    
    logger.info("\nExpanding window: Training set grows over time (more data)")
    logger.info("Sliding window:   Fixed training set size (more recent data)")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Walk-Forward Cross-Validation Examples")
    logger.info("    Demonstrating Robust Time-Series Evaluation")
    logger.info("="*70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run examples
    try:
        example_basic_walk_forward()
        example_enhanced_backtesting()
        example_expanding_vs_sliding()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info("\nKey Takeaways:")
        logger.info("1. Walk-forward CV respects temporal ordering (no lookahead)")
        logger.info("2. Provides robust performance estimates with confidence intervals")
        logger.info("3. Enhanced backtesting includes realistic trading costs")
        logger.info("4. Both expanding and sliding windows have trade-offs")
        logger.info("\nNext Steps:")
        logger.info("- Integrate with your actual model (LateFusionModel)")
        logger.info("- Run on real stock and news data")
        logger.info("- Use results for publication-quality reporting")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
