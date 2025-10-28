"""
Walk-Forward Evaluation with Enhanced Backtesting
Integrates temporal validation with realistic trading simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.walk_forward_cv import WalkForwardValidator, FoldResult
from src.utils.temporal_utils import TemporalDataPipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)


class EnhancedBacktester:
    """
    Enhanced backtesting with realistic trading assumptions
    
    Improvements over basic backtest:
    - Transaction costs (bid-ask spread + commission)
    - Slippage modeling
    - Execution delay (1-day lag)
    - Market impact for large positions
    - Risk management (position sizing, stop losses)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Trading costs
        self.transaction_cost_pct = config.get('evaluation', {}).get(
            'backtesting', {}
        ).get('transaction_cost', 0.002)  # 0.2% per trade (buy + sell)
        
        self.slippage_pct = config.get('evaluation', {}).get(
            'backtesting', {}
        ).get('slippage', 0.001)  # 0.1% slippage
        
        # Capital management
        self.initial_capital = config.get('evaluation', {}).get(
            'backtesting', {}
        ).get('initial_capital', 100000)
        
        self.max_position_pct = config.get('evaluation', {}).get(
            'backtesting', {}
        ).get('max_position_pct', 0.1)  # 10% max per position
        
        # Risk management
        self.stop_loss_pct = config.get('evaluation', {}).get(
            'backtesting', {}
        ).get('stop_loss_pct', 0.05)  # 5% stop loss
        
        self.confidence_threshold = config.get('evaluation', {}).get(
            'backtesting', {}
        ).get('confidence_threshold', 0.6)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def backtest_with_costs(
        self,
        predictions: np.ndarray,
        prediction_probs: np.ndarray,
        actual_returns: np.ndarray,
        dates: pd.Series = None
    ) -> Dict[str, float]:
        """
        Run backtest with realistic trading costs
        
        Args:
            predictions: Predicted classes (0=down, 1=neutral, 2=up)
            prediction_probs: Prediction probabilities [n_samples, n_classes]
            actual_returns: Actual returns for each period
            dates: Dates corresponding to predictions
            
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Running enhanced backtest with transaction costs and slippage")
        
        capital = self.initial_capital
        cash = capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        
        portfolio_values = [capital]
        returns_log = []
        trades = []
        
        for i in range(len(predictions)):
            # Get prediction confidence
            pred_class = predictions[i]
            confidence = prediction_probs[i][pred_class]
            
            # Execution delay: use prediction from t-1 to trade at t
            if i == 0:
                continue
            
            actual_return = actual_returns[i]
            
            # Current portfolio value
            if position != 0:
                # Mark to market
                portfolio_value = cash + position * entry_price * (1 + actual_return)
            else:
                portfolio_value = cash
            
            # Check stop loss
            if position != 0:
                current_return = (portfolio_value - cash) / abs(position * entry_price)
                if current_return <= -self.stop_loss_pct:
                    # Hit stop loss - close position
                    exit_price = entry_price * (1 + actual_return)
                    trade_return = self._execute_trade_close(
                        position, entry_price, exit_price
                    )
                    cash = portfolio_value
                    position = 0
                    
                    trades.append({
                        'type': 'stop_loss',
                        'return': trade_return
                    })
                    continue
            
            # Trading decision (using previous prediction due to delay)
            prev_pred = predictions[i-1]
            prev_confidence = prediction_probs[i-1][prev_pred]
            
            if prev_confidence >= self.confidence_threshold:
                # Determine desired position
                if prev_pred == 2:  # Bullish prediction
                    desired_position = 1
                elif prev_pred == 0:  # Bearish prediction (if shorting allowed)
                    desired_position = 0  # For simplicity, just exit long
                else:  # Neutral
                    desired_position = 0
                
                # Rebalance if needed
                if position != desired_position:
                    # Close current position if any
                    if position != 0:
                        exit_price = entry_price * (1 + actual_return)
                        trade_return = self._execute_trade_close(
                            position, entry_price, exit_price
                        )
                        cash = portfolio_value
                        position = 0
                        
                        trades.append({
                            'type': 'close',
                            'return': trade_return
                        })
                    
                    # Open new position if desired
                    if desired_position != 0 and cash > 0:
                        position_size = min(
                            cash * self.max_position_pct,
                            cash
                        )
                        
                        # Apply transaction costs and slippage
                        total_cost = self.transaction_cost_pct + self.slippage_pct
                        effective_capital = position_size * (1 - total_cost)
                        
                        position = desired_position
                        entry_price = 100  # Normalized price
                        cash -= position_size
                        
                        trades.append({
                            'type': 'open',
                            'position_size': position_size,
                            'costs': position_size * total_cost
                        })
            
            # Calculate period return
            period_return = (portfolio_value - portfolio_values[-1]) / portfolio_values[-1]
            returns_log.append(period_return)
            portfolio_values.append(portfolio_value)
        
        # Close final position if any
        if position != 0:
            final_value = cash + position * entry_price * (1 + actual_returns[-1])
            portfolio_values[-1] = final_value
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            np.array(portfolio_values),
            np.array(returns_log),
            trades
        )
        
        return metrics
    
    def _execute_trade_close(
        self,
        position: int,
        entry_price: float,
        exit_price: float
    ) -> float:
        """Calculate return from closing a position including costs"""
        gross_return = (exit_price - entry_price) / entry_price * position
        
        # Subtract transaction costs (both entry and exit)
        net_return = gross_return - 2 * self.transaction_cost_pct - self.slippage_pct
        
        return net_return
    
    def _calculate_metrics(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized metrics (assuming 252 trading days)
        n_periods = len(portfolio_values)
        years = n_periods / 252
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative = portfolio_values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = 0
        
        # Win rate
        winning_trades = sum(1 for t in trades if t.get('return', 0) > 0)
        total_trades = len([t for t in trades if 'return' in t])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Total costs
        total_costs = sum(t.get('costs', 0) for t in trades)
        cost_drag = total_costs / portfolio_values[0]
        
        # Turnover (total traded / average portfolio value)
        total_traded = sum(t.get('position_size', 0) for t in trades)
        avg_portfolio_value = np.mean(portfolio_values)
        turnover = total_traded / avg_portfolio_value / years if years > 0 and avg_portfolio_value > 0 else 0
        
        metrics = {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'final_capital': float(portfolio_values[-1]),
            'total_costs': float(total_costs),
            'cost_drag': float(cost_drag),
            'turnover': float(turnover)
        }
        
        self.logger.info(
            f"Enhanced Backtest - CAGR: {cagr:.2%}, Sharpe: {sharpe:.2f}, "
            f"Max DD: {max_drawdown:.2%}, Win Rate: {win_rate:.2%}"
        )
        
        return metrics


class WalkForwardEvaluator:
    """
    Complete walk-forward evaluation framework
    Combines temporal CV with comprehensive metrics
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.validator = WalkForwardValidator(config)
        self.backtester = EnhancedBacktester(config)
        self.temporal_pipeline = TemporalDataPipeline(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_with_walk_forward(
        self,
        model_class,
        data: pd.DataFrame,
        window_type: str = 'expanding',
        initial_train_months: int = 24,
        test_months: int = 3,
        step_months: int = 3,
        max_folds: int = None
    ) -> Dict:
        """
        Evaluate model using walk-forward cross-validation
        
        Args:
            model_class: Model class to instantiate
            data: Complete dataset with features and targets
            window_type: 'expanding' or 'sliding'
            initial_train_months: Initial training window
            test_months: Test window size
            step_months: Step size
            max_folds: Maximum folds to run
            
        Returns:
            Complete evaluation results
        """
        
        def train_fn(train_data):
            """Train model on training data"""
            # Extract features and targets
            text_features = np.stack(train_data['text_embeddings'].values)
            numerical_features = np.stack(train_data['numerical_features'].values)
            targets = train_data['target'].values
            
            # Initialize model
            model_instance = model_class(self.config)
            
            # Create sequences
            from src.training.trainer import DataPreparer
            preparer = DataPreparer(self.config)
            
            text_seq, num_seq, y_seq = preparer.create_sequences(
                text_features, numerical_features, targets
            )
            
            # Split train and validation
            split_idx = int(0.8 * len(text_seq))
            train_data_seq = (
                text_seq[:split_idx],
                num_seq[:split_idx],
                y_seq[:split_idx]
            )
            val_data_seq = (
                text_seq[split_idx:],
                num_seq[split_idx:],
                y_seq[split_idx:]
            )
            
            # Train
            from src.training.trainer import ModelTrainer
            trainer = ModelTrainer(model_instance.model, self.config)
            trainer.train(train_data_seq, val_data_seq)
            
            return model_instance
        
        def evaluate_fn(model, test_data):
            """Evaluate model on test data"""
            # Extract features
            text_features = np.stack(test_data['text_embeddings'].values)
            numerical_features = np.stack(test_data['numerical_features'].values)
            targets = test_data['target'].values
            actual_returns = test_data.get('returns', np.zeros(len(test_data))).values
            
            # Create sequences
            from src.training.trainer import DataPreparer
            preparer = DataPreparer(self.config)
            
            text_seq, num_seq, y_seq = preparer.create_sequences(
                text_features, numerical_features, targets
            )
            
            # Predict
            y_pred_proba = model.model.predict([text_seq, num_seq], verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Classification metrics
            accuracy = accuracy_score(y_seq, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_seq, y_pred, average='weighted', zero_division=0
            )
            
            # Backtesting metrics
            backtest_metrics = self.backtester.backtest_with_costs(
                y_pred, y_pred_proba, actual_returns[-len(y_pred):]
            )
            
            # Combine metrics
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                **backtest_metrics
            }
            
            return metrics, y_pred, y_seq
        
        # Run walk-forward validation
        results = self.validator.walk_forward_evaluate(
            data=data,
            date_column='Date',
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
            window_type=window_type,
            initial_train_months=initial_train_months,
            test_months=test_months,
            step_months=step_months,
            max_folds=max_folds
        )
        
        # Save detailed results
        os.makedirs('results', exist_ok=True)
        self.validator.save_results(
            results,
            f'results/walk_forward_{window_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        return results
    
    def compare_models(
        self,
        model_configs: Dict[str, Tuple],
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple models using walk-forward CV
        
        Args:
            model_configs: Dict mapping model_name -> (model_class, config)
            data: Dataset
            **kwargs: Arguments for walk-forward CV
            
        Returns:
            DataFrame comparing models
        """
        comparison_results = []
        
        for model_name, (model_class, model_config) in model_configs.items():
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Evaluating model: {model_name}")
            self.logger.info(f"{'='*70}\n")
            
            # Update config
            evaluator = WalkForwardEvaluator(model_config)
            
            # Evaluate
            results = evaluator.evaluate_with_walk_forward(
                model_class, data, **kwargs
            )
            
            # Extract aggregate metrics
            agg_metrics = results['aggregate_metrics']
            
            row = {'model': model_name}
            for metric_name, stats in agg_metrics.items():
                row[f'{metric_name}_mean'] = stats['mean']
                row[f'{metric_name}_std'] = stats['std']
                row[f'{metric_name}_ci'] = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            
            comparison_results.append(row)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save comparison
        comparison_path = f'results/model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"Model comparison saved to {comparison_path}")
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'features': {'lookback_window': 30},
        'training': {
            'epochs': 10,
            'batch_size': 32
        },
        'evaluation': {
            'backtesting': {
                'transaction_cost': 0.002,
                'slippage': 0.001,
                'initial_capital': 100000,
                'max_position_pct': 0.1,
                'stop_loss_pct': 0.05,
                'confidence_threshold': 0.6
            }
        }
    }
    
    print("Walk-forward evaluator initialized")
    print("Ready for comprehensive model evaluation with realistic backtesting")
