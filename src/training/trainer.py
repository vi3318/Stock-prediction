"""
Training Pipeline with Evaluation Metrics
Handles model training, validation, and comprehensive evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error
)
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, List, Optional
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DataPreparer:
    """Prepares data for model training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_window = config.get('features', {}).get('lookback_window', 30)
        self.validation_split = config.get('training', {}).get('validation_split', 0.2)
        self.test_split = config.get('training', {}).get('test_split', 0.1)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.text_scaler = None
        self.numerical_scaler = StandardScaler()
    
    def create_sequences(
        self,
        text_features: np.ndarray,
        numerical_features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction
        
        Args:
            text_features: Text embeddings (num_samples, embedding_dim)
            numerical_features: Numerical features (num_samples, num_features)
            targets: Target variables (num_samples,)
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (text_sequences, numerical_sequences, sequence_targets)
        """
        text_sequences = []
        numerical_sequences = []
        sequence_targets = []
        
        for i in range(sequence_length, len(targets)):
            text_sequences.append(text_features[i-sequence_length:i])
            numerical_sequences.append(numerical_features[i-sequence_length:i])
            sequence_targets.append(targets[i])
        
        return (
            np.array(text_sequences),
            np.array(numerical_sequences),
            np.array(sequence_targets)
        )
    
    def prepare_data(
        self,
        text_features: np.ndarray,
        numerical_features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple:
        """
        Prepare and split data for training
        
        Args:
            text_features: Text embeddings
            numerical_features: Numerical features
            targets: Target variables
            
        Returns:
            Tuple of train, validation, and test sets
        """
        self.logger.info("Preparing data for training")
        
        # Create sequences
        text_seq, num_seq, target_seq = self.create_sequences(
            text_features,
            numerical_features,
            targets,
            self.lookback_window
        )
        
        self.logger.info(f"Created {len(target_seq)} sequences")
        
        # Split into train, val, test
        # First split off test set
        train_val_size = 1.0 - self.test_split
        
        (text_train_val, text_test,
         num_train_val, num_test,
         y_train_val, y_test) = train_test_split(
            text_seq, num_seq, target_seq,
            test_size=self.test_split,
            shuffle=False  # Important for time series
        )
        
        # Then split train into train and validation
        val_size_adjusted = self.validation_split / train_val_size
        
        (text_train, text_val,
         num_train, num_val,
         y_train, y_val) = train_test_split(
            text_train_val, num_train_val, y_train_val,
            test_size=val_size_adjusted,
            shuffle=False
        )
        
        # Scale numerical features
        num_train_shape = num_train.shape
        num_train_2d = num_train.reshape(-1, num_train.shape[-1])
        num_train_2d = self.numerical_scaler.fit_transform(num_train_2d)
        num_train = num_train_2d.reshape(num_train_shape)
        
        num_val_2d = num_val.reshape(-1, num_val.shape[-1])
        num_val_2d = self.numerical_scaler.transform(num_val_2d)
        num_val = num_val_2d.reshape(num_val.shape)
        
        num_test_2d = num_test.reshape(-1, num_test.shape[-1])
        num_test_2d = self.numerical_scaler.transform(num_test_2d)
        num_test = num_test_2d.reshape(num_test.shape)
        
        self.logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        
        return (
            (text_train, num_train, y_train),
            (text_val, num_val, y_val),
            (text_test, num_test, y_test)
        )


class ModelTrainer:
    """Handles model training"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.history = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train(
        self,
        train_data: Tuple,
        val_data: Tuple,
        callbacks: List
    ) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            train_data: (text_train, num_train, y_train)
            val_data: (text_val, num_val, y_val)
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        text_train, num_train, y_train = train_data
        text_val, num_val, y_val = val_data
        
        epochs = self.config.get('training', {}).get('epochs', 100)
        batch_size = self.config.get('training', {}).get('batch_size', 64)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Calculate class weights if classification
        class_weight = None
        if self.config.get('model', {}).get('output', {}).get('task') == 'classification':
            if self.config.get('training', {}).get('class_weights') == 'balanced':
                class_weight = self._calculate_class_weights(y_train)
        
        # Train model
        self.history = self.model.fit(
            [text_train, num_train],
            y_train,
            validation_data=([text_val, num_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        self.logger.info("Training completed")
        return self.history
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict:
        """Calculate class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def save_history(self, path: str):
        """Save training history"""
        if self.history:
            history_dict = self.history.history
            with open(path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            self.logger.info(f"Training history saved to {path}")


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.task = config.get('model', {}).get('output', {}).get('task', 'classification')
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        self.logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        self.logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        return metrics
    
    def evaluate(
        self,
        model,
        test_data: Tuple
    ) -> Dict:
        """
        Complete evaluation
        
        Args:
            model: Trained model
            test_data: (text_test, num_test, y_test)
            
        Returns:
            Dictionary of all metrics
        """
        text_test, num_test, y_test = test_data
        
        self.logger.info("Evaluating model on test set")
        
        # Get predictions
        y_pred_raw = model.predict([text_test, num_test])
        
        if self.task == 'classification':
            y_pred = np.argmax(y_pred_raw, axis=1)
            metrics = self.evaluate_classification(y_test, y_pred, y_pred_raw)
        else:
            y_pred = y_pred_raw.flatten()
            metrics = self.evaluate_regression(y_test, y_pred)
        
        return metrics


class BacktestingEngine:
    """Backtesting for trading simulation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('evaluation', {}).get('backtesting', {}).get('initial_capital', 100000)
        self.transaction_cost = config.get('evaluation', {}).get('backtesting', {}).get('transaction_cost', 0.001)
        self.confidence_threshold = config.get('evaluation', {}).get('backtesting', {}).get('confidence_threshold', 0.6)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def backtest_classification(
        self,
        predictions: np.ndarray,
        prediction_proba: np.ndarray,
        actual_returns: np.ndarray
    ) -> Dict:
        """
        Backtest classification predictions
        
        Args:
            predictions: Predicted classes (0: down, 1: neutral, 2: up)
            prediction_proba: Prediction probabilities
            actual_returns: Actual returns
            
        Returns:
            Dictionary of trading metrics
        """
        capital = self.initial_capital
        positions = []
        returns = []
        
        for i in range(len(predictions)):
            # Get prediction confidence
            confidence = np.max(prediction_proba[i])
            
            # Only trade if confidence is high enough
            if confidence < self.confidence_threshold:
                positions.append(0)
                returns.append(0)
                continue
            
            pred = predictions[i]
            
            # Trading logic
            if pred == 2:  # Up prediction - buy
                position = 1
            elif pred == 0:  # Down prediction - short (or stay out)
                position = -1
            else:  # Neutral - no position
                position = 0
            
            # Calculate return
            trade_return = position * actual_returns[i]
            
            # Apply transaction cost if trading
            if position != 0:
                trade_return -= self.transaction_cost
            
            returns.append(trade_return)
            positions.append(position)
            capital *= (1 + trade_return)
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = np.sum(returns > 0)
        total_trades = np.sum(np.array(positions) != 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        metrics = {
            'total_return': float(total_return),
            'final_capital': float(capital),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades)
        }
        
        self.logger.info(f"Backtest - Return: {total_return:.2%}, Sharpe: {sharpe:.2f}, Win Rate: {win_rate:.2%}")
        
        return metrics


class TrainingPipeline:
    """Complete training and evaluation pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.data_preparer = DataPreparer(config)
        self.evaluator = ModelEvaluator(config)
        self.backtester = BacktestingEngine(config)
    
    def run(
        self,
        model,
        text_features: np.ndarray,
        numerical_features: np.ndarray,
        targets: np.ndarray,
        actual_returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run complete training and evaluation pipeline
        
        Args:
            model: Model instance
            text_features: Text embeddings
            numerical_features: Numerical features
            targets: Target variables
            actual_returns: Actual returns for backtesting
            
        Returns:
            Dictionary of all results
        """
        self.logger.info("Starting training pipeline")
        
        # Prepare data
        train_data, val_data, test_data = self.data_preparer.prepare_data(
            text_features, numerical_features, targets
        )
        
        # Get callbacks
        callbacks = model.get_callbacks()
        
        # Train model
        trainer = ModelTrainer(model.model, self.config)
        history = trainer.train(train_data, val_data, callbacks)
        
        # Save history
        os.makedirs('results', exist_ok=True)
        trainer.save_history('results/training_history.json')
        
        # Evaluate
        metrics = self.evaluator.evaluate(model.model, test_data)
        
        # Backtesting (if returns provided)
        if actual_returns is not None and self.config.get('model', {}).get('output', {}).get('task') == 'classification':
            text_test, num_test, y_test = test_data
            y_pred_proba = model.model.predict([text_test, num_test])
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Align returns with test data
            test_returns = actual_returns[-len(y_test):]
            
            backtest_metrics = self.backtester.backtest_classification(
                y_pred, y_pred_proba, test_returns
            )
            metrics['backtesting'] = backtest_metrics
        
        # Save metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = f'results/evaluation_metrics_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Results saved to {metrics_path}")
        
        return metrics


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'features': {'lookback_window': 30},
        'training': {
            'epochs': 50,
            'batch_size': 64,
            'validation_split': 0.2,
            'test_split': 0.1
        },
        'model': {
            'output': {'task': 'classification', 'classes': 3}
        },
        'evaluation': {
            'backtesting': {
                'initial_capital': 100000,
                'transaction_cost': 0.001,
                'confidence_threshold': 0.6
            }
        }
    }
    
    print("Training pipeline initialized")
