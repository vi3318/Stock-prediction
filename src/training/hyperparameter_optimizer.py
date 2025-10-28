"""
Hyperparameter Optimization using Optuna
Automated tuning for all model hyperparameters
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Callable
import logging
from pathlib import Path
import json

from src.models.advanced_models import create_model


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimization
    """
    
    def __init__(
        self,
        model_type: str,
        num_features: int,
        device: str = 'cpu',
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """
        Args:
            model_type: 'baseline', 'bilstm', 'transformer', 'hybrid'
            num_features: Number of input features
            device: 'cpu' or 'cuda'
            study_name: Name for Optuna study (for resume)
            storage: SQLite database path for persistence
        """
        self.model_type = model_type
        self.num_features = num_features
        self.device = device
        self.study_name = study_name or f"{model_type}_optimization"
        self.storage = storage
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Training data (set via set_data)
        self.train_loader = None
        self.val_loader = None
        
        # Best results
        self.best_params = None
        self.best_accuracy = 0.0
    
    def set_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32
    ):
        """
        Set training and validation data
        
        Args:
            X_train: Training features [n_samples, seq_len, num_features]
            y_train: Training labels [n_samples]
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size (can be overridden in optimization)
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        # Create loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.logger.info(f"Data set: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Define hyperparameter search space
        
        Args:
            trial: Optuna trial object
        
        Returns:
            config: Dictionary of hyperparameters
        """
        config = {}
        
        # Learning rate (log scale)
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Architecture hyperparameters
        if self.model_type in ['bilstm', 'hybrid']:
            config['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            config['num_layers'] = trial.suggest_int('num_layers', 2, 4)
            config['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16])
        elif self.model_type == 'transformer':
            config['d_model'] = trial.suggest_categorical('d_model', [256, 512, 768])
            config['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
            config['num_layers'] = trial.suggest_int('num_layers', 3, 6)
            config['dim_feedforward'] = trial.suggest_categorical('dim_feedforward', [1024, 2048, 4096])
        else:  # baseline
            config['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            config['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        
        # Regularization
        config['dropout'] = trial.suggest_float('dropout', 0.2, 0.5)
        config['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Training hyperparameters
        config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Gradient clipping
        config['grad_clip'] = trial.suggest_float('grad_clip', 0.5, 5.0)
        
        # Model-specific parameters
        if self.model_type == 'hybrid':
            config['temporal_decay_init'] = trial.suggest_float('temporal_decay_init', 0.85, 0.99)
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna
        
        Args:
            trial: Optuna trial
        
        Returns:
            accuracy: Validation accuracy
        """
        # Get hyperparameters
        config = self.define_search_space(trial)
        
        # Update batch size if needed
        if config['batch_size'] != self.train_loader.batch_size:
            train_dataset = self.train_loader.dataset
            val_dataset = self.val_loader.dataset
            self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Create model
        model_config = {k: v for k, v in config.items() 
                        if k not in ['learning_rate', 'batch_size', 'weight_decay', 'grad_clip']}
        model_config['use_finbert'] = False  # Disable for faster optimization
        
        model = create_model(self.model_type, self.num_features, **model_config)
        model = model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop (limited epochs for optimization speed)
        max_epochs = 20
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.model_type == 'hybrid':
                    outputs, _ = model(batch_X)
                else:
                    outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if self.model_type == 'hybrid':
                        outputs, _ = model(batch_X)
                    else:
                        outputs = model(batch_X)
                    
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            val_acc = val_correct / val_total
            
            # Report intermediate value
            trial.report(val_acc, epoch)
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_acc
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True
    ) -> Dict:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            timeout: Time limit in seconds
            n_jobs: Number of parallel jobs (1 = sequential)
            show_progress_bar: Show progress bar
        
        Returns:
            best_params: Dictionary of best hyperparameters
        """
        self.logger.info(f"Starting optimization for {self.model_type} model")
        self.logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )
        
        # Get best results
        self.best_params = study.best_params
        self.best_accuracy = study.best_value
        
        self.logger.info(f"\nOptimization complete!")
        self.logger.info(f"Best accuracy: {self.best_accuracy:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        # Save results
        self.save_results(study)
        
        return self.best_params
    
    def save_results(self, study: optuna.Study, output_dir: str = 'results/optimization'):
        """
        Save optimization results
        
        Args:
            study: Optuna study object
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        results = {
            'model_type': self.model_type,
            'best_accuracy': self.best_accuracy,
            'best_params': self.best_params,
            'n_trials': len(study.trials),
            'study_name': self.study_name
        }
        
        results_file = output_path / f'{self.model_type}_best_params.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save all trials
        trials_df = study.trials_dataframe()
        trials_file = output_path / f'{self.model_type}_all_trials.csv'
        trials_df.to_csv(trials_file, index=False)
        
        self.logger.info(f"All trials saved to {trials_file}")
        
        # Visualization (if plotly available)
        try:
            import plotly
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice
            )
            
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_html(output_path / f'{self.model_type}_optimization_history.html')
            
            # Parameter importances
            fig = plot_param_importances(study)
            fig.write_html(output_path / f'{self.model_type}_param_importances.html')
            
            # Slice plot
            fig = plot_slice(study)
            fig.write_html(output_path / f'{self.model_type}_param_slice.html')
            
            self.logger.info(f"Visualizations saved to {output_path}")
        except ImportError:
            self.logger.warning("Plotly not available, skipping visualizations")
    
    def load_best_params(self, filepath: str) -> Dict:
        """
        Load best parameters from file
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            best_params: Dictionary of parameters
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_params = results['best_params']
        self.best_accuracy = results['best_accuracy']
        
        self.logger.info(f"Loaded best params from {filepath}")
        self.logger.info(f"Best accuracy: {self.best_accuracy:.4f}")
        
        return self.best_params


def quick_optimize(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    device: str = 'cpu'
) -> Dict:
    """
    Quick optimization helper function
    
    Args:
        model_type: 'baseline', 'bilstm', 'transformer', 'hybrid'
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials
        device: 'cpu' or 'cuda'
    
    Returns:
        best_params: Dictionary of best hyperparameters
    """
    num_features = X_train.shape[2]
    
    optimizer = HyperparameterOptimizer(
        model_type=model_type,
        num_features=num_features,
        device=device
    )
    
    optimizer.set_data(X_train, y_train, X_val, y_val)
    
    best_params = optimizer.optimize(n_trials=n_trials)
    
    return best_params


if __name__ == "__main__":
    # Test optimization with dummy data
    print("Testing Hyperparameter Optimization...")
    
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    n_train = 500
    n_val = 100
    seq_len = 60
    num_features = 50
    
    X_train = np.random.randn(n_train, seq_len, num_features).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train)
    X_val = np.random.randn(n_val, seq_len, num_features).astype(np.float32)
    y_val = np.random.randint(0, 2, n_val)
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    
    # Test with baseline model (faster)
    print("\nRunning optimization (5 trials for demo)...")
    best_params = quick_optimize(
        model_type='baseline',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=5,  # Small for testing
        device='cpu'
    )
    
    print("\n✅ Optimization complete!")
    print(f"Best parameters: {best_params}")
