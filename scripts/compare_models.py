"""
Model Comparison Framework
Compares baseline vs enhanced models to demonstrate accuracy improvement
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.models.advanced_models import create_model
from src.training.enhanced_trainer import EnhancedTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare baseline and enhanced models
    """
    
    def __init__(self, output_dir: str = 'results/comparison'):
        """
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def train_baseline_sklearn(
        self,
        X_text_train: Optional[np.ndarray],
        X_num_train: np.ndarray,
        y_train: np.ndarray,
        X_text_val: Optional[np.ndarray],
        X_num_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Train baseline sklearn model
        
        Args:
            X_text_train: Training text embeddings [n_samples, seq_len, 768] (not used by sklearn)
            X_num_train: Training numerical features [n_samples, seq_len, num_features]
            y_train: Training labels
            X_text_val: Validation text embeddings (not used by sklearn)
            X_num_val: Validation numerical features
            y_val: Validation labels
            model_type: 'random_forest' or 'logistic_regression'
        
        Returns:
            metrics: Evaluation metrics
        """
        logger.info(f"\nTraining baseline {model_type}...")
        
        # Flatten numerical sequences to 2D for sklearn (ignore text)
        X_train_flat = X_num_train.reshape(X_num_train.shape[0], -1)
        X_val_flat = X_num_val.reshape(X_num_val.shape[0], -1)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X_train_flat, y_train)
        
        # Predict
        y_pred = model.predict(X_val_flat)
        y_proba = model.predict_proba(X_val_flat)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_proba)
        metrics['model_type'] = model_type
        
        logger.info(f"{model_type} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def train_baseline_lstm(
        self,
        X_text_train: Optional[np.ndarray],
        X_num_train: np.ndarray,
        y_train: np.ndarray,
        X_text_val: Optional[np.ndarray],
        X_num_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30
    ) -> Dict:
        """
        Train baseline LSTM model
        
        Args:
            X_text_train: Training text embeddings (not used for baseline)
            X_num_train: Training numerical features
            y_train: Training labels
            X_text_val: Validation text embeddings (not used for baseline)
            X_num_val: Validation numerical features
            y_val: Validation labels
            epochs: Number of training epochs
        
        Returns:
            metrics: Evaluation metrics
        """
        logger.info("\nTraining baseline LSTM...")
        print("Note: FinBERT disabled for baseline (use numerical features only)")
        
        num_features = X_num_train.shape[2]
        
        # Create model (no FinBERT)
        model = create_model(
            model_type='baseline',
            num_features=num_features,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        )
        
        # Create trainer
        trainer = EnhancedTrainer(
            model,
            device=self.device,
            output_dir=str(self.output_dir / 'baseline_lstm')
        )
        
        # Create data loaders (numerical only, no text)
        train_loader, val_loader = trainer.create_dataloaders(
            X_num_train=X_num_train,
            y_train=y_train,
            X_num_val=X_num_val,
            y_val=y_val,
            X_text_train=None,  # No text for baseline
            X_text_val=None,
            batch_size=32
        )
        
        # Train
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=0.001,
            early_stopping_patience=10,
            log_interval=5
        )
        
        # Evaluate
        metrics = trainer.evaluate(val_loader)
        metrics['model_type'] = 'baseline_lstm'
        metrics['history'] = history
        
        logger.info(f"Baseline LSTM - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def train_enhanced_model(
        self,
        model_type: str,
        X_text_train: Optional[np.ndarray],
        X_num_train: np.ndarray,
        y_train: np.ndarray,
        X_text_val: Optional[np.ndarray],
        X_num_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        use_finbert: bool = True,
        **model_kwargs
    ) -> Dict:
        """
        Train enhanced model
        
        Args:
            model_type: 'bilstm', 'transformer', or 'hybrid'
            X_text_train: Training text embeddings
            X_num_train: Training numerical features
            y_train: Training labels
            X_text_val: Validation text embeddings
            X_num_val: Validation numerical features
            y_val: Validation labels
            epochs: Number of training epochs
            use_finbert: Whether to use FinBERT embeddings
            **model_kwargs: Additional model arguments
        
        Returns:
            metrics: Evaluation metrics
        """
        logger.info(f"\nTraining enhanced {model_type} model...")
        
        # If text embeddings not available, disable FinBERT
        if X_text_train is None:
            use_finbert = False
            logger.info(f"FinBERT disabled for {model_type} (no text embeddings available)")
        
        num_features = X_num_train.shape[2]
        
        # Default hyperparameters (can be overridden)
        default_params = {
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.3,
            'use_finbert': use_finbert
        }
        
        if model_type == 'transformer':
            default_params.update({
                'd_model': 512,
                'nhead': 8,
                'num_layers': 4,
                'dim_feedforward': 2048
            })
        elif model_type == 'hybrid':
            default_params.update({
                'num_heads': 8,
                'temporal_decay_init': 0.95
            })
        
        # Merge with user params
        model_params = {**default_params, **model_kwargs}
        
        # Print FinBERT status
        if use_finbert and model_type == 'hybrid':
            print(f"Loading FinBERT for HybridModel...")
        else:
            print(f"FinBERT disabled for {model_type}Model (use_finbert=False)")
        
        # Create model
        model = create_model(
            model_type=model_type,
            num_features=num_features,
            **model_params
        )
        
        # Create trainer
        trainer = EnhancedTrainer(
            model,
            device=self.device,
            output_dir=str(self.output_dir / f'enhanced_{model_type}')
        )
        
        # Create data loaders with both text and numerical features
        train_loader, val_loader = trainer.create_dataloaders(
            X_num_train=X_num_train,
            y_train=y_train,
            X_num_val=X_num_val,
            y_val=y_val,
            X_text_train=X_text_train if use_finbert else None,
            X_text_val=X_text_val if use_finbert else None,
            batch_size=32
        )
        
        # Train
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=0.0005,
            scheduler_type='plateau',
            early_stopping_patience=15,
            log_interval=5
        )
        
        # Evaluate
        metrics = trainer.evaluate(val_loader)
        metrics['model_type'] = f'enhanced_{model_type}'
        metrics['history'] = history
        
        logger.info(f"Enhanced {model_type} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            roc_auc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
    
    def run_full_comparison(
        self,
        X_text_train: Optional[np.ndarray],
        X_num_train: np.ndarray,
        y_train: np.ndarray,
        X_text_val: Optional[np.ndarray],
        X_num_val: np.ndarray,
        y_val: np.ndarray,
        baseline_epochs: int = 30,
        enhanced_epochs: int = 50,
        use_sklearn_baselines: bool = True,
        enhanced_models: List[str] = ['bilstm', 'transformer', 'hybrid']
    ):
        """
        Run full comparison of all models
        
        Args:
            X_text_train: Training text embeddings [n_samples, seq_len, 768]
            X_num_train: Training numerical features [n_samples, seq_len, num_features]
            y_train: Training labels
            X_text_val: Validation text embeddings
            X_num_val: Validation numerical features
            y_val: Validation labels
            baseline_epochs: Epochs for baseline models
            enhanced_epochs: Epochs for enhanced models
            use_sklearn_baselines: Whether to include sklearn baselines
            enhanced_models: List of enhanced models to train
        """
        logger.info(f"\n{'='*80}")
        logger.info("RUNNING FULL MODEL COMPARISON")
        logger.info(f"{'='*80}\n")
        
        # Train baseline models
        if use_sklearn_baselines:
            self.results['random_forest'] = self.train_baseline_sklearn(
                X_text_train, X_num_train, y_train,
                X_text_val, X_num_val, y_val,
                'random_forest'
            )
            
            self.results['logistic_regression'] = self.train_baseline_sklearn(
                X_text_train, X_num_train, y_train,
                X_text_val, X_num_val, y_val,
                'logistic_regression'
            )
        
        # Train baseline LSTM
        self.results['baseline_lstm'] = self.train_baseline_lstm(
            X_text_train, X_num_train, y_train,
            X_text_val, X_num_val, y_val,
            epochs=baseline_epochs
        )
        
        # Train enhanced models
        for model_type in enhanced_models:
            # Use FinBERT only for hybrid model if text embeddings available
            use_finbert = (model_type == 'hybrid' and X_text_train is not None)
            
            self.results[f'enhanced_{model_type}'] = self.train_enhanced_model(
                model_type,
                X_text_train, X_num_train, y_train,
                X_text_val, X_num_val, y_val,
                epochs=enhanced_epochs,
                use_finbert=use_finbert
            )
        
        # Generate comparison report
        self.generate_comparison_report()
        
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON COMPLETE")
        logger.info(f"{'='*80}\n")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        logger.info("\nGenerating comparison report...")
        
        # Create summary table
        summary_data = []
        for model_name, metrics in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', 0.0)
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info(f"{'='*80}\n")
        logger.info(df.to_string(index=False))
        logger.info(f"\n{'='*80}\n")
        
        # Calculate improvement
        if 'baseline_lstm' in self.results:
            baseline_acc = self.results['baseline_lstm']['accuracy']
            best_model = df.iloc[0]
            best_acc = best_model['Accuracy']
            improvement = ((best_acc - baseline_acc) / baseline_acc) * 100
            
            logger.info(f"BEST MODEL: {best_model['Model']}")
            logger.info(f"Baseline Accuracy: {baseline_acc:.4f}")
            logger.info(f"Best Accuracy: {best_acc:.4f}")
            logger.info(f"Improvement: {improvement:.2f}%")
            logger.info(f"Absolute Improvement: {(best_acc - baseline_acc)*100:.2f} percentage points\n")
        
        # Save to CSV
        csv_path = self.output_dir / 'comparison_summary.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary saved to: {csv_path}")
        
        # Save full results
        results_path = self.output_dir / 'full_results.json'
        # Remove non-serializable items
        save_results = {}
        for model, metrics in self.results.items():
            save_results[model] = {k: v for k, v in metrics.items() 
                                    if k not in ['predictions', 'labels', 'probabilities', 'history']}
        
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"Full results saved to: {results_path}")
        
        # Generate plots
        self._plot_accuracy_comparison(df)
        self._plot_metrics_comparison(df)
        self._plot_confusion_matrices()
        
        if any('history' in m for m in self.results.values()):
            self._plot_training_comparison()
    
    def _plot_accuracy_comparison(self, df: pd.DataFrame):
        """Plot accuracy comparison bar chart"""
        plt.figure(figsize=(12, 6))
        
        colors = ['red' if 'baseline' in m or m in ['random_forest', 'logistic_regression'] 
                  else 'green' for m in df['Model']]
        
        bars = plt.bar(range(len(df)), df['Accuracy'], color=colors, alpha=0.7)
        plt.xticks(range(len(df)), df['Model'], rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        plot_path = self.output_dir / 'accuracy_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Accuracy plot saved to: {plot_path}")
    
    def _plot_metrics_comparison(self, df: pd.DataFrame):
        """Plot all metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            colors = ['red' if 'baseline' in m or m in ['random_forest', 'logistic_regression'] 
                      else 'green' for m in df['Model']]
            
            bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.7)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['Model'], rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'all_metrics_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics plot saved to: {plot_path}")
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.3f}')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'confusion_matrices.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to: {plot_path}")
    
    def _plot_training_comparison(self):
        """Plot training curves comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for model_name, metrics in self.results.items():
            if 'history' not in metrics:
                continue
            
            history = metrics['history']
            label = model_name
            
            # Validation accuracy
            axes[0].plot(history['val_acc'], label=label, linewidth=2)
            
            # Validation loss
            axes[1].plot(history['val_loss'], label=label, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Validation Accuracy Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'training_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to: {plot_path}")


def quick_comparison(
    X_text_train: Optional[np.ndarray],
    X_num_train: np.ndarray,
    y_train: np.ndarray,
    X_text_val: Optional[np.ndarray],
    X_num_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = 'results/quick_comparison'
):
    """
    Quick comparison with shorter training
    
    Args:
        X_text_train: Training text embeddings [n_samples, seq_len, 768]
        X_num_train: Training numerical features [n_samples, seq_len, num_features]
        y_train: Training labels
        X_text_val: Validation text embeddings
        X_num_val: Validation numerical features
        y_val: Validation labels
        output_dir: Output directory
    """
    comparator = ModelComparator(output_dir)
    
    comparator.run_full_comparison(
        X_text_train, X_num_train, y_train,
        X_text_val, X_num_val, y_val,
        baseline_epochs=10,
        enhanced_epochs=15,
        use_sklearn_baselines=True,
        enhanced_models=['bilstm', 'hybrid']  # Skip transformer for speed
    )
    
    return comparator.results


if __name__ == "__main__":
    # Test comparison framework
    print("Testing Model Comparison Framework...")
    
    # Create dummy data
    n_train = 500
    n_val = 200
    seq_len = 60
    num_features = 50
    
    X_num_train = np.random.randn(n_train, seq_len, num_features).astype(np.float32)
    X_text_train = None  # No text embeddings for test
    y_train = np.random.randint(0, 2, n_train)
    X_num_val = np.random.randn(n_val, seq_len, num_features).astype(np.float32)
    X_text_val = None
    y_val = np.random.randint(0, 2, n_val)
    
    # Run quick comparison
    print("\nRunning quick comparison (short epochs for testing)...")
    results = quick_comparison(
        X_text_train, X_num_train, y_train,
        X_text_val, X_num_val, y_val,
        output_dir='results/test_comparison'
    )
    
    print("\n✅ Model comparison framework working correctly!")
    print(f"Results saved to: results/test_comparison/")
