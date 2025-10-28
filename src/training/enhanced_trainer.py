"""
Enhanced Training Pipeline
Supports longer sequences, advanced optimization, and comprehensive logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class EnhancedTrainer:
    """
    Enhanced training pipeline with advanced features
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        output_dir: str = 'models/saved'
    ):
        """
        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
            output_dir: Directory to save models and logs
        """
        # Initialize logger first
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Handle CUDA device initialization more gracefully
        if device == 'cuda':
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = 'cpu'
            else:
                try:
                    # Test CUDA device
                    torch.cuda.init()
                    torch.cuda.empty_cache()
                    self.logger.info(f"CUDA initialized successfully. Device: {torch.cuda.get_device_name()}")
                except Exception as e:
                    self.logger.error(f"CUDA initialization failed: {e}. Falling back to CPU.")
                    device = 'cpu'
        
        self.device = device
        
        # Move model to device with error handling
        try:
            self.model = model.to(device)
            self.logger.info(f"Model moved to device: {device}")
        except RuntimeError as e:
            if 'CachingAllocator' in str(e):
                self.logger.error(f"CUDA memory allocator error: {e}")
                self.logger.error("This may be caused by incompatible CUDA environment variables.")
                self.logger.error("Try unsetting PYTORCH_CUDA_ALLOC_CONF or CUDA-related environment variables.")
                raise
            else:
                raise
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders
        
        Args:
            X_train: Training features [n_samples, seq_len, num_features]
            y_train: Training labels [n_samples]
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            num_workers: Number of workers for data loading
        
        Returns:
            train_loader, val_loader
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
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        self.logger.info(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_type: str = 'plateau',  # 'plateau', 'cosine', or None
        early_stopping_patience: int = 15,
        grad_clip: float = 1.0,
        warmup_epochs: int = 5,
        save_best_only: bool = True,
        log_interval: int = 10
    ) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            scheduler_type: LR scheduler ('plateau', 'cosine', None)
            early_stopping_patience: Patience for early stopping
            grad_clip: Gradient clipping threshold
            warmup_epochs: Linear warmup epochs
            save_best_only: Only save best model
            log_interval: Logging frequency
        
        Returns:
            history: Training history dictionary
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info("STARTING TRAINING")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Scheduler: {scheduler_type}")
        self.logger.info(f"Early stopping patience: {early_stopping_patience}")
        self.logger.info(f"{'='*60}\n")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=learning_rate * 0.01
            )
        else:
            scheduler = None
        
        # Loss function with class weighting for imbalanced data
        # Calculate class weights from training data
        train_labels = []
        for _, batch_y in train_loader:
            train_labels.extend(batch_y.cpu().numpy())
        train_labels = np.array(train_labels)
        
        # Count classes
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        
        # Calculate weights: higher weight for minority class
        class_weights = torch.FloatTensor([
            total_samples / (2 * class_counts[0]),
            total_samples / (2 * class_counts[1])
        ]).to(self.device)
        
        self.logger.info(f"Class distribution: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        self.logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Warmup learning rate
            if epoch < warmup_epochs and scheduler_type is not None:
                warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Training phase
            train_metrics = self._train_epoch(
                train_loader,
                optimizer,
                criterion,
                grad_clip
            )
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Val Acc: {val_metrics['accuracy']:.4f} "
                    f"Val F1: {val_metrics['f1']:.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Learning rate scheduling
            if scheduler is not None and epoch >= warmup_epochs:
                if scheduler_type == 'plateau':
                    scheduler.step(val_metrics['accuracy'])
                else:
                    scheduler.step()
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.best_val_acc = best_val_acc
                patience_counter = 0
                
                if save_best_only:
                    self._save_checkpoint(epoch, optimizer, best_val_acc, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
                break
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("TRAINING COMPLETE")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        self.logger.info(f"{'='*60}\n")
        
        # Save final history
        self._save_history()
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        grad_clip: float
    ) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Handle different model outputs
            outputs = self.model(batch_X)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract just the logits
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Extract just the logits
                
                loss = criterion(outputs, batch_y)
                
                # Metrics
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on test set
        
        Args:
            test_loader: Test data loader
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.logger.info("Running comprehensive evaluation...")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Extract just the logits
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except:
            roc_auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        # Print metrics
        self.logger.info(f"\n{'='*60}")
        self.logger.info("EVALUATION RESULTS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Accuracy:  {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall:    {recall:.4f}")
        self.logger.info(f"F1 Score:  {f1:.4f}")
        self.logger.info(f"ROC-AUC:   {roc_auc:.4f}")
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
        self.logger.info(f"FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
        self.logger.info(f"{'='*60}\n")
        
        # Save metrics
        self._save_metrics(metrics)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        val_acc: float,
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.output_dir / 'best_model.pt'
            self.best_model_path = path
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def _save_history(self):
        """Save training history"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")
        
        # Plot training curves
        self._plot_training_curves()
    
    def _save_metrics(self, metrics: Dict):
        """Save evaluation metrics"""
        # Remove non-serializable items
        save_metrics = {k: v for k, v in metrics.items() 
                        if k not in ['predictions', 'labels', 'probabilities']}
        
        metrics_path = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)
        
        self.logger.info(f"Evaluation metrics saved: {metrics_path}")
    
    def _plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_path}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plot_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved: {plot_path}")
    
    def load_best_model(self):
        """Load best saved model"""
        if self.best_model_path is None or not self.best_model_path.exists():
            self.logger.warning("No best model found")
            return
        
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Best model loaded from {self.best_model_path}")
        self.logger.info(f"Validation accuracy: {checkpoint['val_acc']:.4f}")


if __name__ == "__main__":
    # Test trainer
    print("Testing Enhanced Trainer...")
    
    logging.basicConfig(level=logging.INFO)
    
    from src.models.advanced_models import SimpleBaselineModel
    
    # Create dummy data
    n_train = 500
    n_val = 100
    seq_len = 60
    num_features = 50
    
    X_train = np.random.randn(n_train, seq_len, num_features).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train)
    X_val = np.random.randn(n_val, seq_len, num_features).astype(np.float32)
    y_val = np.random.randint(0, 2, n_val)
    
    # Create model
    model = SimpleBaselineModel(num_features)
    
    # Create trainer
    trainer = EnhancedTrainer(model, device='cpu')
    
    # Create data loaders
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Train (short for testing)
    print("\nTraining for 5 epochs...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=5,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    # Evaluate
    print("\nEvaluating...")
    metrics = trainer.evaluate(val_loader)
    
    print("\n✅ Enhanced trainer working correctly!")
