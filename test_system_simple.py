"""
Simple test script with dummy data to validate the system
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_models():
    """Test all models with dummy data"""
    from scripts.compare_models import ModelComparator
    
    logger.info("\n" + "="*80)
    logger.info("TESTING ENHANCED SYSTEM WITH DUMMY DATA")
    logger.info("="*80 + "\n")
    
    # Create dummy data
    n_train = 500
    n_val = 200
    seq_len = 60
    num_features = 50
    
    logger.info(f"Generating dummy data...")
    logger.info(f"  Train samples: {n_train}")
    logger.info(f"  Val samples: {n_val}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  Features: {num_features}")
    
    # Generate data with some pattern (not completely random)
    X_train = np.random.randn(n_train, seq_len, num_features).astype(np.float32)
    # Add some trend to make it learnable
    for i in range(n_train):
        trend = np.linspace(0, 1, seq_len).reshape(-1, 1)
        X_train[i] += trend * np.random.randn(1, num_features) * 0.5
    
    # Create labels with some correlation to features
    y_train = (X_train[:, -1, 0] > X_train[:, -1, 0].mean()).astype(int)
    
    X_val = np.random.randn(n_val, seq_len, num_features).astype(np.float32)
    for i in range(n_val):
        trend = np.linspace(0, 1, seq_len).reshape(-1, 1)
        X_val[i] += trend * np.random.randn(1, num_features) * 0.5
    y_val = (X_val[:, -1, 0] > X_val[:, -1, 0].mean()).astype(int)
    
    logger.info(f"  Train positive class: {y_train.mean():.2%}")
    logger.info(f"  Val positive class: {y_val.mean():.2%}")
    
    # Test model comparison
    logger.info("\nRunning model comparison...")
    
    comparator = ModelComparator(output_dir='results/test_comparison')
    
    # Run comparison with reduced epochs for speed
    comparator.run_full_comparison(
        X_train, y_train, X_val, y_val,
        baseline_epochs=5,      # Very short for testing
        enhanced_epochs=10,     # Very short for testing
        use_sklearn_baselines=True,
        enhanced_models=['bilstm']  # Just test one enhanced model
    )
    
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*80 + "\n")
    
    # Print results
    for model_name, metrics in comparator.results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}\n")
    
    # Best model
    best_model = max(comparator.results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"Best Model: {best_model[0]}")
    logger.info(f"Best Accuracy: {best_model[1]['accuracy']:.4f}")
    
    logger.info("\n✅ System test complete!")
    logger.info(f"Results saved to: results/test_comparison/")
    
    return comparator.results


def test_individual_model():
    """Test a single model quickly"""
    from src.models.advanced_models import create_model
    from src.training.enhanced_trainer import EnhancedTrainer
    import torch
    
    logger.info("\n" + "="*80)
    logger.info("TESTING SINGLE MODEL")
    logger.info("="*80 + "\n")
    
    # Dummy data
    n_train = 200
    n_val = 50
    seq_len = 60
    num_features = 50
    
    X_train = np.random.randn(n_train, seq_len, num_features).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train)
    X_val = np.random.randn(n_val, seq_len, num_features).astype(np.float32)
    y_val = np.random.randint(0, 2, n_val)
    
    # Create model
    logger.info("Creating Hybrid model...")
    model = create_model(
        model_type='hybrid',
        num_features=num_features,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        use_finbert=False  # Disable for speed
    )
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params:,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    trainer = EnhancedTrainer(model, device=device, output_dir='results/test_single')
    
    # Create loaders
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    # Train for just a few epochs
    logger.info("\nTraining for 5 epochs...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=5,
        learning_rate=0.001,
        early_stopping_patience=10,
        log_interval=1
    )
    
    # Evaluate
    logger.info("\nEvaluating...")
    metrics = trainer.evaluate(val_loader)
    
    logger.info("\n✅ Single model test complete!")
    logger.info(f"Final validation accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'single'])
    args = parser.parse_args()
    
    try:
        if args.mode == 'full':
            results = test_models()
        else:
            results = test_individual_model()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
