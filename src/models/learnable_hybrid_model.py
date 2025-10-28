"""
Enhanced Hybrid Model with Learnable Temporal Parameters
Extends LateFusionModel to include learnable event weights and decay
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.hybrid_model import TextEncoder, NumericalEncoder, FusionLayer
from src.features.learnable_temporal import LearnableTemporalWeights

logger = logging.getLogger(__name__)


class LateFusionModelWithLearnableWeights(keras.Model):
    """
    Late Fusion Model with Learnable Temporal Parameters
    
    Key Enhancements:
    - Learnable event importance weights
    - Learnable temporal decay half-life
    - Integrated into training objective
    - Visualization of learned parameters
    """
    
    def __init__(
        self,
        config: Dict,
        event_types: List[str] = None,
        use_learnable_weights: bool = True
    ):
        """
        Initialize enhanced model
        
        Args:
            config: Configuration dictionary
            event_types: List of event type names
            use_learnable_weights: Whether to use learnable temporal weights
        """
        super().__init__()
        
        self.config = config
        self.use_learnable_weights = use_learnable_weights
        
        # Default event types if not provided
        if event_types is None:
            event_types = [
                'earnings', 'merger', 'acquisition', 'partnership',
                'lawsuit', 'product_launch', 'regulatory', 'dividend',
                'ceo_change', 'restructuring'
            ]
        self.event_types = event_types
        
        # Model architecture parameters
        model_config = config.get('model', {})
        text_config = model_config.get('text_encoder', {})
        num_config = model_config.get('numerical_encoder', {})
        fusion_config = model_config.get('fusion', {})
        output_config = model_config.get('output', {})
        
        # Text encoder
        self.text_encoder = TextEncoder(
            lstm_units=text_config.get('lstm_units', 128),
            lstm_layers=text_config.get('lstm_layers', 2),
            dropout=text_config.get('dropout', 0.3),
            use_attention=text_config.get('use_attention', True),
            attention_heads=text_config.get('attention_heads', 4)
        )
        
        # Numerical encoder
        self.numerical_encoder = NumericalEncoder(
            gru_units=num_config.get('gru_units', 64),
            gru_layers=num_config.get('gru_layers', 2),
            dropout=num_config.get('dropout', 0.3)
        )
        
        # Learnable temporal weights layer
        if use_learnable_weights:
            self.temporal_weights = LearnableTemporalWeights(
                event_types=event_types,
                initial_half_life=config.get('temporal', {}).get('half_life_days', 3.0)
            )
        else:
            self.temporal_weights = None
        
        # Fusion layer
        self.fusion = FusionLayer(
            hidden_units=fusion_config.get('hidden_units', [128, 64]),
            dropout=fusion_config.get('dropout', 0.3),
            use_batch_norm=fusion_config.get('use_batch_norm', True)
        )
        
        # Output layer
        num_classes = output_config.get('classes', 3)
        self.output_layer = layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass with learnable temporal weights
        
        Args:
            inputs: Can be:
                - Tuple of (text_features, numerical_features)
                - Dict with 'text', 'numerical', and optionally temporal info
            training: Whether in training mode
            
        Returns:
            Model predictions
        """
        # Handle different input formats
        if isinstance(inputs, (list, tuple)):
            text_input, numerical_input = inputs
            temporal_info = None
        elif isinstance(inputs, dict):
            text_input = inputs['text']
            numerical_input = inputs['numerical']
            temporal_info = inputs.get('temporal_info', None)
        else:
            raise ValueError("Inputs must be tuple/list or dictionary")
        
        # Apply learnable temporal weights if available
        if self.use_learnable_weights and temporal_info is not None:
            temporal_weights = self.temporal_weights(temporal_info)
            # Apply weights to text embeddings
            # temporal_weights shape: [batch, 1]
            # text_input shape: [batch, seq_len, embed_dim]
            text_input = text_input * temporal_weights[:, tf.newaxis, :]
        
        # Encode branches
        text_encoded = self.text_encoder(text_input, training=training)
        numerical_encoded = self.numerical_encoder(numerical_input, training=training)
        
        # Fuse and predict
        fused = self.fusion([text_encoded, numerical_encoded], training=training)
        output = self.output_layer(fused)
        
        return output
    
    def get_learned_weights(self) -> Optional[Dict]:
        """Get learned temporal weights"""
        if self.use_learnable_weights and self.temporal_weights:
            return self.temporal_weights.get_learned_weights()
        return None
    
    def save_learned_weights(self, filepath: str = 'results/learned_weights.json'):
        """Save learned weights to file"""
        if self.use_learnable_weights and self.temporal_weights:
            return self.temporal_weights.save_learned_weights(filepath)
        logger.warning("No learnable weights to save")
        return None
    
    def visualize_learned_weights(self, save_path: str = 'results/learned_weights.png'):
        """Visualize learned temporal parameters"""
        if self.use_learnable_weights and self.temporal_weights:
            self.temporal_weights.visualize_learned_weights(save_path)
        else:
            logger.warning("No learnable weights to visualize")
    
    def get_callbacks(self, model_path: str = 'models/saved_models/best_model.h5'):
        """Get training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('training', {}).get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Custom callback to log learned weights
            LearnedWeightsLogger(self)
        ]
        
        return callbacks


class LearnedWeightsLogger(keras.callbacks.Callback):
    """
    Custom callback to log learned temporal weights during training
    """
    
    def __init__(self, model_instance):
        super().__init__()
        self.model_instance = model_instance
        self.weight_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Log learned weights at end of each epoch"""
        if self.model_instance.use_learnable_weights:
            weights = self.model_instance.get_learned_weights()
            
            if weights and epoch % 5 == 0:  # Log every 5 epochs
                logger.info(f"\nEpoch {epoch} - Learned Parameters:")
                logger.info(f"  Half-life: {weights['temporal_half_life_days']:.2f} days")
                
                # Show top 3 event weights
                event_weights = weights['event_weights']
                top_events = sorted(
                    event_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                logger.info("  Top 3 Event Weights:")
                for event, weight in top_events:
                    logger.info(f"    {event}: {weight:.3f}")
                
                self.weight_history.append({
                    'epoch': epoch,
                    **weights
                })
    
    def on_train_end(self, logs=None):
        """Save weight evolution at end of training"""
        if self.weight_history:
            import json
            os.makedirs('results', exist_ok=True)
            
            with open('results/weight_evolution.json', 'w') as f:
                json.dump(self.weight_history, f, indent=2)
            
            logger.info("Weight evolution saved to results/weight_evolution.json")
            
            # Visualize final learned weights
            self.model_instance.visualize_learned_weights(
                'results/final_learned_weights.png'
            )


def compare_manual_vs_learned(
    config: Dict,
    data: Tuple,
    epochs: int = 20
):
    """
    Compare model performance with manual vs. learned temporal weights
    
    Args:
        config: Configuration dictionary
        data: Training data tuple (X_text, X_num, y)
        epochs: Number of training epochs
        
    Returns:
        Comparison results
    """
    logger.info("="*70)
    logger.info("COMPARING MANUAL VS. LEARNABLE TEMPORAL WEIGHTS")
    logger.info("="*70)
    
    X_text, X_num, y = data
    
    # Split data
    split_idx = int(0.8 * len(X_text))
    train_data = (X_text[:split_idx], X_num[:split_idx], y[:split_idx])
    val_data = (X_text[split_idx:], X_num[split_idx:], y[split_idx:])
    
    results = {}
    
    # Train with manual weights (baseline)
    logger.info("\n1. Training with MANUAL weights...")
    model_manual = LateFusionModelWithLearnableWeights(
        config,
        use_learnable_weights=False
    )
    
    model_manual.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_manual = model_manual.fit(
        [train_data[0], train_data[1]],
        train_data[2],
        validation_data=([val_data[0], val_data[1]], val_data[2]),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    results['manual'] = {
        'val_accuracy': history_manual.history['val_accuracy'][-1],
        'val_loss': history_manual.history['val_loss'][-1]
    }
    
    # Train with learnable weights
    logger.info("\n2. Training with LEARNABLE weights...")
    model_learned = LateFusionModelWithLearnableWeights(
        config,
        use_learnable_weights=True
    )
    
    model_learned.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_learned = model_learned.fit(
        [train_data[0], train_data[1]],
        train_data[2],
        validation_data=([val_data[0], val_data[1]], val_data[2]),
        epochs=epochs,
        batch_size=32,
        callbacks=model_learned.get_callbacks(),
        verbose=1
    )
    
    results['learned'] = {
        'val_accuracy': history_learned.history['val_accuracy'][-1],
        'val_loss': history_learned.history['val_loss'][-1],
        'learned_weights': model_learned.get_learned_weights()
    }
    
    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)
    logger.info(f"Manual Weights    - Val Acc: {results['manual']['val_accuracy']:.4f}, "
                f"Val Loss: {results['manual']['val_loss']:.4f}")
    logger.info(f"Learnable Weights - Val Acc: {results['learned']['val_accuracy']:.4f}, "
                f"Val Loss: {results['learned']['val_loss']:.4f}")
    
    improvement = (results['learned']['val_accuracy'] - results['manual']['val_accuracy']) * 100
    logger.info(f"\nImprovement: {improvement:+.2f}%")
    
    if results['learned']['learned_weights']:
        logger.info("\nLearned Parameters:")
        logger.info(f"  Half-life: {results['learned']['learned_weights']['temporal_half_life_days']:.2f} days")
        logger.info(f"  Event Weights: {results['learned']['learned_weights']['event_weights']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'model': {
            'text_encoder': {'lstm_units': 64, 'lstm_layers': 2},
            'numerical_encoder': {'gru_units': 32, 'gru_layers': 2},
            'fusion': {'hidden_units': [64, 32]},
            'output': {'classes': 3}
        },
        'temporal': {
            'half_life_days': 3.0
        },
        'training': {
            'early_stopping_patience': 10
        }
    }
    
    # Create model
    model = LateFusionModelWithLearnableWeights(config, use_learnable_weights=True)
    
    # Create dummy data
    batch_size = 32
    seq_len = 30
    embed_dim = 768
    num_features = 50
    
    X_text = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    X_num = np.random.randn(batch_size, seq_len, num_features).astype(np.float32)
    
    # Build model
    model([X_text, X_num])
    
    print(f"\nModel created with {len(model.event_types)} learnable event weights")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test learned weights extraction
    weights = model.get_learned_weights()
    print("\nInitial learned weights:")
    print(f"  Half-life: {weights['temporal_half_life_days']:.2f} days")
    print(f"  Event types: {model.event_types}")
    
    print("\nEnhanced model with learnable temporal parameters ready!")
