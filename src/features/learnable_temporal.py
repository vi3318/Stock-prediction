"""
Learnable Temporal Parameters Module
Replaces manual event weights and decay parameters with learnable ones
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class LearnableTemporalWeights(keras.layers.Layer):
    """
    Learnable weights for event importance and temporal decay
    
    Key Innovation:
    - Event weights learned during training (not manually specified)
    - Temporal decay half-life learned from data
    - Softplus constraint ensures positive weights
    """
    
    def __init__(
        self,
        event_types: List[str],
        initial_event_weights: Optional[Dict[str, float]] = None,
        initial_half_life: float = 3.0,
        name: str = 'learnable_temporal_weights',
        **kwargs
    ):
        """
        Initialize learnable temporal parameters
        
        Args:
            event_types: List of event type names
            initial_event_weights: Optional initial weights (warm start)
            initial_half_life: Initial decay half-life in days
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.event_types = event_types
        self.num_event_types = len(event_types)
        self.initial_half_life = initial_half_life
        
        # Initialize event weight values (before softplus)
        if initial_event_weights:
            # Use inverse softplus to get initial values
            # softplus(x) = log(1 + exp(x)), so x = log(exp(w) - 1)
            init_values = [
                np.log(np.exp(initial_event_weights.get(et, 1.5)) - 1.0)
                for et in event_types
            ]
        else:
            # Default: small positive values that will become ~1.5 after softplus
            init_values = [0.5] * self.num_event_types
        
        # Learnable event importance weights (raw, before softplus)
        self.event_weights_raw = self.add_weight(
            name='event_weights_raw',
            shape=(self.num_event_types,),
            initializer=keras.initializers.Constant(init_values),
            trainable=True
        )
        
        # Learnable temporal decay half-life (raw, before softplus)
        # Initialize so softplus(x) ≈ initial_half_life
        init_half_life_raw = np.log(np.exp(initial_half_life) - 1.0)
        self.half_life_raw = self.add_weight(
            name='half_life_raw',
            shape=(1,),
            initializer=keras.initializers.Constant([init_half_life_raw]),
            trainable=True
        )
        
        # Learnable source credibility weights
        self.source_weight_raw = self.add_weight(
            name='source_credibility_raw',
            shape=(1,),
            initializer=keras.initializers.Constant([0.5]),
            trainable=True
        )
    
    def get_event_weights(self) -> tf.Tensor:
        """Get positive event weights via softplus"""
        # Softplus ensures weights are positive
        # Minimum value is 1.0 (no importance boost)
        return tf.nn.softplus(self.event_weights_raw) + 1.0
    
    def get_half_life(self) -> tf.Tensor:
        """Get positive half-life via softplus"""
        # Ensure half-life is positive and reasonable (0.5 to 30 days)
        half_life = tf.nn.softplus(self.half_life_raw)
        return tf.clip_by_value(half_life, 0.5, 30.0)
    
    def get_source_weight(self) -> tf.Tensor:
        """Get source credibility weight"""
        return tf.nn.softplus(self.source_weight_raw) + 1.0
    
    def call(self, inputs):
        """
        Apply learnable weights to temporal features
        
        Args:
            inputs: Dictionary with:
                - 'event_indicators': [batch, num_event_types] binary indicators
                - 'days_elapsed': [batch, 1] days since publication
                - 'source_credibility': [batch, 1] source credibility score
                
        Returns:
            combined_weight: [batch, 1] temporal importance weight
        """
        event_indicators = inputs['event_indicators']  # [batch, num_events]
        days_elapsed = inputs['days_elapsed']  # [batch, 1]
        source_credibility = inputs.get('source_credibility', None)
        
        # Get learnable parameters
        event_weights = self.get_event_weights()  # [num_events]
        half_life = self.get_half_life()  # [1]
        
        # Calculate event importance
        # For each sample, take max of applicable event weights
        weighted_events = event_indicators * event_weights  # [batch, num_events]
        event_importance = tf.reduce_max(weighted_events, axis=1, keepdims=True)  # [batch, 1]
        # Add base weight of 1.0 if no events
        event_importance = tf.maximum(event_importance, 1.0)
        
        # Calculate temporal decay
        # decay = 0.5^(days_elapsed / half_life)
        decay_exponent = days_elapsed / half_life
        temporal_decay = tf.pow(0.5, decay_exponent)  # [batch, 1]
        
        # Apply max history cutoff (30 days)
        temporal_decay = tf.where(
            days_elapsed > 30.0,
            tf.zeros_like(temporal_decay),
            temporal_decay
        )
        
        # Combine weights
        combined_weight = event_importance * temporal_decay  # [batch, 1]
        
        # Apply source credibility if provided
        if source_credibility is not None:
            source_weight = self.get_source_weight()
            combined_weight = combined_weight * (source_credibility * source_weight)
        
        return combined_weight
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'event_types': self.event_types,
            'initial_half_life': self.initial_half_life,
            'num_event_types': self.num_event_types
        })
        return config
    
    def get_learned_weights(self) -> Dict:
        """
        Get learned parameter values for inspection
        
        Returns:
            Dictionary with learned weights
        """
        event_weights = self.get_event_weights().numpy()
        half_life = self.get_half_life().numpy()[0]
        
        learned_weights = {
            'event_weights': {
                event_type: float(weight)
                for event_type, weight in zip(self.event_types, event_weights)
            },
            'temporal_half_life_days': float(half_life),
            'source_weight': float(self.get_source_weight().numpy()[0])
        }
        
        return learned_weights
    
    def save_learned_weights(self, filepath: str = 'results/learned_weights.json'):
        """Save learned weights to JSON file"""
        weights = self.get_learned_weights()
        
        with open(filepath, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"Learned weights saved to {filepath}")
        return weights
    
    def visualize_learned_weights(self, save_path: str = 'results/learned_weights.png'):
        """
        Visualize learned event weights
        
        Args:
            save_path: Path to save visualization
        """
        weights = self.get_learned_weights()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Event importance weights
        event_weights = weights['event_weights']
        event_names = list(event_weights.keys())
        event_values = list(event_weights.values())
        
        colors = ['#e74c3c' if v > 2.0 else '#3498db' if v > 1.5 else '#95a5a6' 
                  for v in event_values]
        
        axes[0].barh(event_names, event_values, color=colors)
        axes[0].axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1.0)')
        axes[0].set_xlabel('Importance Weight', fontsize=12)
        axes[0].set_title('Learned Event Importance Weights', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Temporal decay curve
        half_life = weights['temporal_half_life_days']
        days = np.linspace(0, 30, 100)
        decay = 0.5 ** (days / half_life)
        
        axes[1].plot(days, decay, linewidth=2.5, color='#2ecc71')
        axes[1].fill_between(days, 0, decay, alpha=0.3, color='#2ecc71')
        axes[1].axvline(x=half_life, color='red', linestyle='--', 
                       label=f'Half-life: {half_life:.2f} days')
        axes[1].set_xlabel('Days Since Publication', fontsize=12)
        axes[1].set_ylabel('Temporal Weight', fontsize=12)
        axes[1].set_title('Learned Temporal Decay Function', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim(0, 30)
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Weight visualization saved to {save_path}")
        logger.info(f"Learned half-life: {half_life:.2f} days")
        logger.info(f"Top 3 important events: {sorted(event_weights.items(), key=lambda x: x[1], reverse=True)[:3]}")


class TemporalFeatureLayer(keras.layers.Layer):
    """
    Complete temporal feature processing layer with learnable weights
    Integrates into model architecture
    """
    
    def __init__(
        self,
        event_types: List[str],
        sequence_length: int = 30,
        learnable: bool = True,
        **kwargs
    ):
        """
        Args:
            event_types: List of event type names
            sequence_length: Length of temporal sequences
            learnable: Whether to use learnable weights (vs. fixed)
        """
        super().__init__(**kwargs)
        
        self.event_types = event_types
        self.sequence_length = sequence_length
        self.learnable = learnable
        
        if learnable:
            self.temporal_weights = LearnableTemporalWeights(event_types)
        else:
            self.temporal_weights = None
    
    def call(self, inputs):
        """
        Process temporal features
        
        Args:
            inputs: Dictionary with temporal information
            
        Returns:
            weighted_features: Temporally weighted features
        """
        if self.learnable:
            weights = self.temporal_weights(inputs)
        else:
            # Use fixed weights if not learnable
            weights = tf.ones_like(inputs['days_elapsed'])
        
        # Apply weights to features
        # Assuming inputs also contains the actual features
        features = inputs.get('features', inputs['days_elapsed'])
        weighted_features = features * weights
        
        return weighted_features
    
    def get_learned_weights(self):
        """Get learned weights if available"""
        if self.learnable and self.temporal_weights:
            return self.temporal_weights.get_learned_weights()
        return None


def create_temporal_weight_regularizer(target_sparsity: float = 0.3):
    """
    Create L1 regularizer to encourage sparse event weights
    
    Args:
        target_sparsity: Target sparsity level (0 to 1)
        
    Returns:
        Regularization loss function
    """
    def regularizer(weights):
        """L1 regularization on event weights"""
        # Encourage some event types to have weight close to 1.0 (unimportant)
        l1_loss = tf.reduce_mean(tf.abs(weights - 1.0))
        return target_sparsity * l1_loss
    
    return regularizer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Define event types
    event_types = [
        'earnings', 'merger', 'acquisition', 'partnership',
        'lawsuit', 'product_launch', 'regulatory', 'dividend'
    ]
    
    # Create learnable weights layer
    learnable_layer = LearnableTemporalWeights(event_types)
    
    # Create dummy data
    batch_size = 32
    dummy_inputs = {
        'event_indicators': tf.random.uniform((batch_size, len(event_types)), 0, 1) > 0.7,
        'days_elapsed': tf.random.uniform((batch_size, 1), 0, 15),
        'source_credibility': tf.random.uniform((batch_size, 1), 0.8, 1.5)
    }
    
    # Forward pass
    weights = learnable_layer(dummy_inputs)
    
    print(f"Output shape: {weights.shape}")
    print(f"Sample weights: {weights[:5].numpy().flatten()}")
    
    # Get learned weights
    learned = learnable_layer.get_learned_weights()
    print("\nLearned parameters:")
    print(json.dumps(learned, indent=2))
    
    # Visualize
    import os
    os.makedirs('results', exist_ok=True)
    learnable_layer.visualize_learned_weights()
    
    print("\nLearnable temporal weights module ready!")
