"""
Hybrid Deep Learning Model Architecture
Implements late fusion model combining text embeddings and numerical features
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class TextEncoder(layers.Layer):
    """
    Encodes text embeddings using BiLSTM with attention
    """
    
    def __init__(
        self, 
        lstm_units: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        attention_heads: int = 4,
        name: str = "text_encoder",
        **kwargs
    ):
        super(TextEncoder, self).__init__(name=name, **kwargs)
        
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        
        # Build LSTM layers
        self.lstm_stack = []
        for i in range(lstm_layers):
            self.lstm_stack.append(
                layers.Bidirectional(
                    layers.LSTM(
                        lstm_units,
                        return_sequences=True if (i < lstm_layers - 1 or use_attention) else False,
                        dropout=dropout,
                        recurrent_dropout=dropout * 0.5,
                        name=f'lstm_{i}'
                    ),
                    name=f'bidirectional_lstm_{i}'
                )
            )
        
        # Attention mechanism
        if use_attention:
            self.attention = layers.MultiHeadAttention(
                num_heads=attention_heads,
                key_dim=lstm_units * 2,  # *2 for bidirectional
                dropout=dropout,
                name='multi_head_attention'
            )
            self.layer_norm = layers.LayerNormalization(name='attention_layer_norm')
            self.global_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')
        
        self.dropout_layer = layers.Dropout(dropout, name='encoder_dropout')
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Input tensor (batch_size, sequence_length, embedding_dim)
            training: Whether in training mode
            
        Returns:
            Encoded tensor
        """
        x = inputs
        
        # Pass through LSTM stack
        for lstm_layer in self.lstm_stack:
            x = lstm_layer(x, training=training)
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention
            attn_output = self.attention(x, x, training=training)
            x = self.layer_norm(x + attn_output)
            # Pool to fixed size
            x = self.global_pool(x)
        
        x = self.dropout_layer(x, training=training)
        
        return x


class NumericalEncoder(layers.Layer):
    """
    Encodes numerical features using GRU
    """
    
    def __init__(
        self,
        gru_units: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.2,
        name: str = "numerical_encoder",
        **kwargs
    ):
        super(NumericalEncoder, self).__init__(name=name, **kwargs)
        
        self.gru_units = gru_units
        self.gru_layers = gru_layers
        self.dropout = dropout
        
        # Build GRU layers
        self.gru_stack = []
        for i in range(gru_layers):
            self.gru_stack.append(
                layers.GRU(
                    gru_units,
                    return_sequences=True if i < gru_layers - 1 else False,
                    dropout=dropout,
                    recurrent_dropout=dropout * 0.5,
                    name=f'gru_{i}'
                )
            )
        
        self.dropout_layer = layers.Dropout(dropout, name='numerical_dropout')
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Input tensor (batch_size, sequence_length, num_features)
            training: Whether in training mode
            
        Returns:
            Encoded tensor
        """
        x = inputs
        
        # Pass through GRU stack
        for gru_layer in self.gru_stack:
            x = gru_layer(x, training=training)
        
        x = self.dropout_layer(x, training=training)
        
        return x


class FusionLayer(layers.Layer):
    """
    Fuses text and numerical features
    """
    
    def __init__(
        self,
        dense_layers: List[int] = [128, 64],
        dropout: float = 0.4,
        activation: str = 'relu',
        name: str = "fusion_layer",
        **kwargs
    ):
        super(FusionLayer, self).__init__(name=name, **kwargs)
        
        self.dense_layers_config = dense_layers
        self.dropout = dropout
        self.activation = activation
        
        # Build dense layers
        self.dense_stack = []
        for i, units in enumerate(dense_layers):
            self.dense_stack.append(
                layers.Dense(units, activation=activation, name=f'fusion_dense_{i}')
            )
            self.dense_stack.append(
                layers.Dropout(dropout, name=f'fusion_dropout_{i}')
            )
            self.dense_stack.append(
                layers.BatchNormalization(name=f'fusion_bn_{i}')
            )
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: List of [text_features, numerical_features]
            training: Whether in training mode
            
        Returns:
            Fused tensor
        """
        # Concatenate features
        x = layers.Concatenate(name='concat')(inputs)
        
        # Pass through dense layers
        for layer in self.dense_stack:
            x = layer(x, training=training)
        
        return x


class LateFusionModel:
    """
    Late fusion model combining text and numerical encoders
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        
        # Extract configuration
        self.text_config = config.get('model', {}).get('text_encoder', {})
        self.numerical_config = config.get('model', {}).get('numerical_encoder', {})
        self.fusion_config = config.get('model', {}).get('fusion', {})
        self.output_config = config.get('model', {}).get('output', {})
        
        # Task configuration
        self.task = self.output_config.get('task', 'classification')
        self.num_classes = self.output_config.get('classes', 3)
    
    def build_model(
        self,
        text_sequence_length: int,
        text_embedding_dim: int,
        numerical_sequence_length: int,
        numerical_features: int
    ) -> Model:
        """
        Build the complete late fusion model
        
        Args:
            text_sequence_length: Length of text sequence
            text_embedding_dim: Dimension of text embeddings
            numerical_sequence_length: Length of numerical sequence
            numerical_features: Number of numerical features
            
        Returns:
            Keras Model
        """
        self.logger.info("Building late fusion model")
        
        # Input layers
        text_input = layers.Input(
            shape=(text_sequence_length, text_embedding_dim),
            name='text_input'
        )
        numerical_input = layers.Input(
            shape=(numerical_sequence_length, numerical_features),
            name='numerical_input'
        )
        
        # Text encoder branch
        text_encoder = TextEncoder(
            lstm_units=self.text_config.get('lstm_units', 128),
            lstm_layers=self.text_config.get('lstm_layers', 2),
            dropout=self.text_config.get('dropout', 0.3),
            use_attention=self.text_config.get('attention', True),
            attention_heads=self.text_config.get('attention_heads', 4)
        )
        text_features = text_encoder(text_input)
        
        # Numerical encoder branch
        numerical_encoder = NumericalEncoder(
            gru_units=self.numerical_config.get('gru_units', 64),
            gru_layers=self.numerical_config.get('gru_layers', 2),
            dropout=self.numerical_config.get('dropout', 0.2)
        )
        numerical_features = numerical_encoder(numerical_input)
        
        # Fusion layer
        fusion = FusionLayer(
            dense_layers=self.fusion_config.get('dense_layers', [128, 64]),
            dropout=self.fusion_config.get('dropout', 0.4),
            activation=self.fusion_config.get('activation', 'relu')
        )
        fused_features = fusion([text_features, numerical_features])
        
        # Output layer
        if self.task == 'classification':
            output = layers.Dense(
                self.num_classes,
                activation='softmax',
                name='output'
            )(fused_features)
        else:  # regression
            output = layers.Dense(
                1,
                activation='linear',
                name='output'
            )(fused_features)
        
        # Create model
        self.model = Model(
            inputs=[text_input, numerical_input],
            outputs=output,
            name='late_fusion_stock_predictor'
        )
        
        self.logger.info("Model architecture created")
        self.logger.info(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        class_weights: Optional[Dict] = None
    ):
        """
        Compile the model with appropriate loss and metrics
        
        Args:
            learning_rate: Learning rate
            optimizer: Optimizer name
            class_weights: Class weights for imbalanced data
        """
        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'adamw':
            opt = keras.optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Choose loss and metrics
        if self.task == 'classification':
            loss = 'sparse_categorical_crossentropy'
            metrics = [
                'accuracy',
                keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        else:  # regression
            loss = 'mse'
            metrics = [
                'mae',
                keras.metrics.RootMeanSquaredError(name='rmse'),
                keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with {optimizer} optimizer and {loss} loss")
    
    def get_callbacks(
        self,
        model_path: str = 'models/saved_models/best_model.h5',
        patience: int = 15,
        min_delta: float = 0.001
    ) -> List:
        """
        Get training callbacks
        
        Args:
            model_path: Path to save best model
            patience: Early stopping patience
            min_delta: Minimum improvement threshold
            
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            self.logger.warning("Model not built yet")


class EarlyFusionModel:
    """
    Alternative: Early fusion model (combines features before encoding)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        
        self.task = config.get('model', {}).get('output', {}).get('task', 'classification')
        self.num_classes = config.get('model', {}).get('output', {}).get('classes', 3)
    
    def build_model(
        self,
        sequence_length: int,
        text_embedding_dim: int,
        numerical_features: int
    ) -> Model:
        """
        Build early fusion model
        
        Args:
            sequence_length: Sequence length
            text_embedding_dim: Text embedding dimension
            numerical_features: Number of numerical features
            
        Returns:
            Keras Model
        """
        # Input layers
        text_input = layers.Input(
            shape=(sequence_length, text_embedding_dim),
            name='text_input'
        )
        numerical_input = layers.Input(
            shape=(sequence_length, numerical_features),
            name='numerical_input'
        )
        
        # Concatenate at input level
        combined = layers.Concatenate(axis=-1, name='early_fusion')([text_input, numerical_input])
        
        # Shared encoder
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.3)
        )(combined)
        x = layers.Bidirectional(
            layers.LSTM(64, dropout=0.3)
        )(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        # Output
        if self.task == 'classification':
            output = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        else:
            output = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = Model(
            inputs=[text_input, numerical_input],
            outputs=output,
            name='early_fusion_stock_predictor'
        )
        
        return self.model


def create_model(config: Dict, fusion_type: str = 'late') -> LateFusionModel:
    """
    Factory function to create model
    
    Args:
        config: Configuration dictionary
        fusion_type: 'late' or 'early'
        
    Returns:
        Model instance
    """
    if fusion_type == 'late':
        return LateFusionModel(config)
    elif fusion_type == 'early':
        return EarlyFusionModel(config)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'model': {
            'text_encoder': {
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout': 0.3,
                'attention': True,
                'attention_heads': 4
            },
            'numerical_encoder': {
                'gru_units': 64,
                'gru_layers': 2,
                'dropout': 0.2
            },
            'fusion': {
                'dense_layers': [128, 64],
                'dropout': 0.4,
                'activation': 'relu'
            },
            'output': {
                'task': 'classification',
                'classes': 3
            }
        }
    }
    
    model_builder = LateFusionModel(config)
    model = model_builder.build_model(
        text_sequence_length=10,
        text_embedding_dim=768,
        numerical_sequence_length=30,
        numerical_features=50
    )
    model_builder.compile_model()
    model_builder.summary()
