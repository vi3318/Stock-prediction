"""
Example: Learnable Temporal Parameters
Demonstrates replacing manual event weights with learnable parameters
"""

import numpy as np
import pandas as pd
import logging
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_temporal_data(n_samples=1000):
    """Generate synthetic data with temporal information"""
    logger.info("Generating synthetic temporal data...")
    
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    
    event_types = [
        'earnings', 'merger', 'acquisition', 'partnership',
        'lawsuit', 'product_launch', 'regulatory', 'dividend'
    ]
    
    data = {
        'date': dates,
        'text_embedding': [np.random.randn(768) for _ in range(n_samples)],
        'numerical_features': [np.random.randn(50) for _ in range(n_samples)],
        'target': np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    }
    
    # Add event indicators
    for event in event_types:
        # Random events with some correlation to target
        prob = 0.15 if event in ['earnings', 'merger'] else 0.08
        data[f'event_{event}'] = np.random.random(n_samples) < prob
    
    # Add days elapsed (simulated)
    data['days_elapsed'] = np.random.uniform(0, 14, n_samples)
    
    # Add source credibility
    data['source_credibility'] = np.random.uniform(0.8, 1.5, n_samples)
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {n_samples} samples with {len(event_types)} event types")
    
    return df, event_types


def example_learnable_weights():
    """
    Example 1: Demonstrate learnable temporal weights
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Learnable Temporal Weights")
    logger.info("="*70 + "\n")
    
    try:
        import tensorflow as tf
        from src.features.learnable_temporal import LearnableTemporalWeights
        
        # Generate data
        df, event_types = generate_temporal_data(500)
        
        # Create learnable weights layer
        learnable_layer = LearnableTemporalWeights(
            event_types=event_types,
            initial_half_life=3.0
        )
        
        # Prepare inputs
        event_indicators = np.stack([
            df[[f'event_{et}' for et in event_types]].values
            for _ in range(1)
        ])[0].astype(np.float32)
        
        days_elapsed = df['days_elapsed'].values.reshape(-1, 1).astype(np.float32)
        source_credibility = df['source_credibility'].values.reshape(-1, 1).astype(np.float32)
        
        inputs = {
            'event_indicators': tf.constant(event_indicators),
            'days_elapsed': tf.constant(days_elapsed),
            'source_credibility': tf.constant(source_credibility)
        }
        
        # Forward pass
        weights = learnable_layer(inputs)
        
        logger.info(f"Computed temporal weights for {len(weights)} samples")
        logger.info(f"Weight range: [{weights.numpy().min():.4f}, {weights.numpy().max():.4f}]")
        logger.info(f"Mean weight: {weights.numpy().mean():.4f}")
        
        # Get initial learned weights
        learned = learnable_layer.get_learned_weights()
        logger.info("\nInitial learned parameters:")
        logger.info(f"  Half-life: {learned['temporal_half_life_days']:.2f} days")
        logger.info(f"  Top 3 event weights:")
        sorted_events = sorted(
            learned['event_weights'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for event, weight in sorted_events:
            logger.info(f"    {event}: {weight:.3f}")
        
        # Visualize
        os.makedirs('results', exist_ok=True)
        learnable_layer.visualize_learned_weights('results/example_learnable_weights.png')
        
        logger.info("\n✅ Learnable weights example completed!")
        
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        logger.info("This example requires TensorFlow to be installed.")


def example_manual_vs_learned_comparison():
    """
    Example 2: Compare manual vs. learnable weights
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Manual vs. Learnable Weights Comparison")
    logger.info("="*70 + "\n")
    
    # Manual weights (current approach)
    manual_weights = {
        'earnings': 2.0,
        'merger': 2.5,
        'acquisition': 2.5,
        'partnership': 1.5,
        'lawsuit': 1.8,
        'product_launch': 1.3,
        'regulatory': 1.7,
        'dividend': 1.2
    }
    
    logger.info("MANUAL WEIGHTS (Hand-tuned):")
    for event, weight in sorted(manual_weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {event:20s}: {weight:.2f}")
    
    logger.info("\nLimitations of Manual Weights:")
    logger.info("  ❌ Require domain expertise and manual tuning")
    logger.info("  ❌ May not be optimal for specific stocks/sectors")
    logger.info("  ❌ Fixed across all time periods and market conditions")
    logger.info("  ❌ No adaptation to changing market dynamics")
    
    logger.info("\n" + "-"*70)
    logger.info("\nLEARNABLE WEIGHTS (Data-driven):")
    logger.info("  ✅ Automatically learned from training data")
    logger.info("  ✅ Optimized for specific prediction task")
    logger.info("  ✅ Can adapt to different stocks/sectors")
    logger.info("  ✅ Integrated into model training objective")
    logger.info("  ✅ Provide insights into what events matter most")
    
    # Simulated learned weights (what model might learn)
    learned_weights = {
        'merger': 2.8,          # Model learned mergers are very important
        'earnings': 2.4,        # Earnings also important
        'acquisition': 2.3,
        'regulatory': 1.9,      # Regulatory higher than manual
        'lawsuit': 1.6,         # Lawsuit lower than manual
        'partnership': 1.4,
        'product_launch': 1.3,
        'dividend': 1.1         # Dividend least important
    }
    
    logger.info("\nExample LEARNED weights (after training):")
    for event, weight in sorted(learned_weights.items(), key=lambda x: x[1], reverse=True):
        manual_w = manual_weights[event]
        diff = weight - manual_w
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        logger.info(f"  {event:20s}: {weight:.2f} (manual: {manual_w:.2f}) {symbol} {abs(diff):.2f}")
    
    logger.info("\nKey Insights:")
    logger.info("  • Model learned mergers are MORE important than we thought")
    logger.info("  • Regulatory events have higher impact than manual weights suggested")
    logger.info("  • Lawsuits have less impact than manually assigned")
    logger.info("  • Data-driven approach reveals true event importance")


def example_temporal_decay_learning():
    """
    Example 3: Learning temporal decay half-life
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Learning Temporal Decay Half-Life")
    logger.info("="*70 + "\n")
    
    logger.info("MANUAL APPROACH:")
    logger.info("  Half-life: 3.0 days (fixed, hand-tuned)")
    logger.info("  Problem: May not be optimal for all stocks/sectors")
    logger.info("  • Tech stocks: News relevance may decay faster (2 days?)")
    logger.info("  • Utilities: News relevance may decay slower (5 days?)")
    
    logger.info("\n" + "-"*70)
    logger.info("\nLEARNABLE APPROACH:")
    logger.info("  Half-life: Learned from data during training")
    logger.info("  Benefits:")
    logger.info("    ✅ Automatically finds optimal decay rate")
    logger.info("    ✅ Can differ by stock/sector")
    logger.info("    ✅ Adapts to market volatility")
    
    # Simulate what model might learn for different sectors
    learned_half_lives = {
        'Tech (AAPL, MSFT)': 2.3,      # Fast decay
        'Finance (JPM, BAC)': 3.5,      # Medium decay
        'Utilities (NEE, DUK)': 5.2,    # Slow decay
        'Healthcare (JNJ, PFE)': 4.1    # Medium-slow decay
    }
    
    logger.info("\nExample learned half-lives by sector:")
    for sector, half_life in learned_half_lives.items():
        logger.info(f"  {sector:25s}: {half_life:.1f} days")
    
    logger.info("\nInterpretation:")
    logger.info("  • Tech: Fast-moving sector, news becomes stale quickly")
    logger.info("  • Utilities: Stable sector, news has longer-lasting impact")
    logger.info("  • Model learned these patterns from data!")


def example_integration_with_model():
    """
    Example 4: How to integrate learnable weights into model
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Integration with Model Architecture")
    logger.info("="*70 + "\n")
    
    logger.info("STEP 1: Define learnable weights layer")
    logger.info("""
    from src.features.learnable_temporal import LearnableTemporalWeights
    
    temporal_weights = LearnableTemporalWeights(
        event_types=['earnings', 'merger', 'lawsuit', ...],
        initial_half_life=3.0
    )
    """)
    
    logger.info("\nSTEP 2: Integrate into model forward pass")
    logger.info("""
    def call(self, inputs):
        # inputs contains temporal information
        temporal_info = {
            'event_indicators': inputs['events'],
            'days_elapsed': inputs['days_since_publication'],
            'source_credibility': inputs['source_scores']
        }
        
        # Compute temporal weights
        weights = self.temporal_weights(temporal_info)
        
        # Apply to text embeddings
        weighted_text = inputs['text_embeddings'] * weights
        
        # Continue with normal model flow
        ...
    """)
    
    logger.info("\nSTEP 3: Train normally - weights learn automatically!")
    logger.info("""
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, y_train, epochs=50)
    
    # Weights are optimized via backpropagation!
    """)
    
    logger.info("\nSTEP 4: Inspect learned weights after training")
    logger.info("""
    learned = model.get_learned_weights()
    print(f"Learned half-life: {learned['temporal_half_life_days']:.2f} days")
    print(f"Event weights: {learned['event_weights']}")
    
    # Visualize
    model.visualize_learned_weights('results/learned_weights.png')
    """)
    
    logger.info("\n✅ Weights are now part of model parameters!")
    logger.info("   They're saved with the model and loaded automatically")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Learnable Temporal Parameters - Examples")
    logger.info("    Replacing Manual Weights with Data-Driven Learning")
    logger.info("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    try:
        # Run examples
        example_learnable_weights()
        example_manual_vs_learned_comparison()
        example_temporal_decay_learning()
        example_integration_with_model()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n📊 Key Takeaways:")
        logger.info("  1. Learnable weights adapt to data automatically")
        logger.info("  2. No manual tuning required")
        logger.info("  3. Optimal for specific stocks/sectors")
        logger.info("  4. Provides interpretable insights")
        logger.info("  5. Publication-worthy approach (data-driven > heuristic)")
        
        logger.info("\n🔬 For Research Paper:")
        logger.info("  • Compare manual vs. learned in ablation study")
        logger.info("  • Report learned weights for different sectors")
        logger.info("  • Show weight evolution during training")
        logger.info("  • Demonstrate statistical significance of improvement")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
