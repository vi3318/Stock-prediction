"""
Example: Advanced Explainability
Demonstrates temporal heatmaps, event contributions, and counterfactual analysis
"""

import numpy as np
import pandas as pd
import logging
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_temporal_heatmap():
    """
    Example 1: Temporal feature importance heatmap
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: Temporal Feature Importance Heatmap")
    logger.info("="*70 + "\n")
    
    from src.explainability.advanced_explain import TemporalHeatmapGenerator
    
    # Initialize
    generator = TemporalHeatmapGenerator(lookback_days=30)
    
    # Mock data
    prediction_date = datetime(2023, 6, 15)
    
    # Simulate events over past 30 days
    events = []
    for i in range(20):
        days_ago = np.random.randint(0, 30)
        event = {
            'id': f'event_{i}',
            'timestamp': prediction_date - timedelta(days=days_ago),
            'event_type': np.random.choice(['earnings', 'merger', 'product_launch']),
            'headline': f'News event {i}',
            'features': np.random.randn(10)
        }
        events.append(event)
    
    # Mock attention weights and temporal decay
    attention_weights = np.random.dirichlet(np.ones(len(events)))
    temporal_decay_fn = lambda days: np.exp(-0.3 * days)
    
    # Compute temporal importance
    importance_matrix = generator.compute_temporal_importance(
        events, prediction_date, attention_weights, temporal_decay_fn
    )
    
    logger.info(f"Importance matrix shape: {importance_matrix.shape}")
    logger.info(f"  Rows (days): {importance_matrix.shape[0]}")
    logger.info(f"  Columns (features): {importance_matrix.shape[1]}")
    logger.info(f"  Max importance: {importance_matrix.max():.4f}")
    logger.info(f"  Mean importance: {importance_matrix.mean():.4f}")
    
    # Create visualization
    feature_names = ['Sentiment', 'Volume', 'Momentum', 'RSI', 'MACD', 
                    'Event_Type', 'Source', 'Entities', 'Relations', 'Topics']
    
    os.makedirs('results/explanations', exist_ok=True)
    generator.visualize_temporal_heatmap(
        importance_matrix,
        feature_names,
        prediction_date,
        'results/explanations/temporal_heatmap_example.png'
    )
    
    logger.info("\n✅ Temporal heatmap shows:")
    logger.info("   • How feature importance changes over time")
    logger.info("   • Recent events have higher importance (temporal decay)")
    logger.info("   • Which features were most relevant on which days")


def example_event_contribution():
    """
    Example 2: Event contribution analysis
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: Event Contribution Analysis")
    logger.info("="*70 + "\n")
    
    from src.explainability.advanced_explain import EventContributionAnalyzer
    
    # Initialize
    analyzer = EventContributionAnalyzer(top_k=10)
    
    # Mock events with realistic headlines
    events = [
        {
            'id': 'evt1',
            'timestamp': datetime(2023, 6, 14),
            'event_type': 'earnings',
            'headline': 'Company beats Q2 earnings expectations',
            'sentiment': 0.85
        },
        {
            'id': 'evt2',
            'timestamp': datetime(2023, 6, 12),
            'event_type': 'product_launch',
            'headline': 'New flagship product unveiled at tech conference',
            'sentiment': 0.65
        },
        {
            'id': 'evt3',
            'timestamp': datetime(2023, 6, 10),
            'event_type': 'guidance',
            'headline': 'CEO provides strong guidance for next quarter',
            'sentiment': 0.72
        },
        {
            'id': 'evt4',
            'timestamp': datetime(2023, 6, 8),
            'event_type': 'analyst_rating',
            'headline': 'Analyst upgrades stock to Buy rating',
            'sentiment': 0.55
        },
        {
            'id': 'evt5',
            'timestamp': datetime(2023, 6, 5),
            'event_type': 'rumor',
            'headline': 'Rumors of potential merger circulate',
            'sentiment': 0.40
        }
    ]
    
    # Mock model outputs
    attention_weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
    temporal_weights = np.array([0.95, 0.85, 0.75, 0.65, 0.50])
    prediction = np.array([0.1, 0.2, 0.7])  # Strong UP prediction
    
    # Analyze contributions
    contributions = analyzer.analyze_event_contributions(
        events, attention_weights, temporal_weights, prediction
    )
    
    logger.info("Top Contributing Events:")
    logger.info("-" * 70)
    
    for i, contrib in enumerate(contributions, 1):
        logger.info(f"\n{i}. {contrib.headline}")
        logger.info(f"   Event Type: {contrib.event_type}")
        logger.info(f"   Date: {contrib.timestamp.strftime('%Y-%m-%d')}")
        logger.info(f"   Contribution Score: {contrib.contribution_score:.4f}")
        logger.info(f"   Attention Weight: {contrib.attention_weight:.3f}")
        logger.info(f"   Temporal Weight: {contrib.temporal_weight:.3f}")
        logger.info(f"   Sentiment: {contrib.sentiment:+.2f}")
    
    # Create visualization
    analyzer.visualize_event_contributions(
        contributions,
        'results/explanations/event_contributions_example.png'
    )
    
    logger.info("\n✅ Event contribution analysis shows:")
    logger.info("   • Which specific news events drove the prediction")
    logger.info("   • Relative importance of each event")
    logger.info("   • How attention and temporal weighting combine")


def example_attention_visualization():
    """
    Example 3: Word-level attention visualization
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Word-Level Attention Visualization")
    logger.info("="*70 + "\n")
    
    from src.explainability.advanced_explain import AttentionVisualizer
    
    # Initialize
    visualizer = AttentionVisualizer()
    
    # Mock article text
    article_text = """
    Apple Inc reported strong quarterly earnings that beat analyst expectations.
    Revenue grew 15 percent year-over-year driven by robust iPhone sales.
    The company also announced a new product line targeting enterprise customers.
    CEO Tim Cook expressed optimism about future growth prospects.
    Shares surged in after-hours trading following the positive results.
    """
    
    # Mock word-level attention weights (higher for important words)
    words = article_text.split()
    n_words = len(words)
    
    # Simulate realistic attention pattern
    # Higher attention on: earnings, beat, revenue, grew, iPhone, announced, product, surged
    base_attention = np.random.uniform(0.1, 0.3, n_words)
    
    important_word_indices = [
        i for i, word in enumerate(words) 
        if any(kw in word.lower() for kw in ['earnings', 'beat', 'revenue', 'grew', 
                                              'iphone', 'announced', 'product', 'surged'])
    ]
    
    for idx in important_word_indices:
        base_attention[idx] = np.random.uniform(0.7, 1.0)
    
    logger.info(f"Article length: {n_words} words")
    logger.info(f"High-attention words: {len(important_word_indices)}")
    logger.info(f"Average attention: {base_attention.mean():.3f}")
    
    # Create visualization
    visualizer.visualize_article_attention(
        article_text,
        base_attention,
        'results/explanations/word_attention_example.png',
        max_words=100
    )
    
    logger.info("\n✅ Word attention visualization shows:")
    logger.info("   • Which words the model focused on")
    logger.info("   • Brighter colors = higher attention")
    logger.info("   • Useful for debugging and trust")


def example_counterfactual_analysis():
    """
    Example 4: Counterfactual analysis
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Counterfactual Analysis")
    logger.info("="*70 + "\n")
    
    from src.explainability.advanced_explain import CounterfactualAnalyzer
    
    # Mock model
    class MockModel:
        def predict(self, X):
            # Simulate: removing certain features changes prediction
            if X[:, 0] == 0:  # First feature removed
                # Prediction changes from UP to NEUTRAL
                return np.array([[0.3, 0.5, 0.2]])
            else:
                # Original: strong UP prediction
                return np.array([[0.1, 0.2, 0.7]])
    
    # Initialize
    analyzer = CounterfactualAnalyzer(MockModel())
    
    # Mock features (1 sample, 10 features)
    original_features = np.random.randn(1, 10)
    
    # Analyze counterfactuals for each event
    event_names = [
        'Strong earnings report',
        'Product launch announcement',
        'Positive analyst rating',
        'CEO interview',
        'Market rumors'
    ]
    
    # Mock feature matrix
    features = np.random.randn(1, len(event_names))
    
    results = analyzer.analyze_all_counterfactuals(features, event_names)
    
    logger.info("Counterfactual Analysis Results:")
    logger.info("-" * 70)
    logger.info("Question: What if each event didn't happen?\n")
    
    for result in results:
        logger.info(f"Event: {result['event_name']}")
        logger.info(f"  Original: Class {result['original_class']} "
                   f"(confidence: {result['original_confidence']:.1%})")
        logger.info(f"  Without event: Class {result['counterfactual_class']} "
                   f"(confidence: {result['counterfactual_confidence']:.1%})")
        
        if result['prediction_changed']:
            logger.info(f"  ⚠️  PREDICTION CHANGED! This event is CRITICAL")
        else:
            logger.info(f"  ℹ️  Prediction unchanged (confidence change: {result['confidence_change']:+.1%})")
        
        logger.info(f"  Impact magnitude: {result['impact_magnitude']:.3f}\n")
    
    # Visualize
    analyzer.visualize_counterfactual_impact(
        results,
        'results/explanations/counterfactual_example.png',
        top_k=5
    )
    
    logger.info("✅ Counterfactual analysis shows:")
    logger.info("   • Which events are critical vs marginal")
    logger.info("   • Red bars = prediction would change without this event")
    logger.info("   • Blue bars = prediction stays same")


def example_complete_explanation_pipeline():
    """
    Example 5: Complete explanation for a single prediction
    """
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 5: Complete Explanation Pipeline")
    logger.info("="*70 + "\n")
    
    from src.explainability.advanced_explain import ExplainabilityPipeline
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.array([[0.1, 0.2, 0.7]])
    
    # Mock config
    config = {
        'explainability': {
            'lookback_days': 30,
            'top_k_events': 10
        }
    }
    
    # Initialize pipeline
    pipeline = ExplainabilityPipeline(MockModel(), config)
    
    # Mock prediction scenario
    prediction_date = datetime(2023, 6, 15)
    
    events = [
        {
            'id': 'evt1',
            'timestamp': prediction_date - timedelta(days=1),
            'event_type': 'earnings',
            'headline': 'Company reports record Q2 earnings',
            'sentiment': 0.9,
            'features': np.random.randn(10)
        },
        {
            'id': 'evt2',
            'timestamp': prediction_date - timedelta(days=3),
            'event_type': 'product_launch',
            'headline': 'Revolutionary new product announced',
            'sentiment': 0.7,
            'features': np.random.randn(10)
        },
        {
            'id': 'evt3',
            'timestamp': prediction_date - timedelta(days=7),
            'event_type': 'guidance',
            'headline': 'Strong forward guidance provided',
            'sentiment': 0.6,
            'features': np.random.randn(10)
        }
    ]
    
    features = np.random.randn(1, 50)
    prediction = np.array([[0.1, 0.2, 0.7]])
    actual_class = 2  # Prediction was correct
    
    # Generate explanation
    explanation = pipeline.explain_prediction(
        prediction_date, events, features, prediction, actual_class
    )
    
    # Display explanation
    logger.info("Complete Explanation:")
    logger.info("="*70)
    logger.info(f"Date: {explanation.prediction_date.strftime('%Y-%m-%d')}")
    logger.info(f"Predicted Class: {explanation.predicted_class} (UP)")
    logger.info(f"Confidence: {explanation.predicted_proba[0, explanation.predicted_class]:.1%}")
    logger.info(f"Actual Class: {explanation.actual_class} (UP)")
    logger.info(f"Result: ✅ CORRECT\n")
    
    logger.info("Top Contributing Events:")
    for i, event in enumerate(explanation.top_events[:3], 1):
        logger.info(f"  {i}. {event.headline}")
        logger.info(f"     Score: {event.contribution_score:.4f}\n")
    
    logger.info("Feature Importance:")
    for feature, importance in explanation.feature_importance.items():
        logger.info(f"  {feature:20s}: {importance:.3f}")
    
    if explanation.counterfactual_scenario:
        logger.info(f"\nCounterfactual: {explanation.counterfactual_scenario}")
        logger.info(f"  → Prediction would be: Class {explanation.counterfactual_prediction}")
    
    # Save explanation
    pipeline.save_explanation(explanation, 'example_prediction_20230615')
    
    logger.info("\n✅ Complete explanation includes:")
    logger.info("   • Top contributing news events")
    logger.info("   • Feature importance scores")
    logger.info("   • Temporal importance heatmap")
    logger.info("   • Counterfactual analysis")
    logger.info("   • All visualizations saved for paper")


def main():
    """Run all examples"""
    logger.info("="*70)
    logger.info("    Advanced Explainability - Examples")
    logger.info("    Temporal Heatmaps & Event Contribution Analysis")
    logger.info("="*70)
    
    os.makedirs('results/explanations', exist_ok=True)
    
    try:
        example_temporal_heatmap()
        example_event_contribution()
        example_attention_visualization()
        example_counterfactual_analysis()
        example_complete_explanation_pipeline()
        
        logger.info("\n" + "="*70)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        logger.info("\n🔍 Why Explainability Matters:")
        logger.info("  1. Builds trust in model predictions")
        logger.info("  2. Identifies which news actually matters")
        logger.info("  3. Debugs model behavior and biases")
        logger.info("  4. Provides actionable insights for traders")
        logger.info("  5. Essential for regulatory compliance")
        
        logger.info("\n📊 For Your Research Paper:")
        logger.info("  • Include temporal heatmaps to show dynamic importance")
        logger.info("  • Show specific examples of event contributions")
        logger.info("  • Use counterfactual analysis to prove causality")
        logger.info("  • Demonstrate model focuses on meaningful signals")
        logger.info("  • Compare attention patterns across different scenarios")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
