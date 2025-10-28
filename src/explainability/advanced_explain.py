"""
Advanced Explainability Module
Provides temporal heatmaps, event contribution analysis, and counterfactual explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EventContribution:
    """Contribution of a single event to prediction"""
    event_id: str
    timestamp: datetime
    event_type: str
    headline: str
    contribution_score: float
    attention_weight: float
    temporal_weight: float
    sentiment: float


@dataclass
class TemporalExplanation:
    """Explanation for a single prediction with temporal context"""
    prediction_date: datetime
    predicted_class: int
    predicted_proba: np.ndarray
    actual_class: Optional[int]
    
    # Event contributions
    top_events: List[EventContribution]
    
    # Feature importance
    feature_importance: Dict[str, float]
    
    # Temporal analysis
    temporal_importance: np.ndarray  # Importance over time window
    
    # Counterfactual
    counterfactual_prediction: Optional[int] = None
    counterfactual_scenario: Optional[str] = None


class TemporalHeatmapGenerator:
    """
    Generates temporal heatmaps showing how feature importance changes over time
    """
    
    def __init__(self, lookback_days: int = 30):
        """
        Initialize temporal heatmap generator
        
        Args:
            lookback_days: Number of days to look back for temporal analysis
        """
        self.lookback_days = lookback_days
        logger.info(f"Initialized TemporalHeatmapGenerator (lookback={lookback_days} days)")
    
    def compute_temporal_importance(
        self,
        events: List[Dict],
        prediction_date: datetime,
        model_attention_weights: np.ndarray,
        temporal_decay_fn: callable
    ) -> np.ndarray:
        """
        Compute feature importance over time window
        
        Args:
            events: List of news events with timestamps
            prediction_date: Date of prediction
            model_attention_weights: Attention weights from model
            temporal_decay_fn: Function to compute temporal decay
            
        Returns:
            Array of shape (n_days, n_features) with importance scores
        """
        # Create time bins
        time_bins = []
        for i in range(self.lookback_days):
            bin_date = prediction_date - timedelta(days=i)
            time_bins.append(bin_date)
        
        time_bins = list(reversed(time_bins))
        
        # Initialize importance matrix
        n_features = len(events[0].get('features', [])) if events else 10
        importance_matrix = np.zeros((self.lookback_days, n_features))
        
        # Compute importance for each event
        for event_idx, event in enumerate(events):
            event_date = event.get('timestamp', prediction_date)
            
            # Find time bin
            days_diff = (prediction_date - event_date).days
            if 0 <= days_diff < self.lookback_days:
                bin_idx = self.lookback_days - 1 - days_diff
                
                # Get attention weight for this event
                if event_idx < len(model_attention_weights):
                    attention = model_attention_weights[event_idx]
                else:
                    attention = 0.0
                
                # Apply temporal decay
                temporal_weight = temporal_decay_fn(days_diff)
                
                # Update importance
                event_features = event.get('features', np.zeros(n_features))
                importance_matrix[bin_idx, :] += attention * temporal_weight * np.abs(event_features)
        
        return importance_matrix
    
    def visualize_temporal_heatmap(
        self,
        importance_matrix: np.ndarray,
        feature_names: List[str],
        prediction_date: datetime,
        output_path: str
    ):
        """
        Create temporal heatmap visualization
        
        Args:
            importance_matrix: Matrix of shape (n_days, n_features)
            feature_names: Names of features
            prediction_date: Date of prediction
            output_path: Path to save visualization
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create date labels
        date_labels = []
        for i in range(self.lookback_days):
            date = prediction_date - timedelta(days=self.lookback_days - 1 - i)
            date_labels.append(date.strftime('%m/%d'))
        
        # Create heatmap
        sns.heatmap(
            importance_matrix.T,
            cmap='YlOrRd',
            xticklabels=date_labels,
            yticklabels=feature_names,
            cbar_kws={'label': 'Importance Score'},
            ax=ax,
            vmin=0,
            vmax=np.percentile(importance_matrix, 95)  # Cap at 95th percentile
        )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(f'Temporal Feature Importance Heatmap\nPrediction Date: {prediction_date.strftime("%Y-%m-%d")}',
                    fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal heatmap to {output_path}")
        plt.close()


class EventContributionAnalyzer:
    """
    Analyzes which specific news events contributed most to predictions
    """
    
    def __init__(self, top_k: int = 10):
        """
        Initialize event contribution analyzer
        
        Args:
            top_k: Number of top events to analyze
        """
        self.top_k = top_k
        logger.info(f"Initialized EventContributionAnalyzer (top_k={top_k})")
    
    def analyze_event_contributions(
        self,
        events: List[Dict],
        attention_weights: np.ndarray,
        temporal_weights: np.ndarray,
        prediction: np.ndarray
    ) -> List[EventContribution]:
        """
        Analyze contribution of each event to the prediction
        
        Args:
            events: List of news events
            attention_weights: Attention weights from model (n_events,)
            temporal_weights: Temporal decay weights (n_events,)
            prediction: Model prediction probabilities (n_classes,)
            
        Returns:
            List of EventContribution objects sorted by importance
        """
        contributions = []
        
        for i, event in enumerate(events):
            # Get weights
            attention = attention_weights[i] if i < len(attention_weights) else 0.0
            temporal = temporal_weights[i] if i < len(temporal_weights) else 0.0
            
            # Compute contribution score (attention * temporal * prediction confidence)
            contribution_score = attention * temporal * np.max(prediction)
            
            contribution = EventContribution(
                event_id=event.get('id', f'event_{i}'),
                timestamp=event.get('timestamp', datetime.now()),
                event_type=event.get('event_type', 'unknown'),
                headline=event.get('headline', 'Unknown headline'),
                contribution_score=contribution_score,
                attention_weight=attention,
                temporal_weight=temporal,
                sentiment=event.get('sentiment', 0.0)
            )
            
            contributions.append(contribution)
        
        # Sort by contribution score
        contributions.sort(key=lambda x: x.contribution_score, reverse=True)
        
        return contributions[:self.top_k]
    
    def visualize_event_contributions(
        self,
        contributions: List[EventContribution],
        output_path: str
    ):
        """
        Visualize top event contributions
        
        Args:
            contributions: List of EventContribution objects
            output_path: Path to save visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Contribution scores
        headlines = [c.headline[:40] + '...' if len(c.headline) > 40 else c.headline 
                    for c in contributions]
        scores = [c.contribution_score for c in contributions]
        
        colors = ['green' if c.sentiment > 0 else 'red' if c.sentiment < 0 else 'gray' 
                 for c in contributions]
        
        ax1.barh(range(len(headlines)), scores, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(headlines)))
        ax1.set_yticklabels(headlines, fontsize=9)
        ax1.set_xlabel('Contribution Score', fontsize=11)
        ax1.set_title('Top Contributing News Events', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Attention vs Temporal weights
        attention_weights = [c.attention_weight for c in contributions]
        temporal_weights = [c.temporal_weight for c in contributions]
        
        x = np.arange(len(contributions))
        width = 0.35
        
        ax2.bar(x - width/2, attention_weights, width, label='Attention Weight', alpha=0.8, color='steelblue')
        ax2.bar(x + width/2, temporal_weights, width, label='Temporal Weight', alpha=0.8, color='coral')
        
        ax2.set_xlabel('Event Rank', fontsize=11)
        ax2.set_ylabel('Weight Value', fontsize=11)
        ax2.set_title('Attention vs Temporal Weighting', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'#{i+1}' for i in range(len(contributions))])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved event contributions to {output_path}")
        plt.close()


class AttentionVisualizer:
    """
    Visualizes attention weights for individual news articles
    """
    
    def __init__(self):
        logger.info("Initialized AttentionVisualizer")
    
    def visualize_article_attention(
        self,
        article_text: str,
        word_attention_weights: np.ndarray,
        output_path: str,
        max_words: int = 100
    ):
        """
        Visualize attention weights for words in an article
        
        Args:
            article_text: Full article text
            word_attention_weights: Attention weight for each word
            output_path: Path to save visualization
            max_words: Maximum number of words to display
        """
        # Split into words
        words = article_text.split()[:max_words]
        weights = word_attention_weights[:len(words)]
        
        # Normalize weights for visualization
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create word positions
        n_words_per_row = 10
        n_rows = (len(words) + n_words_per_row - 1) // n_words_per_row
        
        for i, (word, weight) in enumerate(zip(words, weights_norm)):
            row = i // n_words_per_row
            col = i % n_words_per_row
            
            # Color based on attention weight
            color = plt.cm.YlOrRd(weight)
            
            # Add word with background color
            ax.text(
                col, n_rows - row - 1, word,
                ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='none')
            )
        
        ax.set_xlim(-0.5, n_words_per_row - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.axis('off')
        ax.set_title('Word-Level Attention Weights\n(Brighter = Higher Attention)',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_label('Attention Weight', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved attention visualization to {output_path}")
        plt.close()
    
    def create_attention_heatmap(
        self,
        articles: List[Dict],
        article_attention_weights: np.ndarray,
        output_path: str
    ):
        """
        Create heatmap showing attention across multiple articles
        
        Args:
            articles: List of article dictionaries
            article_attention_weights: Attention weight for each article
            output_path: Path to save visualization
        """
        fig, ax = plt.subplots(figsize=(12, max(6, len(articles) * 0.4)))
        
        # Prepare data
        headlines = [art.get('headline', 'Unknown')[:60] for art in articles]
        dates = [art.get('timestamp', datetime.now()).strftime('%m/%d') for art in articles]
        
        # Create data matrix (1 column per article)
        data = article_attention_weights.reshape(-1, 1)
        
        # Create labels with date and headline
        labels = [f"{date}: {headline}" for date, headline in zip(dates, headlines)]
        
        # Create heatmap
        sns.heatmap(
            data,
            cmap='YlOrRd',
            yticklabels=labels,
            xticklabels=['Attention'],
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            vmin=0,
            vmax=article_attention_weights.max()
        )
        
        ax.set_title('Article-Level Attention Weights', fontsize=14, fontweight='bold')
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved article attention heatmap to {output_path}")
        plt.close()


class CounterfactualAnalyzer:
    """
    Performs counterfactual analysis: "What if this news didn't exist?"
    """
    
    def __init__(self, model):
        """
        Initialize counterfactual analyzer
        
        Args:
            model: Trained model for predictions
        """
        self.model = model
        logger.info("Initialized CounterfactualAnalyzer")
    
    def analyze_counterfactual(
        self,
        original_features: np.ndarray,
        removed_event_idx: int,
        event_name: str
    ) -> Dict[str, Any]:
        """
        Analyze prediction change when removing a specific event
        
        Args:
            original_features: Original feature matrix
            removed_event_idx: Index of event to remove
            event_name: Name of the event
            
        Returns:
            Dictionary with counterfactual analysis results
        """
        # Original prediction
        original_pred = self.model.predict(original_features)
        original_class = np.argmax(original_pred)
        original_confidence = original_pred[0, original_class]
        
        # Create counterfactual features (zero out the removed event)
        counterfactual_features = original_features.copy()
        counterfactual_features[:, removed_event_idx] = 0
        
        # Counterfactual prediction
        counterfactual_pred = self.model.predict(counterfactual_features)
        counterfactual_class = np.argmax(counterfactual_pred)
        counterfactual_confidence = counterfactual_pred[0, counterfactual_class]
        
        # Compute impact
        prediction_changed = (original_class != counterfactual_class)
        confidence_change = counterfactual_confidence - original_confidence
        
        result = {
            'event_name': event_name,
            'original_class': int(original_class),
            'original_confidence': float(original_confidence),
            'counterfactual_class': int(counterfactual_class),
            'counterfactual_confidence': float(counterfactual_confidence),
            'prediction_changed': prediction_changed,
            'confidence_change': float(confidence_change),
            'impact_magnitude': abs(confidence_change)
        }
        
        return result
    
    def analyze_all_counterfactuals(
        self,
        features: np.ndarray,
        event_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze counterfactuals for all events
        
        Args:
            features: Feature matrix
            event_names: Names of all events
            
        Returns:
            List of counterfactual analysis results
        """
        results = []
        
        for i, event_name in enumerate(event_names):
            result = self.analyze_counterfactual(features, i, event_name)
            results.append(result)
        
        # Sort by impact magnitude
        results.sort(key=lambda x: x['impact_magnitude'], reverse=True)
        
        return results
    
    def visualize_counterfactual_impact(
        self,
        counterfactual_results: List[Dict[str, Any]],
        output_path: str,
        top_k: int = 10
    ):
        """
        Visualize counterfactual impact
        
        Args:
            counterfactual_results: List of counterfactual analysis results
            output_path: Path to save visualization
            top_k: Number of top events to show
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top k most impactful events
        top_results = counterfactual_results[:top_k]
        
        event_names = [r['event_name'][:40] + '...' if len(r['event_name']) > 40 
                      else r['event_name'] for r in top_results]
        impacts = [r['confidence_change'] for r in top_results]
        changed = [r['prediction_changed'] for r in top_results]
        
        # Color: red if prediction changed, blue otherwise
        colors = ['red' if c else 'steelblue' for c in changed]
        
        ax.barh(range(len(event_names)), impacts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(event_names)))
        ax.set_yticklabels(event_names, fontsize=9)
        ax.set_xlabel('Confidence Change (Counterfactual - Original)', fontsize=11)
        ax.set_title('Counterfactual Impact: "What if this event didn\'t exist?"',
                    fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Prediction Changed'),
            Patch(facecolor='steelblue', alpha=0.7, label='Prediction Same')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved counterfactual impact to {output_path}")
        plt.close()


class ExplainabilityPipeline:
    """
    Complete explainability pipeline integrating all analysis components
    """
    
    def __init__(
        self,
        model,
        config: Dict,
        output_dir: str = 'results/explanations'
    ):
        """
        Initialize explainability pipeline
        
        Args:
            model: Trained model
            config: Configuration dictionary
            output_dir: Directory to save explanations
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        
        # Initialize components
        self.heatmap_generator = TemporalHeatmapGenerator(
            lookback_days=config.get('explainability', {}).get('lookback_days', 30)
        )
        self.event_analyzer = EventContributionAnalyzer(
            top_k=config.get('explainability', {}).get('top_k_events', 10)
        )
        self.attention_visualizer = AttentionVisualizer()
        self.counterfactual_analyzer = CounterfactualAnalyzer(model)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized ExplainabilityPipeline (output_dir={output_dir})")
    
    def explain_prediction(
        self,
        prediction_date: datetime,
        events: List[Dict],
        features: np.ndarray,
        prediction: np.ndarray,
        actual_class: Optional[int] = None
    ) -> TemporalExplanation:
        """
        Generate complete explanation for a prediction
        
        Args:
            prediction_date: Date of prediction
            events: List of news events
            features: Feature matrix
            prediction: Model prediction
            actual_class: Actual class (if available)
            
        Returns:
            TemporalExplanation object
        """
        logger.info(f"Generating explanation for prediction on {prediction_date}")
        
        # Get model internals (mock for now - replace with actual model outputs)
        attention_weights = np.random.dirichlet(np.ones(len(events)))
        temporal_weights = np.array([
            np.exp(-0.3 * (prediction_date - e.get('timestamp', prediction_date)).days)
            for e in events
        ])
        
        # Analyze event contributions
        top_events = self.event_analyzer.analyze_event_contributions(
            events, attention_weights, temporal_weights, prediction
        )
        
        # Compute feature importance (mock)
        feature_importance = {
            'sentiment': np.random.uniform(0.2, 0.4),
            'volume': np.random.uniform(0.1, 0.3),
            'price_momentum': np.random.uniform(0.15, 0.35),
            'event_importance': np.random.uniform(0.2, 0.4)
        }
        
        # Compute temporal importance
        temporal_decay_fn = lambda days: np.exp(-0.3 * days)
        temporal_importance = self.heatmap_generator.compute_temporal_importance(
            events, prediction_date, attention_weights, temporal_decay_fn
        )
        
        # Counterfactual analysis (top event)
        counterfactual_pred = None
        counterfactual_scenario = None
        if len(top_events) > 0:
            cf_result = self.counterfactual_analyzer.analyze_counterfactual(
                features, 0, top_events[0].headline
            )
            counterfactual_pred = cf_result['counterfactual_class']
            counterfactual_scenario = f"Without '{top_events[0].headline[:50]}...'"
        
        explanation = TemporalExplanation(
            prediction_date=prediction_date,
            predicted_class=int(np.argmax(prediction)),
            predicted_proba=prediction,
            actual_class=actual_class,
            top_events=top_events,
            feature_importance=feature_importance,
            temporal_importance=temporal_importance,
            counterfactual_prediction=counterfactual_pred,
            counterfactual_scenario=counterfactual_scenario
        )
        
        return explanation
    
    def save_explanation(
        self,
        explanation: TemporalExplanation,
        prediction_id: str
    ):
        """
        Save explanation to disk with visualizations
        
        Args:
            explanation: TemporalExplanation object
            prediction_id: Unique identifier for this prediction
        """
        pred_dir = os.path.join(self.output_dir, prediction_id)
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save JSON summary
        summary = {
            'prediction_date': explanation.prediction_date.isoformat(),
            'predicted_class': explanation.predicted_class,
            'predicted_proba': explanation.predicted_proba.tolist(),
            'actual_class': explanation.actual_class,
            'top_events': [asdict(e) for e in explanation.top_events],
            'feature_importance': explanation.feature_importance,
            'counterfactual_prediction': explanation.counterfactual_prediction,
            'counterfactual_scenario': explanation.counterfactual_scenario
        }
        
        json_path = os.path.join(pred_dir, 'explanation.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved explanation JSON to {json_path}")
        
        # Create visualizations
        feature_names = list(explanation.feature_importance.keys())
        
        # Temporal heatmap
        heatmap_path = os.path.join(pred_dir, 'temporal_heatmap.png')
        self.heatmap_generator.visualize_temporal_heatmap(
            explanation.temporal_importance,
            feature_names,
            explanation.prediction_date,
            heatmap_path
        )
        
        # Event contributions
        if explanation.top_events:
            events_path = os.path.join(pred_dir, 'event_contributions.png')
            self.event_analyzer.visualize_event_contributions(
                explanation.top_events,
                events_path
            )
        
        logger.info(f"Saved all explanations to {pred_dir}")


if __name__ == "__main__":
    # Mock usage example
    logger.info("Explainability Pipeline - Example Usage\n")
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.random.dirichlet(np.ones(3), size=1)
    
    # Mock config
    config = {
        'explainability': {
            'lookback_days': 30,
            'top_k_events': 10
        }
    }
    
    # Create pipeline
    pipeline = ExplainabilityPipeline(MockModel(), config)
    
    # Mock data
    prediction_date = datetime(2023, 6, 15)
    events = [
        {
            'id': 'evt1',
            'timestamp': prediction_date - timedelta(days=1),
            'event_type': 'earnings',
            'headline': 'Company reports strong Q2 earnings',
            'sentiment': 0.8,
            'features': np.random.randn(10)
        },
        {
            'id': 'evt2',
            'timestamp': prediction_date - timedelta(days=3),
            'event_type': 'product_launch',
            'headline': 'New product launch announced',
            'sentiment': 0.6,
            'features': np.random.randn(10)
        }
    ]
    
    features = np.random.randn(1, 50)
    prediction = np.array([[0.1, 0.2, 0.7]])  # UP with 70% confidence
    
    # Generate explanation
    explanation = pipeline.explain_prediction(
        prediction_date, events, features, prediction, actual_class=2
    )
    
    # Save explanation
    pipeline.save_explanation(explanation, 'example_prediction')
    
    logger.info("\n✅ Explainability pipeline demo complete!")
    logger.info("   Check results/explanations/example_prediction/")
