"""
Temporal Relevance and Event Weighting System
Implements time-decay functions and contextual event importance scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalRelevanceCalculator:
    """Calculates temporal relevance of news articles"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.decay_function = config.get('temporal', {}).get('decay_function', 'exponential')
        self.half_life_days = config.get('temporal', {}).get('half_life_days', 3)
        self.max_history_days = config.get('temporal', {}).get('max_history_days', 30)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def exponential_decay(self, days_elapsed: float) -> float:
        """
        Exponential decay function
        
        Args:
            days_elapsed: Number of days since publication
            
        Returns:
            Decay weight (0 to 1)
        """
        if days_elapsed > self.max_history_days:
            return 0.0
        
        # Exponential decay: weight = 0.5^(days / half_life)
        decay_weight = 0.5 ** (days_elapsed / self.half_life_days)
        return decay_weight
    
    def linear_decay(self, days_elapsed: float) -> float:
        """
        Linear decay function
        
        Args:
            days_elapsed: Number of days since publication
            
        Returns:
            Decay weight (0 to 1)
        """
        if days_elapsed > self.max_history_days:
            return 0.0
        
        # Linear decay
        decay_weight = 1.0 - (days_elapsed / self.max_history_days)
        return max(0.0, decay_weight)
    
    def step_decay(self, days_elapsed: float) -> float:
        """
        Step decay function (categorical time windows)
        
        Args:
            days_elapsed: Number of days since publication
            
        Returns:
            Decay weight (0 to 1)
        """
        if days_elapsed <= 1:
            return 1.0
        elif days_elapsed <= 3:
            return 0.8
        elif days_elapsed <= 7:
            return 0.5
        elif days_elapsed <= 14:
            return 0.3
        elif days_elapsed <= 30:
            return 0.1
        else:
            return 0.0
    
    def calculate_temporal_weight(self, published_date: datetime, reference_date: datetime) -> float:
        """
        Calculate temporal weight for a news article
        
        Args:
            published_date: When the news was published
            reference_date: Reference date (usually trading date)
            
        Returns:
            Temporal weight
        """
        days_elapsed = (reference_date - published_date).total_seconds() / 86400
        
        if days_elapsed < 0:
            # Future news (shouldn't happen, but handle gracefully)
            return 0.0
        
        if self.decay_function == 'exponential':
            return self.exponential_decay(days_elapsed)
        elif self.decay_function == 'linear':
            return self.linear_decay(days_elapsed)
        elif self.decay_function == 'step':
            return self.step_decay(days_elapsed)
        else:
            return self.exponential_decay(days_elapsed)
    
    def add_temporal_weights(
        self, 
        news_df: pd.DataFrame, 
        reference_date_col: str = 'reference_date',
        published_date_col: str = 'published_at'
    ) -> pd.DataFrame:
        """
        Add temporal weights to news DataFrame
        
        Args:
            news_df: DataFrame with news
            reference_date_col: Column with reference dates
            published_date_col: Column with publication dates
            
        Returns:
            DataFrame with temporal_weight column
        """
        news_df['temporal_weight'] = news_df.apply(
            lambda row: self.calculate_temporal_weight(
                row[published_date_col],
                row[reference_date_col]
            ),
            axis=1
        )
        
        return news_df


class EventImportanceScorer:
    """Scores news importance based on event types and context"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.event_multipliers = config.get('temporal', {}).get(
            'event_weight_multipliers', 
            {
                'earnings': 2.0,
                'merger': 2.5,
                'acquisition': 2.5,
                'partnership': 1.5,
                'lawsuit': 1.8,
                'product_launch': 1.3,
                'regulatory': 1.7
            }
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_event_importance(self, events: Dict[str, bool]) -> float:
        """
        Calculate importance score based on detected events
        
        Args:
            events: Dictionary of event types and their presence
            
        Returns:
            Importance multiplier (>= 1.0)
        """
        importance = 1.0
        
        for event_type, is_present in events.items():
            if is_present and event_type in self.event_multipliers:
                # Use maximum multiplier if multiple events
                importance = max(importance, self.event_multipliers[event_type])
        
        return importance
    
    def calculate_sentiment_strength(
        self, 
        sentiment_score: float, 
        sentiment_confidence: float = None
    ) -> float:
        """
        Calculate sentiment strength weight
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            sentiment_confidence: Confidence in sentiment (0 to 1)
            
        Returns:
            Sentiment strength multiplier
        """
        # Strong sentiment (positive or negative) is more impactful
        strength = abs(sentiment_score)
        
        if sentiment_confidence is not None:
            # Weight by confidence
            strength *= sentiment_confidence
        
        # Scale to multiplier range (1.0 to 2.0)
        return 1.0 + strength
    
    def calculate_source_credibility(self, source: str) -> float:
        """
        Calculate credibility weight based on news source
        
        Args:
            source: News source name
            
        Returns:
            Credibility multiplier (0.5 to 2.0)
        """
        # High credibility sources
        high_credibility = {
            'reuters', 'bloomberg', 'wall street journal', 'wsj',
            'financial times', 'ft', 'cnbc', 'seeking alpha'
        }
        
        # Medium credibility
        medium_credibility = {
            'yahoo finance', 'marketwatch', 'forbes', 'benzinga'
        }
        
        source_lower = source.lower()
        
        if any(hc in source_lower for hc in high_credibility):
            return 1.5
        elif any(mc in source_lower for mc in medium_credibility):
            return 1.2
        else:
            return 1.0
    
    def add_importance_scores(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add importance scores to news DataFrame
        
        Args:
            news_df: DataFrame with news and events
            
        Returns:
            DataFrame with importance_score column
        """
        # Calculate event importance
        news_df['event_importance'] = news_df['events'].apply(
            self.calculate_event_importance
        )
        
        # Calculate sentiment strength
        news_df['sentiment_strength'] = news_df['sentiment_score'].apply(
            self.calculate_sentiment_strength
        )
        
        # Calculate source credibility
        news_df['source_credibility'] = news_df['source'].apply(
            self.calculate_source_credibility
        )
        
        # Combined importance score
        news_df['importance_score'] = (
            news_df['event_importance'] * 
            news_df['sentiment_strength'] * 
            news_df['source_credibility']
        )
        
        # Normalize to 0-1 range for easier interpretation
        max_score = news_df['importance_score'].max()
        if max_score > 0:
            news_df['importance_score_normalized'] = (
                news_df['importance_score'] / max_score
            )
        else:
            news_df['importance_score_normalized'] = 1.0
        
        return news_df


class NewsAggregator:
    """Aggregates news features for time-series alignment"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def aggregate_news_for_dates(
        self, 
        news_df: pd.DataFrame,
        stock_dates: pd.DatetimeIndex,
        ticker: str
    ) -> pd.DataFrame:
        """
        Aggregate news features for each trading date
        
        Args:
            news_df: DataFrame with processed news
            stock_dates: Trading dates to aggregate for
            ticker: Stock ticker
            
        Returns:
            DataFrame with aggregated news features per date
        """
        aggregated_features = []
        
        for date in stock_dates:
            # Get news up to this date with temporal decay
            news_df['reference_date'] = date
            
            # Calculate temporal weights
            temporal_calc = TemporalRelevanceCalculator(self.config)
            news_with_weights = temporal_calc.add_temporal_weights(news_df.copy())
            
            # Filter to relevant time window
            relevant_news = news_with_weights[
                news_with_weights['temporal_weight'] > 0
            ].copy()
            
            if len(relevant_news) == 0:
                # No news for this date
                features = self._create_empty_features(date, ticker)
            else:
                # Calculate weighted features
                features = self._aggregate_weighted_features(
                    relevant_news, date, ticker
                )
            
            aggregated_features.append(features)
        
        return pd.DataFrame(aggregated_features)
    
    def _create_empty_features(self, date: datetime, ticker: str) -> Dict:
        """Create empty feature dict when no news available"""
        return {
            'date': date,
            'ticker': ticker,
            'news_count': 0,
            'avg_sentiment': 0.0,
            'avg_sentiment_positive': 0.0,
            'avg_sentiment_negative': 0.0,
            'avg_sentiment_neutral': 0.0,
            'weighted_sentiment': 0.0,
            'importance_score': 0.0,
            'event_earnings': 0,
            'event_merger': 0,
            'event_partnership': 0,
            'event_lawsuit': 0,
            'event_product_launch': 0,
            'event_regulatory': 0,
            'embedding_mean': np.zeros(768)  # FinBERT dimension
        }
    
    def _aggregate_weighted_features(
        self, 
        news_df: pd.DataFrame, 
        date: datetime, 
        ticker: str
    ) -> Dict:
        """Aggregate features with temporal and importance weighting"""
        
        # Combined weight: temporal * importance
        news_df['combined_weight'] = (
            news_df['temporal_weight'] * 
            news_df.get('importance_score_normalized', 1.0)
        )
        
        # Normalize weights
        total_weight = news_df['combined_weight'].sum()
        if total_weight > 0:
            news_df['normalized_weight'] = (
                news_df['combined_weight'] / total_weight
            )
        else:
            news_df['normalized_weight'] = 1.0 / len(news_df)
        
        # Weighted sentiment
        weighted_sentiment = (
            news_df['sentiment_score'] * news_df['normalized_weight']
        ).sum()
        
        # Event counts
        event_counts = {}
        for event_type in ['earnings', 'merger', 'partnership', 'lawsuit', 
                          'product_launch', 'regulatory']:
            event_counts[f'event_{event_type}'] = news_df['events'].apply(
                lambda x: x.get(event_type, False)
            ).sum()
        
        # Weighted average embedding
        embeddings = np.array(news_df['embedding'].tolist())
        weights = news_df['normalized_weight'].values.reshape(-1, 1)
        weighted_embedding = (embeddings * weights).sum(axis=0)
        
        features = {
            'date': date,
            'ticker': ticker,
            'news_count': len(news_df),
            'avg_sentiment': news_df['sentiment_score'].mean(),
            'avg_sentiment_positive': news_df['sentiment_positive'].mean(),
            'avg_sentiment_negative': news_df['sentiment_negative'].mean(),
            'avg_sentiment_neutral': news_df['sentiment_neutral'].mean(),
            'weighted_sentiment': weighted_sentiment,
            'importance_score': news_df.get('importance_score_normalized', pd.Series([0])).mean(),
            **event_counts,
            'embedding_mean': weighted_embedding
        }
        
        return features


class TemporalFeatureEngineer:
    """Main class for temporal feature engineering"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.temporal_calculator = TemporalRelevanceCalculator(config)
        self.importance_scorer = EventImportanceScorer(config)
        self.aggregator = NewsAggregator(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_news_with_temporal_features(
        self, 
        news_df: pd.DataFrame,
        stock_df: pd.DataFrame,
        ticker: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete temporal feature engineering pipeline
        
        Args:
            news_df: DataFrame with processed news (with NLP features)
            stock_df: DataFrame with stock prices
            ticker: Stock ticker
            
        Returns:
            Tuple of (enhanced_news_df, aggregated_features_df)
        """
        self.logger.info(f"Processing temporal features for {ticker}")
        
        # Ensure datetime types
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        # Add importance scores
        news_df = self.importance_scorer.add_importance_scores(news_df)
        
        # Aggregate for each trading date
        aggregated_df = self.aggregator.aggregate_news_for_dates(
            news_df,
            stock_df['Date'],
            ticker
        )
        
        self.logger.info(f"Created {len(aggregated_df)} aggregated feature records")
        
        return news_df, aggregated_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'temporal': {
            'decay_function': 'exponential',
            'half_life_days': 3,
            'max_history_days': 30,
            'event_weight_multipliers': {
                'earnings': 2.0,
                'merger': 2.5
            }
        }
    }
    
    engineer = TemporalFeatureEngineer(config)
    print("Temporal feature engineer initialized")
