"""
Temporal Data Utilities
Handles strict time-based splitting, timestamp alignment, and lookahead prevention
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalDataSplitter:
    """
    Enforces strict temporal ordering in data splits to prevent lookahead bias
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def temporal_train_val_test_split(
        self,
        data: pd.DataFrame,
        date_column: str = 'Date',
        train_end: str = '2021-12-31',
        val_end: str = '2022-12-31',
        test_end: str = '2023-12-31'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test based on strict time boundaries
        
        CRITICAL: This prevents lookahead bias by ensuring:
        - Training data comes ONLY from past
        - No overlap between splits
        - Temporal ordering is preserved
        
        Args:
            data: DataFrame with temporal data
            date_column: Column containing dates
            train_end: End date for training (inclusive)
            val_end: End date for validation (inclusive)
            test_end: End date for testing (inclusive)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure datetime type
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date (critical for temporal integrity)
        data = data.sort_values(date_column).reset_index(drop=True)
        
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)
        test_end_dt = pd.to_datetime(test_end)
        
        # Split with strict boundaries
        train_df = data[data[date_column] <= train_end_dt].copy()
        val_df = data[
            (data[date_column] > train_end_dt) & 
            (data[date_column] <= val_end_dt)
        ].copy()
        test_df = data[
            (data[date_column] > val_end_dt) & 
            (data[date_column] <= test_end_dt)
        ].copy()
        
        self.logger.info(f"Temporal split results:")
        self.logger.info(f"  Train: {len(train_df)} samples ({train_df[date_column].min()} to {train_df[date_column].max()})")
        self.logger.info(f"  Val:   {len(val_df)} samples ({val_df[date_column].min()} to {val_df[date_column].max()})")
        self.logger.info(f"  Test:  {len(test_df)} samples ({test_df[date_column].min()} to {test_df[date_column].max()})")
        
        # Validation checks
        self._validate_temporal_split(train_df, val_df, test_df, date_column)
        
        return train_df, val_df, test_df
    
    def _validate_temporal_split(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_column: str
    ):
        """
        Verify no temporal contamination between splits
        """
        train_max = train_df[date_column].max()
        val_min = val_df[date_column].min()
        val_max = val_df[date_column].max()
        test_min = test_df[date_column].min()
        
        assert train_max < val_min, f"Temporal contamination: train_max ({train_max}) >= val_min ({val_min})"
        assert val_max < test_min, f"Temporal contamination: val_max ({val_max}) >= test_min ({test_min})"
        
        self.logger.info("✓ Temporal split validation passed - no lookahead bias")


class NewsTimestampAligner:
    """
    Ensures news articles are only used AFTER their publication date
    Prevents critical lookahead bias in financial prediction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def filter_news_by_timestamp(
        self,
        news_df: pd.DataFrame,
        target_date: datetime,
        published_col: str = 'published_at',
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Filter news to only those published BEFORE target date
        
        CRITICAL LOOKAHEAD PREVENTION:
        - Only news published strictly before target_date is used
        - Lookback window prevents using news from distant past
        
        Args:
            news_df: DataFrame with news articles
            target_date: Prediction target date
            published_col: Column with publication timestamps
            lookback_days: Maximum days to look back
            
        Returns:
            Filtered news DataFrame
        """
        news_df[published_col] = pd.to_datetime(news_df[published_col])
        
        # News must be published BEFORE target date
        # (strictly less than, not less than or equal)
        cutoff_start = target_date - timedelta(days=lookback_days)
        
        filtered = news_df[
            (news_df[published_col] < target_date) &  # STRICT: before target
            (news_df[published_col] >= cutoff_start)   # Within lookback window
        ].copy()
        
        return filtered
    
    def align_news_to_trading_days(
        self,
        news_df: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
        published_col: str = 'published_at',
        lookback_days: int = 30
    ) -> Dict[datetime, pd.DataFrame]:
        """
        For each trading day, return ONLY the news available before that day
        
        This is the GOLD STANDARD for preventing lookahead bias
        
        Args:
            news_df: DataFrame with all news
            trading_dates: Trading days to predict
            published_col: Publication date column
            lookback_days: Lookback window
            
        Returns:
            Dictionary mapping each trading day to its available news
        """
        aligned_news = {}
        
        for target_date in trading_dates:
            available_news = self.filter_news_by_timestamp(
                news_df,
                target_date,
                published_col,
                lookback_days
            )
            aligned_news[target_date] = available_news
        
        self.logger.info(f"Aligned news for {len(trading_dates)} trading days")
        self.logger.info(f"Average news per day: {np.mean([len(v) for v in aligned_news.values()]):.1f}")
        
        return aligned_news
    
    def verify_no_lookahead(
        self,
        aligned_news: Dict[datetime, pd.DataFrame],
        published_col: str = 'published_at'
    ) -> bool:
        """
        Verify no news article is used before it was published
        
        Returns:
            True if no lookahead detected, raises AssertionError otherwise
        """
        violations = []
        
        for target_date, news_subset in aligned_news.items():
            if len(news_subset) > 0:
                max_pub_date = news_subset[published_col].max()
                
                if max_pub_date >= target_date:
                    violations.append({
                        'target_date': target_date,
                        'max_pub_date': max_pub_date,
                        'violation_count': (news_subset[published_col] >= target_date).sum()
                    })
        
        if violations:
            self.logger.error(f"LOOKAHEAD BIAS DETECTED: {len(violations)} violations")
            for v in violations[:5]:  # Show first 5
                self.logger.error(f"  Target: {v['target_date']}, Max Pub: {v['max_pub_date']}")
            raise AssertionError("Lookahead bias detected! News used before publication.")
        
        self.logger.info("✓ No lookahead bias detected - all news properly timestamped")
        return True


class SequenceGenerator:
    """
    Generates sequences with strict temporal ordering
    """
    
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_temporal_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dates: pd.DatetimeIndex,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences ensuring temporal ordering
        
        IMPORTANT: Sequences are created in order - past predicts future
        
        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,)
            dates: Corresponding dates
            sequence_length: Length of lookback window
            
        Returns:
            Tuple of (feature_sequences, targets, sequence_dates)
        """
        if sequence_length is None:
            sequence_length = self.lookback_window
        
        feature_sequences = []
        target_sequences = []
        date_sequences = []
        
        for i in range(sequence_length, len(features)):
            # Sequence from [i-sequence_length : i]
            # Predicts target at time i
            feature_sequences.append(features[i-sequence_length:i])
            target_sequences.append(targets[i])
            date_sequences.append(dates[i])
        
        feature_sequences = np.array(feature_sequences)
        target_sequences = np.array(target_sequences)
        date_sequences = np.array(date_sequences)
        
        self.logger.info(f"Created {len(feature_sequences)} temporal sequences")
        self.logger.info(f"Sequence shape: {feature_sequences.shape}")
        
        return feature_sequences, target_sequences, date_sequences
    
    def verify_sequence_ordering(
        self,
        sequences: np.ndarray,
        dates: np.ndarray
    ) -> bool:
        """
        Verify sequences maintain temporal order
        """
        # Check that dates are monotonically increasing
        date_diffs = np.diff(dates.astype('datetime64[D]').astype(int))
        
        if np.any(date_diffs < 0):
            self.logger.error("Temporal ordering violated!")
            return False
        
        self.logger.info("✓ Sequence temporal ordering verified")
        return True


class TemporalDataPipeline:
    """
    Complete pipeline with temporal integrity guarantees
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.splitter = TemporalDataSplitter(config)
        self.aligner = NewsTimestampAligner()
        self.sequence_gen = SequenceGenerator(
            lookback_window=config.get('features', {}).get('lookback_window', 30)
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def prepare_temporal_data(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame,
        train_end: str = '2021-12-31',
        val_end: str = '2022-12-31',
        test_end: str = '2023-12-31'
    ) -> Dict:
        """
        Complete temporal data preparation with all safeguards
        
        Returns:
            Dictionary with train/val/test splits and metadata
        """
        self.logger.info("="*70)
        self.logger.info("TEMPORAL DATA PREPARATION PIPELINE")
        self.logger.info("="*70)
        
        # Step 1: Split stock data temporally
        stock_train, stock_val, stock_test = self.splitter.temporal_train_val_test_split(
            stock_df,
            date_column='Date',
            train_end=train_end,
            val_end=val_end,
            test_end=test_end
        )
        
        # Step 2: Align news to each split's trading days
        self.logger.info("\nAligning news to training days...")
        train_aligned_news = self.aligner.align_news_to_trading_days(
            news_df,
            stock_train['Date'],
            lookback_days=30
        )
        
        self.logger.info("Aligning news to validation days...")
        val_aligned_news = self.aligner.align_news_to_trading_days(
            news_df,
            stock_val['Date'],
            lookback_days=30
        )
        
        self.logger.info("Aligning news to test days...")
        test_aligned_news = self.aligner.align_news_to_trading_days(
            news_df,
            stock_test['Date'],
            lookback_days=30
        )
        
        # Step 3: Verify no lookahead bias
        self.logger.info("\nVerifying temporal integrity...")
        self.aligner.verify_no_lookahead(train_aligned_news)
        self.aligner.verify_no_lookahead(val_aligned_news)
        self.aligner.verify_no_lookahead(test_aligned_news)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("✓ TEMPORAL DATA PREPARATION COMPLETE - NO LOOKAHEAD BIAS")
        self.logger.info("="*70)
        
        return {
            'stock_train': stock_train,
            'stock_val': stock_val,
            'stock_test': stock_test,
            'news_train': train_aligned_news,
            'news_val': val_aligned_news,
            'news_test': test_aligned_news,
            'train_dates': (stock_train['Date'].min(), stock_train['Date'].max()),
            'val_dates': (stock_val['Date'].min(), stock_val['Date'].max()),
            'test_dates': (stock_test['Date'].min(), stock_test['Date'].max())
        }


if __name__ == "__main__":
    # Example usage demonstrating temporal integrity
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
    
    stock_df = pd.DataFrame({
        'Date': dates,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    news_dates = pd.date_range('2019-01-01', '2023-12-31', freq='6H')
    news_df = pd.DataFrame({
        'published_at': news_dates,
        'title': ['News ' + str(i) for i in range(len(news_dates))],
        'sentiment': np.random.randn(len(news_dates))
    })
    
    # Run pipeline
    config = {'features': {'lookback_window': 30}}
    pipeline = TemporalDataPipeline(config)
    
    result = pipeline.prepare_temporal_data(
        stock_df,
        news_df,
        train_end='2021-12-31',
        val_end='2022-12-31',
        test_end='2023-12-31'
    )
    
    print("\nTemporal split summary:")
    print(f"Train period: {result['train_dates']}")
    print(f"Val period: {result['val_dates']}")
    print(f"Test period: {result['test_dates']}")
