"""
Numerical Feature Engineering Module
Creates technical indicators and time-series features from OHLC data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """Calculates technical indicators for stock data"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not TA_AVAILABLE:
            self.logger.warning("ta library not available. Some indicators may not work.")
    
    def add_moving_averages(
        self, 
        df: pd.DataFrame, 
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Add Simple and Exponential Moving Averages
        
        Args:
            df: DataFrame with price data
            windows: List of window sizes
            
        Returns:
            DataFrame with MA columns
        """
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        return df
    
    def add_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        window: int = 20, 
        num_std: int = 2
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands
        
        Args:
            df: DataFrame with price data
            window: Rolling window size
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Band columns
        """
        if TA_AVAILABLE:
            indicator = ta.volatility.BollingerBands(
                close=df['Close'], 
                window=window, 
                window_dev=num_std
            )
            df['BB_High'] = indicator.bollinger_hband()
            df['BB_Mid'] = indicator.bollinger_mavg()
            df['BB_Low'] = indicator.bollinger_lband()
            df['BB_Width'] = indicator.bollinger_wband()
        else:
            # Manual calculation
            sma = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            df['BB_High'] = sma + (std * num_std)
            df['BB_Mid'] = sma
            df['BB_Low'] = sma - (std * num_std)
            df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index
        
        Args:
            df: DataFrame with price data
            window: RSI window
            
        Returns:
            DataFrame with RSI column
        """
        if TA_AVAILABLE:
            df['RSI'] = ta.momentum.RSIIndicator(
                close=df['Close'], 
                window=window
            ).rsi()
        else:
            # Manual calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def add_macd(
        self, 
        df: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line window
            
        Returns:
            DataFrame with MACD columns
        """
        if TA_AVAILABLE:
            indicator = ta.trend.MACD(
                close=df['Close'],
                window_slow=slow,
                window_fast=fast,
                window_sign=signal
            )
            df['MACD'] = indicator.macd()
            df['MACD_Signal'] = indicator.macd_signal()
            df['MACD_Diff'] = indicator.macd_diff()
        else:
            # Manual calculation
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            df['MACD'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (volatility indicator)
        
        Args:
            df: DataFrame with OHLC data
            window: ATR window
            
        Returns:
            DataFrame with ATR column
        """
        if TA_AVAILABLE:
            df['ATR'] = ta.volatility.AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=window
            ).average_true_range()
        else:
            # Manual calculation
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=window).mean()
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume indicator columns
        """
        # Volume moving averages
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        if TA_AVAILABLE:
            # Money Flow Index
            df['MFI'] = ta.volume.MFIIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['Volume'],
                window=14
            ).money_flow_index()
        
        return df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all indicators
        """
        self.logger.info("Adding technical indicators")
        
        df = self.add_moving_averages(df)
        df = self.add_bollinger_bands(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_atr(df)
        df = self.add_volume_indicators(df)
        
        self.logger.info("Technical indicators added")
        return df


class PriceFeatureEngineer:
    """Creates price-based features for prediction"""
    
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add various return metrics
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with return columns
        """
        # Simple returns
        df['Returns_1d'] = df['Close'].pct_change(1)
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        
        # Log returns
        df['Log_Returns_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Returns_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        
        # Cumulative returns
        df['Cumulative_Returns'] = (1 + df['Returns_1d']).cumprod() - 1
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility metrics
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volatility columns
        """
        # Historical volatility (rolling std of returns)
        df['Volatility_10d'] = df['Returns_1d'].rolling(window=10).std()
        df['Volatility_20d'] = df['Returns_1d'].rolling(window=20).std()
        df['Volatility_30d'] = df['Returns_1d'].rolling(window=30).std()
        
        # Parkinson volatility (using high-low range)
        df['Parkinson_Volatility'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['High'] / df['Low']) ** 2).rolling(window=20).mean()
        )
        
        # Garman-Klass volatility
        df['GK_Volatility'] = np.sqrt(
            0.5 * (np.log(df['High'] / df['Low']) ** 2) -
            (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
        ).rolling(window=20).mean()
        
        return df
    
    def add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features about price position relative to historical range
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with position features
        """
        # Price relative to moving averages
        df['Price_to_SMA_20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA_50'] = df['Close'] / df['SMA_50']
        
        # 52-week high/low
        df['52W_High'] = df['High'].rolling(window=252).max()
        df['52W_Low'] = df['Low'].rolling(window=252).min()
        df['Distance_from_52W_High'] = (df['Close'] - df['52W_High']) / df['52W_High']
        df['Distance_from_52W_Low'] = (df['Close'] - df['52W_Low']) / df['52W_Low']
        
        # Stochastic oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with momentum columns
        """
        # Rate of Change
        df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['ROC_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        
        # Momentum
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
        
        # Williams %R
        high_max = df['High'].rolling(window=14).max()
        low_min = df['Low'].rolling(window=14).min()
        df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        return df
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all price-based features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all features
        """
        self.logger.info("Adding price features")
        
        df = self.add_returns(df)
        df = self.add_volatility_features(df)
        df = self.add_price_position_features(df)
        df = self.add_momentum_features(df)
        
        self.logger.info("Price features added")
        return df


class TargetVariableCreator:
    """Creates target variables for prediction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prediction_horizon = config.get('features', {}).get('prediction_horizon', 1)
        self.task = config.get('model', {}).get('output', {}).get('task', 'classification')
        self.num_classes = config.get('model', {}).get('output', {}).get('classes', 3)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_regression_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create regression target (future price change %)
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with target column
        """
        df['Target_Return'] = (
            df['Close'].shift(-self.prediction_horizon) / df['Close'] - 1
        ) * 100
        
        return df
    
    def create_classification_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create classification target
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with target column
        """
        # Calculate future returns
        future_return = (
            df['Close'].shift(-self.prediction_horizon) / df['Close'] - 1
        )
        
        if self.num_classes == 2:
            # Binary: up (1) or down (0)
            df['Target'] = (future_return > 0).astype(int)
        
        elif self.num_classes == 3:
            # Ternary: down (0), neutral (1), up (2)
            threshold = df['Returns_1d'].std() * 0.5  # Adaptive threshold
            
            df['Target'] = pd.cut(
                future_return,
                bins=[-np.inf, -threshold, threshold, np.inf],
                labels=[0, 1, 2]
            ).astype(int)
        
        else:
            # Multiple classes based on return magnitude
            percentiles = np.linspace(0, 100, self.num_classes + 1)
            bins = np.percentile(future_return.dropna(), percentiles)
            df['Target'] = pd.cut(
                future_return,
                bins=bins,
                labels=range(self.num_classes),
                include_lowest=True
            ).astype(int)
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create appropriate target variable based on task
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with target column
        """
        if self.task == 'regression':
            df = self.create_regression_target(df)
        else:
            df = self.create_classification_target(df)
        
        return df


class NumericalFeaturePipeline:
    """Complete numerical feature engineering pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.price_engineer = PriceFeatureEngineer(
            lookback_window=config.get('features', {}).get('lookback_window', 30)
        )
        self.target_creator = TargetVariableCreator(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete processing pipeline for stock data
        
        Args:
            df: DataFrame with raw OHLC data
            
        Returns:
            DataFrame with all features and target
        """
        self.logger.info(f"Processing {len(df)} records")
        
        # Add technical indicators
        df = self.indicator_calculator.add_all_indicators(df)
        
        # Add price features
        df = self.price_engineer.add_all_features(df)
        
        # Create target variable
        df = self.target_creator.create_target(df)
        
        # Drop NaN rows (from indicators and target)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        self.logger.info(f"Processing complete. Dropped {dropped} NaN rows. {len(df)} records remaining.")
        
        return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'features': {
            'lookback_window': 30,
            'prediction_horizon': 1
        },
        'model': {
            'output': {
                'task': 'classification',
                'classes': 3
            }
        }
    }
    
    pipeline = NumericalFeaturePipeline(config)
    print("Numerical feature pipeline initialized")
