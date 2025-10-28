"""
Enhanced Feature Engineering Module
Adds advanced technical indicators, sentiment trends, and market correlation features
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class EnhancedNumericalFeatures:
    """
    Enhanced technical and numerical feature engineering
    Adds momentum indicators, volume analysis, and market correlations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced momentum indicators
        
        Indicators added:
        - Stochastic Oscillator (%K and %D)
        - Williams %R
        - ADX (Average Directional Index)
        - OBV (On-Balance Volume)
        - CMF (Chaikin Money Flow)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added momentum features
        """
        self.logger.info("Adding momentum indicators...")
        
        # Stochastic Oscillator (14-period)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['stochastic_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
        
        # Williams %R (14-period)
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-10)
        
        # ADX (Average Directional Index) - Trend Strength
        df = self._calculate_adx(df, period=14)
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        # CMF (Chaikin Money Flow)
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        mf_volume = mfm * df['Volume']
        df['cmf'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # Rate of Change (ROC)
        df['roc_5'] = df['Close'].pct_change(periods=5) * 100
        df['roc_10'] = df['Close'].pct_change(periods=10) * 100
        df['roc_20'] = df['Close'].pct_change(periods=20) * 100
        
        self.logger.info(f"✅ Added {8} momentum indicators")
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        # True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smoothed indicators
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based indicators
        
        Indicators added:
        - ATR (Average True Range)
        - Bollinger Bandwidth
        - Keltner Channels
        - Historical Volatility (multiple timeframes)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        self.logger.info("Adding volatility indicators...")
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        df['atr_20'] = tr.rolling(window=20).mean()
        
        # Bollinger Bandwidth
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['bb_bandwidth'] = (2 * std_20) / (sma_20 + 1e-10)
        df['bb_percent'] = (df['Close'] - (sma_20 - 2*std_20)) / (4*std_20 + 1e-10)
        
        # Keltner Channels
        ema_20 = df['Close'].ewm(span=20).mean()
        df['keltner_upper'] = ema_20 + 2*df['atr_20']
        df['keltner_lower'] = ema_20 - 2*df['atr_20']
        df['keltner_percent'] = (df['Close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'] + 1e-10)
        
        # Historical Volatility (different timeframes)
        df['hist_vol_10'] = df['Returns'].rolling(window=10).std() * np.sqrt(252)  # Annualized
        df['hist_vol_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['hist_vol_60'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)
        
        self.logger.info(f"✅ Added volatility indicators")
        return df
    
    def add_market_correlation_features(
        self,
        stock_df: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        window: int = 20
    ) -> pd.DataFrame:
        """
        Add market correlation features (S&P 500, VIX, Nasdaq)
        
        Args:
            stock_df: Main stock DataFrame
            market_data: Dictionary with market indices data
            window: Rolling correlation window
            
        Returns:
            DataFrame with market correlation features
        """
        self.logger.info("Adding market correlation features...")
        
        df = stock_df.copy()
        
        # Ensure Date column is timezone-naive
        df['Date'] = pd.to_datetime(df['Date'])
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        # S&P 500 correlation
        if 'sp500' in market_data and not market_data['sp500'].empty:
            sp500 = market_data['sp500'].copy()
            
            # Flatten multi-level columns if present
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500.columns = sp500.columns.get_level_values(0)
            
            sp500.index = pd.to_datetime(sp500.index)
            
            # Remove timezone info if present
            if sp500.index.tz is not None:
                sp500.index = sp500.index.tz_localize(None)
            
            # Align on dates
            merged = df.set_index('Date').join(sp500[['Returns']], how='left', rsuffix='_sp500')
            
            # S&P 500 returns
            df['sp500_returns'] = merged['Returns_sp500'].values
            
            # Rolling correlation
            df['sp500_correlation'] = merged['Returns'].rolling(window=window).corr(merged['Returns_sp500']).values
            
            # Beta (stock sensitivity to market)
            cov = merged['Returns'].rolling(window=window).cov(merged['Returns_sp500'])
            var = merged['Returns_sp500'].rolling(window=window).var()
            df['beta_sp500'] = (cov / (var + 1e-10)).values
            
            self.logger.info("✅ Added S&P 500 features")
        
        # VIX features
        if 'vix' in market_data and not market_data['vix'].empty:
            vix = market_data['vix'].copy()
            
            # Flatten multi-level columns if present
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            
            vix.index = pd.to_datetime(vix.index)
            
            # Remove timezone info if present
            if vix.index.tz is not None:
                vix.index = vix.index.tz_localize(None)
            
            merged = df.set_index('Date').join(vix[['Close']], how='left', rsuffix='_vix')
            df['vix_level'] = merged['Close_vix'].values
            df['vix_change'] = merged['Close_vix'].pct_change().values
            df['vix_sma_10'] = merged['Close_vix'].rolling(window=10).mean().values
            
            # Market stress indicator (high VIX = high stress)
            df['market_stress'] = (df['vix_level'] > df['vix_sma_10']).astype(int)
            
            self.logger.info("✅ Added VIX features")
        
        # Nasdaq correlation
        if 'nasdaq' in market_data and not market_data['nasdaq'].empty:
            nasdaq = market_data['nasdaq'].copy()
            
            # Flatten multi-level columns if present
            if isinstance(nasdaq.columns, pd.MultiIndex):
                nasdaq.columns = nasdaq.columns.get_level_values(0)
            
            nasdaq.index = pd.to_datetime(nasdaq.index)
            
            # Remove timezone info if present
            if nasdaq.index.tz is not None:
                nasdaq.index = nasdaq.index.tz_localize(None)
            
            merged = df.set_index('Date').join(nasdaq[['Returns']], how='left', rsuffix='_nasdaq')
            df['nasdaq_returns'] = merged['Returns_nasdaq'].values
            df['nasdaq_correlation'] = merged['Returns'].rolling(window=window).corr(merged['Returns_nasdaq']).values
            
            self.logger.info("✅ Added Nasdaq features")
        
        return df
    
    def add_sector_features(
        self,
        stock_df: pd.DataFrame,
        sector_df: Optional[pd.DataFrame],
        window: int = 20
    ) -> pd.DataFrame:
        """
        Add sector ETF correlation features
        
        Args:
            stock_df: Main stock DataFrame
            sector_df: Sector ETF DataFrame
            window: Rolling correlation window
            
        Returns:
            DataFrame with sector features
        """
        if sector_df is None or sector_df.empty:
            self.logger.warning("No sector data available")
            return stock_df
        
        self.logger.info("Adding sector features...")
        
        df = stock_df.copy()
        
        # Ensure Date column is timezone-naive
        df['Date'] = pd.to_datetime(df['Date'])
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        sector = sector_df.copy()
        
        # Flatten multi-level columns if present
        if isinstance(sector.columns, pd.MultiIndex):
            sector.columns = sector.columns.get_level_values(0)
        
        sector.index = pd.to_datetime(sector.index)
        
        # Remove timezone info if present
        if sector.index.tz is not None:
            sector.index = sector.index.tz_localize(None)
        
        # Merge on dates
        merged = df.set_index('Date').join(sector, how='left')
        
        # Check if expected columns exist
        if 'Returns' in merged.columns:
            # Sector returns
            sector_returns_col = 'Sector_Returns' if 'Sector_Returns' in merged.columns else 'Returns'
            if sector_returns_col in merged.columns:
                df['sector_returns'] = merged[sector_returns_col].values
            
            # Sector correlation (use stock returns if available)
            if 'Returns' in df.columns and 'Returns' in merged.columns:
                stock_returns = merged['Returns']
                sector_returns = merged[sector_returns_col]
                df['sector_correlation'] = stock_returns.rolling(window=window).corr(sector_returns).values
            
            # Relative strength to sector
            sector_close_col = 'Sector_Close' if 'Sector_Close' in merged.columns else 'Close'
            if sector_close_col in merged.columns and 'Close' in merged.columns:
                # Use merged Close (from stock) and sector Close
                rel_strength = merged['Close'] / merged[sector_close_col]
                df['rel_strength_sector'] = rel_strength.values
                df['rel_strength_change'] = rel_strength.pct_change().values
            
            self.logger.info("✅ Added sector features")
        else:
            self.logger.warning("Sector data missing expected columns")
        
        return df
    
    def add_macro_features(
        self,
        stock_df: pd.DataFrame,
        macro_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Add macroeconomic indicator features
        
        Args:
            stock_df: Main stock DataFrame
            macro_data: Dictionary with macro indicators
            
        Returns:
            DataFrame with macro features
        """
        self.logger.info("Adding macroeconomic features...")
        
        df = stock_df.copy()
        
        # Ensure Date column is timezone-naive
        df['Date'] = pd.to_datetime(df['Date'])
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        df_indexed = df.set_index('Date')
        
        # 10-Year Treasury Yield
        if 'treasury_10y' in macro_data and not macro_data['treasury_10y'].empty:
            treasury = macro_data['treasury_10y'].copy()
            
            # Flatten multi-level columns if present
            if isinstance(treasury.columns, pd.MultiIndex):
                treasury.columns = treasury.columns.get_level_values(0)
            
            treasury.index = pd.to_datetime(treasury.index)
            
            # Remove timezone info if present
            if treasury.index.tz is not None:
                treasury.index = treasury.index.tz_localize(None)
            
            merged = df_indexed.join(treasury, how='left', rsuffix='_treasury')
            df['treasury_10y'] = merged['Close_treasury'].ffill().values
            df['treasury_change'] = merged['Close_treasury'].diff().ffill().values
            
            # Yield curve steepness proxy (would need more yields for true curve)
            df['treasury_volatility'] = merged['Close_treasury'].rolling(window=20).std().values
            
            self.logger.info("✅ Added Treasury features")
        
        # US Dollar Index
        if 'dxy' in macro_data and not macro_data['dxy'].empty:
            dxy = macro_data['dxy'].copy()
            
            # Flatten multi-level columns if present
            if isinstance(dxy.columns, pd.MultiIndex):
                dxy.columns = dxy.columns.get_level_values(0)
            
            dxy.index = pd.to_datetime(dxy.index)
            
            # Remove timezone info if present
            if dxy.index.tz is not None:
                dxy.index = dxy.index.tz_localize(None)
            
            merged = df_indexed.join(dxy, how='left', rsuffix='_dxy')
            df['dollar_index'] = merged['Close_dxy'].ffill().values
            df['dollar_change'] = merged['Close_dxy'].pct_change().ffill().values
            df['dollar_strength'] = (df['dollar_index'] > df['dollar_index'].rolling(20).mean()).astype(int)
            
            self.logger.info("✅ Added Dollar Index features")
        
        # Fed Funds proxy
        if 'fed_funds' in macro_data and not macro_data['fed_funds'].empty:
            fed = macro_data['fed_funds'].copy()
            
            # Flatten multi-level columns if present
            if isinstance(fed.columns, pd.MultiIndex):
                fed.columns = fed.columns.get_level_values(0)
            
            fed.index = pd.to_datetime(fed.index)
            
            # Remove timezone info if present
            if fed.index.tz is not None:
                fed.index = fed.index.tz_localize(None)
            
            merged = df_indexed.join(fed, how='left', rsuffix='_fed')
            df['fed_funds_rate'] = merged['Close_fed'].ffill().values
            df['fed_rate_change'] = merged['Close_fed'].diff().ffill().values
            
            self.logger.info("✅ Added Fed Funds features")
        
        return df
    
    def add_competitor_features(
        self,
        stock_df: pd.DataFrame,
        competitor_data: Dict[str, pd.DataFrame],
        window: int = 20
    ) -> pd.DataFrame:
        """
        Add competitor correlation and divergence features
        
        Args:
            stock_df: Main stock DataFrame
            competitor_data: Dictionary of competitor DataFrames
            window: Rolling correlation window
            
        Returns:
            DataFrame with competitor features
        """
        if not competitor_data:
            self.logger.warning("No competitor data available")
            return stock_df
        
        self.logger.info(f"Adding competitor features for {len(competitor_data)} companies...")
        
        df = stock_df.copy()
        
        # Ensure Date column is timezone-naive
        df['Date'] = pd.to_datetime(df['Date'])
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        df_indexed = df.set_index('Date')
        
        # Collect all competitor returns
        comp_returns_list = []
        comp_names = []
        
        for comp_name, comp_df in competitor_data.items():
            try:
                comp = comp_df.copy()
                
                # Flatten multi-level columns if present
                if isinstance(comp.columns, pd.MultiIndex):
                    comp.columns = comp.columns.get_level_values(0)
                
                # Ensure Date column exists
                if 'Date' not in comp.columns:
                    comp = comp.reset_index()
                
                comp['Date'] = pd.to_datetime(comp['Date'])
                comp_indexed = comp.set_index('Date')
                
                # Remove timezone info if present
                if comp_indexed.index.tz is not None:
                    comp_indexed.index = comp_indexed.index.tz_localize(None)
                
                # Look for Returns column (might be named differently)
                returns_col = None
                for col in comp_indexed.columns:
                    if 'return' in col.lower():
                        returns_col = col
                        break
                
                if returns_col is None and 'Close' in comp_indexed.columns:
                    # Calculate returns if not present
                    comp_indexed['Returns'] = comp_indexed['Close'].pct_change()
                    returns_col = 'Returns'
                
                if returns_col:
                    merged = df_indexed.join(comp_indexed[[returns_col]], how='left', rsuffix=f'_{comp_name}')
                    comp_returns = merged[f'{returns_col}_{comp_name}'].ffill()
                    
                    comp_returns_list.append(comp_returns)
                    comp_names.append(comp_name)
                    
                    # Individual correlation
                    if 'Returns' in df_indexed.columns:
                        df[f'corr_{comp_name[:10]}'] = df_indexed['Returns'].rolling(window=window).corr(comp_returns).values
            except Exception as e:
                self.logger.warning(f"Failed to process competitor {comp_name}: {e}")
                continue
        
        if comp_returns_list:
            # Average competitor return
            comp_returns_df = pd.DataFrame(comp_returns_list).T
            df['competitor_avg_return'] = comp_returns_df.mean(axis=1).values
            
            # Sentiment divergence (stock vs competitors)
            df['return_divergence'] = (df['Returns'] - df['competitor_avg_return']).values
            
            # Competitor consensus (low std = all moving together)
            df['competitor_consensus'] = comp_returns_df.std(axis=1).values
            
            # Relative strength vs competitors
            df['rel_strength_comps'] = (df['Returns'].rolling(20).mean() - 
                                         df['competitor_avg_return'].rolling(20).mean()).values
            
            self.logger.info(f"✅ Added features for {len(competitor_data)} competitors")
        
        return df
    
    def create_all_features(
        self,
        stock_df: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        sector_df: Optional[pd.DataFrame] = None,
        macro_data: Optional[Dict[str, pd.DataFrame]] = None,
        competitor_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Create all enhanced numerical features
        
        Args:
            stock_df: Main stock DataFrame
            market_data: Market indices data
            sector_df: Sector ETF data
            macro_data: Macroeconomic indicators
            competitor_data: Competitor stock data
            
        Returns:
            DataFrame with all features
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info("CREATING ENHANCED NUMERICAL FEATURES")
        self.logger.info(f"{'='*60}\n")
        
        # Validate required OHLCV columns before computing indicators
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required_cols if c not in stock_df.columns]
        if missing_cols:
            msg = ("Missing required OHLCV columns for feature engineering: "
                   f"{missing_cols}. Ensure data collection succeeded and that the stock DataFrame\n"
                   "contains Date, Open, High, Low, Close, Volume columns (case-sensitive).")
            self.logger.error(msg)
            raise ValueError(msg)

        # Copy and ensure Returns exists (many indicators expect Returns)
        df = stock_df.copy()
        if 'Returns' not in df.columns and 'Close' in df.columns:
            self.logger.info("'Returns' column not found — computing from 'Close' via pct_change()")
            df['Returns'] = df['Close'].pct_change()
        initial_features = len(df.columns)
        
        # Add momentum indicators
        df = self.add_momentum_indicators(df)
        
        # Add volatility indicators
        df = self.add_volatility_indicators(df)
        
        # Add market correlations
        if market_data:
            df = self.add_market_correlation_features(df, market_data)
        
        # Add sector features
        if sector_df is not None:
            df = self.add_sector_features(df, sector_df)
        
        # Add macro features
        if macro_data:
            df = self.add_macro_features(df, macro_data)
        
        # Add competitor features
        if competitor_data:
            df = self.add_competitor_features(df, competitor_data)
        
        final_features = len(df.columns)
        added_features = final_features - initial_features
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"FEATURE ENGINEERING COMPLETE")
        self.logger.info(f"Initial features: {initial_features}")
        self.logger.info(f"Final features: {final_features}")
        self.logger.info(f"Added features: {added_features}")
        self.logger.info(f"{'='*60}\n")
        
        return df


class EnhancedSentimentFeatures:
    """
    Enhanced sentiment feature engineering
    Adds temporal trends, volatility, and regime indicators
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_sentiment_trends(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'sentiment'
    ) -> pd.DataFrame:
        """
        Add sentiment trend features
        
        Features added:
        - 3-day rolling sentiment
        - 7-day rolling sentiment
        - 14-day rolling sentiment
        - Sentiment momentum (change)
        - Sentiment volatility
        - Sentiment regime (positive/negative/neutral)
        
        Args:
            df: DataFrame with sentiment scores
            sentiment_col: Name of sentiment column
            
        Returns:
            DataFrame with trend features
        """
        self.logger.info("Adding sentiment trend features...")
        
        if sentiment_col not in df.columns:
            self.logger.warning(f"Column {sentiment_col} not found")
            return df
        
        # Rolling averages
        df['sentiment_3d'] = df[sentiment_col].rolling(window=3, min_periods=1).mean()
        df['sentiment_7d'] = df[sentiment_col].rolling(window=7, min_periods=1).mean()
        df['sentiment_14d'] = df[sentiment_col].rolling(window=14, min_periods=1).mean()
        
        # Sentiment momentum (trend direction)
        df['sentiment_momentum'] = df[sentiment_col].diff()
        df['sentiment_momentum_3d'] = df['sentiment_3d'].diff()
        
        # Sentiment volatility (stability of sentiment)
        df['sentiment_volatility'] = df[sentiment_col].rolling(window=7, min_periods=1).std()
        df['sentiment_volatility_14d'] = df[sentiment_col].rolling(window=14, min_periods=1).std()
        
        # Sentiment regime classification
        df['sentiment_regime'] = pd.cut(
            df['sentiment_7d'],
            bins=[-np.inf, -0.1, 0.1, np.inf],
            labels=['negative', 'neutral', 'positive']
        )
        df['sentiment_regime_encoded'] = df['sentiment_regime'].map({
            'negative': -1,
            'neutral': 0,
            'positive': 1
        })
        
        # Sentiment acceleration (second derivative)
        df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
        
        # Crossing indicators (sentiment crosses MA)
        df['sent_above_7d_ma'] = (df[sentiment_col] > df['sentiment_7d']).astype(int)
        df['sent_above_14d_ma'] = (df[sentiment_col] > df['sentiment_14d']).astype(int)
        
        self.logger.info("✅ Added 13 sentiment trend features")
        return df
    
    def add_competitor_sentiment_features(
        self,
        main_df: pd.DataFrame,
        competitor_sentiments: Dict[str, pd.DataFrame],
        main_sentiment_col: str = 'sentiment'
    ) -> pd.DataFrame:
        """
        Add competitor sentiment comparison features
        
        Args:
            main_df: Main stock DataFrame with sentiment
            competitor_sentiments: Dict of competitor sentiment DataFrames
            main_sentiment_col: Main sentiment column name
            
        Returns:
            DataFrame with competitor sentiment features
        """
        if not competitor_sentiments:
            self.logger.warning("No competitor sentiment data")
            return main_df
        
        self.logger.info(f"Adding competitor sentiment features for {len(competitor_sentiments)} companies...")
        
        df = main_df.copy()
        
        # Aggregate competitor sentiments
        comp_sent_list = []
        for comp_name, comp_df in competitor_sentiments.items():
            if 'sentiment' in comp_df.columns:
                comp_sent_list.append(comp_df['sentiment'])
        
        if comp_sent_list:
            comp_sent_df = pd.DataFrame(comp_sent_list).T
            
            # Average competitor sentiment
            df['competitor_avg_sentiment'] = comp_sent_df.mean(axis=1).values[:len(df)]
            
            # Sentiment divergence (contrarian indicator)
            if main_sentiment_col in df.columns:
                df['sentiment_divergence'] = (df[main_sentiment_col] - 
                                               df['competitor_avg_sentiment'])
            
            # Competitor sentiment consensus
            df['competitor_sent_consensus'] = comp_sent_df.std(axis=1).values[:len(df)]
            
            # Relative sentiment strength
            df['relative_sentiment'] = (df[main_sentiment_col] - 
                                         df['competitor_avg_sentiment'].rolling(7).mean())
            
            self.logger.info("✅ Added competitor sentiment features")
        
        return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100),
        'Returns': np.random.randn(100) * 0.02
    })
    
    # Create features
    feature_eng = EnhancedNumericalFeatures()
    df_enhanced = feature_eng.add_momentum_indicators(df)
    df_enhanced = feature_eng.add_volatility_indicators(df_enhanced)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Enhanced features: {len(df_enhanced.columns)}")
    print(f"Added features: {len(df_enhanced.columns) - len(df.columns)}")
