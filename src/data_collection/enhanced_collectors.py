"""
Enhanced Data Collection Module
Supports extended date ranges, market-wide data, competitor news, and macroeconomic indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Tuple, Optional
import time
import os
from .collectors import StockDataCollector, NewsDataCollector

logger = logging.getLogger(__name__)


# Competitor/Correlation mappings
COMPETITOR_MAP = {
    'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META'],          # Big Tech
    'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'ORCL'],          # Tech Giants
    'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN'],          # Tech/Ad
    'NVDA': ['AMD', 'INTC', 'TSM', 'QCOM'],             # Semiconductors
    'AMD': ['NVDA', 'INTC', 'MU', 'QCOM'],              # Chips
    'TSLA': ['F', 'GM', 'RIVN', 'LCID'],                # Auto/EV
    'JPM': ['BAC', 'WFC', 'C', 'GS'],                   # Banks
    'BAC': ['JPM', 'WFC', 'C', 'USB'],                  # Banks
    'XOM': ['CVX', 'COP', 'SLB', 'BP'],                 # Energy
    'JNJ': ['PFE', 'MRK', 'ABBV', 'UNH'],               # Healthcare
    'WMT': ['TGT', 'COST', 'HD', 'LOW'],                # Retail
}

# Sector ETF mappings
SECTOR_MAP = {
    'AAPL': 'XLK',   # Technology
    'MSFT': 'XLK',   # Technology
    'GOOGL': 'XLK',  # Technology
    'NVDA': 'XLK',   # Technology
    'AMD': 'XLK',    # Technology
    'TSLA': 'XLY',   # Consumer Discretionary
    'JPM': 'XLF',    # Financial
    'BAC': 'XLF',    # Financial
    'WFC': 'XLF',    # Financial
    'XOM': 'XLE',    # Energy
    'CVX': 'XLE',    # Energy
    'JNJ': 'XLV',    # Healthcare
    'PFE': 'XLV',    # Healthcare
    'WMT': 'XLP',    # Consumer Staples
    'HD': 'XLY',     # Consumer Discretionary
}


class EnhancedStockDataCollector(StockDataCollector):
    """
    Enhanced stock collector with market-wide and macroeconomic data
    """
    
    def __init__(self):
        super().__init__()
        self._setup_proxy()
    
    def _setup_proxy(self):
        """Setup proxy settings from environment variables"""
        self.proxy_settings = {}
        
        # Check for proxy environment variables
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        
        if http_proxy:
            self.proxy_settings['http'] = http_proxy
            self.logger.info(f"Using HTTP proxy: {http_proxy}")
        
        if https_proxy:
            self.proxy_settings['https'] = https_proxy
            self.logger.info(f"Using HTTPS proxy: {https_proxy}")
        
        if self.proxy_settings:
            # Configure requests session with proxy
            self.session = requests.Session()
            self.session.proxies.update(self.proxy_settings)
            # Set yfinance to use our session
            yf.Ticker._session = self.session
        else:
            self.session = None
    
    def test_yfinance_connectivity(self, ticker: str = 'AAPL') -> bool:
        """
        Test yfinance connectivity and diagnose issues
        
        Args:
            ticker: Test ticker symbol
            
        Returns:
            True if connectivity works, False otherwise
        """
        self.logger.info(f"Testing yfinance connectivity with {ticker}...")
        
        try:
            # Test basic ticker info
            test_ticker = yf.Ticker(ticker)
            info = test_ticker.info
            
            if info and 'regularMarketPrice' in info:
                self.logger.info(f"✅ yfinance connectivity OK - {ticker} price: ${info['regularMarketPrice']:.2f}")
                return True
            else:
                self.logger.warning(f"⚠️ yfinance returned empty info for {ticker}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ yfinance connectivity failed: {e}")
            
            # Provide troubleshooting guidance
            if "Expecting value: line 1 column 1" in str(e):
                self.logger.error("This error suggests network/proxy/firewall issues:")
                self.logger.error("  - Check if you're behind a corporate proxy/firewall")
                self.logger.error("  - Set HTTP_PROXY and HTTPS_PROXY environment variables")
                self.logger.error("  - Example: set HTTP_PROXY=http://proxy.company.com:8080")
                self.logger.error("  - Example: set HTTPS_PROXY=http://proxy.company.com:8080")
            elif "No timezone found" in str(e):
                self.logger.error("This suggests the ticker may be delisted or invalid")
                self.logger.error("  - Verify the ticker symbol is correct")
                self.logger.error("  - Try a different ticker like 'MSFT' or 'GOOGL'")
            
            return False
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Enhanced fetch_stock_data with proxy support and error handling
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
            
            # Use proxy-aware session if configured
            if self.session:
                # Temporarily set yfinance session
                original_session = yf.Ticker._session
                yf.Ticker._session = self.session
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Restore original session
            if self.session:
                yf.Ticker._session = original_session
            
            if df.empty:
                self.logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            
            # Rename Date column for consistency
            if 'Date' not in df.columns and df.index.name == 'Date':
                df.reset_index(inplace=True)
            if 'Date' not in df.columns:
                df['Date'] = df.index
                df.reset_index(drop=True, inplace=True)
            
            # Calculate additional features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            
            # Try fallback: check for cached data
            cached_path = f"data/raw/stocks/{ticker}_stock_data_extended.csv"
            if os.path.exists(cached_path):
                self.logger.info(f"Attempting to load cached data from {cached_path}")
                try:
                    cached_df = pd.read_csv(cached_path)
                    if not cached_df.empty and len(cached_df) > 0:
                        self.logger.info(f"✅ Loaded {len(cached_df)} records from cache")
                        return cached_df
                except Exception as cache_e:
                    self.logger.error(f"Failed to load cached data: {cache_e}")
            
            return pd.DataFrame()
    
    def fetch_extended_stock_data(
        self,
        ticker: str,
        days: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch extended historical data (500-1000+ days recommended)
        
        Args:
            ticker: Stock ticker symbol
            days: Number of trading days to fetch (default: 1000 = ~4 years)
            start_date: Optional start date (overrides days)
            end_date: Optional end date (default: today)
            
        Returns:
            DataFrame with extended OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Calculate start_date from days (approximate with calendar days)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=int(days * 1.4))  # 1.4x for weekends/holidays
            start_date = start_dt.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching {days} days of data for {ticker}: {start_date} to {end_date}")
        
        # Fetch primary stock data
        df = self.fetch_stock_data(ticker, start_date, end_date)
        
        if df.empty:
            return df
        
        actual_days = len(df)
        self.logger.info(f"✅ Fetched {actual_days} trading days for {ticker}")
        
        return df
    
    def fetch_market_data(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market-wide indices: S&P 500, VIX, Nasdaq
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with market data
        """
        self.logger.info(f"Fetching market-wide data from {start_date} to {end_date}")
        
        market_data = {}
        
        # S&P 500
        try:
            sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            if not sp500.empty:
                market_data['sp500'] = sp500[['Close', 'Volume']].copy()
                market_data['sp500']['Returns'] = market_data['sp500']['Close'].pct_change()
                self.logger.info(f"✅ Fetched S&P 500: {len(sp500)} days")
        except Exception as e:
            self.logger.error(f"Failed to fetch S&P 500: {e}")
        
        # VIX (Volatility Index)
        try:
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix.empty:
                market_data['vix'] = vix[['Close']].copy()
                market_data['vix']['Change'] = market_data['vix']['Close'].pct_change()
                self.logger.info(f"✅ Fetched VIX: {len(vix)} days")
        except Exception as e:
            self.logger.error(f"Failed to fetch VIX: {e}")
        
        # Nasdaq
        try:
            nasdaq = yf.download('^IXIC', start=start_date, end=end_date, progress=False)
            if not nasdaq.empty:
                market_data['nasdaq'] = nasdaq[['Close']].copy()
                market_data['nasdaq']['Returns'] = market_data['nasdaq']['Close'].pct_change()
                self.logger.info(f"✅ Fetched Nasdaq: {len(nasdaq)} days")
        except Exception as e:
            self.logger.error(f"Failed to fetch Nasdaq: {e}")
        
        return market_data
    
    def fetch_sector_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch sector ETF data for the given ticker
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with sector ETF data
        """
        sector_etf = SECTOR_MAP.get(ticker)
        
        if not sector_etf:
            self.logger.warning(f"No sector mapping for {ticker}")
            return None
        
        try:
            self.logger.info(f"Fetching sector ETF {sector_etf} for {ticker}")
            sector_data = yf.download(sector_etf, start=start_date, end=end_date, progress=False)
            
            if not sector_data.empty:
                sector_df = sector_data[['Close']].copy()
                sector_df.columns = ['Sector_Close']
                sector_df['Sector_Returns'] = sector_df['Sector_Close'].pct_change()
                self.logger.info(f"✅ Fetched sector {sector_etf}: {len(sector_df)} days")
                return sector_df
        except Exception as e:
            self.logger.error(f"Failed to fetch sector data: {e}")
        
        return None
    
    def fetch_macro_data(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch macroeconomic indicators: Treasury yields, Dollar index
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with macro data
        """
        self.logger.info(f"Fetching macroeconomic data from {start_date} to {end_date}")
        
        macro_data = {}
        
        # 10-Year Treasury Yield
        try:
            treasury_10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)
            if not treasury_10y.empty:
                macro_data['treasury_10y'] = treasury_10y[['Close']].copy()
                macro_data['treasury_10y']['Change'] = macro_data['treasury_10y']['Close'].diff()
                self.logger.info(f"✅ Fetched 10Y Treasury: {len(treasury_10y)} days")
        except Exception as e:
            self.logger.error(f"Failed to fetch Treasury data: {e}")
        
        # US Dollar Index (DXY)
        try:
            dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)
            if not dxy.empty:
                macro_data['dxy'] = dxy[['Close']].copy()
                macro_data['dxy']['Change'] = macro_data['dxy']['Close'].pct_change()
                self.logger.info(f"✅ Fetched DXY: {len(dxy)} days")
        except Exception as e:
            self.logger.error(f"Failed to fetch DXY: {e}")
        
        # 1-Month Treasury (Fed Funds proxy)
        try:
            fed_proxy = yf.download('^IRX', start=start_date, end=end_date, progress=False)
            if not fed_proxy.empty:
                macro_data['fed_funds'] = fed_proxy[['Close']].copy()
                self.logger.info(f"✅ Fetched Fed Funds proxy: {len(fed_proxy)} days")
        except Exception as e:
            self.logger.error(f"Failed to fetch Fed Funds proxy: {e}")
        
        return macro_data
    
    def fetch_competitor_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for competitors/correlated companies
        
        Args:
            ticker: Primary stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping competitor ticker to DataFrame
        """
        competitors = COMPETITOR_MAP.get(ticker, [])
        
        if not competitors:
            self.logger.warning(f"No competitors defined for {ticker}")
            return {}
        
        self.logger.info(f"Fetching competitor data for {ticker}: {competitors}")
        
        competitor_data = {}
        for comp in competitors:
            try:
                comp_df = self.fetch_stock_data(comp, start_date, end_date)
                if not comp_df.empty:
                    # Keep only essential columns
                    comp_df = comp_df[['Date', 'Close', 'Returns']].copy()
                    comp_df.columns = ['Date', f'{comp}_Close', f'{comp}_Returns']
                    competitor_data[comp] = comp_df
                    self.logger.info(f"✅ Fetched {comp}: {len(comp_df)} days")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Failed to fetch {comp}: {e}")
        
        return competitor_data
    
    def fetch_all_data(
        self,
        ticker: str,
        days: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_market: bool = True,
        include_sector: bool = True,
        include_macro: bool = True,
        include_competitors: bool = True
    ) -> Dict[str, any]:
        """
        Fetch all data: stock, market, sector, macro, competitors
        
        Args:
            ticker: Stock ticker
            days: Number of days (if start_date not provided)
            start_date: Optional start date
            end_date: Optional end date
            include_market: Fetch market indices
            include_sector: Fetch sector ETF
            include_macro: Fetch macro indicators
            include_competitors: Fetch competitor stocks
            
        Returns:
            Dictionary with all data
        """
        # Calculate dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=int(days * 1.4))
            start_date = start_dt.strftime('%Y-%m-%d')
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"COMPREHENSIVE DATA COLLECTION FOR {ticker}")
        self.logger.info(f"Period: {start_date} to {end_date} (target: {days} days)")
        self.logger.info(f"{'='*60}\n")
        
        result = {}
        
        # 1. Primary stock data
        result['stock'] = self.fetch_extended_stock_data(ticker, days, start_date, end_date)
        
        # 2. Market-wide data
        if include_market:
            result['market'] = self.fetch_market_data(start_date, end_date)
        
        # 3. Sector data
        if include_sector:
            result['sector'] = self.fetch_sector_data(ticker, start_date, end_date)
        
        # 4. Macroeconomic data
        if include_macro:
            result['macro'] = self.fetch_macro_data(start_date, end_date)
        
        # 5. Competitor data
        if include_competitors:
            result['competitors'] = self.fetch_competitor_data(ticker, start_date, end_date)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DATA COLLECTION COMPLETE FOR {ticker}")
        self.logger.info(f"{'='*60}\n")
        
        return result


class EnhancedNewsDataCollector(NewsDataCollector):
    """
    Enhanced news collector with competitor news support
    """
    
    def __init__(self):
        super().__init__()
    
    def fetch_competitor_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_per_competitor: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch news for competitor companies
        
        Args:
            ticker: Primary stock ticker
            start_date: Start date
            end_date: End date
            max_per_competitor: Max articles per competitor
            
        Returns:
            Dictionary mapping competitor ticker to news DataFrame
        """
        competitors = COMPETITOR_MAP.get(ticker, [])
        
        if not competitors:
            self.logger.warning(f"No competitors defined for {ticker}")
            return {}
        
        self.logger.info(f"Fetching competitor news for {ticker}: {competitors}")
        
        competitor_news = {}
        for comp in competitors:
            try:
                # Fetch Yahoo Finance news (faster, no API limits)
                comp_articles = self.fetch_yahoo_finance_news(comp, max_articles=max_per_competitor)
                
                if comp_articles:
                    df = pd.DataFrame(comp_articles)
                    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
                    df = df.dropna(subset=['published_at'])
                    df = df[(df['published_at'] >= start_date) & (df['published_at'] <= end_date)]
                    
                    competitor_news[comp] = df
                    self.logger.info(f"✅ Fetched {len(df)} articles for {comp}")
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Failed to fetch news for {comp}: {e}")
        
        return competitor_news
    
    def fetch_all_news_extended(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        include_competitors: bool = True,
        sources: List[str] = ['yahoo_finance', 'finviz']
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch news for ticker AND competitors
        
        Args:
            ticker: Primary stock ticker
            start_date: Start date
            end_date: End date
            include_competitors: Also fetch competitor news
            sources: News sources to use
            
        Returns:
            Dictionary with 'main' and 'competitors' news
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"NEWS COLLECTION FOR {ticker}")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"{'='*60}\n")
        
        result = {}
        
        # Main ticker news
        result['main'] = self.fetch_all_news(ticker, start_date, end_date, sources=sources)
        self.logger.info(f"✅ Main news: {len(result['main'])} articles")
        
        # Competitor news
        if include_competitors:
            result['competitors'] = self.fetch_competitor_news(ticker, start_date, end_date)
            total_comp = sum(len(df) for df in result['competitors'].values())
            self.logger.info(f"✅ Competitor news: {total_comp} articles across {len(result['competitors'])} companies")
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"NEWS COLLECTION COMPLETE")
        self.logger.info(f"{'='*60}\n")
        
        return result


class EnhancedDataManager:
    """
    Comprehensive data manager for enhanced system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stock_collector = EnhancedStockDataCollector()
        self.news_collector = EnhancedNewsDataCollector()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def collect_comprehensive_data(
        self,
        ticker: str,
        days: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_dir: str = 'data/raw'
    ) -> Dict[str, any]:
        """
        Collect ALL data for comprehensive training
        
        Args:
            ticker: Stock ticker
            days: Number of days (default: 1000 = ~4 years)
            start_date: Optional start date
            end_date: Optional end date
            data_dir: Directory to save data
            
        Returns:
            Dictionary with all collected data
        """
        import os
        
        # Create directories
        os.makedirs(f"{data_dir}/stocks", exist_ok=True)
        os.makedirs(f"{data_dir}/news", exist_ok=True)
        os.makedirs(f"{data_dir}/market", exist_ok=True)
        os.makedirs(f"{data_dir}/competitors", exist_ok=True)
        
        # Calculate dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=int(days * 1.4))
            start_date = start_dt.strftime('%Y-%m-%d')
        
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"# COMPREHENSIVE DATA COLLECTION")
        self.logger.info(f"# Ticker: {ticker}")
        self.logger.info(f"# Period: {start_date} to {end_date}")
        self.logger.info(f"# Target Days: {days}")
        self.logger.info(f"{'#'*70}\n")
        
        # Test connectivity first
        self.logger.info("Testing yfinance connectivity...")
        connectivity_ok = self.stock_collector.test_yfinance_connectivity(ticker)
        
        if not connectivity_ok:
            self.logger.warning("yfinance connectivity test failed!")
            self.logger.warning("Will attempt data collection anyway, but it may fail.")
            self.logger.warning("If collection fails, check proxy settings or use cached data.")
        
        # Collect all stock/market data
        stock_data = self.stock_collector.fetch_all_data(
            ticker, days, start_date, end_date,
            include_market=True,
            include_sector=True,
            include_macro=True,
            include_competitors=True
        )
        
        # Collect all news
        news_data = self.news_collector.fetch_all_news_extended(
            ticker, start_date, end_date,
            include_competitors=True
        )
        
        # Save everything
        self._save_data(ticker, stock_data, news_data, data_dir)
        
        # Summary
        self._print_summary(ticker, stock_data, news_data)
        
        return {
            'stock_data': stock_data,
            'news_data': news_data
        }
    
    def _save_data(
        self,
        ticker: str,
        stock_data: Dict,
        news_data: Dict,
        data_dir: str
    ):
        """Save all collected data"""
        # Save main stock data
        if 'stock' in stock_data and not stock_data['stock'].empty:
            path = f"{data_dir}/stocks/{ticker}_stock_data_extended.csv"
            stock_data['stock'].to_csv(path, index=False)
            self.logger.info(f"💾 Saved: {path}")
        
        # Save market data
        if 'market' in stock_data:
            for market_name, df in stock_data['market'].items():
                if not df.empty:
                    path = f"{data_dir}/market/{market_name}_data.csv"
                    df.to_csv(path)
                    self.logger.info(f"💾 Saved: {path}")
        
        # Save sector data
        if 'sector' in stock_data and stock_data['sector'] is not None:
            path = f"{data_dir}/market/{ticker}_sector_data.csv"
            stock_data['sector'].to_csv(path)
            self.logger.info(f"💾 Saved: {path}")
        
        # Save macro data
        if 'macro' in stock_data:
            for macro_name, df in stock_data['macro'].items():
                if not df.empty:
                    path = f"{data_dir}/market/{macro_name}_data.csv"
                    df.to_csv(path)
                    self.logger.info(f"💾 Saved: {path}")
        
        # Save competitor stock data
        if 'competitors' in stock_data:
            for comp, df in stock_data['competitors'].items():
                if not df.empty:
                    path = f"{data_dir}/competitors/{comp}_stock_data.csv"
                    df.to_csv(path, index=False)
                    self.logger.info(f"💾 Saved: {path}")
        
        # Save main news
        if 'main' in news_data and not news_data['main'].empty:
            path = f"{data_dir}/news/{ticker}_news_data_extended.csv"
            news_data['main'].to_csv(path, index=False)
            self.logger.info(f"💾 Saved: {path}")
        
        # Save competitor news
        if 'competitors' in news_data:
            for comp, df in news_data['competitors'].items():
                if not df.empty:
                    path = f"{data_dir}/news/{comp}_news_data.csv"
                    df.to_csv(path, index=False)
                    self.logger.info(f"💾 Saved: {path}")
    
    def _print_summary(
        self,
        ticker: str,
        stock_data: Dict,
        news_data: Dict
    ):
        """Print comprehensive summary"""
        print(f"\n{'='*70}")
        print(f"📊 DATA COLLECTION SUMMARY FOR {ticker}")
        print(f"{'='*70}\n")
        
        # Stock data
        if 'stock' in stock_data and not stock_data['stock'].empty:
            df = stock_data['stock']
            print(f"📈 Main Stock Data:")
            print(f"   Trading Days: {len(df)}")
            print(f"   Date Range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"   Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            print()
        
        # Market data
        if 'market' in stock_data:
            print(f"🌍 Market Data:")
            for name, df in stock_data['market'].items():
                print(f"   {name.upper()}: {len(df)} days")
            print()
        
        # Sector
        if 'sector' in stock_data and stock_data['sector'] is not None:
            print(f"🏢 Sector Data: {len(stock_data['sector'])} days")
            print()
        
        # Macro
        if 'macro' in stock_data:
            print(f"💹 Macro Indicators:")
            for name, df in stock_data['macro'].items():
                print(f"   {name}: {len(df)} days")
            print()
        
        # Competitors
        if 'competitors' in stock_data:
            print(f"🤝 Competitor Stocks:")
            for comp, df in stock_data['competitors'].items():
                print(f"   {comp}: {len(df)} days")
            print()
        
        # News
        if 'main' in news_data:
            print(f"📰 Main News: {len(news_data['main'])} articles")
        
        if 'competitors' in news_data:
            total = sum(len(df) for df in news_data['competitors'].values())
            print(f"📰 Competitor News: {total} articles across {len(news_data['competitors'])} companies")
        
        print(f"\n{'='*70}")
        print(f"✅ DATA COLLECTION COMPLETE!")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example: Collect 1000 days of comprehensive data for AAPL
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = EnhancedDataManager()
    manager.collect_comprehensive_data(
        ticker='AAPL',
        days=1000,  # ~4 years
        data_dir='data/raw'
    )
