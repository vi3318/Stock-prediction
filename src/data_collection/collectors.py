"""
Data Collection Module
Handles fetching financial news and stock price data from various sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collects historical stock price data"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            
            # Calculate additional features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_tickers(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data = {}
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, start_date, end_date)
            if not df.empty:
                data[ticker] = df
            time.sleep(0.5)  # Rate limiting
        
        return data


class NewsDataCollector:
    """Collects financial news from various sources"""
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.newsapi = NewsApiClient(api_key=self.newsapi_key) if self.newsapi_key else None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fetch_newsapi(
        self, 
        query: str, 
        start_date: str, 
        end_date: str,
        language: str = 'en'
    ) -> List[Dict]:
        """
        Fetch news from NewsAPI
        
        Args:
            query: Search query (company name or ticker)
            start_date: Start date
            end_date: End date
            language: Language code
            
        Returns:
            List of news articles
        """
        if not self.newsapi:
            self.logger.warning("NewsAPI key not configured")
            return []
        
        try:
            self.logger.info(f"Fetching NewsAPI articles for {query}")
            all_articles = []
            
            # NewsAPI has a limit of 1 month per request
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            while current_date < end:
                batch_end = min(current_date + timedelta(days=30), end)
                
                response = self.newsapi.get_everything(
                    q=query,
                    from_param=current_date.strftime('%Y-%m-%d'),
                    to=batch_end.strftime('%Y-%m-%d'),
                    language=language,
                    sort_by='relevancy',
                    page_size=100
                )
                
                if response.get('articles'):
                    all_articles.extend(response['articles'])
                
                current_date = batch_end + timedelta(days=1)
                time.sleep(1)  # Rate limiting
            
            # Process articles
            processed_articles = []
            for article in all_articles:
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'author': article.get('author', ''),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', ''),
                    'query': query
                })
            
            self.logger.info(f"Fetched {len(processed_articles)} articles from NewsAPI")
            return processed_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI data: {str(e)}")
            return []
    
    def fetch_yahoo_finance_news(self, ticker: str, max_articles: int = 50) -> List[Dict]:
        """
        Scrape news from Yahoo Finance for a specific ticker
        
        Args:
            ticker: Stock ticker
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        try:
            self.logger.info(f"Fetching Yahoo Finance news for {ticker}")
            stock = yf.Ticker(ticker)
            news = stock.news[:max_articles]
            
            processed_news = []
            for item in news:
                processed_news.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'content': item.get('summary', ''),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'author': '',
                    'published_at': datetime.fromtimestamp(
                        item.get('providerPublishTime', 0)
                    ).isoformat(),
                    'url': item.get('link', ''),
                    'ticker': ticker
                })
            
            self.logger.info(f"Fetched {len(processed_news)} articles from Yahoo Finance")
            return processed_news
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance news: {str(e)}")
            return []
    
    def fetch_finviz_news(self, ticker: str) -> List[Dict]:
        """
        Scrape news from Finviz
        
        Args:
            ticker: Stock ticker
            
        Returns:
            List of news articles
        """
        try:
            self.logger.info(f"Fetching Finviz news for {ticker}")
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_table = soup.find('table', {'id': 'news-table'})
            if not news_table:
                return []
            
            news_items = []
            for row in news_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    date_cell = cells[0].text.strip()
                    news_cell = cells[1]
                    
                    link = news_cell.find('a')
                    if link:
                        news_items.append({
                            'title': link.text.strip(),
                            'description': link.text.strip(),
                            'content': '',
                            'source': 'Finviz',
                            'author': '',
                            'published_at': date_cell,
                            'url': link.get('href', ''),
                            'ticker': ticker
                        })
            
            self.logger.info(f"Fetched {len(news_items)} articles from Finviz")
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching Finviz news: {str(e)}")
            return []
    
    def fetch_all_news(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        sources: List[str] = ['newsapi', 'yahoo_finance', 'finviz']
    ) -> pd.DataFrame:
        """
        Fetch news from all configured sources
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            sources: List of sources to use
            
        Returns:
            DataFrame with all news articles
        """
        all_news = []
        
        if 'newsapi' in sources and self.newsapi:
            all_news.extend(self.fetch_newsapi(ticker, start_date, end_date))
        
        if 'yahoo_finance' in sources:
            all_news.extend(self.fetch_yahoo_finance_news(ticker))
        
        if 'finviz' in sources:
            all_news.extend(self.fetch_finviz_news(ticker))
        
        if not all_news:
            self.logger.warning(f"No news found for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_news)
        
        # Ensure ticker column
        if 'ticker' not in df.columns:
            df['ticker'] = ticker
        
        # Parse dates
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # Remove duplicates based on title
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by date
        df = df.sort_values('published_at')
        
        self.logger.info(f"Total unique articles collected: {len(df)}")
        return df


class DataManager:
    """Manages data collection and storage"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stock_collector = StockDataCollector()
        self.news_collector = NewsDataCollector()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def collect_and_save(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str,
        data_dir: str = 'data/raw'
    ):
        """
        Collect and save both stock and news data
        
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            data_dir: Directory to save data
        """
        os.makedirs(f"{data_dir}/stocks", exist_ok=True)
        os.makedirs(f"{data_dir}/news", exist_ok=True)
        
        for ticker in tickers:
            self.logger.info(f"Processing {ticker}")
            
            # Collect stock data
            stock_data = self.stock_collector.fetch_stock_data(
                ticker, start_date, end_date
            )
            if not stock_data.empty:
                stock_path = f"{data_dir}/stocks/{ticker}_stock_data.csv"
                stock_data.to_csv(stock_path, index=False)
                self.logger.info(f"Saved stock data to {stock_path}")
            
            # Collect news data
            news_data = self.news_collector.fetch_all_news(
                ticker, start_date, end_date,
                sources=self.config.get('data', {}).get('news_sources', ['yahoo_finance'])
            )
            if not news_data.empty:
                news_path = f"{data_dir}/news/{ticker}_news_data.csv"
                news_data.to_csv(news_path, index=False)
                self.logger.info(f"Saved news data to {news_path}")
            
            time.sleep(2)  # Rate limiting between tickers
        
        self.logger.info("Data collection completed")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    collector = DataManager({})
    collector.collect_and_save(
        tickers=['AAPL'],
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
