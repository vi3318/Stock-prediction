"""
Quick Stock Testing Script
Test the system on any stock ticker you want
"""

import argparse
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_stock.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_stock(ticker, days=60):
    """
    Test the system on a specific stock
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA', 'NVDA', 'META')
        days: Number of days of historical data to collect
    """
    from datetime import datetime, timedelta
    
    logger.info("="*70)
    logger.info(f"🧪 TESTING STOCK: {ticker}")
    logger.info("="*70)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_str} to {end_str}")
    logger.info(f"Collecting {days} days of data...\n")
    
    # Step 1: Collect data
    logger.info("Step 1: Collecting stock and news data...")
    logger.info("-" * 70)
    
    import subprocess
    result = subprocess.run([
        sys.executable, 'main.py',
        '--mode', 'collect',
        '--tickers', ticker,
        '--start-date', start_str,
        '--end-date', end_str
    ], capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"❌ Data collection failed for {ticker}")
        return False
    
    logger.info("✅ Data collection complete!\n")
    
    # Step 2: Preprocess
    logger.info("Step 2: Processing with FinBERT and feature engineering...")
    logger.info("-" * 70)
    
    result = subprocess.run([
        sys.executable, 'main.py',
        '--mode', 'preprocess',
        '--tickers', ticker
    ], capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"❌ Preprocessing failed for {ticker}")
        return False
    
    logger.info("✅ Preprocessing complete!\n")
    
    # Step 3: Show results
    logger.info("Step 3: Analyzing processed data...")
    logger.info("-" * 70)
    
    import pandas as pd
    import numpy as np
    
    # Load the processed data
    processed_file = f'data/processed/{ticker}_processed_data.pkl'
    
    if os.path.exists(processed_file):
        data = pd.read_pickle(processed_file)
        
        logger.info(f"\n📊 {ticker} Data Summary:")
        logger.info("=" * 70)
        
        stock_data = data['stock_data']
        news_data = data['news_data']
        embeddings = data['news_embeddings']
        numerical_features = data['numerical_features']
        target = data['target']
        
        # Stock statistics
        logger.info(f"\n📈 Stock Data ({len(stock_data)} days):")
        logger.info(f"   Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
        logger.info(f"   Current price: ${stock_data['Close'].iloc[-1]:.2f}")
        logger.info(f"   Average volume: {stock_data['Volume'].mean():,.0f}")
        
        # Price movement
        total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
        logger.info(f"   Total return: {total_return:+.2f}%")
        
        # News statistics
        logger.info(f"\n📰 News Data ({len(news_data)} articles):")
        if 'title' in news_data.columns:
            logger.info(f"   Sample headlines:")
            for i, title in enumerate(news_data['title'].head(3), 1):
                logger.info(f"   {i}. {title[:80]}...")
        
        # Feature statistics
        logger.info(f"\n🔢 Features Created:")
        logger.info(f"   FinBERT embeddings: {embeddings.shape}")
        logger.info(f"   Numerical features: {numerical_features.shape[1]} features")
        logger.info(f"   Total features per day: {embeddings.shape[1] + numerical_features.shape[1]}")
        
        # Target distribution
        up_days = target.sum()
        down_days = len(target) - target.sum()
        logger.info(f"\n🎯 Target Distribution:")
        logger.info(f"   Up days: {up_days} ({up_days/len(target)*100:.1f}%)")
        logger.info(f"   Down days: {down_days} ({down_days/len(target)*100:.1f}%)")
        
        # Volatility
        returns = numerical_features['returns'].dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        logger.info(f"\n📊 Risk Metrics:")
        logger.info(f"   Daily volatility: {returns.std()*100:.2f}%")
        logger.info(f"   Annualized volatility: {volatility:.2f}%")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ {ticker} is ready for modeling!")
        logger.info("=" * 70)
        
        # Show file locations
        logger.info(f"\n📁 Files created:")
        logger.info(f"   Stock data: data/raw/stocks/{ticker}_stock_data.csv")
        logger.info(f"   News data: data/raw/news/{ticker}_news_data.csv")
        logger.info(f"   Processed: data/processed/{ticker}_processed_data.pkl")
        logger.info(f"   Combined: data/processed/combined_data.pkl")
        
        return True
    else:
        logger.error(f"❌ Processed file not found: {processed_file}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test the stock prediction system on any ticker'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol (e.g., TSLA, NVDA, META, AMZN)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=60,
        help='Number of days of historical data (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Convert ticker to uppercase
    ticker = args.ticker.upper()
    
    logger.info("\n" + "🚀" * 35)
    logger.info("   STOCK PREDICTION SYSTEM - QUICK TEST")
    logger.info("🚀" * 35 + "\n")
    
    success = test_stock(ticker, args.days)
    
    if success:
        logger.info("\n" + "✅" * 35)
        logger.info(f"   {ticker} TEST COMPLETED SUCCESSFULLY!")
        logger.info("✅" * 35)
        
        logger.info("\n💡 Next steps:")
        logger.info(f"   1. View data: ls -la data/raw/stocks/{ticker}*")
        logger.info(f"   2. Check news: ls -la data/raw/news/{ticker}*")
        logger.info(f"   3. Train model: python3 main.py --mode train --tickers {ticker}")
        logger.info(f"   4. Run full pipeline: python3 main.py --mode full --tickers {ticker} --walk-forward")
        
    else:
        logger.error("\n" + "❌" * 35)
        logger.error(f"   {ticker} TEST FAILED")
        logger.error("❌" * 35)
        logger.error("\n💡 Troubleshooting:")
        logger.error("   1. Check if ticker symbol is valid")
        logger.error("   2. Verify internet connection")
        logger.error("   3. Check logs/test_stock.log for details")
        sys.exit(1)
