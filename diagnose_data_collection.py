#!/usr/bin/env python3
"""
Data Collection Diagnostic Script
Tests yfinance connectivity and provides troubleshooting guidance
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_collection.enhanced_collectors import EnhancedStockDataCollector
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check environment variables and proxy settings"""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)

    # Check proxy settings
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')

    if http_proxy:
        print(f"✅ HTTP_PROXY: {http_proxy}")
    else:
        print("❌ HTTP_PROXY: Not set")

    if https_proxy:
        print(f"✅ HTTPS_PROXY: {https_proxy}")
    else:
        print("❌ HTTPS_PROXY: Not set")

    if not http_proxy and not https_proxy:
        print("\n⚠️  No proxy environment variables detected.")
        print("   If you're behind a corporate proxy, you may need to set:")
        print("   - Windows PowerShell: $env:HTTP_PROXY='http://proxy.company.com:8080'")
        print("   - Windows PowerShell: $env:HTTPS_PROXY='http://proxy.company.com:8080'")
        print("   - Or permanently in Windows Environment Variables")

    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")

    # Check if we're in virtual environment
    venv = os.getenv('VIRTUAL_ENV') or os.getenv('CONDA_DEFAULT_ENV')
    if venv:
        print(f"✅ Virtual environment: {venv}")
    else:
        print("⚠️  Not in a virtual environment")


def test_connectivity():
    """Test yfinance connectivity"""
    print("\n" + "="*60)
    print("CONNECTIVITY TEST")
    print("="*60)

    collector = EnhancedStockDataCollector()

    # Test with different tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in test_tickers:
        print(f"\nTesting {ticker}...")
        success = collector.test_yfinance_connectivity(ticker)
        if success:
            break  # If one works, we're good
        print()

    return success


def check_cached_data():
    """Check for cached data files"""
    print("\n" + "="*60)
    print("CACHED DATA CHECK")
    print("="*60)

    data_dirs = [
        'data/raw/stocks',
        'data/raw/news',
        'data/raw/market',
        'data/raw/competitors'
    ]

    found_cache = False
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            if files:
                print(f"✅ {data_dir}: {len(files)} files")
                found_cache = True
                # Show a few examples
                for f in files[:3]:
                    print(f"   - {f}")
                if len(files) > 3:
                    print(f"   ... and {len(files)-3} more")
            else:
                print(f"📁 {data_dir}: Empty")
        else:
            print(f"❌ {data_dir}: Directory doesn't exist")

    if found_cache:
        print("\n💡 Cached data found! The system will use cached data if live fetching fails.")
    else:
        print("\n❌ No cached data found. First run will need live data.")


def provide_guidance():
    """Provide troubleshooting guidance"""
    print("\n" + "="*60)
    print("TROUBLESHOOTING GUIDANCE")
    print("="*60)

    guidance = """
🔧 COMMON FIXES:

1. PROXY SETTINGS (Most Common Issue):
   If you're behind a corporate proxy/firewall, set environment variables:

   Windows PowerShell (temporary):
   ```
   $env:HTTP_PROXY = "http://proxy.company.com:8080"
   $env:HTTPS_PROXY = "http://proxy.company.com:8080"
   ```

   Windows Environment Variables (permanent):
   - Search for "Environment Variables" in Windows search
   - Add HTTP_PROXY and HTTPS_PROXY variables

2. NETWORK/FIREWALL:
   - Try from a different network (home vs work)
   - Check if Yahoo Finance is blocked by your firewall
   - Some corporate networks block financial data APIs

3. PYTHON ENVIRONMENT:
   - Ensure you're using the virtual environment: `activate_venv.bat`
   - Update packages: `pip install --upgrade yfinance requests`

4. FALLBACK OPTIONS:
   - The system will automatically use cached CSV files if they exist
   - Place valid stock data CSVs in `data/raw/stocks/` folder
   - Format: Date,Open,High,Low,Close,Volume columns required

5. ALTERNATIVE DATA SOURCES:
   - Consider using Alpha Vantage API (requires API key)
   - Or manually download data from Yahoo Finance and save as CSV

🚀 TO RUN THE FULL PIPELINE:
```
.\\RUN_EXPERIMENTS_WITH_VENV.bat
```

Or directly:
```
python scripts\\train_full_system.py --ticker AAPL --days 1000
```

📞 IF ISSUES PERSIST:
- Check the logs in `logs/full_training.log`
- Verify your internet connection
- Try a different ticker (MSFT, GOOGL, TSLA)
"""
    print(guidance)


def main():
    """Main diagnostic function"""
    print("🔍 STOCK PREDICTION DATA COLLECTION DIAGNOSTIC")
    print("This script tests yfinance connectivity and provides setup guidance.\n")

    # Run checks
    check_environment()
    connectivity_ok = test_connectivity()
    check_cached_data()
    provide_guidance()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if connectivity_ok:
        print("✅ yfinance connectivity: WORKING")
        print("🎯 Ready to run data collection!")
    else:
        print("❌ yfinance connectivity: FAILED")
        print("🔧 Follow the troubleshooting guidance above.")

    print("\n📝 Next steps:")
    print("1. Fix any proxy/network issues")
    print("2. Run: .\\RUN_EXPERIMENTS_WITH_VENV.bat")
    print("3. Check logs/full_training.log for details")


if __name__ == "__main__":
    main()