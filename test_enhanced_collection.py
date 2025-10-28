"""
Quick Test: Enhanced Data Collection
Tests the new comprehensive data collection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.enhanced_collectors import EnhancedDataManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_enhanced_collection():
    """Test enhanced data collection for AAPL"""
    
    print("\n" + "="*70)
    print("TESTING ENHANCED DATA COLLECTION SYSTEM")
    print("="*70 + "\n")
    
    # Create manager
    manager = EnhancedDataManager()
    
    # Test with 100 days first (faster for demo)
    # For production, use days=1000 for 4 years of data
    print("Collecting 100 days of comprehensive data for AAPL...")
    print("(For production: use days=1000 for ~4 years)\n")
    
    data = manager.collect_comprehensive_data(
        ticker='AAPL',
        days=100,  # Use 1000 for full system
        data_dir='data/raw'
    )
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70 + "\n")
    
    # Show what was collected
    print("📦 Data Collected:")
    print(f"   Stock data: {len(data['stock_data']['stock'])} days")
    
    if 'market' in data['stock_data']:
        print(f"   Market indices: {len(data['stock_data']['market'])} types")
        for name in data['stock_data']['market'].keys():
            print(f"      - {name}")
    
    if 'sector' in data['stock_data'] and data['stock_data']['sector'] is not None:
        print(f"   Sector data: ✅")
    
    if 'macro' in data['stock_data']:
        print(f"   Macro indicators: {len(data['stock_data']['macro'])} types")
        for name in data['stock_data']['macro'].keys():
            print(f"      - {name}")
    
    if 'competitors' in data['stock_data']:
        print(f"   Competitor stocks: {len(data['stock_data']['competitors'])} companies")
        for comp in data['stock_data']['competitors'].keys():
            print(f"      - {comp}")
    
    if 'main' in data['news_data']:
        print(f"   Main news: {len(data['news_data']['main'])} articles")
    
    if 'competitors' in data['news_data']:
        total_news = sum(len(df) for df in data['news_data']['competitors'].values())
        print(f"   Competitor news: {total_news} articles")
    
    print("\n✅ Enhanced data collection is working!")
    print("\nNext steps:")
    print("1. Run enhanced feature engineering on this data")
    print("2. Train advanced models (BiLSTM, Transformer, Hybrid)")
    print("3. Run hyperparameter optimization")
    print("4. Achieve 80%+ accuracy target!")
    print()


if __name__ == "__main__":
    test_enhanced_collection()
