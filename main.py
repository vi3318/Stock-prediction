"""  
Main Execution Script for Stock Price Prediction System
Orchestrates data collection, preprocessing, training, and evaluation
"""

import argparse
import logging
import yaml
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction using News and NLP Parsing'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['collect', 'preprocess', 'train', 'evaluate', 'full'],
        default='full',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Stock tickers to process (e.g., AAPL MSFT)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Use walk-forward cross-validation for evaluation'
    )
    
    parser.add_argument(
        '--window-type',
        type=str,
        choices=['expanding', 'sliding'],
        default='expanding',
        help='Window type for walk-forward CV (expanding or sliding)'
    )
    
    parser.add_argument(
        '--max-folds',
        type=int,
        default=None,
        help='Maximum number of folds for walk-forward CV'
    )
    
    parser.add_argument(
        '--sectors',
        nargs='+',
        choices=['technology', 'financial', 'healthcare', 'energy', 'consumer', 'all'],
        help='Sectors to evaluate (e.g., technology financial) or "all" for all sectors'
    )
    
    parser.add_argument(
        '--multi-sector',
        action='store_true',
        help='Enable multi-sector evaluation to test generalization across sectors'
    )
    
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/raw/stocks', exist_ok=True)
    os.makedirs('data/raw/news', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Display banner
    logger.info("="*70)
    logger.info("   Stock Price Prediction System")
    logger.info("   Using News and NLP Parsing with Deep Learning")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    try:
        if args.mode == 'collect':
            logger.info("Starting data collection...")
            from data_collection.collectors import DataManager
            
            tickers = args.tickers if args.tickers else config['data']['tickers']
            start_date = args.start_date if args.start_date else config['data']['start_date']
            end_date = args.end_date if args.end_date else config['data']['end_date']
            
            data_manager = DataManager(config)
            data_manager.collect_and_save(tickers, start_date, end_date)
            
            logger.info("Data collection completed!")
        
        elif args.mode == 'preprocess':
            logger.info("Starting data preprocessing...")
            
            # Import preprocessing modules
            from preprocessing.nlp_processor import NLPPipeline
            from features.numerical_features import NumericalFeaturePipeline
            from features.temporal_features import TemporalFeatureEngineer
            
            tickers = args.tickers if args.tickers else config['data']['tickers']
            
            for ticker in tickers:
                logger.info(f"Processing {ticker}...")
                
                # Load raw data
                stock_file = f'data/raw/stocks/{ticker}_stock_data.csv'
                news_file = f'data/raw/news/{ticker}_news_data.csv'
                
                if not os.path.exists(stock_file):
                    logger.error(f"Stock data not found: {stock_file}")
                    logger.error("Please run data collection first (--mode collect)")
                    continue
                
                if not os.path.exists(news_file):
                    logger.error(f"News data not found: {news_file}")
                    logger.error("Please run data collection first (--mode collect)")
                    continue
                
                # Load data
                stock_data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                news_data = pd.read_csv(news_file)
                
                logger.info(f"  Stock data: {len(stock_data)} records")
                logger.info(f"  News data: {len(news_data)} articles")
                
                # Initialize processors
                from preprocessing.nlp_processor import FinancialTextEmbedder
                nlp_embedder = FinancialTextEmbedder('ProsusAI/finbert')
                
                # Process news text
                logger.info("  Processing news with FinBERT...")
                news_texts = news_data['title'].fillna('').tolist()
                news_embeddings = nlp_embedder.get_embeddings(news_texts)
                
                # Create simple numerical features 
                logger.info("  Creating numerical features...")
                numerical_features = stock_data.copy()
                
                # Add basic technical indicators
                numerical_features['returns'] = stock_data['Close'].pct_change()
                numerical_features['volatility'] = numerical_features['returns'].rolling(window=20).std()
                numerical_features['sma_20'] = stock_data['Close'].rolling(window=20).mean()
                numerical_features['rsi'] = (numerical_features['returns'].rolling(window=14).apply(
                    lambda x: 100 - 100 / (1 + x[x > 0].sum() / -x[x < 0].sum()) if len(x[x < 0]) > 0 else 50
                ))
                
                # Create temporal features (simple time-based)
                logger.info("  Creating temporal features...")
                temporal_features = pd.DataFrame(index=stock_data.index)
                
                # Convert index to datetime if it's not already
                if not isinstance(stock_data.index, pd.DatetimeIndex):
                    stock_data.index = pd.to_datetime(stock_data.index, utc=True)
                elif stock_data.index.tz is not None:
                    # Convert timezone-aware to UTC then remove timezone
                    stock_data.index = stock_data.index.tz_convert('UTC').tz_localize(None)
                
                temporal_features['day_of_week'] = stock_data.index.dayofweek
                temporal_features['month'] = stock_data.index.month
                temporal_features['quarter'] = stock_data.index.quarter
                
                # Create target variable (next day price direction)
                target = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
                
                # Combine all features
                logger.info("  Combining features...")
                combined_data = {
                    'stock_data': stock_data,
                    'news_data': news_data,
                    'news_embeddings': news_embeddings,
                    'numerical_features': numerical_features,
                    'temporal_features': temporal_features,
                    'target': target,
                    'ticker': ticker
                }
                
                # Save processed data
                os.makedirs('data/processed', exist_ok=True)
                output_file = f'data/processed/{ticker}_processed_data.pkl'
                pd.to_pickle(combined_data, output_file)
                
                logger.info(f"  Saved processed data to {output_file}")
            
            # Create combined dataset
            logger.info("Creating combined dataset...")
            combined_datasets = []
            
            for ticker in tickers:
                processed_file = f'data/processed/{ticker}_processed_data.pkl'
                if os.path.exists(processed_file):
                    data = pd.read_pickle(processed_file)
                    
                    # Extract and prepare the data for machine learning
                    stock_data = data['stock_data']
                    numerical_features = data['numerical_features']
                    temporal_features = data['temporal_features']
                    news_embeddings = data['news_embeddings']
                    target = data['target']
                    
                    # Ensure all dataframes have timezone-naive indices
                    if hasattr(stock_data.index, 'tz') and stock_data.index.tz is not None:
                        stock_data.index = stock_data.index.tz_localize(None)
                    if hasattr(numerical_features.index, 'tz') and numerical_features.index.tz is not None:
                        numerical_features.index = numerical_features.index.tz_localize(None)
                    if hasattr(temporal_features.index, 'tz') and temporal_features.index.tz is not None:
                        temporal_features.index = temporal_features.index.tz_localize(None)
                    
                    # Add news embeddings - aggregate by taking mean for each date
                    # For now, use the first N embeddings or repeat to match stock data length
                    if len(news_embeddings) >= len(stock_data):
                        # Take first N embeddings to match stock data
                        aligned_embeddings = news_embeddings[:len(stock_data)]
                    else:
                        # Repeat embeddings to match stock data length
                        repeat_factor = len(stock_data) // len(news_embeddings) + 1
                        repeated_embeddings = np.tile(news_embeddings, (repeat_factor, 1))
                        aligned_embeddings = repeated_embeddings[:len(stock_data)]
                    
                    embeddings_df = pd.DataFrame(
                        aligned_embeddings, 
                        index=stock_data.index,
                        columns=[f'embedding_{i}' for i in range(aligned_embeddings.shape[1])]
                    )
                    
                    # Create a combined DataFrame - don't duplicate columns from numerical_features
                    ml_data = numerical_features.copy()
                    
                    # Add temporal features (avoiding column overlaps)
                    for col in temporal_features.columns:
                        if col not in ml_data.columns:
                            ml_data[col] = temporal_features[col]
                    
                    # Add embeddings
                    for col in embeddings_df.columns:
                        ml_data[col] = embeddings_df[col]
                    
                    ml_data['target'] = target.values if hasattr(target, 'values') else target
                    ml_data['ticker'] = ticker
                    ml_data['date'] = ml_data.index
                    
                    combined_datasets.append(ml_data)
            
            if combined_datasets:
                # Combine all tickers into one DataFrame
                combined_df = pd.concat(combined_datasets, ignore_index=True)
                combined_file = 'data/processed/combined_data.pkl'
                pd.to_pickle(combined_df, combined_file)
                logger.info(f"Combined dataset saved to {combined_file}")
                logger.info(f"Combined dataset shape: {combined_df.shape}")
            
            logger.info("Data preprocessing completed!")
        
        elif args.mode == 'train':
            logger.info("Starting model training...")
            logger.info("Training pipeline would execute here")
            logger.info("(See src/models/ and src/training/ modules)")
        
        elif args.mode == 'evaluate':
            logger.info("Starting model evaluation...")
            
            if args.multi_sector:
                logger.info("Using multi-sector evaluation...")
                from evaluation.multi_sector_evaluator import MultiSectorEvaluator
                from models.learnable_hybrid_model import LateFusionModelWithLearnableWeights
                
                # Determine sectors
                if args.sectors and 'all' in args.sectors:
                    sectors = None  # Evaluate all sectors
                elif args.sectors:
                    sectors = args.sectors
                else:
                    sectors = ['technology', 'financial']  # Default to 2 sectors
                
                logger.info(f"Evaluating sectors: {sectors if sectors else 'ALL'}")
                
                # Mock data loader (replace with actual data loading)
                def data_loader(ticker):
                    logger.info(f"  Loading data for {ticker}...")
                    # This should load actual stock + news data for the ticker
                    # For now, return mock data structure
                    return {
                        'X': np.random.randn(1000, 50),
                        'y': np.random.randint(0, 3, 1000)
                    }
                
                # Initialize evaluator
                evaluator = MultiSectorEvaluator(
                    model_class=LateFusionModelWithLearnableWeights,
                    config=config,
                    sectors=sectors
                )
                
                # Evaluate all sectors
                sector_results = evaluator.evaluate_all_sectors(data_loader)
                
                # Display comparison
                logger.info("\n" + "="*70)
                logger.info("MULTI-SECTOR EVALUATION RESULTS")
                logger.info("="*70)
                
                comparison_df = evaluator.compare_sectors(sector_results)
                print("\n" + comparison_df.to_string(index=False))
                
                # Test statistical significance
                logger.info("\n" + "-"*70)
                logger.info("STATISTICAL TESTS (Accuracy)")
                logger.info("-"*70)
                
                sig_tests = evaluator.test_sector_differences(sector_results, metric='accuracy')
                
                for (s1, s2), test in sig_tests.items():
                    logger.info(f"\n{s1} vs {s2}:")
                    logger.info(f"  Mean difference: {test['mean_diff']:.4f}")
                    logger.info(f"  p-value: {test['p_value']:.4f}")
                    logger.info(f"  Cohen's d: {test['cohen_d']:.3f}")
                    logger.info(f"  Significant: {'✅ YES' if test['significant'] else '❌ NO'}")
                
                # Save results
                os.makedirs('results/multi_sector', exist_ok=True)
                evaluator.save_results(sector_results, 'results/multi_sector')
                
                # Create visualizations
                evaluator.visualize_sector_comparison(
                    sector_results,
                    'results/multi_sector/sector_comparison.png'
                )
                
                evaluator.visualize_learned_weights(
                    sector_results,
                    'results/multi_sector/learned_weights.png'
                )
                
                logger.info("\n✅ Multi-sector evaluation complete!")
                logger.info("   Results saved to results/multi_sector/")
            
            elif args.walk_forward:
                logger.info("Using walk-forward cross-validation...")
                from evaluation.walk_forward_evaluator import WalkForwardEvaluator
                from models.hybrid_model import LateFusionModel
                
                # Load processed data
                data_path = 'data/processed/combined_data.pkl'
                if not os.path.exists(data_path):
                    logger.error(f"Processed data not found at {data_path}")
                    logger.error("Please run preprocessing first (--mode preprocess)")
                    sys.exit(1)
                
                data = pd.read_pickle(data_path)
                
                # Initialize evaluator
                evaluator = WalkForwardEvaluator(config)
                
                # Run walk-forward evaluation
                results = evaluator.evaluate_with_walk_forward(
                    model_class=LateFusionModel,
                    data=data,
                    window_type=args.window_type,
                    initial_train_months=24,
                    test_months=3,
                    step_months=3,
                    max_folds=args.max_folds
                )
                
                logger.info("\n" + "="*70)
                logger.info("WALK-FORWARD CROSS-VALIDATION RESULTS")
                logger.info("="*70)
                
                # Display aggregate results
                agg_metrics = results['aggregate_metrics']
                for metric_name, stats in agg_metrics.items():
                    logger.info(
                        f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                        f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])"
                    )
                
                logger.info(f"\nTotal folds evaluated: {results['num_folds']}")
                logger.info(f"Window type: {results['window_type']}")
            else:
                logger.info("Using standard holdout evaluation...")
                logger.info("(For more robust evaluation, use --walk-forward flag)")

        
        elif args.mode == 'full':
            logger.info("Running full pipeline...")
            logger.info("This would run: collect -> preprocess -> train -> evaluate")
        
        logger.info("="*70)
        logger.info("Execution completed successfully!")
        logger.info("="*70)
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
