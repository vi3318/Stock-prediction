"""
Full System Training Script
Complete end-to-end pipeline for AAPL stock prediction with 80%+ accuracy target
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import logging
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.data_collection.enhanced_collectors import EnhancedDataManager
from src.features.enhanced_features import EnhancedNumericalFeatures
from src.models.advanced_models import create_model
from src.training.enhanced_trainer import EnhancedTrainer
from src.training.hyperparameter_optimizer import HyperparameterOptimizer
from src.preprocessing.nlp_processor import FinancialTextEmbedder
from scripts.compare_models import ModelComparator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FullTrainingPipeline:
    """
    Complete end-to-end training pipeline
    """
    
    def __init__(
        self,
        ticker: str = 'AAPL',
        days: int = 1000,
        sequence_length: int = 120,
        output_dir: str = 'results/full_training'
    ):
        """
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data
            sequence_length: Sequence length for models (90-180 days)
            output_dir: Output directory for results
        """
        self.ticker = ticker
        self.days = days
        self.sequence_length = sequence_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / 'pipeline_checkpoint.json'
        
        # Data split ratios
        self.test_split = 0.15
        self.validation_split = 0.15
        
        logger.info(f"\n{'='*80}")
        logger.info("FULL TRAINING PIPELINE INITIALIZED")
        logger.info(f"{'='*80}")
        logger.info(f"Ticker: {ticker}")
        logger.info(f"Days: {days}")
        logger.info(f"Sequence Length: {sequence_length}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"{'='*80}\n")
    
    def save_checkpoint(self, step: int, step_name: str):
        """Save pipeline checkpoint"""
        checkpoint = {
            'completed_step': step,
            'step_name': step_name,
            'ticker': self.ticker,
            'days': self.days,
            'sequence_length': self.sequence_length,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save specific data based on completed step
        if step >= 3:
            checkpoint['num_features'] = self.num_features
            checkpoint['train_samples'] = len(self.X_num_train)
            checkpoint['val_samples'] = len(self.X_num_val)
            checkpoint['test_samples'] = len(self.X_num_test)
        
        if step >= 4 and hasattr(self, 'comparison_results'):
            checkpoint['comparison_complete'] = True
        
        if step >= 5 and hasattr(self, 'best_params'):
            checkpoint['best_params'] = self.best_params
            checkpoint['best_model_type'] = self.best_model_type
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"✅ Checkpoint saved: Step {step} ({step_name}) complete")
    
    def load_checkpoint(self):
        """Load pipeline checkpoint if exists"""
        if not self.checkpoint_file.exists():
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        logger.info(f"\n{'='*80}")
        logger.info("CHECKPOINT FOUND")
        logger.info(f"{'='*80}")
        logger.info(f"Last completed step: {checkpoint['completed_step']} - {checkpoint['step_name']}")
        logger.info(f"Timestamp: {checkpoint['timestamp']}")
        logger.info(f"{'='*80}\n")
        
        return checkpoint
    
    def clear_checkpoint(self):
        """Clear checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")
    
    def step1_collect_data(self):
        """Step 1: Collect 1000 days of data"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*80 + "\n")
        
        # Initialize data manager
        data_manager = EnhancedDataManager()
        
        # Collect comprehensive data
        result = data_manager.collect_comprehensive_data(
            ticker=self.ticker,
            days=self.days
        )
        
        stock_df = result['stock_data']['stock']
        news_df = result['news_data']['main']
        
        # Store additional data for feature engineering
        market_data = result['stock_data'].get('market', {})
        sector_df = result['stock_data'].get('sector', None)
        macro_data = result['stock_data'].get('macro', {})
        competitor_data = result['stock_data'].get('competitors', {})
        
        logger.info(f"Stock data collected: {len(stock_df)} rows")
        logger.info(f"News data collected: {len(news_df)} rows")
        logger.info(f"Market data: {len(market_data)} datasets")
        logger.info(f"Sector data: {'Yes' if sector_df is not None else 'No'}")
        logger.info(f"Macro data: {len(macro_data)} datasets")
        logger.info(f"Competitor data: {len(competitor_data)} stocks")
        
        # Save raw data
        stock_df.to_csv(self.output_dir / 'stock_data.csv', index=False)
        news_df.to_csv(self.output_dir / 'news_data.csv', index=False)
        # Validate collected stock data before proceeding
        if stock_df is None or len(stock_df) == 0:
            logger.error(
                "No stock data was collected. Aborting pipeline. "
                "Check network connection, yfinance availability, or existing CSVs in data/raw/stocks/"
            )
            raise RuntimeError(f"No stock data collected for ticker: {self.ticker}")

        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_cols if c not in stock_df.columns]
        if missing:
            logger.error(f"Stock data missing required columns: {missing}. Aborting pipeline.")
            raise RuntimeError(f"Stock data missing required columns: {missing}")

        self.stock_df = stock_df
        self.news_df = news_df
        # Store additional data for feature engineering
        self.market_data = market_data
        self.sector_df = sector_df
        self.macro_data = macro_data
        self.competitor_data = competitor_data
        
        logger.info("✅ Data collection complete\n")
        
        # Save checkpoint
        self.save_checkpoint(1, "Data Collection")
        
        return stock_df, news_df
    
    def step1b_process_news_embeddings(self):
        """Step 1B: Generate FinBERT embeddings for news (Corrected Version with Robust Date Parsing)"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1B: NEWS EMBEDDING GENERATION")
        logger.info("="*80 + "\n")

        # Check if news_df exists and has data
        if not hasattr(self, 'news_df') or self.news_df is None or self.news_df.empty:
            logger.warning("No news data loaded in self.news_df - skipping embedding generation.")
            self.news_df_with_embeddings = None
            self.news_embeddings = None
            return

        try:
            logger.info(f"Generating FinBERT embeddings for {len(self.news_df)} news articles...")

            # Initialize FinBERT
            embedder = FinancialTextEmbedder() # Uses default 'ProsusAI/finbert'

            # --- Robust Date Handling ---
            published_col_name = None
            # 1. Find the date column
            if 'published_at' in self.news_df.columns:
                published_col_name = 'published_at'
            else:
                # Look for likely Finviz date column (often unnamed or has date-like string)
                for col in self.news_df.columns:
                     # Check if column contains strings matching Finviz date patterns
                     if self.news_df[col].astype(str).str.match(r'(\w{3}-\d{2}-\d{2} \d{2}:\d{2}[AP]M|\d{2}:\d{2}[AP]M|\w{3}-\d{2})').any():
                         published_col_name = col
                         logger.warning(f"Column 'published_at' not found, using likely Finviz date column '{published_col_name}'.")
                         break
                if not published_col_name:
                     # Fallback to finding any column with 'date' or 'time'
                     date_cols = [col for col in self.news_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                     if date_cols:
                          published_col_name = date_cols[0]
                          logger.warning(f"Column 'published_at' not found, using fallback date column '{published_col_name}'.")
                     else:
                          raise ValueError("Cannot find a suitable date column (like 'published_at' or Finviz format) in news_df.")

            # 2. Define Finviz Date Parser
            def parse_finviz_date(date_str):
                """Parses Finviz absolute ('Oct-28-25 05:38PM'), time-only ('05:38PM'), or date-only ('Oct-28') formats."""
                try:
                    # Try absolute format first (e.g., 'Oct-28-25 05:38PM') - Assuming ET timezone
                    dt = pd.to_datetime(date_str, format='%b-%d-%y %I:%M%p', errors='raise').tz_localize('America/New_York')
                    return dt.tz_convert('UTC') # Convert to UTC
                except (ValueError, TypeError):
                    try:
                        # Try time-only format ('05:38PM' means today) - Assuming ET timezone
                        time_part = pd.to_datetime(date_str, format='%I:%M%p', errors='raise').time()
                        # Use current date but keep original time - handle potential timezone issues carefully
                        dt_today_et = pd.Timestamp.now(tz='America/New_York').normalize() # Start of today in ET
                        dt = dt_today_et.replace(hour=time_part.hour, minute=time_part.minute, second=0, microsecond=0)
                        return dt.tz_convert('UTC') # Convert to UTC
                    except (ValueError, TypeError):
                        try:
                            # Try date-only format ('Oct-28' means this year) - Assuming ET timezone, set time to market close (e.g., 4 PM ET)
                            date_part = pd.to_datetime(date_str, format='%b-%d', errors='raise')
                            current_year = pd.Timestamp.now(tz='America/New_York').year
                            dt = date_part.replace(year=current_year).tz_localize('America/New_York')
                            # Set time to something reasonable like 4 PM ET
                            dt = dt.replace(hour=16, minute=0, second=0, microsecond=0)
                            return dt.tz_convert('UTC') # Convert to UTC
                        except (ValueError, TypeError):
                            # Try standard pandas parsing as a fallback, convert to UTC
                            dt_standard = pd.to_datetime(date_str, errors='coerce', utc=True)
                            if pd.isna(dt_standard):
                                 logger.debug(f"Could not parse date: {date_str}") # Log unparseable dates
                            return dt_standard # Return NaT if standard parsing fails

            # 3. Apply robust parsing and cleaning
            self.news_df['parsed_datetime_utc'] = self.news_df[published_col_name].astype(str).apply(parse_finviz_date)

            initial_rows = len(self.news_df)
            self.news_df = self.news_df.dropna(subset=['parsed_datetime_utc'])
            dropped_rows = initial_rows - len(self.news_df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} news articles due to invalid/unparseable dates.")
            if self.news_df.empty:
                logger.warning("No valid news articles remaining after date cleaning.")
                self.news_df_with_embeddings = None
                self.news_embeddings = None
                return

            # 4. Sort by date
            self.news_df = self.news_df.sort_values('parsed_datetime_utc')
            # --- End Date Handling ---

            # Prepare news texts
            title_col = 'title' if 'title' in self.news_df.columns else ''
            desc_col = 'description' if 'description' in self.news_df.columns else ''
            if not title_col and not desc_col:
                 raise ValueError("News DataFrame must contain 'title' or 'description'.")

            texts = []
            for _, row in self.news_df.iterrows():
                 title_text = row.get(title_col, '') or ''
                 desc_text = row.get(desc_col, '') or ''
                 texts.append(f"{title_text} {desc_text}".strip())

            # --- Correct call to process_news_batch ---
            self.news_df['temp_combined_text'] = texts
            # Make sure the embedder uses the specified text column
            embeddings, sentiments = embedder.process_news_batch(self.news_df, text_column='temp_combined_text')
            self.news_df = self.news_df.drop(columns=['temp_combined_text'])
            # --- End Correct Call ---

            if len(embeddings) == len(self.news_df):
                self.news_df['embedding'] = list(embeddings)
                self.news_df['sentiment_score'] = sentiments[:, 2] - sentiments[:, 0]
                self.news_df_with_embeddings = self.news_df.copy() # Store the updated df
                self.news_embeddings = embeddings

                logger.info(f"✅ Embeddings generated successfully. Shape: {embeddings.shape}")

                # --- Add 'Date' column (normalized, timezone-naive UTC date) for alignment ---
                self.news_df_with_embeddings['Date'] = self.news_df_with_embeddings['parsed_datetime_utc'].dt.tz_localize(None).dt.normalize()
                logger.info(f"   Added 'Date' column for alignment. Date range: {self.news_df_with_embeddings['Date'].min().date()} to {self.news_df_with_embeddings['Date'].max().date()}")
                # --- End Add Date Column ---

                # Save embeddings DataFrame with dates
                embeddings_path = self.output_dir / 'news_embeddings.pkl'
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(self.news_df_with_embeddings, f)
                logger.info(f"   Saved DataFrame with embeddings and dates to: {embeddings_path}")
            else:
                logger.error(f"Mismatch in embedding count ({len(embeddings)}) and news DataFrame rows ({len(self.news_df)}) after date cleaning. Skipping embedding assignment.")
                self.news_df_with_embeddings = None
                self.news_embeddings = None

        except ImportError as e:
            logger.error(f"Failed to import embedding class: {e}", exc_info=True)
            logger.warning("Continuing without text embeddings...")
            self.news_df_with_embeddings = None
            self.news_embeddings = None
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            logger.warning("Continuing without text embeddings...")
            self.news_df_with_embeddings = None
            self.news_embeddings = None
    
    def step2_create_features(self):
        """Step 2: Create 80-100 enhanced features"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80 + "\n")
        
        # Create enhanced features
        feature_engineer = EnhancedNumericalFeatures()
        
        logger.info("Creating enhanced numerical features...")
        logger.info(f"  - Stock data: {len(self.stock_df)} rows")
        logger.info(f"  - Market data available: {len(self.market_data)} datasets")
        logger.info(f"  - Sector data available: {'Yes' if self.sector_df is not None else 'No'}")
        logger.info(f"  - Macro data available: {len(self.macro_data)} datasets")
        logger.info(f"  - Competitor data available: {len(self.competitor_data)} stocks")
        
        features_df = feature_engineer.create_all_features(
            stock_df=self.stock_df,
            market_data=self.market_data,
            sector_df=self.sector_df,
            macro_data=self.macro_data,
            competitor_data=self.competitor_data
        )
        
        logger.info(f"\n🎯 Total features created: {len(features_df.columns)}")
        logger.info(f"   Target range: 80-100+ features")
        logger.info(f"  - Regime Indicators: ~10")
        
        # Save features
        features_df.to_csv(self.output_dir / 'features.csv', index=False)
        
        self.features_df = features_df
        
        logger.info("✅ Feature engineering complete\n")
        
        # Save checkpoint
        self.save_checkpoint(2, "Feature Engineering")
        
        return features_df
    
    def step3_create_sequences(self):
        """Step 3: Create sequences and labels, correctly aligning embeddings"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: SEQUENCE CREATION")
        logger.info("="*80 + "\n")

        # --- Ensure 'Date' column exists and is datetime in features_df ---
        if 'Date' not in self.features_df.columns and 'date' in self.features_df.columns:
             self.features_df = self.features_df.rename(columns={'date': 'Date'})
        if 'Date' not in self.features_df.columns:
             raise ValueError("Feature DataFrame must have a 'Date' column.")
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'], errors='coerce')
        self.features_df = self.features_df.dropna(subset=['Date'])
        # Ensure timezone naive for merging/lookup
        if hasattr(self.features_df['Date'].dtype, 'tz') and self.features_df['Date'].dt.tz is not None:
            self.features_df['Date'] = self.features_df['Date'].dt.tz_localize(None)
        self.features_df = self.features_df.sort_values('Date').reset_index(drop=True)
        # --- End Date Handling ---

        # Drop initial NaN rows created by indicators/returns
        initial_len = len(self.features_df)
        clean_df = self.features_df.dropna(axis=0, how='any') # Drop rows with ANY NaNs
        dropped_rows = initial_len - len(clean_df)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} initial rows containing NaNs.")
        if len(clean_df) < self.sequence_length + 1:
             raise ValueError(f"Not enough data ({len(clean_df)} rows) after dropping NaNs to create sequences of length {self.sequence_length}.")
        logger.info(f"Clean samples available for sequencing: {len(clean_df)}")

        # Create target (next day price up/down)
        target_col = 'Close' if 'Close' in clean_df.columns else 'close'
        if target_col not in clean_df.columns:
             raise ValueError("No 'close' or 'Close' column found for target creation.")
        clean_df['target'] = (clean_df[target_col].shift(-1) > clean_df[target_col]).astype(int)

        # Remove last row (no target)
        clean_df = clean_df.iloc[:-1]
        if len(clean_df) < self.sequence_length:
             raise ValueError(f"Not enough data ({len(clean_df)} rows) after target creation to create sequences of length {self.sequence_length}.")

        # Get feature columns (exclude target and date-related columns)
        exclude_cols = ['target', 'Date', 'date', 'timestamp', 'Ticker', 'published_at', 'embedding', 'sentiment_score'] # Add embedding/sentiment if they ended up here
        feature_cols = [c for c in clean_df.columns if c not in exclude_cols and not c.startswith('embedding_')] # Exclude individual embedding dims if flattened

        # Filter out non-numeric columns that might have slipped through
        numeric_feature_cols = clean_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
        if len(numeric_feature_cols) != len(feature_cols):
             dropped = set(feature_cols) - set(numeric_feature_cols)
             logger.warning(f"Dropping non-numeric columns from features: {dropped}")
             feature_cols = numeric_feature_cols

        logger.info(f"Using {len(feature_cols)} numerical feature columns for sequences.")
        if not feature_cols:
             raise ValueError("No valid numerical feature columns found for sequence creation.")

        # Create separate numerical and text sequences
        X_num_list = []  # Numerical features only
        X_text_list = [] # Text embeddings (if available)
        y_list = []
        dates_list = [] # Keep track of the target date for each sequence

        # --- Prepare Embeddings Lookup ---
        has_text_embeddings = hasattr(self, 'news_df_with_embeddings') and self.news_df_with_embeddings is not None and not self.news_df_with_embeddings.empty and 'embedding' in self.news_df_with_embeddings.columns
        daily_embeddings_dict = {}
        embedding_dim = 0
        if has_text_embeddings:
            logger.info("News embeddings available - creating multimodal sequences")
            # Ensure Date is datetime and timezone naive
            self.news_df_with_embeddings['Date'] = pd.to_datetime(self.news_df_with_embeddings['Date']).dt.normalize()
            if hasattr(self.news_df_with_embeddings['Date'].dtype, 'tz') and self.news_df_with_embeddings['Date'].dt.tz is not None:
                 self.news_df_with_embeddings['Date'] = self.news_df_with_embeddings['Date'].dt.tz_localize(None)

            # Group embeddings by date (average if multiple news per day)
            daily_embeddings_grouped = self.news_df_with_embeddings.groupby('Date')['embedding'].apply(
                 lambda x: np.mean(np.vstack(x.tolist()), axis=0) if not x.empty else None # Handle potential empty groups
            )
            daily_embeddings_dict = daily_embeddings_grouped.to_dict()
            # Get embedding dimension dynamically
            first_valid_embedding = next((emb for emb in daily_embeddings_dict.values() if emb is not None), None)
            if first_valid_embedding is not None:
                 embedding_dim = first_valid_embedding.shape[0]
                 logger.info(f"Detected embedding dimension: {embedding_dim}")
            else:
                 logger.warning("Could not determine embedding dimension from available news. Disabling text features.")
                 has_text_embeddings = False # Disable if no valid embeddings found
                 embedding_dim = 768 # Default fallback if needed elsewhere, but features won't be used
        else:
            logger.warning("No valid news embeddings found or loaded - using numerical features only")
            embedding_dim = 768 # Default dimension if needed, but text features won't be used
        # --- End Embeddings Lookup Prep ---


        # --- Create Sequences ---
        num_samples = len(clean_df)
        for i in range(self.sequence_length, num_samples):
            # Target date for this sequence's prediction is date at index i
            target_date = clean_df['Date'].iloc[i]

            # Numerical features sequence: indices [i - sequence_length] to [i - 1]
            num_sequence = clean_df[feature_cols].iloc[i-self.sequence_length : i].values
            X_num_list.append(num_sequence)

            # Text embeddings sequence (align with stock dates)
            if has_text_embeddings:
                text_sequence = []
                # Dates corresponding to the numerical sequence window
                sequence_dates = clean_df['Date'].iloc[i-self.sequence_length : i]

                for date in sequence_dates:
                    # Normalize date just in case
                    lookup_date = date.normalize()
                    # Get embedding for this date, or use zeros if no news
                    embedding = daily_embeddings_dict.get(lookup_date, None)
                    if embedding is not None:
                        text_sequence.append(embedding)
                    else:
                        # Use zeros if no embedding found for that day
                        text_sequence.append(np.zeros(embedding_dim, dtype=np.float32))

                # Ensure the sequence has the correct length (should be guaranteed by loop)
                if len(text_sequence) == self.sequence_length:
                    X_text_list.append(np.array(text_sequence, dtype=np.float32))
                else:
                    # This case should ideally not happen if data is clean
                    logger.error(f"Text sequence length mismatch at index {i}. Expected {self.sequence_length}, got {len(text_sequence)}. Skipping sequence.")
                    # Remove the corresponding numerical sequence and skip target
                    X_num_list.pop()
                    continue # Skip to next iteration

            # Target value corresponds to the state at index i
            y_list.append(clean_df['target'].iloc[i])
            dates_list.append(target_date) # Store the date this sequence predicts for

        # --- End Sequence Creation ---

        # Convert lists to numpy arrays
        X_num = np.array(X_num_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        if has_text_embeddings:
             # Check if X_text_list was actually populated
             if X_text_list:
                 X_text = np.array(X_text_list, dtype=np.float32)
                 # Verify shapes match after potential skipping
                 if X_text.shape[0] != X_num.shape[0]:
                      logger.error(f"Mismatch between Num ({X_num.shape[0]}) and Text ({X_text.shape[0]}) sequence counts. Check alignment logic.")
                      # Attempt to reconcile - might indicate deeper issues
                      min_len = min(X_num.shape[0], X_text.shape[0])
                      X_num = X_num[:min_len]
                      X_text = X_text[:min_len]
                      y = y[:min_len]
                      dates_list = dates_list[:min_len]
                      logger.warning(f"Trimmed sequences to minimum length: {min_len}")
             else:
                 logger.warning("Text sequence list is empty despite embeddings being available. Check date alignment. Disabling text features.")
                 X_text = None
                 has_text_embeddings = False # Update flag
        else:
             X_text = None

        if X_num.shape[0] == 0:
             raise ValueError("Sequence creation resulted in zero valid sequences. Check data and parameters.")

        logger.info(f"Final sequences created: {X_num.shape[0]}")
        logger.info(f"Numerical sequence shape: {X_num.shape}")
        if X_text is not None:
             logger.info(f"Text embedding sequence shape: {X_text.shape}")
        else:
             logger.info("No text embedding sequences created.")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Positive class ratio in final sequences: {y.mean():.3f}")

        # --- Train/Val/Test Split (Temporal) ---
        n_total = len(y)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.validation_split)
        n_train = n_total - n_test - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
             raise ValueError(f"Insufficient data for split: Train={n_train}, Val={n_val}, Test={n_test}. Need more data or smaller sequence length/splits.")

        # Temporal split indices
        train_end_idx = n_train
        val_end_idx = n_train + n_val

        # Split numerical features
        X_num_train = X_num[:train_end_idx]
        X_num_val = X_num[train_end_idx:val_end_idx]
        X_num_test = X_num[val_end_idx:]

        # Split text features (if available)
        if X_text is not None:
             X_text_train = X_text[:train_end_idx]
             X_text_val = X_text[train_end_idx:val_end_idx]
             X_text_test = X_text[val_end_idx:]
        else:
             X_text_train, X_text_val, X_text_test = None, None, None

        # Split labels
        y_train = y[:train_end_idx]
        y_val = y[train_end_idx:val_end_idx]
        y_test = y[val_end_idx:]
        # --- End Split ---

        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(y_train)} sequences")
        logger.info(f"  Val:   {len(y_val)} sequences")
        logger.info(f"  Test:  {len(y_test)} sequences")

        # Store class attributes
        self.X_num_train, self.X_text_train, self.y_train = X_num_train, X_text_train, y_train
        self.X_num_val, self.X_text_val, self.y_val = X_num_val, X_text_val, y_val
        self.X_num_test, self.X_text_test, self.y_test = X_num_test, X_text_test, y_test
        self.num_features = X_num_train.shape[2] # Number of numerical features per timestep

        logger.info("✅ Sequence creation and splitting complete\n")

        # Save checkpoint
        self.save_checkpoint(3, "Sequence Creation")

        return (X_num_train, X_text_train, y_train,
                X_num_val, X_text_val, y_val,
                X_num_test, X_text_test, y_test)
    
    def step4_run_comparison(self):
        """Step 4: Run baseline vs enhanced comparison"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: MODEL COMPARISON")
        logger.info("="*80 + "\n")
        
        comparator = ModelComparator(
            output_dir=str(self.output_dir / 'comparison')
        )
        
        # Run comparison with separate text and numerical arrays
        comparator.run_full_comparison(
            X_text_train=self.X_text_train,
            X_num_train=self.X_num_train,
            y_train=self.y_train,
            X_text_val=self.X_text_val,
            X_num_val=self.X_num_val,
            y_val=self.y_val,
            baseline_epochs=30,
            enhanced_epochs=50,
            use_sklearn_baselines=True,
            enhanced_models=['bilstm', 'hybrid']  # Skip transformer initially
        )
        
        self.comparison_results = comparator.results
        
        logger.info("✅ Model comparison complete\n")
        
        # Save checkpoint
        self.save_checkpoint(4, "Model Comparison")
        
        return comparator.results
    
    def step5_optimize_best_model(self, n_trials: int = 50):
        """Step 5: Hyperparameter optimization on best model"""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80 + "\n")
        
        # Find best model from comparison (excluding sklearn baselines)
        dl_models = {k: v for k, v in self.comparison_results.items() 
                     if k not in ['random_forest', 'logistic_regression']}
        
        if not dl_models:
            logger.warning("No deep learning models to optimize, using hybrid model")
            best_model_name = 'hybrid'
        else:
            # Find best DL model by F1 score (more robust than accuracy for imbalanced data)
            best_model_name = max(dl_models.items(), key=lambda x: x[1].get('f1', 0))[0]
            best_f1 = dl_models[best_model_name]['f1']
            
            logger.info(f"Best performing DL model from comparison: {best_model_name}")
            logger.info(f"F1 Score: {best_f1:.4f}")
            
            # Map comparison names to model types
            model_type_map = {
                'baseline_lstm': 'baseline',
                'enhanced_bilstm': 'bilstm',
                'enhanced_transformer': 'transformer',
                'enhanced_hybrid': 'hybrid'
            }
            best_model_name = model_type_map.get(best_model_name, 'hybrid')
        
        logger.info(f"Optimizing {best_model_name} model...")
        
        optimizer = HyperparameterOptimizer(
            model_type=best_model_name,
            num_features=self.num_features,
            device=self.device,
            study_name=f'{self.ticker}_{best_model_name}_optimization'
        )
        
        # Set data (use numerical features only for optimization)
        # TODO: Update HyperparameterOptimizer to support separate text/numerical arrays
        optimizer.set_data(
            self.X_num_train,
            self.y_train,
            self.X_num_val,
            self.y_val
        )
        
        # Run optimization
        best_params = optimizer.optimize(n_trials=n_trials)
        
        # Results are automatically saved by optimizer.optimize()
        # No need to call save_results again
        
        self.best_params = best_params
        self.best_model_type = best_model_name
        
        logger.info(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ Hyperparameter optimization complete\n")
        
        # Save checkpoint
        self.save_checkpoint(5, "Hyperparameter Optimization")
        
        return best_params
    
    def step6_train_final_model(self):
        """Step 6: Train final optimized model"""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: FINAL MODEL TRAINING")
        logger.info("="*80 + "\n")
        
        # Separate model parameters from trainer parameters
        model_params = {}
        trainer_params = {}
        
        # Model-specific parameters
        model_param_names = ['hidden_dim', 'num_layers', 'dropout', 'num_heads', 'temporal_decay_init', 'd_model', 'nhead', 'dim_feedforward']
        
        for key, value in self.best_params.items():
            if key in model_param_names:
                model_params[key] = value
            else:
                trainer_params[key] = value
        
        # Use the best model type from optimization
        model_type = getattr(self, 'best_model_type', 'hybrid')
        logger.info(f"Training final {model_type} model with optimized parameters...")
        
        # Add use_finbert flag based on model type
        if model_type == 'hybrid' and self.X_text_train is not None:
            model_params['use_finbert'] = True
        else:
            model_params['use_finbert'] = False
        
        # Create optimized model
        model = create_model(
            model_type=model_type,
            num_features=self.num_features,
            **model_params
        )
        
        # Create trainer
        trainer = EnhancedTrainer(
            model,
            device=self.device,
            output_dir=str(self.output_dir / 'final_model')
        )
        
        # Create data loaders with separate arrays
        train_loader, val_loader = trainer.create_dataloaders(
            X_num_train=self.X_num_train,
            y_train=self.y_train,
            X_num_val=self.X_num_val,
            y_val=self.y_val,
            X_text_train=self.X_text_train,
            X_text_val=self.X_text_val,
            batch_size=trainer_params.get('batch_size', 32)
        )
        
        # Train with best hyperparameters
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=100,
            learning_rate=trainer_params.get('learning_rate', 0.0005),
            scheduler_type='plateau',
            early_stopping_patience=20,
            grad_clip=trainer_params.get('grad_clip', 1.0)
        )
        
        self.final_trainer = trainer
        self.final_history = history
        
        logger.info("✅ Final model training complete\n")
        
        # Save checkpoint
        self.save_checkpoint(6, "Final Model Training")
        
        return trainer, history
    
    def step7_evaluate_final_model(self):
        """Step 7: Comprehensive evaluation on test set"""
        logger.info("\n" + "="*80)
        logger.info("STEP 7: FINAL EVALUATION")
        logger.info("="*80 + "\n")
        
        # Create test loader with separate arrays
        if self.X_text_test is not None:
            test_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.X_text_test),
                torch.FloatTensor(self.X_num_test),
                torch.LongTensor(self.y_test)
            )
        else:
            test_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.X_num_test),
                torch.LongTensor(self.y_test)
            )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )
        
        # Load best model
        self.final_trainer.load_best_model()
        
        # Evaluate
        test_metrics = self.final_trainer.evaluate(test_loader)
        
        self.test_metrics = test_metrics
        
        logger.info("✅ Final evaluation complete\n")
        
        # Save checkpoint
        self.save_checkpoint(7, "Final Evaluation")
        
        return test_metrics
    
    def step8_generate_report(self):
        """Step 8: Generate comprehensive final report"""
        logger.info("\n" + "="*80)
        logger.info("STEP 8: GENERATING FINAL REPORT")
        logger.info("="*80 + "\n")
        
        report = {
            'ticker': self.ticker,
            'training_date': datetime.now().isoformat(),
            'data_summary': {
                'total_days': self.days,
                'sequence_length': self.sequence_length,
                'num_features': self.num_features,
                'train_samples': len(self.X_num_train),
                'val_samples': len(self.X_num_val),
                'test_samples': len(self.X_num_test),
                'has_text_embeddings': self.X_text_train is not None
            },
            'model_comparison': {},
            'best_hyperparameters': self.best_params,
            'final_test_metrics': {
                'accuracy': self.test_metrics['accuracy'],
                'precision': self.test_metrics['precision'],
                'recall': self.test_metrics['recall'],
                'f1': self.test_metrics['f1'],
                'roc_auc': self.test_metrics['roc_auc']
            },
            'improvement_analysis': {}
        }
        
        # Add comparison results
        for model_name, metrics in self.comparison_results.items():
            report['model_comparison'][model_name] = {
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1']
            }
        
        # Calculate improvement
        if 'baseline_lstm' in self.comparison_results:
            baseline_acc = self.comparison_results['baseline_lstm']['accuracy']
            final_acc = self.test_metrics['accuracy']
            improvement_pct = ((final_acc - baseline_acc) / baseline_acc) * 100
            improvement_abs = (final_acc - baseline_acc) * 100
            
            report['improvement_analysis'] = {
                'baseline_accuracy': baseline_acc,
                'final_accuracy': final_acc,
                'improvement_percentage': improvement_pct,
                'improvement_absolute': improvement_abs,
                'target_achieved': final_acc >= 0.80
            }
            
            logger.info(f"\n{'='*80}")
            logger.info("IMPROVEMENT ANALYSIS")
            logger.info(f"{'='*80}")
            logger.info(f"Baseline Accuracy: {baseline_acc:.4f}")
            logger.info(f"Final Accuracy:    {final_acc:.4f}")
            logger.info(f"Improvement:       {improvement_pct:.2f}%")
            logger.info(f"Absolute Gain:     {improvement_abs:.2f} percentage points")
            logger.info(f"80% Target:        {'ACHIEVED' if final_acc >= 0.80 else 'Not reached'}")
            logger.info(f"{'='*80}\n")
        
        # Save report
        report_path = self.output_dir / 'FINAL_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved: {report_path}")
        
        # Generate human-readable report
        self._generate_markdown_report(report)
        
        logger.info("✅ Report generation complete\n")
        
        return report
    
    def _generate_markdown_report(self, report: dict):
        """Generate markdown report"""
        md_content = f"""# Full Training Report: {self.ticker}

**Training Date:** {report['training_date']}

## Data Summary

- **Ticker:** {self.ticker}
- **Historical Days:** {report['data_summary']['total_days']}
- **Sequence Length:** {report['data_summary']['sequence_length']}
- **Number of Features:** {report['data_summary']['num_features']}
- **Training Samples:** {report['data_summary']['train_samples']}
- **Validation Samples:** {report['data_summary']['val_samples']}
- **Test Samples:** {report['data_summary']['test_samples']}

## Model Comparison Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
"""
        
        for model, metrics in report['model_comparison'].items():
            md_content += f"| {model} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} |\n"
        
        md_content += f"""
## Final Test Results

- **Accuracy:** {report['final_test_metrics']['accuracy']:.4f}
- **Precision:** {report['final_test_metrics']['precision']:.4f}
- **Recall:** {report['final_test_metrics']['recall']:.4f}
- **F1 Score:** {report['final_test_metrics']['f1']:.4f}
- **ROC-AUC:** {report['final_test_metrics']['roc_auc']:.4f}

## Improvement Analysis

"""
        
        if report['improvement_analysis']:
            ia = report['improvement_analysis']
            md_content += f"""- **Baseline Accuracy:** {ia['baseline_accuracy']:.4f}
- **Final Accuracy:** {ia['final_accuracy']:.4f}
- **Improvement:** {ia['improvement_percentage']:.2f}%
- **Absolute Gain:** {ia['improvement_absolute']:.2f} percentage points
- **80% Target:** {'ACHIEVED' if ia['target_achieved'] else 'Not reached'}

"""
        
        md_content += f"""## Best Hyperparameters

```json
{json.dumps(report['best_hyperparameters'], indent=2)}
```

## Conclusion

"""
        
        if report['improvement_analysis'] and report['improvement_analysis']['target_achieved']:
            md_content += "**Successfully achieved 80%+ accuracy target!**\n\n"
        
        md_content += f"""The enhanced system demonstrates significant improvement over baseline models through:
1. Advanced model architecture (Hybrid: FinBERT + BiLSTM + Attention)
2. 80-100 enhanced features
3. Hyperparameter optimization
4. Longer sequence context ({self.sequence_length} days)
5. Sophisticated training pipeline

For usage instructions, see `QUICKSTART.md` and `ENHANCED_SYSTEM_USAGE.md`.
"""
        
        # Save markdown
        md_path = self.output_dir / 'FINAL_REPORT.md'
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved: {md_path}")
        
        # Save final checkpoint
        self.save_checkpoint(8, "Report Generation")
    
    def run_full_pipeline(self, skip_optimization: bool = False, n_trials: int = 50, start_from_step: int = 1):
        """
        Run complete end-to-end pipeline
        
        Args:
            skip_optimization: Skip hyperparameter optimization (use defaults)
            n_trials: Number of optimization trials
            start_from_step: Step to start from (1-8, for resuming)
        """
        try:
            # Check for checkpoint
            checkpoint = self.load_checkpoint()
            if checkpoint and start_from_step == 1:
                logger.info(f"Previous run found. Last completed: Step {checkpoint['completed_step']}")
                user_input = input("Resume from last checkpoint? (y/n): ").strip().lower()
                if user_input == 'y':
                    start_from_step = checkpoint['completed_step'] + 1
                    logger.info(f"Resuming from step {start_from_step}")
                    
                    # Load checkpoint data if resuming
                    if start_from_step > 5 and 'best_params' in checkpoint:
                        self.best_params = checkpoint['best_params']
                        self.best_model_type = checkpoint['best_model_type']
                else:
                    logger.info("Starting fresh pipeline")
                    self.clear_checkpoint()
            
            # Step 1: Collect data
            if start_from_step <= 1:
                self.step1_collect_data()
                self.step1b_process_news_embeddings()
            else:
                logger.info("Step 1: Skipped (loading from saved data)")
                self.stock_df = pd.read_csv(self.output_dir / 'stock_data.csv')
                self.news_df = pd.read_csv(self.output_dir / 'news_data.csv')
                # Try to load embeddings if available
                try:
                    import pickle
                    embeddings_path = self.output_dir / 'news_embeddings.pkl'
                    if embeddings_path.exists():
                        with open(embeddings_path, 'rb') as f:
                            self.news_df_with_embeddings = pickle.load(f)
                        logger.info(f"Loaded {len(self.news_df_with_embeddings)} news embeddings from checkpoint")
                    else:
                        self.news_df_with_embeddings = None
                except Exception as e:
                    logger.warning(f"Could not load embeddings: {e}")
                    self.news_df_with_embeddings = None
                # Initialize empty data for feature engineering (not critical for checkpoint resume)
                self.market_data = {}
                self.sector_df = None
                self.macro_data = {}
                self.competitor_data = {}
                logger.warning("Market/sector/macro/competitor data not loaded (not saved in checkpoint)")
                logger.warning("If you need full feature engineering, please run from step 1")
            
            # Step 2: Create features
            if start_from_step <= 2:
                self.step2_create_features()
            else:
                logger.info("Step 2: Skipped (loading from saved features)")
                self.features_df = pd.read_csv(self.output_dir / 'features.csv')
            
            # Step 3: Create sequences
            if start_from_step <= 3:
                self.step3_create_sequences()
            else:
                logger.info("Step 3: Skipped (would need to reload sequences)")
                logger.warning("Cannot skip sequence creation - recreating sequences...")
                self.step3_create_sequences()
            
            # Step 4: Model comparison
            if start_from_step <= 4:
                self.step4_run_comparison()
            else:
                logger.info("Step 4: Skipped (loading comparison results)")
                try:
                    import json
                    with open(self.output_dir / 'comparison' / 'full_results.json', 'r') as f:
                        self.comparison_results = json.load(f)
                except FileNotFoundError:
                    logger.warning("Comparison results not found - running comparison...")
                    self.step4_run_comparison()
            
            # Step 5: Optimize (optional)
            if start_from_step <= 5:
                if not skip_optimization:
                    self.step5_optimize_best_model(n_trials=n_trials)
                else:
                    # Use default parameters
                    self.best_params = {
                        'hidden_dim': 256,
                        'num_layers': 3,
                        'dropout': 0.3,
                        'learning_rate': 0.0005,
                        'batch_size': 32,
                        'grad_clip': 1.0,
                        'num_heads': 8,
                        'temporal_decay_init': 0.95
                    }
                    self.best_model_type = 'bilstm'  # Default to BiLSTM (simpler than hybrid)
                    logger.info("Using default hyperparameters (optimization skipped)")
                    self.save_checkpoint(5, "Hyperparameter Optimization (Skipped)")
            else:
                logger.info("Step 5: Skipped (using saved hyperparameters)")
            
            # Step 6: Train final model
            if start_from_step <= 6:
                self.step6_train_final_model()
            else:
                logger.info("Step 6: Skipped (loading trained model)")
                # Would need to reload model here
                logger.warning("Cannot skip final model training - retraining...")
                self.step6_train_final_model()
            
            # Step 7: Evaluate
            if start_from_step <= 7:
                self.step7_evaluate_final_model()
            else:
                logger.info("Step 7: Skipped (loading evaluation results)")
            
            # Step 8: Generate report
            if start_from_step <= 8:
                report = self.step8_generate_report()
            else:
                logger.info("Step 8: Already completed")
                report = None
            
            logger.info(f"\n{'='*80}")
            logger.info("FULL PIPELINE COMPLETE!")
            logger.info(f"{'='*80}\n")
            logger.info(f"Results saved to: {self.output_dir}")
            if hasattr(self, 'test_metrics'):
                logger.info(f"Final test accuracy: {self.test_metrics['accuracy']:.4f}")
            
            # Clear checkpoint on successful completion
            self.clear_checkpoint()
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def quick_test():
    """Quick test with small dataset"""
    logger.info("Running quick test with small dataset...")
    
    pipeline = FullTrainingPipeline(
        ticker='AAPL',
        days=200,  # Smaller for testing
        sequence_length=60,
        output_dir='results/quick_test'
    )
    
    # Run with optimization skipped
    report = pipeline.run_full_pipeline(skip_optimization=True)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Full Training Pipeline')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--days', type=int, default=1000, help='Number of days')
    parser.add_argument('--sequence-length', type=int, default=120, help='Sequence length')
    parser.add_argument('--n-trials', type=int, default=50, help='Optimization trials')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        pipeline = FullTrainingPipeline(
            ticker=args.ticker,
            days=args.days,
            sequence_length=args.sequence_length
        )
        
        pipeline.run_full_pipeline(
            skip_optimization=args.skip_optimization,
            n_trials=args.n_trials
        )
