"""
NLP Preprocessing and Feature Extraction Module
Handles text cleaning, entity recognition, relation extraction, and embedding generation
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep important financial symbols
        text = re.sub(r'[^a-zA-Z0-9\s\$\%\.\,\-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords while preserving financial terms"""
        
        # Financial terms to preserve
        financial_terms = {'up', 'down', 'over', 'under', 'above', 'below', 
                          'more', 'less', 'high', 'low', 'beat', 'miss'}
        
        words = word_tokenize(text)
        filtered_words = [
            word for word in words 
            if word not in self.stop_words or word in financial_terms
        ]
        
        return ' '.join(filtered_words)
    
    def preprocess_batch(self, texts: List[str], remove_stops: bool = False) -> List[str]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of texts
            remove_stops: Whether to remove stopwords
            
        Returns:
            List of cleaned texts
        """
        processed = [self.clean_text(text) for text in texts]
        
        if remove_stops:
            processed = [self.remove_stopwords(text) for text in processed]
        
        return processed


class EntityExtractor:
    """Extracts named entities and financial entities"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.logger.warning(f"Spacy model {model_name} not found. Run: python -m spacy download {model_name}")
            self.nlp = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Financial entity patterns
        self.financial_patterns = {
            'PRICE': r'\$\d+(?:\.\d+)?(?:K|M|B|T)?',
            'PERCENTAGE': r'\d+(?:\.\d+)?%',
            'REVENUE': r'revenue|sales|earnings|profit|loss',
            'METRIC': r'EPS|P/E|ROE|ROA|EBITDA|margin'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geopolitical entity
            'MONEY': [],
            'PERCENT': [],
            'DATE': [],
            'PRODUCT': []
        }
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # Extract financial entities using regex
        for pattern_name, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[pattern_name] = matches
        
        return entities
    
    def extract_company_mentions(self, text: str, known_companies: List[str] = None) -> List[str]:
        """
        Extract company mentions from text
        
        Args:
            text: Input text
            known_companies: List of known company names/tickers
            
        Returns:
            List of company mentions
        """
        mentions = []
        
        # Extract ORG entities
        if self.nlp:
            doc = self.nlp(text)
            mentions.extend([ent.text for ent in doc.ents if ent.label_ == 'ORG'])
        
        # Match against known companies
        if known_companies:
            text_lower = text.lower()
            for company in known_companies:
                if company.lower() in text_lower:
                    mentions.append(company)
        
        return list(set(mentions))


class RelationExtractor:
    """Extracts relationships and events from text"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Event patterns
        self.event_patterns = {
            'earnings': [
                r'earnings? report',
                r'quarterly results?',
                r'beat estimates?',
                r'miss(?:ed)? estimates?',
                r'EPS',
                r'revenue grew?'
            ],
            'merger': [
                r'merg(?:e|er|ing)',
                r'acquisition',
                r'acquire[sd]?',
                r'takeover',
                r'buyout'
            ],
            'partnership': [
                r'partner(?:ship)?',
                r'collaborat(?:e|ion)',
                r'joint venture',
                r'alliance'
            ],
            'product_launch': [
                r'launch(?:ed|ing)?',
                r'introduc(?:e|ed|ing)',
                r'unveiled?',
                r'new product',
                r'release[sd]?'
            ],
            'lawsuit': [
                r'lawsuit',
                r'legal action',
                r'sued?',
                r'litigation',
                r'settlement'
            ],
            'regulatory': [
                r'regulat(?:or|ion|ory)',
                r'SEC',
                r'FDA approval',
                r'investigation',
                r'compliance'
            ]
        }
    
    def detect_events(self, text: str) -> Dict[str, bool]:
        """
        Detect event types in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of event types and whether they're present
        """
        events = {}
        text_lower = text.lower()
        
        for event_type, patterns in self.event_patterns.items():
            detected = False
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected = True
                    break
            events[event_type] = detected
        
        return events
    
    def extract_sentiment_relations(self, text: str, entities: List[str]) -> Dict[str, float]:
        """
        Extract sentiment for each entity mentioned
        
        Args:
            text: Input text
            entities: List of entities to analyze
            
        Returns:
            Dictionary mapping entity to sentiment score
        """
        sentiments = {}
        
        for entity in entities:
            # Find sentences mentioning the entity
            sentences = text.split('.')
            relevant_sentences = [s for s in sentences if entity.lower() in s.lower()]
            
            if relevant_sentences:
                # Calculate average sentiment
                scores = [TextBlob(s).sentiment.polarity for s in relevant_sentences]
                sentiments[entity] = np.mean(scores) if scores else 0.0
        
        return sentiments


class FinancialTextEmbedder:
    """Generates embeddings using financial-domain models"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            # FinBERT requires specific trust settings and proper device handling
            self.logger.info(f"Loading FinBERT model: {model_name}")
            
            # Load tokenizer with proper settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=True,
                trust_remote_code=True
            )
            
            # Load base model for embeddings
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load sentiment classification model (same model, different head)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,  # FinBERT outputs: negative, neutral, positive
                trust_remote_code=True
            )
            
            # Device setup with proper error handling
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # Clear cache before loading
                torch.cuda.empty_cache()
            else:
                self.device = torch.device('cpu')
                self.logger.warning("CUDA not available, using CPU for FinBERT (slower)")
            
            # Move models to device
            self.model.to(self.device)
            self.sentiment_model.to(self.device)
            
            # Set to eval mode by default
            self.model.eval()
            self.sentiment_model.eval()
            
            self.logger.info(f"✓ FinBERT loaded successfully on {self.device}")
            self.logger.info(f"  - Model: {model_name}")
            self.logger.info(f"  - Vocab size: {len(self.tokenizer)}")
            self.logger.info(f"  - Max length: {self.tokenizer.model_max_length}")
            
        except Exception as e:
            self.logger.error(f"✗ Error loading FinBERT: {str(e)}")
            self.logger.error(f"  Model: {model_name}")
            self.logger.error(f"  Please ensure transformers library is up to date:")
            self.logger.error(f"  pip install --upgrade transformers torch")
            raise RuntimeError(f"Failed to load FinBERT: {e}") from e
    
    def get_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Generate contextual embeddings for texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Array of embeddings (batch_size, hidden_dim)
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)
    
    def get_sentiment_scores(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Get sentiment scores (negative, neutral, positive) for texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Array of sentiment probabilities (batch_size, 3)
        """
        self.sentiment_model.eval()
        sentiments = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                ).to(self.device)
                
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiments.append(probs.cpu().numpy()[0])
        
        return np.array(sentiments)
    
    def process_news_batch(
        self, 
        news_df: pd.DataFrame, 
        text_column: str = 'title'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of news articles
        
        Args:
            news_df: DataFrame with news articles
            text_column: Column containing text
            
        Returns:
            Tuple of (embeddings, sentiment_scores)
        """
        texts = news_df[text_column].fillna('').tolist()
        
        embeddings = self.get_embeddings(texts)
        sentiments = self.get_sentiment_scores(texts)
        
        return embeddings, sentiments


class NewsProcessor:
    """Complete NLP processing pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.embedder = FinancialTextEmbedder(
            config.get('nlp', {}).get('model_name', 'ProsusAI/finbert')
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire news DataFrame with NLP pipeline
        
        Args:
            news_df: DataFrame with news data
            
        Returns:
            Enhanced DataFrame with NLP features
        """
        self.logger.info(f"Processing {len(news_df)} news articles")
        
        # Combine title and description
        news_df['combined_text'] = (
            news_df['title'].fillna('') + ' ' + 
            news_df['description'].fillna('')
        )
        
        # Clean text
        news_df['cleaned_text'] = self.preprocessor.preprocess_batch(
            news_df['combined_text'].tolist()
        )
        
        # Extract entities and events
        news_df['entities'] = news_df['cleaned_text'].apply(
            self.entity_extractor.extract_entities
        )
        news_df['events'] = news_df['cleaned_text'].apply(
            self.relation_extractor.detect_events
        )
        
        # Get embeddings and sentiments
        embeddings, sentiments = self.embedder.process_news_batch(
            news_df, 'cleaned_text'
        )
        
        # Add to dataframe
        news_df['embedding'] = list(embeddings)
        news_df['sentiment_negative'] = sentiments[:, 0]
        news_df['sentiment_neutral'] = sentiments[:, 1]
        news_df['sentiment_positive'] = sentiments[:, 2]
        news_df['sentiment_score'] = sentiments[:, 2] - sentiments[:, 0]  # positive - negative
        
        self.logger.info("NLP processing completed")
        return news_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    pipeline = NewsProcessor({'nlp': {'model_name': 'ProsusAI/finbert'}})
    
    sample_df = pd.DataFrame({
        'title': ['Apple reports record earnings', 'Tesla faces lawsuit'],
        'description': ['Q4 earnings beat estimates', 'Legal action over autopilot claims']
    })
    
    result = pipeline.process_news_dataframe(sample_df)
    print(result[['title', 'sentiment_score', 'events']])
