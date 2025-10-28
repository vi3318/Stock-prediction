"""
Utility functions for the stock prediction system
"""

import os
import json
import pickle
import logging
from typing import Any, Dict
import numpy as np
import pandas as pd


def setup_logging(log_dir: str = 'logs', level: int = logging.INFO):
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/app.log'),
            logging.StreamHandler()
        ]
    )


def save_object(obj: Any, path: str):
    """
    Save object to pickle file
    
    Args:
        obj: Object to save
        path: File path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: str) -> Any:
    """
    Load object from pickle file
    
    Args:
        path: File path
        
    Returns:
        Loaded object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, path: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        path: File path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    """
    Load dictionary from JSON file
    
    Args:
        path: File path
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str):
    """
    Ensure directory exists
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from prices
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change()


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        Maximum drawdown (negative value)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return np.min(drawdown)


def print_banner():
    """Print project banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Stock Price Prediction Using News and NLP Parsing       ║
║                                                              ║
║     A Novel Deep Learning Framework with Temporal            ║
║     Relevance Modeling and Multimodal Fusion                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == "__main__":
    print_banner()
    print("Utility functions loaded successfully")
