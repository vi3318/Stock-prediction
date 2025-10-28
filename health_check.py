#!/usr/bin/env python3
"""
System Health Check and Quick Test Script
Run this to verify your setup is working correctly
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header(title):
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print('='*50)

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def check_python_version():
    print_header("Python Environment Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python 3.8+ required")
        return False

def check_virtual_environment():
    print_header("Virtual Environment Check")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Virtual environment is active")
        print(f"Environment path: {sys.prefix}")
        return True
    else:
        print_warning("No virtual environment detected")
        print("Recommended: source venv/bin/activate")
        return False

def check_dependencies():
    print_header("Dependency Check")
    
    critical_packages = {
        'tensorflow': '2.13.0',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'yfinance': '0.2.28',
        'pandas': '2.0.0',
        'numpy': '1.24.0',
        'sklearn': '1.3.0',  # Import name is sklearn, not scikit-learn
        'matplotlib': '3.7.0',
        'shap': '0.42.0'
    }
    
    success_count = 0
    total_count = len(critical_packages)
    
    for package, min_version in critical_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{package}: {version}")
            success_count += 1
        except ImportError:
            print_error(f"{package}: NOT INSTALLED")
    
    print(f"\nDependency status: {success_count}/{total_count} packages available")
    
    if success_count == total_count:
        print_success("All critical dependencies available")
        return True
    else:
        print_error("Missing dependencies. Run: pip install -r requirements.txt")
        return False

def check_directories():
    print_header("Directory Structure Check")
    
    required_dirs = [
        'data/raw/stocks',
        'data/raw/news', 
        'data/processed/features',
        'data/processed/embeddings',
        'models/saved',
        'logs',
        'results/figures',
        'results/reports'
    ]
    
    success_count = 0
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print_success(f"{dir_path}")
            success_count += 1
        else:
            print_error(f"{dir_path} - MISSING")
            print(f"   Create with: mkdir -p {dir_path}")
    
    print(f"\nDirectory status: {success_count}/{len(required_dirs)} directories exist")
    
    if success_count == len(required_dirs):
        print_success("All required directories exist")
        return True
    else:
        print_warning("Some directories missing - system will create them automatically")
        return False

def check_gpu_availability():
    print_header("GPU Availability Check")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_success(f"TensorFlow: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print_warning("TensorFlow: No GPUs detected (CPU only)")
    except:
        print_error("TensorFlow GPU check failed")
    
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"PyTorch: CUDA available ({torch.cuda.device_count()} devices)")
            print(f"   Device: {torch.cuda.get_device_name()}")
        else:
            print_warning("PyTorch: CUDA not available (CPU only)")
    except:
        print_error("PyTorch CUDA check failed")
    
    print("Note: CPU-only is fine for this system")

def test_data_collection():
    print_header("Data Collection Test")
    
    try:
        import yfinance as yf
        
        # Test basic yfinance functionality
        ticker = yf.Ticker('AAPL')
        info = ticker.info
        print_success(f"Retrieved AAPL info: {info.get('longName', 'Apple Inc.')}")
        
        # Test recent data
        hist = ticker.history(period='5d')
        if len(hist) > 0:
            latest_close = hist['Close'].iloc[-1]
            print_success(f"Retrieved {len(hist)} days of recent data")
            print(f"   Latest AAPL close: ${latest_close:.2f}")
            return True
        else:
            print_error("No recent data retrieved")
            return False
            
    except Exception as e:
        print_error(f"Data collection test failed: {e}")
        return False

def test_nlp_models():
    print_header("NLP Models Test")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("Loading FinBERT model...")
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        
        print_success("FinBERT model loaded successfully")
        
        # Test tokenization
        text = "Apple reports strong quarterly earnings"
        tokens = tokenizer(text, return_tensors='pt')
        print_success(f"Tokenized sample text: {len(tokens['input_ids'][0])} tokens")
        
        return True
        
    except Exception as e:
        print_error(f"NLP model test failed: {e}")
        print("This might take a few minutes on first run (downloading model)")
        return False

def test_core_imports():
    print_header("Core System Imports Test")
    
    try:
        # Test critical system imports
        from data_collection.collectors import StockDataCollector
        print_success("StockDataCollector import")
        
        from preprocessing.nlp_processor import NLPPipeline  
        print_success("NLPPipeline import")
        
        # Skip model imports for now since they may need config
        print_success("Core preprocessing modules import successfully")
        
        return True
        
    except ImportError as e:
        print_error(f"Core import failed: {e}")
        print("Check that all files exist in src/ directory")
        return False

def run_quick_functional_test():
    print_header("Quick Functional Test")
    
    try:
        # Test data creation
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'close': 150 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        print_success(f"Created sample dataset: {len(sample_data)} rows")
        
        # Test basic ML components
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X = sample_data[['close', 'volume']].values
        y = (np.random.randn(len(X)) > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        print_success(f"ML pipeline test: {accuracy:.2f} accuracy")
        
        return True
        
    except Exception as e:
        print_error(f"Functional test failed: {e}")
        return False

def check_memory_and_disk():
    print_header("System Resources Check")
    
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"Total RAM: {memory_gb:.1f} GB")
        print(f"Available RAM: {available_gb:.1f} GB")
        print(f"Memory usage: {memory.percent}%")
        
        if available_gb >= 4:
            print_success("Sufficient memory available")
        elif available_gb >= 2:
            print_warning("Limited memory - reduce batch sizes if needed")
        else:
            print_error("Low memory - system may be slow")
        
        # Disk check
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        print(f"Free disk space: {disk_free_gb:.1f} GB")
        
        if disk_free_gb >= 5:
            print_success("Sufficient disk space")
        else:
            print_warning("Low disk space - clean up if needed")
            
        return True
        
    except Exception as e:
        print_error(f"Resource check failed: {e}")
        return False

def provide_next_steps(all_tests_passed):
    print_header("Next Steps")
    
    if all_tests_passed:
        print_success("🎉 All tests passed! Your system is ready to use.")
        print("\nRecommended next steps:")
        print("1. Run quick test:")
        print("   python3 main.py --mode full --tickers AAPL --start-date 2023-01-01 --end-date 2023-03-31")
        print("\n2. Run examples:")
        print("   python3 examples/learnable_weights_example.py")
        print("   python3 examples/explainability_example.py")
        print("\n3. See SETUP_AND_TESTING_GUIDE.md for detailed instructions")
        
    else:
        print_warning("Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Create missing directories: mkdir -p data/raw/stocks data/raw/news logs results")
        print("3. Activate virtual environment: source venv/bin/activate")
        print("4. Check internet connection for data/model downloads")

def main():
    print("🚀 Stock Prediction System - Health Check")
    print("This script will verify your setup is working correctly")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment), 
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("GPU Availability", check_gpu_availability),
        ("Data Collection", test_data_collection),
        ("NLP Models", test_nlp_models),
        ("Core Imports", test_core_imports),
        ("Functional Test", run_quick_functional_test),
        ("System Resources", check_memory_and_disk)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_error(f"{check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed_count = sum(1 for name, result in results if result)
    total_count = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    print(f"\nOverall: {passed_count}/{total_count} checks passed")
    
    all_tests_passed = passed_count == total_count
    provide_next_steps(all_tests_passed)
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)