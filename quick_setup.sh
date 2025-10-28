#!/bin/bash

# Quick Setup and Test Script for Stock Prediction System
# This script will guide you through the complete setup process

set -e  # Exit on any error

echo "🚀 Stock Prediction System - Quick Setup & Test"
echo "================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Should contain main.py and requirements.txt"
    exit 1
fi

echo "✅ Project directory verified"
echo ""

# Function to print status
print_status() {
    echo "📋 $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

print_warning() {
    echo "⚠️  $1"
}

# Step 1: Check Python version
print_status "Step 1/8: Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
major_version=$(echo $python_version | cut -d'.' -f1)
minor_version=$(echo $python_version | cut -d'.' -f2)

if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 8 ]; then
    print_success "Python $python_version is compatible"
else
    print_error "Python 3.8+ required, found $python_version"
    echo "Install Python 3.8+ and try again"
    exit 1
fi

# Step 2: Create virtual environment
print_status "Step 2/8: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Step 3: Activate virtual environment
print_status "Step 3/8: Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 4: Upgrade pip and install dependencies  
print_status "Step 4/8: Installing dependencies..."
echo "This may take 5-10 minutes..."

pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

echo "Installing core packages..."
pip install -r requirements.txt

# Handle common missing packages
echo "Installing additional packages..."
pip install pyyaml PyYAML > /dev/null 2>&1

print_success "All dependencies installed"

# Step 5: Download required models
print_status "Step 5/8: Downloading NLP models..."
echo "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm > /dev/null 2>&1
print_success "spaCy model downloaded"

echo "Testing FinBERT download (this may take a few minutes)..."
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
print('✅ FinBERT model cached')
" 2>/dev/null || print_warning "FinBERT will download on first use"

# Step 6: Create required directories
print_status "Step 6/8: Creating directory structure..."
mkdir -p data/raw/stocks
mkdir -p data/raw/news  
mkdir -p data/processed/features
mkdir -p data/processed/embeddings
mkdir -p models/saved
mkdir -p logs
mkdir -p results/figures
mkdir -p results/reports

print_success "Directory structure created"

# Step 7: Run health check
print_status "Step 7/8: Running system health check..."
echo ""
python3 health_check.py

if [ $? -eq 0 ]; then
    print_success "Health check passed!"
else
    print_error "Health check failed - see output above"
    echo ""
    echo "Common solutions:"
    echo "1. Make sure you're in the virtual environment: source venv/bin/activate"
    echo "2. Check internet connection for downloads"
    echo "3. Re-run: pip install -r requirements.txt"
    exit 1
fi

# Step 8: Run quick system test
print_status "Step 8/8: Running quick system test..."
echo ""
echo "Testing data collection..."

python3 -c "
import sys
import os
sys.path.append('./src')

# Test basic functionality
import yfinance as yf
ticker = yf.Ticker('AAPL')
hist = ticker.history(period='5d')
print(f'✅ Retrieved {len(hist)} days of AAPL data')

# Test file I/O
import pandas as pd
test_file = 'data/raw/stocks/test.csv'
hist.to_csv(test_file)
loaded = pd.read_csv(test_file, index_col=0)
print(f'✅ File I/O test passed: {len(loaded)} rows')
os.remove(test_file)

print('✅ Quick system test completed successfully!')
"

if [ $? -eq 0 ]; then
    print_success "Quick test passed!"
else
    print_error "Quick test failed"
    exit 1
fi

echo ""
echo "🎉 SETUP COMPLETE! 🎉"
echo "===================="
echo ""
echo "Your system is ready to use. Next steps:"
echo ""
echo "1. Run a quick demo:"
echo "   python3 main.py --mode full --tickers AAPL --start-date 2023-01-01 --end-date 2023-03-31"
echo ""
echo "2. Try the examples:"
echo "   python3 examples/learnable_weights_example.py"
echo "   python3 examples/explainability_example.py"
echo ""
echo "3. See detailed guide:"
echo "   open SETUP_AND_TESTING_GUIDE.md"
echo ""
echo "4. Re-activate environment later with:"
echo "   source venv/bin/activate"
echo ""
print_success "System ready for research! 🚀"