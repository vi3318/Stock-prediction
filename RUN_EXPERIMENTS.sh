#!/bin/bash

################################################################################
# RUN ALL EXPERIMENTS FOR REAL RESULTS
# This will take 5-8 hours total
# Make sure you have stable internet connection
################################################################################

echo "=================================="
echo "STARTING FULL EXPERIMENT PIPELINE"
echo "=================================="
echo ""
echo "Estimated time: 5-8 hours"
echo "Start time: $(date)"
echo ""

# Navigate to project directory
cd /Users/vidharia/Documents/Projects/dl

################################################################################
# EXPERIMENT 1: Full Training with Optuna (2-4 hours)
################################################################################
echo ""
echo "=========================================="
echo "STEP 1/5: Full Training on 1000 days AAPL"
echo "=========================================="
echo "Expected time: 2-4 hours"
echo "This will:"
echo "  - Download real AAPL stock data (1000 trading days)"
echo "  - Download real financial news about Apple"
echo "  - Extract FinBERT embeddings from news"
echo "  - Train hybrid BiLSTM-Transformer model"
echo "  - Run Optuna hyperparameter optimization (100 trials)"
echo ""

python3 scripts/train_full_system.py --ticker AAPL --days 1000 --n-trials 100

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo "✅ STEP 1 COMPLETE"
echo ""

################################################################################
# EXPERIMENT 2: Baseline Model Comparison (30-45 minutes)
################################################################################
echo ""
echo "=========================================="
echo "STEP 2/5: Baseline Model Comparisons"
echo "=========================================="
echo "Expected time: 30-45 minutes"
echo "This will train and compare:"
echo "  - Random Forest"
echo "  - Logistic Regression"
echo "  - Baseline LSTM"
echo "  - BiLSTM + Attention"
echo "  - Transformer"
echo "  - Your Hybrid Model"
echo ""

python3 scripts/compare_models.py --ticker AAPL --days 1000

if [ $? -ne 0 ]; then
    echo "ERROR: Baseline comparison failed!"
    exit 1
fi

echo "✅ STEP 2 COMPLETE"
echo ""

################################################################################
# EXPERIMENT 3: Multi-Stock Evaluation (1-2 hours)
################################################################################
echo ""
echo "=========================================="
echo "STEP 3/5: Multi-Stock Evaluation"
echo "=========================================="
echo "Expected time: 1-2 hours"
echo "This will test on 15 stocks across 5 sectors:"
echo "  - Tech: AAPL, MSFT, GOOGL"
echo "  - Finance: JPM, GS, BAC"
echo "  - Healthcare: JNJ, PFE, UNH"
echo "  - Energy: XOM, CVX, COP"
echo "  - Retail: WMT, AMZN, TGT"
echo ""

python3 examples/multi_sector_example.py

if [ $? -ne 0 ]; then
    echo "ERROR: Multi-stock evaluation failed!"
    exit 1
fi

echo "✅ STEP 3 COMPLETE"
echo ""

################################################################################
# EXPERIMENT 4: Ablation Study (30 minutes)
################################################################################
echo ""
echo "=========================================="
echo "STEP 4/5: Ablation Study"
echo "=========================================="
echo "Expected time: 30 minutes"
echo "This will test contribution of each component:"
echo "  - Baseline model"
echo "  - + Extended data (1000 days)"
echo "  - + Rich features (80 features)"
echo "  - + BiLSTM architecture"
echo "  - + Attention mechanism"
echo "  - + FinBERT embeddings"
echo "  - + Optuna optimization"
echo ""

python3 examples/ablation_study_example.py

if [ $? -ne 0 ]; then
    echo "ERROR: Ablation study failed!"
    exit 1
fi

echo "✅ STEP 4 COMPLETE"
echo ""

################################################################################
# EXPERIMENT 5: Statistical Significance Testing (15 minutes)
################################################################################
echo ""
echo "=========================================="
echo "STEP 5/5: Statistical Significance Tests"
echo "=========================================="
echo "Expected time: 15 minutes"
echo "This will run:"
echo "  - t-tests between models"
echo "  - McNemar's test"
echo "  - Confidence interval calculations"
echo "  - Effect size measurements"
echo ""

python3 examples/significance_testing_example.py

if [ $? -ne 0 ]; then
    echo "ERROR: Significance testing failed!"
    exit 1
fi

echo "✅ STEP 5 COMPLETE"
echo ""

################################################################################
# GENERATE FIGURES AND TABLES WITH REAL DATA
################################################################################
echo ""
echo "=========================================="
echo "GENERATING PUBLICATION FIGURES & TABLES"
echo "=========================================="
echo "This will create publication-quality:"
echo "  - 8 figures (PNG + PDF)"
echo "  - 11 LaTeX tables"
echo "Using YOUR REAL experimental results"
echo ""

python3 scripts/generate_paper_figures.py

if [ $? -ne 0 ]; then
    echo "ERROR: Figure generation failed!"
    exit 1
fi

python3 scripts/generate_latex_tables.py

if [ $? -ne 0 ]; then
    echo "ERROR: Table generation failed!"
    exit 1
fi

echo "✅ FIGURES AND TABLES GENERATED"
echo ""

################################################################################
# SUMMARY
################################################################################
echo ""
echo "=========================================="
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results are in:"
echo "  📊 Figures: results/paper/figures/"
echo "  📋 Tables: results/paper/tables/"
echo "  📈 Metrics: results/test_single/evaluation_metrics.json"
echo ""
echo "Next steps:"
echo "  1. Check your actual accuracy: cat results/test_single/evaluation_metrics.json"
echo "  2. View figures: open results/paper/figures/"
echo "  3. Update paper with REAL results"
echo "  4. Compile LaTeX paper"
echo ""
echo "✅ You now have 100% REAL, GENUINE results for publication!"
echo ""
