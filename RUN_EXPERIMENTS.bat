@echo off
REM ================================================================================
REM RUN ALL EXPERIMENTS FOR REAL RESULTS - WINDOWS VERSION
REM This will take 5-8 hours total
REM Make sure you have stable internet connection and GPU is available
REM ================================================================================

setlocal enabledelayedexpansion

REM Define checkmark symbol
set checkmark=[OK]

echo ==================================
echo STARTING FULL EXPERIMENT PIPELINE
echo ==================================
echo.
echo Estimated time: 5-8 hours
echo Start time: %date% %time%
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo ERROR: Virtual environment is not activated!
    echo Please run: activate_venv.bat
    echo Then run this script again.
    pause
    exit /b 1
)

REM Verify Python and packages are available
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please check your installation.
    pause
    exit /b 1
)

REM Verify GPU availability
echo Checking GPU availability...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU Detected: {torch.cuda.get_device_name(0)}')" 2>nul
if errorlevel 1 (
    echo WARNING: GPU not detected! Training will be VERY slow on CPU.
    echo Press Ctrl+C to cancel, or any key to continue anyway...
    pause
)

REM Set GPU optimization environment variables
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_ENABLE_AUTO_MIXED_PRECISION=1
REM set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512  # Removed - causing CachingAllocator error

REM Create necessary directories
if not exist "logs\" mkdir logs
if not exist "results\" mkdir results
if not exist "results\test_single\" mkdir results\test_single
if not exist "results\comparison\" mkdir results\comparison
if not exist "results\multi_sector\" mkdir results\multi_sector
if not exist "results\paper\" mkdir results\paper
if not exist "results\paper\figures\" mkdir results\paper\figures
if not exist "results\paper\tables\" mkdir results\paper\tables
if not exist "results\full_training\" mkdir results\full_training

REM ================================================================================
REM EXPERIMENT 1: Full Training with Optuna (2-4 hours)
REM ================================================================================
echo.
echo ==========================================
echo STEP 1/5: Full Training on 1000 days AAPL
echo ==========================================
echo Expected time: 2-4 hours
echo This will:
echo   - Download real AAPL stock data (1000 trading days)
echo   - Download real financial news about Apple
echo   - Extract FinBERT embeddings from news
echo   - Train hybrid BiLSTM-Transformer model
echo   - Run Optuna hyperparameter optimization (100 trials)
echo.

python scripts\train_full_system.py --ticker AAPL --days 1000 --n-trials 100

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo Check logs\full_training.log for details
    pause
    exit /b 1
)

echo.
echo %checkmark% STEP 1 COMPLETE
echo.

REM ================================================================================
REM EXPERIMENT 2: Baseline Model Comparison (30-45 minutes)
REM ================================================================================
echo.
echo ==========================================
echo STEP 2/5: Baseline Model Comparisons
echo ==========================================
echo Expected time: 30-45 minutes
echo This will train and compare:
echo   - Random Forest
echo   - Logistic Regression
echo   - Baseline LSTM
echo   - BiLSTM + Attention
echo   - Transformer
echo   - Your Hybrid Model
echo.

python scripts\compare_models.py --ticker AAPL --days 1000

if errorlevel 1 (
    echo.
    echo ERROR: Baseline comparison failed!
    echo Check logs for details
    pause
    exit /b 1
)

echo.
echo %checkmark% STEP 2 COMPLETE
echo.

REM ================================================================================
REM EXPERIMENT 3: Multi-Stock Evaluation (1-2 hours)
REM ================================================================================
echo.
echo ==========================================
echo STEP 3/5: Multi-Stock Evaluation
echo ==========================================
echo Expected time: 1-2 hours
echo This will test on 15 stocks across 5 sectors:
echo   - Tech: AAPL, MSFT, GOOGL
echo   - Finance: JPM, GS, BAC
echo   - Healthcare: JNJ, PFE, UNH
echo   - Energy: XOM, CVX, COP
echo   - Retail: WMT, AMZN, TGT
echo.

python examples\multi_sector_example.py

if errorlevel 1 (
    echo.
    echo ERROR: Multi-stock evaluation failed!
    echo This might be due to data availability issues
    echo Check if you have NewsAPI key in .env file
    pause
    exit /b 1
)

echo.
echo %checkmark% STEP 3 COMPLETE
echo.

REM ================================================================================
REM EXPERIMENT 4: Ablation Study (30 minutes)
REM ================================================================================
echo.
echo ==========================================
echo STEP 4/5: Ablation Study
echo ==========================================
echo Expected time: 30 minutes
echo This will test contribution of each component:
echo   - Baseline model
echo   - + Extended data (1000 days)
echo   - + Rich features (80 features)
echo   - + BiLSTM architecture
echo   - + Attention mechanism
echo   - + FinBERT embeddings
echo   - + Optuna optimization
echo.

python examples\ablation_study_example.py

if errorlevel 1 (
    echo.
    echo ERROR: Ablation study failed!
    echo Check logs for details
    pause
    exit /b 1
)

echo.
echo %checkmark% STEP 4 COMPLETE
echo.

REM ================================================================================
REM EXPERIMENT 5: Statistical Significance Testing (15 minutes)
REM ================================================================================
echo.
echo ==========================================
echo STEP 5/5: Statistical Significance Tests
echo ==========================================
echo Expected time: 15 minutes
echo This will run:
echo   - t-tests between models
echo   - McNemar's test
echo   - Confidence interval calculations
echo   - Effect size measurements
echo.

python examples\significance_testing_example.py

if errorlevel 1 (
    echo.
    echo ERROR: Significance testing failed!
    echo Check logs for details
    pause
    exit /b 1
)

echo.
echo %checkmark% STEP 5 COMPLETE
echo.

REM ================================================================================
REM GENERATE FIGURES AND TABLES WITH REAL DATA
REM ================================================================================
echo.
echo ==========================================
echo GENERATING PUBLICATION FIGURES ^& TABLES
echo ==========================================
echo This will create publication-quality:
echo   - 8 figures (PNG + PDF)
echo   - 11 LaTeX tables
echo Using YOUR REAL experimental results
echo.

python scripts\generate_paper_figures.py

if errorlevel 1 (
    echo.
    echo WARNING: Figure generation had issues
    echo Some figures might be missing
    echo Continuing anyway...
)

python scripts\generate_latex_tables.py

if errorlevel 1 (
    echo.
    echo WARNING: Table generation had issues
    echo Some tables might be missing
    echo Continuing anyway...
)

echo.
echo %checkmark% FIGURES AND TABLES GENERATED
echo.

REM ================================================================================
REM SUMMARY
REM ================================================================================
echo.
echo ==========================================
echo      ALL EXPERIMENTS COMPLETE!
echo ==========================================
echo End time: %date% %time%
echo.
echo Results are in:
echo   Full Training: results\full_training\
echo   Figures: results\paper\figures\
echo   Tables: results\paper\tables\
echo   Metrics: results\test_single\evaluation_metrics.json
echo.
echo Next steps:
echo   1. Check your actual accuracy:
echo      type results\full_training\FINAL_REPORT.json
echo   2. View training curves:
echo      start results\full_training\training_curves.png
echo   3. View figures:
echo      start results\paper\figures
echo   4. Update paper with REAL results
echo   5. Compile LaTeX paper
echo.
echo %checkmark% You now have 100%% REAL, GENUINE results for publication!
echo.

REM Open results folder
echo Opening results folder...
start explorer results

REM Display final metrics
echo.
echo ==========================================
echo YOUR ACTUAL RESULTS:
echo ==========================================
if exist "results\full_training\FINAL_REPORT.json" (
    type results\full_training\FINAL_REPORT.json
) else (
    echo Full training report not found. Check logs\full_training.log for details.
)
echo.

pause
