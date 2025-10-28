@echo off
REM ================================================================================
REM WINDOWS QUICK SETUP - Stock Prediction System
REM Automated setup for Dell G16 RTX 3070 with Windows 11
REM ================================================================================

echo ==========================================
echo Stock Prediction System - Quick Setup
echo ==========================================
echo.
echo This will:
echo   1. Check Python installation
echo   2. Create virtual environment
echo   3. Install CUDA-enabled PyTorch
echo   4. Install all dependencies
echo   5. Download required models
echo   6. Verify GPU availability
echo.
echo Estimated time: 10-15 minutes
echo.
pause

REM ================================================================================
REM STEP 1: Check Python
REM ================================================================================
echo.
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.8-3.10 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM ================================================================================
REM STEP 2: Create Virtual Environment
REM ================================================================================
echo.
echo [2/6] Creating virtual environment...
if exist "venv\" (
    echo Virtual environment already exists.
    choice /C YN /M "Do you want to recreate it? (This will delete the existing one)"
    if errorlevel 2 goto skip_venv_creation
    if errorlevel 1 (
        echo Removing old virtual environment...
        rmdir /s /q venv
    )
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

:skip_venv_creation

REM ================================================================================
REM STEP 3: Activate Virtual Environment
REM ================================================================================
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM ================================================================================
REM STEP 4: Upgrade pip
REM ================================================================================
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM ================================================================================
REM STEP 5: Install PyTorch with CUDA 11.8 Support
REM ================================================================================
echo.
echo [5/6] Installing PyTorch with CUDA support...
echo This may take 5-10 minutes...
echo.

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch!
    pause
    exit /b 1
)

echo.
echo Verifying PyTorch GPU support...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if errorlevel 1 (
    echo WARNING: PyTorch installed but GPU check failed!
    echo You might need to install CUDA drivers.
    echo See WINDOWS_GPU_SETUP.md for details.
    pause
)
echo.

REM ================================================================================
REM STEP 6: Install TensorFlow with GPU Support
REM ================================================================================
echo.
echo [6/6] Installing TensorFlow...
pip install tensorflow==2.17.0

if errorlevel 1 (
    echo WARNING: TensorFlow installation had issues
    echo Continuing anyway...
)
echo.

REM ================================================================================
REM STEP 7: Install Core Dependencies
REM ================================================================================
echo.
echo [7/9] Installing core dependencies...
pip install transformers==4.30.0 sentence-transformers==2.2.2
echo.

REM ================================================================================
REM STEP 8: Install Data Collection Libraries
REM ================================================================================
echo.
echo [8/9] Installing data collection libraries...
pip install yfinance==0.2.28 pandas-datareader alpha-vantage newsapi-python
pip install requests beautifulsoup4 newspaper3k
echo.

REM ================================================================================
REM STEP 9: Install Data Processing Libraries
REM ================================================================================
echo.
echo [9/9] Installing data processing libraries...
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install spacy nltk textblob
pip install shap lime
pip install python-dotenv tqdm pyyaml ta joblib
pip install optuna streamlit pytest black flake8
echo.

REM ================================================================================
REM STEP 10: Download spaCy Language Model
REM ================================================================================
echo.
echo [10/10] Downloading spaCy language model...
python -m spacy download en_core_web_sm

if errorlevel 1 (
    echo WARNING: spaCy model download failed
    echo You can try again later with: python -m spacy download en_core_web_sm
    pause
)
echo.

REM ================================================================================
REM STEP 11: Create Required Directories
REM ================================================================================
echo.
echo Creating required directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\cache" mkdir data\cache
if not exist "logs" mkdir logs
if not exist "results" mkdir results
if not exist "models\saved_models" mkdir models\saved_models
if not exist "visualizations" mkdir visualizations
echo.

REM ================================================================================
REM STEP 12: Verify Installation
REM ================================================================================
echo.
echo ==========================================
echo VERIFYING INSTALLATION
echo ==========================================
echo.

echo Checking PyTorch...
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo Checking TensorFlow...
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__, '| GPU:', len(tf.config.list_physical_devices('GPU')))" 2>nul

echo Checking Transformers...
python -c "import transformers; print('Transformers:', transformers.__version__)"

echo Checking yfinance...
python -c "import yfinance; print('yfinance: OK')"

echo Checking spaCy...
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy: OK')"

echo.

REM ================================================================================
REM STEP 13: GPU Information
REM ================================================================================
echo.
echo ==========================================
echo GPU INFORMATION
echo ==========================================
echo.

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found!
    echo Make sure NVIDIA drivers are installed.
    echo.
) else (
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo.
)

REM ================================================================================
REM SUMMARY
REM ================================================================================
echo.
echo ==========================================
echo SETUP COMPLETE!
echo ==========================================
echo.
echo Next steps:
echo   1. (Optional) Add NewsAPI key to .env file:
echo      echo NEWS_API_KEY=your_key_here ^> .env
echo.
echo   2. Activate environment (do this every time):
echo      activate_venv.bat
echo.
echo   3. Run experiments:
echo      RUN_EXPERIMENTS.bat
echo.
echo   Or test individual components:
echo      python scripts\train_full_system.py --ticker AAPL --days 250 --n-trials 20
echo.
echo For detailed instructions, see: WINDOWS_GPU_SETUP.md
echo.
echo Virtual environment location: %CD%\venv
echo.

REM Create activation helper
echo @echo off > activate_venv.bat
echo call venv\Scripts\activate.bat >> activate_venv.bat
echo echo Virtual environment activated! >> activate_venv.bat
echo echo. >> activate_venv.bat
echo echo Ready to run experiments. >> activate_venv.bat
echo echo Use: RUN_EXPERIMENTS.bat >> activate_venv.bat
echo. >> activate_venv.bat

echo Created activation script: activate_venv.bat
echo.

pause
