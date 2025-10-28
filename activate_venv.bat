@echo off
call venv\Scripts\activate.bat
set VIRTUAL_ENV=%CD%\venv
echo Virtual environment activated!
echo.
echo Checking for missing packages...

REM Check if transformers is installed
python -c "import transformers" 2>nul
if errorlevel 1 (
    echo Installing transformers...
    pip install transformers
    if errorlevel 1 (
        echo WARNING: Could not install transformers automatically.
        echo You may need to install Rust compiler or use pre-compiled wheels.
    )
)

echo.
echo Ready to run experiments.
echo Use: RUN_EXPERIMENTS.bat
echo.

REM If a parameter is passed, run it
if "%1"=="run_experiments" (
    echo Starting experiments...
    call RUN_EXPERIMENTS.bat
)