@echo off
REM ================================================================================
REM Run Experiments with Virtual Environment
REM This ensures the virtual environment is activated before running experiments
REM ================================================================================

echo Activating virtual environment and running experiments...
echo.

call activate_venv.bat

REM Clear Python cache to ensure latest code is used
echo Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo Done.
echo.

REM Create logs directory
if not exist "logs\" mkdir logs

echo Starting training - AAPL, 1000 days, 100 optimization trials
echo.
echo ================================================================================
echo.

REM Run with unbuffered Python output
python -u scripts\train_full_system.py --ticker AAPL --days 1000 --n-trials 100

set TRAINING_ERROR=%errorlevel%

echo.
echo ================================================================================
if %TRAINING_ERROR% equ 0 (
    echo Training completed successfully!
) else (
    echo ERROR: Training failed with exit code %TRAINING_ERROR%
)
echo ================================================================================
echo.

pause