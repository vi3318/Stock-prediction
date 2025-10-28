@echo off
REM ================================================================================
REM Run Experiments with Virtual Environment
REM This ensures the virtual environment is activated before running experiments
REM ================================================================================

echo Activating virtual environment and running experiments...
echo.

call activate_venv.bat run_experiments

echo.
echo Experiments completed!
pause