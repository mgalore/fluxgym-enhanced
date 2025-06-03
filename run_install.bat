@echo off
echo FluxGym Installation Script for Windows
echo =======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python found, starting installation...
echo.

REM Run the installation script
python install.py

echo.
echo Installation script completed.
echo Check the output above for any errors.
echo.
pause 