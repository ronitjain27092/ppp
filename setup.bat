@echo off
REM Setup script for Malware Detection XAI project

echo.
echo ========================================
echo Malware Detection XAI - Setup Script
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take 5-10 minutes...
echo.
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

REM Check installation
echo.
echo Verifying installation...
python -c "import tensorflow; import shap; import streamlit" && (
    echo.
    echo ========================================
    echo SUCCESS! All dependencies installed.
    echo ========================================
    echo.
    echo To start the application, run:
    echo.
    echo .\venv\Scripts\activate.bat
    echo streamlit run app.py
    echo.
) || (
    echo.
    echo ERROR: Some dependencies failed to install.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

pause
