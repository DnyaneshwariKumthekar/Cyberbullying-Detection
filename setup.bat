@echo off
REM Cyberbullying Detection System - Windows Setup Script
REM Run this file to automatically setup everything!

echo ============================================================
echo   Cyberbullying Detection System - Automated Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo ============================================================
echo   Step 1: Creating Virtual Environment
echo ============================================================
echo.

if exist "venv" (
    echo [INFO] Virtual environment already exists
) else (
    echo [ACTION] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

echo ============================================================
echo   Step 2: Activating Virtual Environment
echo ============================================================
echo.

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

echo ============================================================
echo   Step 3: Installing Dependencies
echo ============================================================
echo.

echo [ACTION] This may take 3-5 minutes...
echo.
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo [OK] All dependencies installed
echo.

echo ============================================================
echo   Step 4: Downloading NLTK Data
echo ============================================================
echo.

python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True)"
if errorlevel 1 (
    echo [WARNING] NLTK download had some issues, but continuing...
) else (
    echo [OK] NLTK data downloaded
)
echo.

echo ============================================================
echo   Step 5: Running Complete Pipeline
echo ============================================================
echo.

echo [ACTION] This will take 10-15 minutes...
echo [ACTION] Setting up datasets, preprocessing, and training models
echo.

python scripts\run_pipeline.py
if errorlevel 1 (
    echo [ERROR] Pipeline execution failed
    echo Please check the error messages above
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo [SUCCESS] Cyberbullying Detection System is ready!
echo.
echo Next steps:
echo   1. Launch dashboard: streamlit run dashboard\streamlit_app.py
echo   2. Or run: launch_dashboard.bat
echo.
echo Press any key to launch the dashboard now...
pause >nul

echo.
echo ============================================================
echo   Launching Dashboard...
echo ============================================================
echo.
echo Dashboard will open in your browser
echo Press Ctrl+C to stop the server
echo.

streamlit run dashboard\streamlit_app.py

pause