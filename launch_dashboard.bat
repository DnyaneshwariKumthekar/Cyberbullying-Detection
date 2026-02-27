@echo off
REM Quick launcher for Cyberbullying Detection Dashboard

echo ============================================================
echo   Cyberbullying Detection System - Dashboard Launcher
echo ============================================================
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARNING] Virtual environment not found
    echo Run setup.bat first or create venv manually
)
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit not installed
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Check if dashboard file exists
if not exist "dashboard\streamlit_app.py" (
    echo [ERROR] Dashboard file not found
    echo Please ensure dashboard\streamlit_app.py exists
    pause
    exit /b 1
)

echo [INFO] Starting dashboard...
echo [INFO] Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Launch dashboard
streamlit run dashboard\streamlit_app.py

pause