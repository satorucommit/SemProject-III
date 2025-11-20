@echo off
REM NASA Patent Forecasting - Web Application Startup Script

echo ================================================================
echo 🚀 NASA PATENT FORECASTING - WEB APPLICATION
echo ================================================================

echo.
echo 📦 Checking virtual environment...
if exist "nasa_patent_env\Scripts\activate.bat" (
    echo ✅ Virtual environment found
    call nasa_patent_env\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found
    echo    Please run setup_venv.bat first
    pause
    exit /b 1
)

echo.
echo 📦 Checking required packages...
python -c "import flask; import flask_cors" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Required packages found
) else (
    echo ⚠️  Required packages not found
    echo    Installing Flask dependencies...
    pip install flask flask-cors
    if %errorlevel% neq 0 (
        echo ❌ Failed to install required packages
        pause
        exit /b 1
    )
    echo ✅ Packages installed successfully
)

echo.
echo 📡 Starting web application...
echo    Host: http://127.0.0.1:5000
echo    Press Ctrl+C to stop the server
echo ================================================================

python web_app.py

echo.
echo 👋 Server stopped
pause