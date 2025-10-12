@echo off
REM NASA Patent Forecasting - Virtual Environment Setup Script
REM This script creates and sets up a virtual environment for the project

echo ================================================================
echo 🚀 NASA PATENT FORECASTING - VIRTUAL ENVIRONMENT SETUP
echo ================================================================

echo.
echo 📦 Creating virtual environment...
python -m venv nasa_patent_env

echo.
echo 🔧 Activating virtual environment...
call nasa_patent_env\Scripts\activate.bat

echo.
echo 📥 Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Virtual environment setup completed!
echo.
echo 💡 TO USE THE VIRTUAL ENVIRONMENT:
echo    1. Activate:   nasa_patent_env\Scripts\activate.bat
echo    2. Run app:    python main.py
echo    3. Run demo:   python simple_demo.py  
echo    4. Deactivate: deactivate
echo.
echo 🎉 Setup complete! Your environment is ready.
pause