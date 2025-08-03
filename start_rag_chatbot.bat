@echo off
setlocal enabledelayedexpansion

:: Set colors for output
for /f %%A in ('"prompt $H &echo on &for %%B in (1) do rem"') do set BS=%%A

:: RAG Chatbot Startup Script
echo.
echo ==========================================
echo    🤖 RAG Chatbot - All-in-One Startup
echo ==========================================
echo.

:: Check if Python is installed
echo [1/6] 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found! Please install Python 3.8+ first.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo ✅ Python found!
echo.

:: Check if we're in the right directory
echo [2/6] 📁 Checking project directory...
if not exist "app_corrected_final.py" (
    echo ❌ ERROR: app_corrected_final.py not found!
    echo    Make sure you're running this from the RAG directory.
    echo    Current directory: %CD%
    pause
    exit /b 1
)
echo ✅ Project files found!
echo.

:: Check for .env file
echo [3/6] 🔑 Checking environment configuration...
if not exist ".env" (
    echo ⚠️  WARNING: .env file not found!
    echo    Creating basic .env template...
    echo # RAG Chatbot Environment Configuration > .env
    echo # Required: >> .env
    echo OPENAI_API_KEY=your-openai-api-key-here >> .env
    echo. >> .env
    echo # Optional - For Confluence support: >> .env
    echo #CONFLUENCE_USERNAME=your-email@company.com >> .env
    echo #CONFLUENCE_API_TOKEN=your-confluence-api-token >> .env
    echo.
    echo 📝 Created .env file template. Please edit it with your API keys.
    echo    You can continue without it, but OpenAI features won't work.
    echo.
)

:: Check if API key is configured
findstr /C:"OPENAI_API_KEY=sk-" .env >nul 2>&1
if errorlevel 1 (
    echo ⚠️  OpenAI API key not configured in .env file
    echo    Some features will be disabled.
) else (
    echo ✅ OpenAI API key found!
)
echo.

:: Install/Update dependencies
echo [4/6] 📦 Installing/Updating dependencies...
echo    This might take a moment...

if exist "requirements_ultimate_fixed.txt" (
    echo    Using requirements_ultimate_fixed.txt...
    pip install -r requirements_ultimate_fixed.txt --quiet --disable-pip-version-check
) else (
    echo ❌ ERROR: requirements_ultimate_fixed.txt not found!
    echo    Cannot install dependencies.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo ❌ ERROR: Failed to install dependencies!
    echo    Try running manually: pip install -r requirements_ultimate_fixed.txt
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully!
echo.

:: Check Tesseract OCR (optional)
echo [5/6] 🔍 Checking OCR support...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Tesseract OCR not found - OCR features will be disabled
    echo    For full OCR support, install from:
    echo    https://github.com/UB-Mannheim/tesseract/wiki
) else (
    echo ✅ Tesseract OCR found - Full OCR support available!
)
echo.

:: Create necessary directories
echo [6/6] 📂 Setting up directories...
if not exist "uploads" mkdir uploads
if not exist "templates" mkdir templates
if not exist "chroma_db" mkdir chroma_db
if not exist "extracted_images" mkdir extracted_images
echo ✅ Directories ready!
echo.

:: Display system status
echo ==========================================
echo    🚀 STARTING RAG CHATBOT
echo ==========================================
echo.
python scripts\system_check.py

echo.
echo 🌐 Starting server on http://localhost:5000
echo 🗑️ Database management available in web UI
echo 📝 Logs will appear below...
echo.
echo ==========================================
echo.

:: Start the application
python app_corrected_final.py

:: If the app exits, show status
echo.
echo ==========================================
echo    📋 APPLICATION STATUS
echo ==========================================

if errorlevel 1 (
    echo ❌ Application exited with error code: %errorlevel%
    echo.
    echo 🔧 Troubleshooting Tips:
    echo    • Check if port 5000 is already in use
    echo    • Verify your .env file configuration
    echo    • Check Python dependencies
    echo    • Review error messages above
    echo.
    echo 💡 Quick fixes:
    echo    • Database issues: python clear_database.py
    echo    • Dependencies: pip install -r requirements_ultimate_fixed.txt
    echo    • Port conflict: Change port in app_corrected_final.py
) else (
    echo ✅ Application exited normally
)

echo.
echo Press any key to exit...
pause >nul 