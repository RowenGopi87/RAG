@echo off
echo.
echo 🚀 Quick Starting RAG Chatbot...
echo.

:: Check if main app exists
if not exist "app_corrected_final.py" (
    echo ❌ ERROR: app_corrected_final.py not found!
    echo    Use start_rag_chatbot.bat for full setup.
    pause
    exit /b 1
)

:: Start the app directly
echo ✅ Starting server on http://localhost:5000
echo 📝 Press Ctrl+C to stop the server
echo.

python app_corrected_final.py 