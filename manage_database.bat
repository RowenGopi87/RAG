@echo off
echo.
echo ==========================================
echo    🗑️ RAG Chatbot - Database Manager
echo ==========================================
echo.

:: Check if clear_database.py exists
if not exist "scripts\clear_database.py" (
    echo ❌ ERROR: scripts\clear_database.py not found!
    echo    Make sure you're in the RAG_Chatbot directory.
    pause
    exit /b 1
)

:: Menu system
:menu
echo Please choose an option:
echo.
echo [1] 📊 Show database status
echo [2] 🗑️ Clear entire database
echo [3] ✅ Verify cleanup
echo [4] 🚪 Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto status
if "%choice%"=="2" goto clear
if "%choice%"=="3" goto verify
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
goto menu

:status
echo.
echo 📊 Getting database status...
python scripts\clear_database.py
echo.
pause
goto menu

:clear
echo.
echo ⚠️  WARNING: This will delete ALL documents, images, and processed data!
set /p confirm="Are you sure? Type 'yes' to confirm: "
if not "%confirm%"=="yes" (
    echo ❌ Cancelled.
    echo.
    goto menu
)

echo.
echo 🗑️ Clearing database...
echo 2 | python scripts\clear_database.py
echo.
pause
goto menu

:verify
echo.
echo ✅ Verifying cleanup...
echo 3 | python scripts\clear_database.py
echo.
pause
goto menu

:exit
echo.
echo 👋 Goodbye!
echo. 