@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  RFQ Orchestrator - Windows Installer
echo ============================================================
echo.

:: ------------------------------------------------------------
:: 1. Check Python is installed and version is acceptable
:: ------------------------------------------------------------
echo [1/4] Checking Python installation...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python not found on PATH.
    echo.
    echo Install Python 3.11 or later from https://www.python.org/downloads/
    echo During installation, check "Add Python to PATH".
    echo Then re-run this script.
    echo.
    pause
    exit /b 1
)

:: Extract major.minor version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if !PY_MAJOR! LSS 3 (
    echo ERROR: Python 3.10 or later is required. Found !PY_VER!
    pause
    exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo ERROR: Python 3.10 or later is required. Found !PY_VER!
    pause
    exit /b 1
)

echo     OK: Python !PY_VER!

:: ------------------------------------------------------------
:: 2. Upgrade pip
:: ------------------------------------------------------------
echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo WARNING: pip upgrade failed. Continuing with existing pip.
)
echo     OK.

:: ------------------------------------------------------------
:: 3. Install required packages
:: ------------------------------------------------------------
echo.
echo [3/4] Installing required packages...
echo     aiohttp, python-dotenv, numpy, scipy, aiolimiter
echo     pyside6, matplotlib, qasync  (GUI)
echo.

python -m pip install ^
    "aiohttp>=3.9" ^
    "python-dotenv>=1.0" ^
    "numpy>=1.24" ^
    "scipy>=1.11" ^
    "aiolimiter>=1.1" ^
    "pyside6>=6.5" ^
    "matplotlib>=3.6" ^
    "qasync>=0.27"

if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [4/4] Verifying imports...
python -c "import aiohttp, dotenv, numpy, scipy, aiolimiter, PySide6, matplotlib, qasync; print('    All imports OK.')"
if errorlevel 1 (
    echo ERROR: Import verification failed. See errors above.
    pause
    exit /b 1
)

:: ------------------------------------------------------------
:: Done
:: ------------------------------------------------------------
echo.
echo ============================================================
echo  Installation complete.
echo.
echo  Next steps:
echo    1. Create a .env file in this folder with your API keys:
echo         COINCALL_SUB_API_KEY=your_key_here
echo         COINCALL_SUB_API_SECRET=your_secret_here
echo         COINCALL_API_KEY=your_main_key_here
echo         COINCALL_API_SECRET=your_main_secret_here
echo.
echo    2a. Run the GUI:
echo         python multileg_rfq_orchestrator_GUI_WIN.py
echo.
echo    2b. Or run the CLI orchestrator:
echo         Edit rfq_orchestrator_win.py, set RFQConfig + leg_specs, then:
echo         python rfq_orchestrator_win.py --account sub
echo.
echo  See GUIDE.md for full configuration reference.
echo ============================================================
echo.
pause
