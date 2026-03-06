@echo off
setlocal EnableDelayedExpansion
title Satellite Drought Detector — Setup ^& Launch

:: ============================================================
::  COLOUR CODES  (uses ANSI — works on Windows 10/11)
:: ============================================================
set "ESC="
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "RED=%ESC%[91m"
set "CYAN=%ESC%[96m"
set "WHITE=%ESC%[97m"
set "DIM=%ESC%[90m"
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"

:: ============================================================
::  BANNER
:: ============================================================
echo.
echo %CYAN%  ==========================================================%RESET%
echo %CYAN%   %BOLD%SATELLITE DROUGHT RISK PREDICTION%RESET%%CYAN%
echo %CYAN%  - VEDANT BAGWALE
echo %CYAN%  ==========================================================%RESET%
echo.

:: ============================================================
::  STEP 0 — LOCATE SCRIPT DIRECTORY
:: ============================================================
cd /d "%~dp0"
echo %DIM%  Working directory: %CD%%RESET%
echo.

:: ============================================================
::  STEP 1 — CHECK PYTHON
:: ============================================================
echo %YELLOW%  [1/6] Checking Python...%RESET%

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%  [ERROR] Python not found in PATH.%RESET%
    echo.
    echo         Please install Python 3.9+ from https://www.python.org/downloads/
    echo         Make sure to tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if !PYMAJOR! lss 3 (
    echo %RED%  [ERROR] Python 3.9+ required. Found: !PYVER!%RESET%
    pause
    exit /b 1
)
if !PYMAJOR! equ 3 if !PYMINOR! lss 9 (
    echo %RED%  [ERROR] Python 3.9+ required. Found: !PYVER!%RESET%
    pause
    exit /b 1
)

echo %GREEN%  [OK] Python !PYVER! found.%RESET%
echo.

:: ============================================================
::  STEP 2 — CHECK PIP
:: ============================================================
echo %YELLOW%  [2/6] Checking pip...%RESET%

python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%  pip not found — installing...%RESET%
    python -m ensurepip --upgrade
    if %errorlevel% neq 0 (
        echo %RED%  [ERROR] Could not install pip. Please install it manually.%RESET%
        pause
        exit /b 1
    )
)

echo %GREEN%  [OK] pip is available.%RESET%
echo.

:: ============================================================
::  STEP 3 — CREATE OR REUSE VIRTUAL ENVIRONMENT
:: ============================================================
echo %YELLOW%  [3/6] Setting up virtual environment...%RESET%

set VENV_DIR=venv

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo %DIM%  Existing venv found — reusing it.%RESET%
) else (
    echo %DIM%  Creating new virtual environment in .\venv ...%RESET%
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo %RED%  [ERROR] Failed to create virtual environment.%RESET%
        echo         Try: python -m pip install virtualenv
        pause
        exit /b 1
    )
    echo %GREEN%  [OK] Virtual environment created.%RESET%
)

:: Activate it
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo %RED%  [ERROR] Could not activate virtual environment.%RESET%
    pause
    exit /b 1
)

echo %GREEN%  [OK] Virtual environment active.%RESET%
echo.

:: ============================================================
::  STEP 4 — UPGRADE PIP INSIDE VENV
:: ============================================================
echo %YELLOW%  [4/6] Upgrading pip inside venv...%RESET%
python -m pip install --upgrade pip --quiet
echo %GREEN%  [OK] pip up to date.%RESET%
echo.

:: ============================================================
::  STEP 5 — INSTALL / VERIFY DEPENDENCIES
:: ============================================================
echo %YELLOW%  [5/6] Installing dependencies...%RESET%
echo.

:: Check if requirements.txt exists
if exist "requirements.txt" (
    echo %DIM%  Found requirements.txt — installing from it...%RESET%
    python -m pip install -r requirements.txt --quiet
    if %errorlevel% neq 0 (
        echo %YELLOW%  requirements.txt install had issues — trying individual packages...%RESET%
        goto INSTALL_MANUAL
    )
    goto CHECK_TF_VERSION
) else (
    echo %YELLOW%  No requirements.txt found — installing packages individually...%RESET%
    goto INSTALL_MANUAL
)

:INSTALL_MANUAL
echo %DIM%  Installing tensorflow...%RESET%
python -m pip install "tensorflow>=2.16.0" --quiet
echo %DIM%  Installing streamlit...%RESET%
python -m pip install streamlit --quiet
echo %DIM%  Installing numpy...%RESET%
python -m pip install numpy --quiet
echo %DIM%  Installing Pillow...%RESET%
python -m pip install Pillow --quiet
echo %DIM%  Installing matplotlib...%RESET%
python -m pip install matplotlib --quiet
echo %DIM%  Installing scikit-learn...%RESET%
python -m pip install scikit-learn --quiet
echo %DIM%  Installing seaborn...%RESET%
python -m pip install seaborn --quiet

:CHECK_TF_VERSION
:: Verify TensorFlow installed and is recent enough
echo.
echo %DIM%  Verifying TensorFlow version...%RESET%
for /f %%v in ('python -c "import tensorflow as tf; print(tf.__version__)" 2^>nul') do set TFVER=%%v

if "!TFVER!"=="" (
    echo %RED%  [ERROR] TensorFlow did not install correctly.%RESET%
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)

:: Extract major.minor from TF version
for /f "tokens=1,2 delims=." %%a in ("!TFVER!") do (
    set TFMAJOR=%%a
    set TFMINOR=%%b
)

if !TFMAJOR! lss 2 (
    echo %YELLOW%  [WARN] TensorFlow !TFVER! is very old. Upgrading to 2.16+...%RESET%
    python -m pip install "tensorflow>=2.16.0" --upgrade --quiet
)
if !TFMAJOR! equ 2 if !TFMINOR! lss 16 (
    echo %YELLOW%  [WARN] TensorFlow !TFVER! may cause model loading errors.%RESET%
    echo %YELLOW%         Upgrading to 2.16+...%RESET%
    python -m pip install "tensorflow>=2.16.0" --upgrade --quiet
    for /f %%v in ('python -c "import tensorflow as tf; print(tf.__version__)" 2^>nul') do set TFVER=%%v
)

echo %GREEN%  [OK] TensorFlow !TFVER!%RESET%

:: Verify Streamlit
for /f %%v in ('python -c "import streamlit; print(streamlit.__version__)" 2^>nul') do set STVER=%%v
if "!STVER!"=="" (
    echo %RED%  [ERROR] Streamlit did not install.%RESET%
    pause
    exit /b 1
)
echo %GREEN%  [OK] Streamlit !STVER!%RESET%

:: Verify other packages
python -c "import numpy, PIL, matplotlib, sklearn, seaborn" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%  [WARN] Some packages may be missing. Attempting reinstall...%RESET%
    python -m pip install numpy Pillow matplotlib scikit-learn seaborn --quiet
)
echo %GREEN%  [OK] All packages verified.%RESET%
echo.

:: ============================================================
::  STEP 6 — CHECK MODEL ARTIFACTS
:: ============================================================
echo %YELLOW%  [6/6] Checking model artifacts...%RESET%
echo.

set MODEL_OK=0
set META_OK=0

:: Check for model file (either format)
if exist "drought_model_artifacts\drought_model.keras" (
    echo %GREEN%  [OK] drought_model.keras found.%RESET%
    set MODEL_OK=1
) else if exist "drought_model_artifacts\drought_model.h5" (
    echo %GREEN%  [OK] drought_model.h5 found.%RESET%
    set MODEL_OK=1
) else (
    echo %RED%  [MISSING] No model file found in .\drought_model_artifacts\%RESET%
    set MODEL_OK=0
)

:: Check metadata
if exist "drought_model_artifacts\metadata.json" (
    echo %GREEN%  [OK] metadata.json found.%RESET%
    set META_OK=1
) else (
    echo %YELLOW%  [WARN] metadata.json not found — app will use built-in fallback.%RESET%
    set META_OK=0
)

:: Check app.py
if not exist "app.py" (
    echo %RED%  [ERROR] app.py not found in %CD%%RESET%
    echo         Make sure you are running this script from the project folder.
    pause
    exit /b 1
)
echo %GREEN%  [OK] app.py found.%RESET%
echo.

:: Warn if model is missing but don't block — app has its own error display
if !MODEL_OK! equ 0 (
    echo %YELLOW%  ============================================================%RESET%
    echo %YELLOW%   WARNING: Model file is missing!%RESET%
    echo %YELLOW%  ============================================================%RESET%
    echo.
    echo   The app will launch but show an error until you add the model.
    echo.
    echo   To get the model files:
    echo     1. Open drought_detection_final.ipynb in Google Colab
    echo     2. Run all cells to train the model
    echo     3. Download drought_model.keras + metadata.json
    echo     4. Place them in:  %CD%\drought_model_artifacts\
    echo.
    echo   Launching app anyway so you can see the debug info...
    echo.
    timeout /t 5 /nobreak >nul
)

:: ============================================================
::  ALL CHECKS PASSED — LAUNCH
:: ============================================================
echo %CYAN%  ==========================================================%RESET%
echo %CYAN%   %BOLD%All checks passed. Launching Streamlit app...%RESET%
echo %CYAN%  ==========================================================%RESET%
echo.
echo %DIM%  The app will open at: http://localhost:8501%RESET%
echo %DIM%  Press Ctrl+C in this window to stop the server.%RESET%
echo.

:: Small pause so user can read the output
timeout /t 2 /nobreak >nul

:: Open browser automatically after a short delay (runs in background)
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8501"

:: Launch Streamlit (this keeps the window open)
python -m streamlit run app.py

:: ============================================================
::  ON EXIT
:: ============================================================
echo.
echo %DIM%  Server stopped. Press any key to close this window.%RESET%
pause >nul

endlocal
