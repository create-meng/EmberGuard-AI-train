@echo off
setlocal
title EmberGuard Demo Launcher

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

set ENV_NAME=yolo
set APP_ENTRY=digital-twin-web\backend\app.py
set CONDA_BAT=
set EMBERGUARD_AUTO_RELOAD=1
set EMBERGUARD_DEV_LIVE_RELOAD=1

echo.
echo ==========================================
echo   EmberGuard Demo Launcher
echo ==========================================
echo ROOT     : %ROOT_DIR%
echo ENV      : %ENV_NAME%
echo APP      : %APP_ENTRY%
echo RELOAD   : backend=%EMBERGUARD_AUTO_RELOAD% frontend=%EMBERGUARD_DEV_LIVE_RELOAD%
echo.

if exist "E:\anaconda3\condabin\conda.bat" set CONDA_BAT=E:\anaconda3\condabin\conda.bat
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat

if not defined CONDA_BAT (
    echo ERROR: conda.bat not found.
    pause
    exit /b 1
)

if not exist "%APP_ENTRY%" (
    echo ERROR: app entry not found: %APP_ENTRY%
    pause
    exit /b 1
)

echo [1/3] Check python in conda env...
call "%CONDA_BAT%" run -n %ENV_NAME% python --version
if errorlevel 1 (
    echo ERROR: python is not available in env %ENV_NAME%
    pause
    exit /b 1
)

echo [2/3] Check app entry...
if not exist "%APP_ENTRY%" (
    echo ERROR: app entry not found: %APP_ENTRY%
    pause
    exit /b 1
)

echo [3/3] Start Flask app...
echo.
echo ==========================================
echo Server is starting...
echo Open this URL in your browser:
echo http://127.0.0.1:5000
echo ==========================================
echo.
call "%CONDA_BAT%" run --live-stream -n %ENV_NAME% python "%APP_ENTRY%"

echo.
echo App exited.
pause
endlocal
