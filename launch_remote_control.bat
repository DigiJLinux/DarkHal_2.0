@echo off
:: Launch LLM_Train Remote Control Application

echo Starting LLM_Train Remote Control...
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python from https://python.org/downloads/
    pause
    exit /b 1
)

:: Launch the remote control application
python remotecontrol.py

if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)