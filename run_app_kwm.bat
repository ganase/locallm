@echo off
setlocal

rem === この bat があるフォルダに移動 ===
cd /d "%~dp0"

echo [INFO] Starting LOCALLM with Miniforge Python...

rem === Miniforge の python.exe を指定 ===
set "PYTHON_EXE=%USERPROFILE%\miniforge3\python.exe"
echo [INFO] Using Python: "%PYTHON_EXE%"
echo.

rem === python が本当にあるか確認 ===
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found at "%PYTHON_EXE%".
    echo Miniforge のインストール場所を確認してください。
    pause
    exit /b 1
)

rem === バージョン表示（確認用） ===
"%PYTHON_EXE%" -c "import sys; print('Python executable:', sys.executable); print('Version:', sys.version)"
echo.

rem === Streamlit アプリ起動 ===
echo [INFO] Starting Streamlit app (app\app_kwm.py)...
"%PYTHON_EXE%" -m streamlit run app\app_kwm.py
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start Streamlit app.
    pause
