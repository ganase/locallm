@echo off
setlocal

rem この bat があるフォルダに移動（C:\TMP\Locallm 想定だが、どこでもOK）
cd /d "%~dp0"

echo [INFO] Locallm (keyword match) starting ...

rem Miniforge の activate.bat を決め打ち
set "MINIFORGE_ACT=%USERPROFILE%\miniforge3\Scripts\activate.bat"

if not exist "%MINIFORGE_ACT%" (
    echo [ERROR] Miniforge not found at:
    echo         "%MINIFORGE_ACT%"
    echo.
    echo Please install Miniforge (miniforge3) and retry.
    echo.
    pause
    goto :EOF
)

echo [INFO] Activating Miniforge base ...
call "%MINIFORGE_ACT%" base

if errorlevel 1 (
    echo [ERROR] Failed to activate Miniforge base.
    echo        Try running the following manually:
    echo          "%MINIFORGE_ACT%" base
    echo.
    pause
    goto :EOF
)

echo [INFO] Launching Streamlit app (keyword match) ...
python -m streamlit run app\app_kwm.py

echo.
echo [INFO] Streamlit exited. Press any key to close.
pause

endlocal
