@echo off
setlocal

REM この bat があるフォルダに移動
cd /d "%~dp0"

echo [INFO] Starting Locallm (Embedding版) with Miniforge Python...

REM Miniforge の base 環境をアクティベート
call "%USERPROFILE%\miniforge3\Scripts\activate.bat"

REM Streamlit で埋め込み版アプリを起動
python -m streamlit run app\app_emb.py

endlocal
