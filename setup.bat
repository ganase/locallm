@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ============================================
echo  Locallm Setup Wizard
echo ============================================
echo.
echo [DEBUG] setup.bat started
echo.

rem この bat が置いてあるフォルダを取得（GitHub で locallm-main でもOK）
set "SCRIPT_DIR=%~dp0"
echo [DEBUG] SCRIPT_DIR = "%SCRIPT_DIR%"
echo.
echo This wizard will:
echo   1. Check Miniforge (base).
echo   2. Copy this folder to C:\TMP\Locallm.
echo   3. Install Python packages (pip install -r requirements.txt).
echo   4. Create desktop shortcuts:
echo        - Locallm.lnk      -> run_app_kwm.bat
echo        - Locallm_emb.lnk  -> run_app_emb.bat
echo.
pause

rem ========================
rem STEP 1: Check Miniforge
rem ========================
echo [1/4] Checking Miniforge...

set "MINIFORGE_ACT=%USERPROFILE%\miniforge3\Scripts\activate.bat"
echo [DEBUG] MINIFORGE_ACT = "%MINIFORGE_ACT%"

if exist "%MINIFORGE_ACT%" (
    echo -> Miniforge found.
) else (
    echo -> Miniforge not found at:
    echo      "%MINIFORGE_ACT%"
    echo.
    echo Please install Miniforge and retry.
    echo   https://github.com/conda-forge/miniforge
    echo.
    pause
    goto :EOF
)
echo.
pause

rem ========================
rem STEP 2: Copy to C:\TMP\Locallm
rem ========================
echo [2/4] Copy files to C:\TMP\Locallm ...
set "TARGET_DIR=C:\TMP\Locallm"
echo [DEBUG] TARGET_DIR = "%TARGET_DIR%"
echo.

if not exist "C:\TMP" (
    echo -> Creating C:\TMP ...
    mkdir "C:\TMP"
)

if exist "%TARGET_DIR%" (
    echo -> Removing old %TARGET_DIR% ...
    rmdir /s /q "%TARGET_DIR%"
)

echo -> Copying from "%SCRIPT_DIR%" to "%TARGET_DIR%" ...
mkdir "%TARGET_DIR%"
xcopy "%SCRIPT_DIR%*" "%TARGET_DIR%\" /E /I /Y
echo -> Copy done.
echo.
pause

rem ========================
rem STEP 3: Install requirements with Miniforge base
rem ========================
echo [3/4] Installing Python packages with Miniforge base ...
echo.

pushd "%TARGET_DIR%"
echo [DEBUG] CURRENT DIR = "%CD%"

echo -> Activating Miniforge base ...
call "%MINIFORGE_ACT%" base
if errorlevel 1 (
    echo -> Failed to activate Miniforge base.
    echo    Try running this manually:
    echo      "%MINIFORGE_ACT%" base
    echo.
    pause
    popd
    goto :EOF
)

echo -> Running: pip install -r requirements.txt
pip install -r requirements.txt
if errorlevel 1 (
    echo -> pip install failed.
    echo    Check error messages above.
    echo.
    pause
    popd
    goto :EOF
)

echo -> Requirements installed successfully.
echo.
pause
popd

rem ========================
rem STEP 4: Create Desktop Shortcut (KWM版)
rem ========================
echo [4/4] Creating desktop shortcuts ...
echo.

set "SHORTCUT_KWM=%USERPROFILE%\Desktop\Locallm.lnk"
set "SHORTCUT_EMB=%USERPROFILE%\Desktop\Locallm_emb.lnk"
set "TARGET_BAT_KWM=%TARGET_DIR%\run_app_kwm.bat"
set "TARGET_BAT_EMB=%TARGET_DIR%\run_app_emb.bat"

echo [DEBUG] SHORTCUT_KWM  = "%SHORTCUT_KWM%"
echo [DEBUG] SHORTCUT_EMB  = "%SHORTCUT_EMB%"
echo [DEBUG] TARGET_BAT_KWM = "%TARGET_BAT_KWM%"
echo [DEBUG] TARGET_BAT_EMB = "%TARGET_BAT_EMB%"
echo.

rem --- KWM ショートカット ---
if exist "%TARGET_BAT_KWM%" (
    echo -> Creating Locallm.lnk ...
    powershell -NoProfile -Command ^
     "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_KWM%');" ^
     "$s.TargetPath='%TARGET_BAT_KWM%';" ^
     "$s.WorkingDirectory='%TARGET_DIR%';" ^
     "$s.IconLocation='%SystemRoot%\System32\shell32.dll,43';" ^
     "$s.Save()"
) else (
    echo -> WARNING: "%TARGET_BAT_KWM%" not found. Skipping Locallm.lnk.
)

rem --- Embedding 版ショートカット ---
if exist "%TARGET_BAT_EMB%" (
    echo -> Creating Locallm_emb.lnk ...
    powershell -NoProfile -Command ^
     "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_EMB%');" ^
     "$s.TargetPath='%TARGET_BAT_EMB%';" ^
     "$s.WorkingDirectory='%TARGET_DIR%';" ^
     "$s.IconLocation='%SystemRoot%\System32\shell32.dll,43';" ^
     "$s.Save()"
) else (
    echo -> WARNING: "%TARGET_BAT_EMB%" not found. Skipping Locallm_emb.lnk.
)

echo.
echo ============================================
echo  Setup finished.
echo.
echo  - C:\TMP\Locallm にコピーしています。
echo  - Desktop に以下のショートカットを作成：
echo      Locallm.lnk      (キーワード検索版)
echo      Locallm_emb.lnk  (埋め込み版, run_app_emb.bat があれば)
echo ============================================
echo.
pause

endlocal
