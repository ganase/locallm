@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ========================
REM  STEP 0: Debug info
REM ========================
echo [DEBUG] setup_locallm_wizard.bat started

set "SCRIPT_DIR=%~dp0"
echo SCRIPT_DIR = "%SCRIPT_DIR%"
echo.
echo This wizard will:
echo   1. Check Miniforge.
echo   2. Copy locallm to C:\TMP\Locallm.
echo   3. Install Python packages (using Miniforge base).
echo   4. Create a desktop shortcut to run_app_kwm.bat.
echo.
pause

REM ========================
REM  STEP 1: Check Miniforge
REM ========================
echo ============================================
echo  Locallm セットアップウィザード
echo ============================================
echo.
echo [1/4] Miniforge の確認
echo    想定パス:
echo      "%USERPROFILE%\miniforge3\Scripts\activate.bat"
echo.

set "MINIFORGE_ACT=%USERPROFILE%\miniforge3\Scripts\activate.bat"

if exist "%MINIFORGE_ACT%" (
    echo -> Miniforge (miniforge3) が見つかりました。
) else (
    echo -> Miniforge が見つかりません。
    echo    以下を参考に Miniforge をインストールしてください:
    echo      https://github.com/conda-forge/miniforge
    echo.
    echo セットアップを終了します。
    pause
    goto :EOF
)
echo.
pause

REM ========================
REM  STEP 2: Copy to C:\TMP\Locallm
REM ========================
echo [2/4] C:\TMP\Locallm へのコピー
echo.

set "TARGET_DIR=C:\TMP\Locallm"

if not exist "C:\TMP" (
    echo -> C:\TMP フォルダを作成します...
    mkdir "C:\TMP"
)

if exist "%TARGET_DIR%" (
    echo -> 既存の %TARGET_DIR% を削除します...
    rmdir /s /q "%TARGET_DIR%"
)

echo -> %TARGET_DIR% にコピー中...
mkdir "%TARGET_DIR%"
xcopy "%SCRIPT_DIR%*" "%TARGET_DIR%\" /E /I /Y >nul

echo -> コピー完了: %TARGET_DIR%
echo.
pause

REM ========================
REM  STEP 3: Install requirements via Miniforge base
REM ========================
echo [3/4] Python パッケージのインストール
echo    Miniforge base 環境で requirements.txt をインストールします。
echo.

pushd "%TARGET_DIR%"

REM Miniforge base をアクティベートして pip インストール
call "%MINIFORGE_ACT%" base
if errorlevel 1 (
    echo -> Miniforge base のアクティベートに失敗しました。
    echo    手動で以下を実行して確認してください:
    echo      "%MINIFORGE_ACT%" base
    echo.
    pause
    popd
    goto :EOF
)

echo -> pip install -r requirements.txt を実行します...
pip install -r requirements.txt
if errorlevel 1 (
    echo -> pip install -r requirements.txt に失敗しました。
    echo    エラーメッセージを確認してください。
    echo.
    pause
    popd
    goto :EOF
)

echo -> パッケージインストール完了。
echo.
pause

popd

REM ========================
REM  STEP 4: Create Desktop Shortcut (run_app_kwm.bat)
REM ========================
echo [4/4] デスクトップショートカットの作成 (キーワード検索版)
echo.

set "SHORTCUT_NAME=Locallm.lnk"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%"
set "TARGET_BAT=%TARGET_DIR%\run_app_kwm.bat"

echo Shortcut path :
echo   %SHORTCUT_PATH%
echo Target (bat)   :
echo   %TARGET_BAT%
echo.

if not exist "%TARGET_BAT%" (
    echo -> %TARGET_BAT% が存在しません。
    echo    セットアップは済みましたが、ショートカットは作成できませんでした。
) else (
    echo -> PowerShell を使ってショートカットを作成します...

    powershell -NoProfile -Command ^
     "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');" ^
     "$s.TargetPath='%TARGET_BAT%';" ^
     "$s.WorkingDirectory='%TARGET_DIR%';" ^
     "$s.IconLocation='%SystemRoot%\System32\shell32.dll,43';" ^
     "$s.Save()"

    if exist "%SHORTCUT_PATH%" (
        echo -> ショートカットを作成しました: %SHORTCUT_PATH%
    ) else (
        echo -> ショートカット作成に失敗しました。PowerShell 実行ポリシー等を確認してください。
    )
)

REM =========================================================
REM 4b. Create Desktop Shortcut for Embedding Version (run_app_emb.bat)
REM =========================================================
echo.
echo [4b/4] 埋め込み検索版ショートカットの作成 (Embedding 版)
set "SHORTCUT_NAME_EMB=Locallm_emb.lnk"
set "SHORTCUT_PATH_EMB=%USERPROFILE%\Desktop\%SHORTCUT_NAME_EMB%"
set "TARGET_BAT_EMB=%TARGET_DIR%\run_app_emb.bat"

echo Shortcut path (embedding) :
echo   %SHORTCUT_PATH_EMB%
echo Target (bat, embedding)  :
echo   %TARGET_BAT_EMB%
echo.

if not exist "%TARGET_BAT_EMB%" (
    echo -> %TARGET_BAT_EMB% が存在しません。
    echo    Embedding 版を使う場合は、run_app_emb.bat が存在することを確認してください。
) else (
    echo -> PowerShell を使って埋め込み版ショートカットを作成します...

    powershell -NoProfile -Command ^
     "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_PATH_EMB%');" ^
     "$s.TargetPath='%TARGET_BAT_EMB%';" ^
     "$s.WorkingDirectory='%TARGET_DIR%';" ^
     "$s.IconLocation='%SystemRoot%\System32\shell32.dll,43';" ^
     "$s.Save()"

    if exist "%SHORTCUT_PATH_EMB%" (
        echo -> 埋め込み版ショートカットを作成しました: %SHORTCUT_PATH_EMB%
    ) else (
        echo -> 埋め込み版ショートカット作成に失敗しました。PowerShell 実行ポリシー等を確認してください。
    )
)

echo.
echo セットアップ完了です。
echo   - キーワード検索版: デスクトップの「Locallm」ショートカット
echo   - 埋め込み検索版:   デスクトップの「Locallm_emb」ショートカット
echo.
pause

endlocal
