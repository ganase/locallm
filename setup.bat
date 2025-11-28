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
echo Press Enter to start STEP 1.


REM ========================
REM  STEP 1: Check Miniforge
REM ========================
echo.
echo ============================================
echo [1/4] Check Miniforge (mamba/conda)
echo ============================================
echo.

set "MAMBA_ROOT_PREFIX=%USERPROFILE%\miniforge3"
set "ACTIVATE_BAT=%MAMBA_ROOT_PREFIX%\Scripts\activate.bat"

echo Expected Miniforge activate.bat:
echo   "%ACTIVATE_BAT%"
echo.

if exist "%ACTIVATE_BAT%" (
    echo -> Found Miniforge activate.bat.
) else (
    echo -> Miniforge was NOT found.
    echo.
    echo Please install Miniforge3 to:
    echo   %MAMBA_ROOT_PREFIX%
    echo and then run this wizard again.
    echo.
    echo Press any key to exit.
    goto END
)

echo.
echo STEP 1 OK. Press Enter to continue to STEP 2.



REM ========================
REM  STEP 2: Copy to C:\TMP\Locallm
REM ========================
echo.
echo ============================================
echo [2/4] Copy files to C:\TMP\Locallm
echo ============================================
echo.

set "TARGET_ROOT=C:\TMP"
set "TARGET_DIR=%TARGET_ROOT%\Locallm"

echo Target root   : %TARGET_ROOT%
echo Target folder : %TARGET_DIR%
echo.

if not exist "%TARGET_ROOT%" (
    echo -> %TARGET_ROOT% does not exist. Creating...
    mkdir "%TARGET_ROOT%"
)

if exist "%TARGET_DIR%" (
    echo -> %TARGET_DIR% already exists.
    echo    If you want to overwrite this folder, type Y and press Enter.
    set /p OVERWRITE="Overwrite existing folder? [y/N]: "
    if /I "%OVERWRITE%"=="Y" (
        echo -> Removing existing folder...
        rmdir /S /Q "%TARGET_DIR%"
    ) else (
        echo -> Keeping existing folder. No copy will be done.
        goto AFTER_COPY
    )
)

echo.
echo -> Copying from "%SCRIPT_DIR%" to "%TARGET_DIR%" ...
xcopy "%SCRIPT_DIR%*" "%TARGET_DIR%\" /E /I /Y >NUL

:AFTER_COPY
echo.
echo STEP 2 OK. Press Enter to continue to STEP 3.



REM ========================
REM  STEP 3: Install requirements (Miniforge base)
REM ========================
echo.
echo ============================================
echo [3/4] Install Python packages (requirements)
echo ============================================
echo.

echo We will:
echo   1. Activate Miniforge base environment.
echo   2. Run install_requirements_conda.bat inside:
echo      %TARGET_DIR%
echo.

pushd "%TARGET_DIR%"
call "%ACTIVATE_BAT%"
if errorlevel 1 (
    echo.
    echo -> Failed to activate Miniforge.
    echo Please try manually:
    echo   "%ACTIVATE_BAT%"
    echo   cd /d "%TARGET_DIR%"
    echo   install_requirements_conda.bat
    echo.
    popd
    goto AFTER_REQ
)

if exist "install_requirements_conda.bat" (
    echo -> Running install_requirements_conda.bat ...
    call install_requirements_conda.bat
) else (
    echo -> install_requirements_conda.bat not found.
    echo Please install packages manually, for example:
    echo   pip install -r requirements.txt
)

popd

:AFTER_REQ
echo.
echo STEP 3 finished (success or manual action required).
echo Press Enter to continue to STEP 4.



REM ========================
REM  STEP 4: Create desktop shortcut
REM ========================
echo.
echo ============================================
echo [4/4] Create desktop shortcut
echo ============================================
echo.

set "SHORTCUT_NAME=Locallm.lnk"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%"
set "TARGET_BAT=%TARGET_DIR%\run_app_kwm.bat"

echo Shortcut path :
echo   %SHORTCUT_PATH%
echo Target (bat)  :
echo   %TARGET_BAT%
echo.

if not exist "%TARGET_BAT%" (
    echo -> %TARGET_BAT% does NOT exist.
    echo Please check that the files were copied correctly.
) else (
    echo -> Creating shortcut via PowerShell...
    powershell -NoProfile -Command ^
     "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');" ^
     "$s.TargetPath='%TARGET_BAT%';" ^
     "$s.WorkingDirectory='%TARGET_DIR%';" ^
     "$s.IconLocation='%SystemRoot%\System32\shell32.dll,43';" ^
     "$s.Save()"

    if exist "%SHORTCUT_PATH%" (
        echo -> Shortcut created: %SHORTCUT_PATH%
    ) else (
        echo -> Failed to create shortcut. Please check PowerShell execution.
    )
)

echo.
echo ============================================
echo Setup wizard finished.
echo ============================================
echo.
echo To start the app next time:
echo   1. Double-click the desktop shortcut:
echo        %SHORTCUT_NAME%
echo   2. Or run manually:
echo        cd /d "%TARGET_DIR%"
echo        run_app_kwm.bat
echo.
goto END

echo.
echo ============================================
echo Setup wizard finished.
echo ============================================
echo.
echo To start the app next time:
echo   1. Double-click the desktop shortcut:
echo        %SHORTCUT_NAME%
echo   2. Or run manually:
echo        cd /d "%TARGET_DIR%"
echo        run_app_kwm.bat
echo.
goto END


:END
echo.
echo [DEBUG] setup_locallm_wizard.bat finished.
echo Press any key to close this window.

endlocal
