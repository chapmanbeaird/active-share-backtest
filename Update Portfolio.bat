@echo off
REM Double-click this file in File Explorer to build the current portfolio from
REM the FactSet export you placed in the data\incoming folder.
REM (Windows equivalent of "Update Portfolio.command" for macOS.)

REM Always run from the folder this script lives in (so file paths and the data
REM cache resolve correctly no matter where it is launched from).
cd /d "%~dp0"

echo ==================================================
echo  Active Share Portfolio - Quarterly Update
echo ==================================================
echo.

REM The project's virtual environment on Windows lives in venv\Scripts\.
REM Use it if it exists; otherwise create it (one time only).
set "VENV_PY=venv\Scripts\python.exe"
if exist "%VENV_PY%" goto run

echo First-time setup: creating the Python environment (one time only)...

REM Prefer the "py" launcher (bundled with python.org installs); fall back to
REM "python" on PATH.
set "PYTHON=py -3"
where py >nul 2>nul || set "PYTHON=python"

%PYTHON% -m venv venv
if errorlevel 1 goto envfail
"%VENV_PY%" -m pip install -q -r requirements.txt
if errorlevel 1 goto envfail

:run
"%VENV_PY%" update_portfolio.py
set "STATUS=%ERRORLEVEL%"

echo.
if "%STATUS%"=="0" (
    echo Finished successfully. Your portfolio Excel is in the results\portfolios folder.
) else if "%STATUS%"=="2" (
    echo Action needed: a few new stocks must be classified ^(see the message above^).
) else (
    echo Something went wrong ^(see the message above^).
)

echo.
echo You can close this window.
pause
exit /b %STATUS%

:envfail
echo.
echo ERROR: could not set up the Python environment.
echo Install Python 3 from https://www.python.org/downloads/ and, during
echo installation, tick "Add python.exe to PATH". Then run this file again.
echo.
pause
exit /b 1
