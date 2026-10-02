@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run START.bat once first so dependencies are installed.
  pause
  exit /b 1
)
.venv\Scripts\python.exe verify_install.py
if errorlevel 1 (
  echo.
  echo Verification failed. Read the first error above.
) else (
  echo.
  echo Verification completed successfully.
)
pause
