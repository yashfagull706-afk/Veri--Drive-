@echo off
title Veri-Drive Gate Security
cd /d "%~dp0"

echo ============================================================
echo   Starting Veri-Drive Gate Security System...
echo   The browser will open automatically in a few seconds.
echo ============================================================

rem Open the dashboard in the default browser once the server is ready
start "" cmd /c "timeout /t 12 /nobreak >nul && start http://127.0.0.1:5000"

rem Run the server (this window stays open - closing it stops the system)
py veri_drive.py

pause

