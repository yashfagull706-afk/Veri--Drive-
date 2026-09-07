@echo off
title Veri-Drive Gate Security
cd /d "%~dp0"

rem Already running? Just open the browser and exit.
curl -s -o nul -m 2 http://127.0.0.1:5000/api/status >nul 2>&1
if not errorlevel 1 (
    echo Veri-Drive is already running - opening browser...
    start "" http://127.0.0.1:5000
    exit /b 0
)

echo ============================================================
echo   Starting Veri-Drive Gate Security System...
echo   The browser will open automatically when the server is ready.
echo ============================================================

rem Open the dashboard as soon as the server answers (max ~60s)
start "" cmd /c "for /L %%i in (1,1,60) do @(curl -s -o nul -m 2 http://127.0.0.1:5000/api/status 2>nul && (start http://127.0.0.1:5000 & exit /b 0)) & timeout /t 1 /nobreak >nul"

rem Run the server (this window stays open - closing it stops the system)
py veri_drive.py

pause
