@echo off
echo ===================================================
echo   TruthLens Frontend Development Server
echo ===================================================
echo.
echo Starting local web server for the frontend...
echo You can access the website at: http://localhost:5500/Truthlens.html
echo.
echo Press Ctrl+C to stop the server.
echo.

python -m http.server 5500
pause
