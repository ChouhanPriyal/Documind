@echo off

echo Starting DocuMind...

start "DocuMind Backend" cmd /k "cd /d %~dp0backend && venv\Scripts\activate && python app.py"

timeout /t 3 /nobreak >nul

start "DocuMind Frontend" cmd /k "cd /d %~dp0frontend && python -m http.server 8000"

timeout /t 2 /nobreak >nul

start http://localhost:8000

echo.
echo DocuMind is running!
echo Frontend: http://localhost:8000
echo Backend:  http://127.0.0.1:5000


