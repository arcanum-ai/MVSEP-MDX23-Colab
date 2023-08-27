:: Check for virtual environment
if exist .venv\ (
    .venv\Scripts\python.exe gui.py
) else (
    :: Check for python installation
    python --version>NUL
    if errorlevel 1 goto noPython

    :: Setup virtualenv
    python -m virtualenv .venv
    .venv\Scripts\pip.exe install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
    .venv\Scripts\python.exe gui.py
    exit 0
)
:noPython
echo.
echo Error: Python not installed
pause>nul
exit 1
