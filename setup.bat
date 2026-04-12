@echo off
echo.
echo ========================================
echo   F1 Tipping Competition -- Setup
echo ========================================
echo.

:: Create venv only if it doesn't already exist
if not exist venv (
    echo   Creating venv...
    python -m venv venv
) else (
    echo   Venv already exists, skipping creation...
)

:: Activate only if not already active
if not defined VIRTUAL_ENV (
    call venv\Scripts\activate.bat
) else (
    echo   Venv already active, skipping activation...
)

python -m pip install --quiet --upgrade pip
python -m pip install -r config/requirements.txt

echo.
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Copy .env.example to .env and add your SurveyMars credentials
echo   2. Run the full pipeline:
echo        python pipeline.py
echo   3. Or see README.md for more options
echo.
cmd /k
