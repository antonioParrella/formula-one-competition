@echo off
echo.
echo   Setting up F1 Leaderboard...
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
python -m pip install -r requirements.txt

echo.
echo   Done! Run the script with:
echo     python f1_leaderboard.py --demo
echo.
cmd /k