#!/bin/zsh

echo ""
echo "  Setting up F1 Leaderboard..."
echo ""

# Create venv only if it doesn't already exist
# -d checks if the directory exists
if [ ! -d "venv" ]; then
    echo "  Creating venv..."
    python3 -m venv venv
else
    echo "  Venv already exists, skipping creation..."
fi

# Activate only if not already active
# -z checks if the variable is empty
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
else
    echo "  Venv already active, skipping activation..."
fi

python3 -m pip install --quiet --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "  Done! Run the script with:"
echo "    python3 f1_leaderboard.py --demo"
echo ""

# This keeps the shell active in the virtual environment
# equivalent to 'cmd /k'
exec $SHELL