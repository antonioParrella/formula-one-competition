#!/bin/bash
# Works on macOS (bash/zsh) and Linux

echo ""
echo "========================================"
echo "  F1 Tipping Competition -- Setup"
echo "========================================"
echo ""

# Create venv only if it doesn't already exist
if [ ! -d "venv" ]; then
    echo "  Creating venv..."
    # Try python3 first, fall back to python
    if command -v python3 &> /dev/null; then
        python3 -m venv venv
    else
        python -m venv venv
    fi
else
    echo "  Venv already exists, skipping creation..."
fi

# Activate
if [ -z "$VIRTUAL_ENV" ]; then
    echo "  Activating venv..."
    source venv/bin/activate
else
    echo "  Venv already active, skipping activation..."
fi

python -m pip install --quiet --upgrade pip
python -m pip install -r config/requirements.txt

echo ""
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Copy .env.example to .env and add your SurveyMars credentials"
echo "  2. Run the full pipeline:"
echo "       python pipeline.py"
echo "  3. Or see README.md for more options"
echo ""
