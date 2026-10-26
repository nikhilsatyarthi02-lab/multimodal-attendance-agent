#!/usr/bin/env bash
# Activity 1.2 - create virtual environment and install libraries (Linux/macOS)
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
echo "Setup complete. Run:  streamlit run app.py"
