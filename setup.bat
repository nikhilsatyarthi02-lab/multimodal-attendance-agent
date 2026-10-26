@echo off
REM Activity 1.2 - create virtual environment and install libraries (Windows)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\check_env.py
echo.
echo Setup complete. Run:  streamlit run app.py
