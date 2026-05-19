@echo off
echo ===================================================
echo   Starting Oasis Student Edge (Gemma 4 E4B)
echo ===================================================

echo [1/3] Checking Python dependencies...
python -m pip install -r requirements.txt

echo [2/3] Ensuring Gemma 4 E4B model is downloaded...
ollama pull batiai/gemma4-e4b:q4

echo [3/3] Starting Offline Tutor UI...
python -m streamlit run student_app2.py

echo.
echo If you see this, something went wrong. Read the error above.
cmd /k

