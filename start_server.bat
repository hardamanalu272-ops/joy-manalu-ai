@echo off
title Joy Manalu AI Studio Server
echo ========================================================
echo   JOY MANALU AI STUDIO - FACE RECOGNITION SERVER
echo ========================================================
echo.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment tidak ditemukan. Membuat .venv...
    uv venv .venv --python 3.12
    uv pip install -r requirements.txt
)

if not exist "embeddings_cache.pkl" (
    echo [INFO] Membangun database embeddings wajah...
    .venv\Scripts\python.exe build_embeddings.py
)

echo.
echo [START] Menjalankan server pada port 5000...
echo Buka di browser Laptop: http://localhost:5000
echo.
.venv\Scripts\python.exe app.py
pause
