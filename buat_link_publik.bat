@echo off
chcp 65001 >nul
title Joy Manalu AI Studio - Launcher

echo.
echo ================================================================
echo   JOY MANALU AI STUDIO - SATU KLIK LAUNCHER
echo ================================================================
echo.

:: Matikan proses lama jika ada
echo [1/4] Membersihkan proses lama...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: Mulai Flask Server di background
echo [2/4] Menjalankan Flask Server...
start "Joy-Flask-Server" /min cmd /c "cd /d d:\potoooo && .venv\Scripts\python.exe app.py > server.log 2>&1"
timeout /t 5 /nobreak >nul
echo       Server Flask aktif di http://localhost:5000

:: Mulai Cloudflare Tunnel dan tangkap URL-nya
echo [3/4] Menyalakan Cloudflare Tunnel (tunggu 15 detik)...
set LOG_FILE=d:\potoooo\tunnel.log
del "%LOG_FILE%" >nul 2>&1
start "Joy-Tunnel" /min cmd /c "cd /d d:\potoooo && cloudflared.exe tunnel --url http://localhost:5000 > tunnel.log 2>&1"

:: Tunggu sampai URL muncul di log
set TUNNEL_URL=
set WAIT=0
:WAIT_LOOP
timeout /t 2 /nobreak >nul
set /a WAIT+=2
if %WAIT% geq 30 goto TIMEOUT

findstr /C:"trycloudflare.com" "%LOG_FILE%" >nul 2>&1
if errorlevel 1 goto WAIT_LOOP

:: Ekstrak URL dari log
for /f "tokens=*" %%i in ('findstr /C:"trycloudflare.com" "%LOG_FILE%"') do (
    set LINE=%%i
)

:: Ambil bagian URL dari baris log
for /f "tokens=8" %%a in ("%LINE%") do set TUNNEL_URL=%%a

goto SHOW_URL

:TIMEOUT
set TUNNEL_URL=Gagal mendapat URL - cek koneksi internet

:SHOW_URL
echo [4/4] Selesai!
echo.
echo ================================================================
echo.
echo   LINK PUBLIK ANDA (bisa dibuka di HP / Laptop / Siapa saja):
echo.
if "%TUNNEL_URL%"=="" (
    echo   Buka tunnel.log untuk cari URL trycloudflare.com
) else (
    echo   >> %TUNNEL_URL%
)
echo.
echo   Link Lokal (Laptop ini):
echo   >> http://localhost:5000
echo.
echo ================================================================
echo.
echo   PENTING: Jangan tutup jendela ini dan jendela Flask/Tunnel!
echo   Selama laptop menyala dan terhubung internet, link aktif.
echo.
echo   Tekan tombol apapun untuk membuka link di browser...
pause >nul

:: Buka browser otomatis
if not "%TUNNEL_URL%"=="" (
    start "" "%TUNNEL_URL%"
) else (
    start "" "http://localhost:5000"
)

echo.
echo   Browser sudah terbuka! Selamat menggunakan Joy Manalu AI Studio.
echo   Biarkan jendela ini tetap terbuka.
echo.
pause
