@echo off
title Joy Manalu AI - Cloudflare HTTPS Tunnel
echo ========================================================
echo   JOY MANALU AI STUDIO - HTTPS TUNNEL RESMI UNTUK WEB & HP
echo ========================================================
echo.
echo Menghubungkan server ke jaringan global Cloudflare...
echo.

if exist "cloudflared.exe" (
    .\cloudflared.exe tunnel --url http://localhost:5000
) else (
    npx -y localtunnel --port 5000
)
pause
