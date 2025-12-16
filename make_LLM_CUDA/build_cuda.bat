@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "build_cuda.ps1"
pause
