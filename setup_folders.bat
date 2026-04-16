@echo off
echo =======================================
echo  NEON-FORECAST AI - Setup Script
echo =======================================

mkdir data\raw
mkdir data\processed
mkdir models\saved
mkdir backend
mkdir frontend
mkdir sample_data

echo.
echo [OK] Folder structure created!
echo.
echo Next step: Run this command to install dependencies:
echo.
echo     pip install -r requirements.txt
echo.
pause