@echo off
setlocal

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH. Please install Python 3.8+ or activate your conda environment.
    exit /b 1
)

echo Building CHEMKIN Rate Viewer v1.3 as a directory package...
python "%~dp0build_package.py"
if errorlevel 1 (
    echo ERROR: package build failed.
    exit /b 1
)

echo.
echo Package directory: %~dp0dist\CHEMKIN_RateCalculator
echo Executable: %~dp0dist\CHEMKIN_RateCalculator\CHEMKIN_RateCalculator.exe
echo User thermo data: %%LOCALAPPDATA%%\CHEMKIN_RateViewer\therm.dat
endlocal
