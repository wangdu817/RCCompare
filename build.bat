@echo off
setlocal

set "PYTHON=C:\Users\17915\anaconda3\envs\gui-env\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: Python interpreter not found: %PYTHON%
    exit /b 1
)

echo Building CHEMKIN Rate Viewer v1.3 as a directory package...
"%PYTHON%" "%~dp0build_package.py"
if errorlevel 1 (
    echo ERROR: package build failed.
    exit /b 1
)

echo.
echo Package directory: %~dp0dist\CHEMKIN_RateCalculator
echo Executable: %~dp0dist\CHEMKIN_RateCalculator\CHEMKIN_RateCalculator.exe
echo User thermo data: %%LOCALAPPDATA%%\CHEMKIN_RateViewer\therm.dat
endlocal
