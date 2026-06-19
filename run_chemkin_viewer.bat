@echo off
REM CHEMKIN Rate Viewer 启动脚本
REM 版本: v1.3
REM 日期: 2026-05-26

echo ========================================
echo CHEMKIN Rate Viewer v1.3
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [信息] Python已安装
echo.

REM 激活conda环境（如果需要）
echo [信息] 正在启动程序...
echo.

REM 运行主程序
python chemkin_pyqt_gui_simple.py

REM 如果程序异常退出，显示错误信息
if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出
    echo.
    echo 可能的原因：
    echo 1. 缺少必需的Python包（PyQt6, numpy, matplotlib）
    echo 2. 模块导入错误
    echo.
    echo 请运行以下命令安装依赖：
    echo   pip install PyQt6 numpy matplotlib pandas openpyxl
    echo.
    pause
)
