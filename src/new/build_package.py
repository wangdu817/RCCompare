"""
打包脚本 - 将程序打包成单个可执行文件
"""
import PyInstaller.__main__
import os
import shutil

# 清理之前的构建
if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('dist'):
    shutil.rmtree('dist')

print("="*70)
print("开始打包 CHEMKIN 反应速率计算器")
print("="*70)

# PyInstaller 参数
args = [
    'chemkin_pyqt_gui_simple.py',  # 主程序
    '--name=CHEMKIN_RateCalculator',  # 程序名称
    '--onedir',  # 打包成单个文件
    '--windowed',  # 不显示控制台窗口
    '--icon=logo.png',  # 图标文件
    '--add-data=logo.png;.',  # 包含logo文件
    '--add-data=therm.dat;.',  # 包含热力学数据文件
    '--clean',  # 清理临时文件
    '--noconfirm',  # 不询问确认
]

print("\n打包参数:")
for arg in args:
    print(f"  {arg}")

print("\n开始打包...")
PyInstaller.__main__.run(args)

print("\n" + "="*70)
print("打包完成！")
print("="*70)
print(f"\n可执行文件位置: dist\\CHEMKIN_RateCalculator.exe")
print(f"文件大小: {os.path.getsize('dist/CHEMKIN_RateCalculator.exe') / 1024 / 1024:.2f} MB")
print("\n注意事项:")
print("1. 首次运行可能需要几秒钟启动")
print("2. 程序会自动加载 therm.dat 文件")
print("3. 如需更新热力学数据，请编辑 therm.dat 文件")
print("\n" + "="*70)

