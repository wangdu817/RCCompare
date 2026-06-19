# CHEMKIN Rate Viewer 打包说明

## v1.3 推荐方式（Windows）

`v1.3` 使用目录模式交付，不生成单文件程序。直接运行 `build.bat`，或使用指定环境执行：

```bat
C:\Users\17915\anaconda3\envs\gui-env\python.exe build_package.py
```

输出目录为 `dist\CHEMKIN_RateCalculator\`，启动文件为
`dist\CHEMKIN_RateCalculator\CHEMKIN_RateCalculator.exe`。

包中的 `therm.dat` 仅用于首次初始化。用户新增或修改的热力学参数永久保存在：

```text
%LOCALAPPDATA%\CHEMKIN_RateViewer\therm.dat
```

程序运行期间保存该文件后，下一次需要热力学数据的计算会自动重新读取。
下文中的旧单文件命令仅供历史参考，不作为 `v1.3` 发布流程。

## 概述

本文档说明如何将 CHEMKIN Rate Viewer 打包为独立的可执行文件，以便在没有 Python 环境的计算机上运行。

---

## 方法1: 使用自动打包脚本（推荐）

### Windows 用户

**最简单的方法**：
1. 双击运行 `build.bat` 文件
2. 等待打包完成（约2-5分钟）
3. 在 `dist/` 目录中找到 `CHEMKIN_Rate_Viewer.exe`

**或者使用 Python 脚本**：
```bash
python build_exe.py
```

### Linux/Mac 用户

```bash
# 安装 PyInstaller
pip install pyinstaller

# 运行打包脚本
python build_exe.py
```

---

## 方法2: 手动打包

### 步骤1: 安装 PyInstaller

```bash
pip install pyinstaller
```

### 步骤2: 运行打包命令

**Windows (单文件模式)**:
```bash
pyinstaller --name="CHEMKIN_Rate_Viewer" ^
    --onefile ^
    --windowed ^
    --add-data="modern_style.qss;." ^
    --add-data="therm.dat;." ^
    --hidden-import=PyQt6.QtCore ^
    --hidden-import=PyQt6.QtGui ^
    --hidden-import=PyQt6.QtWidgets ^
    --hidden-import=matplotlib.backends.backend_qtagg ^
    chemkin_pyqt_gui_simple.py
```

**Linux/Mac (单文件模式)**:
```bash
pyinstaller --name="CHEMKIN_Rate_Viewer" \
    --onefile \
    --windowed \
    --add-data="modern_style.qss:." \
    --add-data="therm.dat:." \
    --hidden-import=PyQt6.QtCore \
    --hidden-import=PyQt6.QtGui \
    --hidden-import=PyQt6.QtWidgets \
    --hidden-import=matplotlib.backends.backend_qtagg \
    chemkin_pyqt_gui_simple.py
```

**Windows (目录模式 - 启动更快)**:
```bash
pyinstaller --name="CHEMKIN_Rate_Viewer" ^
    --windowed ^
    --add-data="modern_style.qss;." ^
    --add-data="therm.dat;." ^
    --hidden-import=PyQt6.QtCore ^
    --hidden-import=PyQt6.QtGui ^
    --hidden-import=PyQt6.QtWidgets ^
    --hidden-import=matplotlib.backends.backend_qtagg ^
    chemkin_pyqt_gui_simple.py
```

### 步骤3: 查找可执行文件

打包完成后，可执行文件位于：
- **单文件模式**: `dist/CHEMKIN_Rate_Viewer.exe` (Windows) 或 `dist/CHEMKIN_Rate_Viewer` (Linux/Mac)
- **目录模式**: `dist/CHEMKIN_Rate_Viewer/` 目录中

---

## 打包选项说明

| 选项 | 说明 |
|------|------|
| `--name` | 可执行文件的名称 |
| `--onefile` | 打包为单个可执行文件（启动较慢，但便于分发） |
| `--windowed` | 不显示控制台窗口（GUI程序推荐） |
| `--add-data` | 添加数据文件（样式表、热力学数据等） |
| `--hidden-import` | 添加隐式导入的模块 |
| `--icon` | 指定程序图标（可选） |
| `--clean` | 清理临时文件 |

---

## 打包模式对比

### 单文件模式 (--onefile)

**优点**:
- ✅ 只有一个 exe 文件，便于分发
- ✅ 不需要额外的 DLL 文件

**缺点**:
- ❌ 首次启动较慢（需要解压临时文件）
- ❌ 文件较大（约 100-200 MB）

**适用场景**: 需要分发给其他用户

### 目录模式 (默认)

**优点**:
- ✅ 启动速度快
- ✅ 便于调试

**缺点**:
- ❌ 包含多个文件和文件夹
- ❌ 分发时需要打包整个目录

**适用场景**: 个人使用或内部使用

---

## 依赖项

打包前请确保已安装以下依赖：

```bash
pip install PyQt6 numpy matplotlib pyinstaller
```

可选依赖（用于 Excel 导出）：
```bash
pip install pandas openpyxl
```

---

## 常见问题

### Q1: 打包后程序无法启动？

**解决方案**:
1. 检查是否有杀毒软件拦截
2. 尝试使用目录模式打包（去掉 `--onefile`）
3. 检查控制台输出（去掉 `--windowed`）

### Q2: 打包后文件太大？

**解决方案**:
1. 使用虚拟环境，只安装必要的包
2. 使用 UPX 压缩（PyInstaller 会自动使用）
3. 排除不必要的模块：
   ```bash
   --exclude-module tkinter --exclude-module PIL
   ```

### Q3: 缺少某些模块？

**解决方案**:
添加隐式导入：
```bash
--hidden-import=模块名
```

### Q4: 样式表或数据文件丢失？

**解决方案**:
确保使用 `--add-data` 添加了所有必要的文件：
```bash
--add-data="modern_style.qss;." --add-data="therm.dat;."
```

### Q5: 打包后程序运行缓慢？

**解决方案**:
1. 使用目录模式而不是单文件模式
2. 将可执行文件添加到杀毒软件白名单
3. 关闭实时保护（仅在测试时）

---

## 分发说明

### 分发单文件版本

1. 将 `dist/CHEMKIN_Rate_Viewer.exe` 复制到目标计算机
2. 确保 `therm.dat` 文件与 exe 在同一目录
3. 双击运行

### 分发目录版本

1. 将整个 `dist/CHEMKIN_Rate_Viewer/` 目录复制到目标计算机
2. 确保 `therm.dat` 文件在目录中
3. 运行目录中的 `CHEMKIN_Rate_Viewer.exe`

### 创建安装包（可选）

可以使用以下工具创建安装程序：
- **Inno Setup** (Windows) - 免费
- **NSIS** (Windows) - 免费
- **InstallForge** (Windows) - 免费

---

## 高级选项

### 添加程序图标

1. 准备一个 `.ico` 文件（Windows）或 `.icns` 文件（Mac）
2. 在打包命令中添加：
   ```bash
   --icon=icon.ico
   ```

### 优化文件大小

```bash
pyinstaller --name="CHEMKIN_Rate_Viewer" \
    --onefile \
    --windowed \
    --strip \
    --upx-dir=/path/to/upx \
    chemkin_pyqt_gui_simple.py
```

### 添加版本信息（Windows）

创建 `version.txt` 文件，然后：
```bash
--version-file=version.txt
```

---

## 测试打包结果

### 基本测试

1. ✅ 程序能否正常启动
2. ✅ 界面是否正常显示
3. ✅ 样式是否正确应用
4. ✅ 能否解析 CHEMKIN 机理
5. ✅ 能否计算和绘图
6. ✅ 能否导出数据

### 兼容性测试

在不同的计算机上测试：
- ✅ 没有 Python 环境的计算机
- ✅ 不同版本的 Windows (7/10/11)
- ✅ 不同的屏幕分辨率

---

## 故障排除

### 查看详细错误信息

去掉 `--windowed` 选项重新打包：
```bash
pyinstaller --name="CHEMKIN_Rate_Viewer" --onefile chemkin_pyqt_gui_simple.py
```

这样可以看到控制台输出的错误信息。

### 清理并重新打包

```bash
# 删除旧文件
rmdir /s /q build dist
del *.spec

# 重新打包
python build_exe.py
```

---

## 更新日志

### v1.2
- ✨ 添加现代化深色主题
- ✨ 改进界面布局和样式
- 📦 优化打包配置

---

## 联系支持

如有问题，请检查：
1. Python 版本 >= 3.8
2. 所有依赖已正确安装
3. 打包命令语法正确
4. 数据文件路径正确

---

**祝打包顺利！** 🎉
