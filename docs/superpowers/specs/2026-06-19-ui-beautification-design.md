# CHEMKIN Rate Viewer 界面美化设计文档

## 概述

将 CHEMKIN Rate Viewer 的 GUI 从当前单调暗灰色界面升级为统一协调的深蓝科研风格，替换 matplotlib 主图为 pyqtgraph 实现原生 Qt 交互式图表，同时保留 matplotlib 用于 publication 质量导出。

## 设计决策

### 图表组件：混合方案
- **主图**：pyqtgraph `PlotWidget`（原生 QWidget，内置暗色主题，交互式缩放/拖拽/十字光标）
- **导出**：matplotlib 静态图（仅在"导出图片"时调用，生成 SVG/PDF 矢量图）
- **理由**：pyqtgraph 渲染速度比 matplotlib 快 10-100x，原生 Qt 集成无 FigureCanvas 包装层，内置 `pg.setConfigOption('background', color)` 一行暗色适配

### 配色方案：深蓝科研风
```
背景层:   #1a1d23 (主背景)  →  #222630 (卡片/面板)  →  #2e3440 (边框/分割线)
文字层:   #e8ecf4 (标题)    →  #b0b8c8 (正文)        →  #7a8599 (辅助/次要)
强调色:   #4a90e2 (主按钮)  →  #6ea8fe (hover)       →  #357abd (pressed)
曲线色:   matplotlib tab10 色板，首色 #4a90e2 与主题呼应
```

## 架构设计

### 文件改动范围

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `chemkin_pyqt_gui_fluent.py` | 重写 | 核心 GUI，pyqtgraph 替换 matplotlib 主图，深蓝配色 QSS |
| `requirements.txt` | 追加 | 添加 `pyqtgraph>=0.13.0` |

### 组件层次

```
FluentWindow (MainWindow)
├── NavigationInterface (侧边栏)
│   ├── Home (FluentIcon.HOME)
│   └── Settings (FluentIcon.SETTING)
├── Home Page (QWidget)
│   ├── Left Panel (SmoothScrollArea, 420px)
│   │   ├── TitleLabel + CaptionLabel (标题)
│   │   ├── SimpleCardWidget (CHEMKIN Input + PlainTextEdit)
│   │   ├── GroupHeaderCardWidget (Controls)
│   │   │   ├── LineEdit × 7 (温度/压力/分组参数)
│   │   │   └── CheckBox × 2
│   │   ├── PrimaryPushButton (Parse, Calculate & Plot)
│   │   ├── PushButton × 3 (Details, Reverse, Export)
│   │   └── CommandBar (图表工具栏)
│   └── Right Panel (QWidget + QSplitter)
│       ├── pyqtgraph PlotWidget (3/5 高度)
│       └── RateTableCard (TableWidget, 2/5 高度)
└── Settings Page
    ├── SwitchButton (Theme toggle)
    └── About card
```

### pyqtgraph 集成方案

```python
import pyqtgraph as pg
pg.setConfigOption('background', '#1a1d23')
pg.setConfigOption('foreground', '#b0b8c8')
pg.setConfigOption('antialias', True)

# PlotWidget 替代 matplotlib FigureCanvas
plot_widget = pg.PlotWidget()
plot_widget.setLabel('bottom', 'Temperature', units='K')
plot_widget.setLabel('left', 'log₁₀(k)')
plot_widget.showGrid(x=True, y=True, alpha=0.3)

# 添加曲线
curve = plot_widget.plot(T_values, k_log10, pen=pg.mkPen('#4a90e2', width=2))

# 十字光标（内置 CrosshairItem 或自定义 InfiniteLine）
vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#7a8599', style=Qt.DashLine))
hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#7a8599', style=Qt.DashLine))
```

### matplotlib 导出方案

保留现有 `_update_plot_with_reverse_rates()` 逻辑，新增 `_export_plot_image(filepath)` 方法：
- 创建临时 matplotlib Figure
- 使用相同数据和配色绘制
- `fig.savefig(filepath, dpi=300, bbox_inches='tight')`
- 支持 SVG/PDF/PNG 格式

### QSS 全局样式

使用 Qt C++ 基类名作为选择器（避免 Python 类名不生效的坑）：
- `QScrollArea#leftPanel` — 左侧滚动区域
- `QWidget#leftPanelContent` — 左侧内容容器
- `#leftPanelContent QFrame` — 卡片组件
- `#leftPanelContent QLabel` — 标签
- `QLineEdit` — 输入框（全局）
- `QPlainTextEdit` — 文本编辑区（全局）
- `QCheckBox` — 复选框（全局）
- `#leftPanelContent QPushButton` — 按钮

### 主题切换机制

`_on_theme_toggled(dark)` 方法依次更新：
1. `setTheme(Theme.DARK/LIGHT)` — QFluentWidgets 内置
2. `_apply_global_dark_qss(dark)` — 全局 QSS
3. pyqtgraph 配置 — `pg.setConfigOption('background', bg_color)`
4. 重绘当前图表 — `_update_plot_with_reverse_rates()`
5. 表格样式 — `table_card.apply_theme(dark)`

### 对话窗口适配

三个对话框（`RateConstantDetailDialog`、`ExperimentalReverseRateDialog`、`NasaPolynomialInputDialog`）继续使用 `MessageBoxBase`，继承全局 QSS 自动获得深蓝配色。

## 验证方案

1. **自动化测试**：`pytest tests/test_v13_regressions.py` — 10/10 通过
2. **功能测试**：
   - 启动 GUI → 左侧面板深蓝背景、浅色文字
   - 点击 Parse → pyqtgraph 绘制 5 条曲线，表格填充数据
   - 鼠标悬停图表 → 十字光标显示坐标值
   - 拖拽/缩放图表 → 交互流畅
   - 切换主题 → 所有组件同步更新
   - 导出图片 → matplotlib 生成高清 SVG
   - 导出 Excel → pandas 生成 xlsx
3. **视觉检查**：启动 GUI 后截图确认配色一致性
