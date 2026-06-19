import sys
import os
import shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QTableWidget, QTableWidgetItem, QStatusBar, QSplitter, QGroupBox,
    QHeaderView, QCheckBox, QDialog, QDialogButtonBox, QSizePolicy,
    QScrollArea, QMenuBar, QMessageBox, QFileDialog, QMenu
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QRect
from PyQt6.QtGui import QFont, QColor, QAction, QKeySequence, QShortcut, QPainter, QPen
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# For Excel export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Excel export will be disabled.")

# Import modules
try:
    from chemkin_parser import parse_chemkin_mechanism
    from rate_calculator import (
        calculate_arrhenius_rate, calculate_plog_rate, calculate_troe_rate,
        get_third_body_concentration, R_cal,
        get_reaction_thermo_properties, calculate_equilibrium_constant_kp,
        calculate_delta_n_gas, calculate_equilibrium_constant_kc,
        calculate_reverse_rate_constant,
        merge_duplicate_reactions, calculate_merged_duplicate_rate
    )
    from thermo_parser import read_nasa_polynomials, append_nasa_polynomial_from_string, is_valid_nasa_polynomial_string
except ImportError as e:
    print(f"Error importing modules: {e}")
    parse_chemkin_mechanism = None
    read_nasa_polynomials = None
    append_nasa_polynomial_from_string = None
    is_valid_nasa_polynomial_string = None
    calculate_arrhenius_rate = None
    calculate_plog_rate = None
    calculate_troe_rate = None
    get_third_body_concentration = None
    get_reaction_thermo_properties = None
    calculate_equilibrium_constant_kp = None
    calculate_delta_n_gas = None
    calculate_equilibrium_constant_kc = None
    calculate_reverse_rate_constant = None
    merge_duplicate_reactions = None
    calculate_merged_duplicate_rate = None


THERMO_DATA_DIRNAME = "CHEMKIN_RateViewer"


def resolve_bundled_thermo_filepath():
    """Return the packaged baseline therm.dat used to initialize user data."""
    if getattr(sys, "frozen", False):
        application_dir = getattr(
            sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable))
        )
        bundled_path = os.path.join(application_dir, "therm.dat")
        if os.path.exists(bundled_path):
            return bundled_path
        adjacent_path = os.path.join(
            os.path.dirname(os.path.abspath(sys.executable)), "therm.dat"
        )
        if os.path.exists(adjacent_path):
            return adjacent_path
    else:
        application_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(application_dir, "therm.dat")


def resolve_thermo_filepath():
    """Return the writable persistent therm.dat path for the current user."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        local_appdata = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return os.path.join(local_appdata, THERMO_DATA_DIRNAME, "therm.dat")


def ensure_persistent_thermo_file(persistent_filepath=None, baseline_filepath=None):
    """Copy the packaged baseline into the user database only on first use."""
    persistent_filepath = persistent_filepath or resolve_thermo_filepath()
    baseline_filepath = baseline_filepath or resolve_bundled_thermo_filepath()
    os.makedirs(os.path.dirname(persistent_filepath), exist_ok=True)
    if not os.path.exists(persistent_filepath):
        shutil.copy2(baseline_filepath, persistent_filepath)
    return persistent_filepath


def get_thermo_file_signature(filepath):
    """Return a cheap signature for detecting externally saved thermo edits."""
    try:
        stat_result = os.stat(filepath)
    except OSError:
        return None
    return stat_result.st_mtime_ns, stat_result.st_size


class ToggleSwitch(QCheckBox):
    """自定义滑动开关控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 28)
        self._circle_position = 3
        self._animation = None  # 保持动画对象引用

        # 连接状态变化信号，更新圆圈位置
        self.toggled.connect(self._on_toggled)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制背景轨道
        if self.isChecked():
            track_color = QColor(76, 175, 80)  # 绿色
        else:
            track_color = QColor(189, 189, 189)  # 灰色

        painter.setBrush(track_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height() / 2, self.height() / 2)

        # 绘制滑动圆圈
        circle_color = QColor(255, 255, 255)  # 白色
        painter.setBrush(circle_color)

        # 添加阴影效果
        shadow_color = QColor(0, 0, 0, 50)
        painter.setPen(QPen(shadow_color, 1))

        circle_x = self._circle_position
        circle_y = 3
        circle_diameter = self.height() - 6
        painter.drawEllipse(circle_x, circle_y, circle_diameter, circle_diameter)

    def hitButton(self, pos):
        """确保整个控件区域都可以点击"""
        return self.contentsRect().contains(pos)

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 不调用父类方法，避免默认的复选框行为
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 切换状态
            self.setChecked(not self.isChecked())
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _on_toggled(self, checked):
        """状态改变时触发动画"""
        self._animate_toggle()

    def _animate_toggle(self):
        """动画效果"""
        # 停止之前的动画
        if self._animation is not None:
            self._animation.stop()

        self._animation = QPropertyAnimation(self, b"circle_position", self)
        self._animation.setDuration(120)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        if self.isChecked():
            self._animation.setEndValue(self.width() - self.height() + 3)
        else:
            self._animation.setEndValue(3)

        self._animation.start()

    def _get_circle_position(self):
        return self._circle_position

    def _set_circle_position(self, pos):
        self._circle_position = pos
        self.update()

    circle_position = pyqtProperty(int, _get_circle_position, _set_circle_position)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 使用主程序入口相同目录下的热力学数据文件
        self.bundled_thermo_filepath = resolve_bundled_thermo_filepath()
        self.thermo_filepath = resolve_thermo_filepath()
        self._thermo_file_signature = None
        self.setWindowTitle("CHEMKIN Rate Viewer - v1.3")
        self.setGeometry(100, 100, 1400, 900)

        # 加载现代化样式
        self._load_stylesheet()

        # Create menu bar
        self._create_menu_bar()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.main_splitter)

        # Left panel
        self.left_panel_widget = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_panel_widget)
        
        self.chemkin_input_label = QLabel("CHEMKIN Input:")
        self.left_panel_layout.addWidget(self.chemkin_input_label)
        
        self.chemkin_input_text_edit = QTextEdit()
        self.chemkin_input_text_edit.setMinimumHeight(200)
        self.chemkin_input_text_edit.setPlaceholderText("Paste CHEMKIN mechanism here...")
        sample_chemkin_input = """! Arrhenius example
UNITS CAL/MOLE
O + H2 <=> H + OH          1.0E10  0.5   15000.0 

! PLOG example (pressure-dependent)
CH4 + O2 = CH3 + HO2      1.0E12  0.0  50000.0
    PLOG / 0.1   1.0E11  0.0  48000.0 /
    PLOG / 1.0   1.0E12  0.0  50000.0 /
    PLOG / 10.0  1.0E13  0.0  52000.0 /
    PLOG / 100.0 1.0E14  0.0  54000.0 /

! TROE example (falloff reaction)
H + O2 (+M) = HO2 (+M)    1.475E12  0.6  0.0
    LOW / 6.366E20  -1.72  524.8 /
    TROE / 0.8  1E-30  1E30 /
    H2/2.0/ H2O/6.0/ CO/1.75/ CO2/3.6/ AR/0.7/
        """
        self.chemkin_input_text_edit.setText(sample_chemkin_input.strip())
        self.left_panel_layout.addWidget(self.chemkin_input_text_edit, 1)

        # Controls
        self.controls_groupbox = QGroupBox("Controls")
        self.controls_layout = QFormLayout(self.controls_groupbox)

        self.temp_min_entry = QLineEdit("300")
        self.controls_layout.addRow("Min Temperature (K):", self.temp_min_entry)
        self.temp_max_entry = QLineEdit("2500")
        self.controls_layout.addRow("Max Temperature (K):", self.temp_max_entry)

        self.pressure_min_entry = QLineEdit("1.0")
        self.controls_layout.addRow("Min Plot Pressure (atm):", self.pressure_min_entry)
        self.pressure_max_entry = QLineEdit("10.0")
        self.controls_layout.addRow("Max Plot Pressure (atm):", self.pressure_max_entry)
        self.pressure_log_steps_entry = QLineEdit("2")
        self.controls_layout.addRow("Log Plot Pressure Steps:", self.pressure_log_steps_entry)
        
        self.table_temperatures_entry = QLineEdit("500, 1000, 1500, 2000")
        self.controls_layout.addRow("Table Temperatures (K):", self.table_temperatures_entry)

        self.group_count_entry = QLineEdit("1")
        self.controls_layout.addRow("Group Number:", self.group_count_entry)

        # 功能1: 组内/组间样式切换复选框
        self.intra_group_linestyle_checkbox = QCheckBox("组内变化线型")
        self.intra_group_linestyle_checkbox.setToolTip("勾选：组内使用不同线型，组间使用不同颜色\n未勾选：组内使用不同颜色，组间使用不同线型（默认）")
        self.intra_group_linestyle_checkbox.stateChanged.connect(self._on_style_mode_changed)
        self.controls_layout.addRow("", self.intra_group_linestyle_checkbox)

        # 功能3: 横坐标转换复选框
        self.use_inverse_temp_checkbox = QCheckBox("使用1000/T作为横坐标")
        self.use_inverse_temp_checkbox.setToolTip("勾选：X轴显示为1000/T (K⁻¹)\n未勾选：X轴显示为温度T (K)")
        self.use_inverse_temp_checkbox.stateChanged.connect(self._on_x_axis_mode_changed)
        self.controls_layout.addRow("", self.use_inverse_temp_checkbox)

        self.left_panel_layout.addWidget(self.controls_groupbox)

        # Buttons
        self.plot_button = QPushButton("Parse, Calculate & Plot")
        self.plot_button.clicked.connect(self._on_plot_button_clicked)
        self.left_panel_layout.addWidget(self.plot_button)

        # 功能2: 查看速率常数详细数据按钮
        self.view_detail_button = QPushButton("查看速率常数详细数据")
        self.view_detail_button.clicked.connect(self._on_view_detail_button_clicked)
        self.view_detail_button.setEnabled(False)
        self.view_detail_button.setToolTip("显示所有反应在不同温度下的速率常数详细数据表")
        self.left_panel_layout.addWidget(self.view_detail_button)

        # 功能5: 实验逆反应速率转换按钮
        self.exp_reverse_rate_button = QPushButton("实验逆反应速率转换")
        self.exp_reverse_rate_button.clicked.connect(self._on_exp_reverse_rate_button_clicked)
        self.exp_reverse_rate_button.setToolTip("根据实验测得的逆反应速率数据，通过平衡常数计算正反应速率")
        self.left_panel_layout.addWidget(self.exp_reverse_rate_button)

        self.save_data_button = QPushButton("Save Data to Excel")
        self.save_data_button.clicked.connect(self._on_save_data_button_clicked)
        self.save_data_button.setEnabled(False)
        if not PANDAS_AVAILABLE:
            self.save_data_button.setToolTip("Excel导出功能需要安装pandas库")
            self.save_data_button.setEnabled(False)
        else:
            self.save_data_button.setToolTip("将当前的速率常数数据导出为Excel文件")
        self.left_panel_layout.addWidget(self.save_data_button)
        
        self.main_splitter.addWidget(self.left_panel_widget)

        # Right panel
        self.right_panel_widget = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_panel_widget)
        self.right_panel_splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot area
        self.plot_area_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_area_widget)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure_canvas = FigureCanvas(self.figure)
        
        self.plot_toolbar = NavigationToolbar(self.figure_canvas, self.plot_area_widget)
        plot_layout.addWidget(self.plot_toolbar)
        plot_layout.addWidget(self.figure_canvas)
        self.plot_axes = self.figure.add_subplot(111)
        self.right_panel_splitter.addWidget(self.plot_area_widget)

        # Table
        self.rate_table_widget = QTableWidget()
        self.table_temperatures = [500.0, 1000.0, 1500.0, 2000.0]  # Default values
        self._setup_table_columns()  # Setup initial columns
        
        self.rate_table_widget.horizontalHeader().setStretchLastSection(False)
        self.rate_table_widget.setAlternatingRowColors(True)
        self.rate_table_widget.setMinimumHeight(150)
        self.rate_table_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.MinimumExpanding
        )
        self._setup_rate_table_widget_appearance()

        self.right_panel_splitter.addWidget(self.rate_table_widget)
        self.right_panel_splitter.setSizes([400, 300])
        self.right_panel_splitter.setStretchFactor(0, 2)
        self.right_panel_splitter.setStretchFactor(1, 1)
        self.right_panel_layout.addWidget(self.right_panel_splitter)
        self.main_splitter.addWidget(self.right_panel_widget)
        self.main_splitter.setSizes([350, 850])

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        self.thermo_data = None
        try:
            ensure_persistent_thermo_file(
                self.thermo_filepath, self.bundled_thermo_filepath
            )
            self._load_thermo_data()
        except Exception as e:
            self.thermo_data = {}
            self.statusBar.showMessage(
                f"Error initializing persistent thermo data: {e}. Reverse rates disabled."
            )

    def _load_stylesheet(self):
        """加载现代化样式表"""
        try:
            # 尝试从文件加载样式
            import os
            style_path = os.path.join(os.path.dirname(__file__), 'modern_style.qss')
            if os.path.exists(style_path):
                with open(style_path, 'r', encoding='utf-8') as f:
                    stylesheet = f.read()
                    self.setStyleSheet(stylesheet)
            else:
                # 如果文件不存在，使用内嵌的样式
                self._apply_embedded_stylesheet()
        except Exception as e:
            print(f"Warning: Could not load stylesheet: {e}")
            # 使用内嵌的样式作为后备
            self._apply_embedded_stylesheet()

    def _apply_embedded_stylesheet(self):
        """应用内嵌的样式表（用于打包后的程序）"""
        stylesheet = """
        /* Modern Dark Theme */
        QMainWindow, QWidget, QDialog {
            background-color: #2b2b2b;
            color: #e0e0e0;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 10pt;
        }

        QGroupBox {
            background-color: #353535;
            border: 2px solid #4a4a4a;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 15px;
            font-weight: bold;
            color: #ffffff;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;
            background-color: #4a90e2;
            border-radius: 4px;
            color: white;
        }

        QLabel {
            color: #e0e0e0;
            background-color: transparent;
        }

        QLineEdit, QTextEdit {
            background-color: #3c3c3c;
            border: 2px solid #4a4a4a;
            border-radius: 6px;
            padding: 8px;
            color: #ffffff;
            selection-background-color: #4a90e2;
        }

        QLineEdit:focus, QTextEdit:focus {
            border: 2px solid #4a90e2;
            background-color: #404040;
        }

        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #4a90e2, stop:1 #357abd);
            border: none;
            border-radius: 6px;
            color: white;
            padding: 10px 20px;
            font-weight: bold;
            min-height: 30px;
        }

        QPushButton:hover {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #5a9ee2, stop:1 #4585c7);
        }

        QPushButton:pressed {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #357abd, stop:1 #2a6aa8);
        }

        QCheckBox {
            color: #e0e0e0;
            spacing: 8px;
        }

        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #4a4a4a;
            border-radius: 4px;
            background-color: #3c3c3c;
        }

        QCheckBox::indicator:checked {
            background-color: #4a90e2;
            border: 2px solid #4a90e2;
        }

        QTableWidget {
            background-color: #3c3c3c;
            alternate-background-color: #404040;
            border: 2px solid #4a4a4a;
            border-radius: 6px;
            gridline-color: #4a4a4a;
            color: #ffffff;
            selection-background-color: #4a90e2;
            font-family: "Consolas", "Courier New", monospace;
            font-size: 9pt;
        }

        QTableWidget::item {
            padding: 6px 8px;
        }

        QHeaderView::section {
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #4a4a4a, stop:1 #3a3a3a);
            color: white;
            padding: 8px;
            border: 1px solid #2b2b2b;
            font-weight: bold;
        }

        QMenuBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-bottom: 1px solid #4a4a4a;
        }

        QMenuBar::item:selected {
            background-color: #4a90e2;
        }

        QMenu {
            background-color: #353535;
            border: 1px solid #4a4a4a;
            color: #e0e0e0;
        }

        QMenu::item:selected {
            background-color: #4a90e2;
        }

        QStatusBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-top: 1px solid #4a4a4a;
        }

        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 14px;
            border-radius: 7px;
        }

        QScrollBar::handle:vertical {
            background-color: #4a4a4a;
            border-radius: 7px;
            min-height: 30px;
        }
        """
        self.setStyleSheet(stylesheet)

    def _setup_table_columns(self):
        """Setup table columns based on current temperature settings"""
        num_temp_points = len(self.table_temperatures)
        self.rate_table_widget.setColumnCount(3 + num_temp_points)
        
        header_labels = ["Reaction", "Pressure (atm)", "Calc Rev?"]
        for T_table in self.table_temperatures:
            header_labels.append(f"k @ {T_table:.0f}K")
        self.rate_table_widget.setHorizontalHeaderLabels(header_labels)

    def _parse_temperature_input(self):
        """Parse temperature input from user and update table_temperatures"""
        try:
            temp_text = self.table_temperatures_entry.text().strip()
            if not temp_text:
                self.table_temperatures = [500.0, 1000.0, 1500.0, 2000.0]  # Default
                return True
                
            # Parse comma-separated temperatures
            temp_strings = [t.strip() for t in temp_text.split(',')]
            temperatures = []
            for temp_str in temp_strings:
                if temp_str:  # Skip empty strings
                    temp_val = float(temp_str)
                    if temp_val > 0:
                        temperatures.append(temp_val)
                    else:
                        raise ValueError(f"Temperature must be positive: {temp_val}")
            
            if not temperatures:
                raise ValueError("No valid temperatures found")
                
            self.table_temperatures = temperatures
            return True
            
        except ValueError as e:
            self.statusBar.showMessage(f"Invalid temperature input: {e}")
            return False

    def _create_menu_bar(self):
        """Create menu bar with Help and About options"""
        menubar = self.menuBar()
        
        help_menu = menubar.addMenu('Help')
        
        help_action = QAction('Usage Instructions', self)
        help_action.setShortcut('F1')
        help_action.triggered.connect(self._show_help_dialog)
        help_menu.addAction(help_action)
        
        help_menu.addSeparator()
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _show_help_dialog(self):
        """Show help dialog"""
        help_text = """CHEMKIN Rate Viewer 使用说明

📝 输入区域：
• CHEMKIN Input: 输入标准CHEMKIN格式的反应机理
  - 支持Arrhenius、PLOG、TROE等反应类型
  - 示例已包含正确的格式参考

⚙️ 控制参数：
• Min/Max Temperature (K): 设置绘图的温度范围
• Min/Max Plot Pressure (atm): 设置PLOG/TROE反应的压力绘图范围
• Log Plot Pressure Steps: 压力对数分布的步数
• Table Temperatures (K): 表格显示的温度点，用逗号分隔
  例如: 300, 500, 1000, 1500, 2000
• Group Number: 反应分组数量
  - 同组内反应使用相同线型、不同颜色
  - 不同组使用不同线型（实线、虚线、点划线、点线）

📊 操作步骤：
1. 输入或修改CHEMKIN反应机理
2. 根据需要调整温度、压力和分组参数
3. 点击'Parse, Calculate & Plot'解析并绘图
4. 在表格中勾选'Calc Rev?'计算逆反应速率
5. 可拖拽调整表格列宽
6. 使用'Save Data to Excel'导出数据

💡 提示：
• 支持重复反应(DUP)的自动合并
• 逆反应计算需要热力学数据(therm.dat)
• 表格颜色与图例颜色对应"""
        
        QMessageBox.information(self, "帮助", help_text)

    def _show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(self, "关于", "CHEMKIN Rate Viewer v1.3\n\n化学反应动力学机理分析工具\n\n开发者：王杜\n单位：中科院工程热物理研究所")

    def _setup_rate_table_widget_appearance(self):
        """设置表格外观和列宽"""
        header = self.rate_table_widget.horizontalHeader()

        # 设置表格字体为等宽字体，提高可读性
        table_font = QFont("Consolas", 9)
        if not table_font.exactMatch():
            table_font = QFont("Courier New", 9)
        self.rate_table_widget.setFont(table_font)

        # 列0: Reaction - 可调整宽度
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.rate_table_widget.setColumnWidth(0, 350)

        # 列1: Pressure (atm) - 固定宽度，增加宽度以完整显示
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.rate_table_widget.setColumnWidth(1, 120)  # 从80增加到120

        # 列2: Calc Rev? - 固定宽度，为滑动开关留出足够空间
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.rate_table_widget.setColumnWidth(2, 100)  # 增加到100以完整显示滑动开关

        # 温度列 - 可调整宽度
        num_temp_points = len(self.table_temperatures)
        for i in range(num_temp_points):
            col_index = 3 + i
            header.setSectionResizeMode(col_index, QHeaderView.ResizeMode.Interactive)
            self.rate_table_widget.setColumnWidth(col_index, 130)  # 从120增加到130

        # 启用交替行颜色
        self.rate_table_widget.setAlternatingRowColors(True)

        # 设置行高，为滑动开关留出足够空间
        self.rate_table_widget.verticalHeader().setDefaultSectionSize(36)  # 从28增加到36
        self.rate_table_widget.verticalHeader().setVisible(False)
        self.rate_table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        header.setStretchLastSection(False)

    def _load_thermo_data(self):
        if read_nasa_polynomials is None:
            self.statusBar.showMessage("Error: Thermo parser not available. Reverse rates disabled.")
            self.thermo_data = {}
            self._thermo_file_signature = None
            return

        try:
            self.thermo_data = read_nasa_polynomials(self.thermo_filepath)
            self._thermo_file_signature = get_thermo_file_signature(self.thermo_filepath)
            if not self.thermo_data:
                self.statusBar.showMessage(f"Warning: No data read from {self.thermo_filepath}.")
                self.thermo_data = {}
            else:
                self.statusBar.showMessage(f"Successfully loaded {len(self.thermo_data)} species from {self.thermo_filepath}.")
        except FileNotFoundError:
            self.statusBar.showMessage(f"Error: {self.thermo_filepath} not found. Reverse rates disabled.")
            self.thermo_data = {}
            self._thermo_file_signature = None
        except Exception as e:
            self.statusBar.showMessage(f"Error parsing {self.thermo_filepath}: {e}. Reverse rates disabled.")
            self.thermo_data = {}
            self._thermo_file_signature = get_thermo_file_signature(self.thermo_filepath)

    def _reload_thermo_data_if_changed(self):
        """Reload persistent thermo data once an external save changes the file."""
        current_signature = get_thermo_file_signature(self.thermo_filepath)
        if current_signature == getattr(self, "_thermo_file_signature", None):
            return False
        self._load_thermo_data()
        return True

    def _calculate_rate_for_reaction(self, reaction_data, T, P_atm):
        k = None
        reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')
        if not (calculate_arrhenius_rate and calculate_plog_rate and calculate_troe_rate and get_third_body_concentration):
            self.statusBar.showMessage("Error: Rate calculation functions not loaded.")
            return None
        
        try:
            if reaction_data.get('duplicate_arrhenius_params') and calculate_merged_duplicate_rate:
                k = calculate_merged_duplicate_rate(reaction_data, T, P_atm)
                if k is not None:
                    return k
            
            if reaction_type == 'ARRHENIUS':
                if reaction_data.get('arrhenius_params'):
                    k = calculate_arrhenius_rate(reaction_data['arrhenius_params'], T)
            elif reaction_type == 'PLOG':
                if reaction_data.get('plog_data'):
                    k = calculate_plog_rate(reaction_data['plog_data'], T, P_atm)
            elif reaction_type == 'TROE':
                if reaction_data.get('troe_data'):
                    M_conc = get_third_body_concentration(P_atm, T)
                    k = calculate_troe_rate(reaction_data['troe_data'], T, P_atm, M_conc=M_conc)
        except Exception as e:
            self.statusBar.showMessage(f"Error calculating rate for {reaction_data.get('equation_string', 'N/A')}: {e}")
            k = None
        return k

    def _update_rate_constant_table(self, parsed_reactions):
        self.rate_table_widget.setRowCount(0)
        
        P_plot_values, use_pressure_range_plotting, P_default = self._get_plot_pressure_settings()
        self.parsed_reactions = parsed_reactions

        current_row = 0
        for reaction_index, reaction_data in enumerate(parsed_reactions):
            eq_display_str = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
            reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')
            
            if use_pressure_range_plotting and reaction_type in ['PLOG', 'TROE']:
                pressures_to_use = P_plot_values
            else:
                pressures_to_use = [P_default]
            
            reaction_data['table_rows'] = []
            reaction_data['is_calculating_reverse'] = False
            reaction_data['original_forward_rates'] = {}
            
            for pressure_index, pressure in enumerate(pressures_to_use):
                self.rate_table_widget.insertRow(current_row)
                
                # Column 0: Reaction string
                if reaction_type in ['PLOG', 'TROE'] and len(pressures_to_use) > 1:
                    if pressure_index == 0:
                        display_text = eq_display_str
                    else:
                        display_text = f"  └─ {reaction_type}"
                else:
                    display_text = eq_display_str
                
                if reaction_data.get('duplicate_arrhenius_params'):
                    num_duplicates = len(reaction_data['duplicate_arrhenius_params'])
                    if pressure_index == 0:
                        display_text += f" [DUP×{num_duplicates}]"
                    
                eq_item = QTableWidgetItem(display_text)
                eq_item.setFlags(eq_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                
                if reaction_data.get('duplicate_arrhenius_params'):
                    tooltip_text = f"合并的重复反应 (DUP关键词)\n"
                    tooltip_text += f"包含 {len(reaction_data['duplicate_arrhenius_params'])} 个重复反应\n"
                    tooltip_text += f"速率常数为各反应之和: k_total = k1 + k2 + ..."
                    eq_item.setToolTip(tooltip_text)
                
                color_hex = self._get_reaction_color(reaction_index, pressure_index)
                color = QColor(color_hex)
                color.setAlpha(50)
                eq_item.setBackground(color)
                
                self.rate_table_widget.setItem(current_row, 0, eq_item)

                # Column 1: Pressure
                if reaction_type in ['PLOG', 'TROE'] and use_pressure_range_plotting:
                    pressure_str = f"{pressure:.1f}"
                elif reaction_type in ['PLOG', 'TROE']:
                    pressure_str = f"{pressure:.1f} (Table)"
                else:
                    pressure_str = f"{pressure:.1f}"
                pressure_item = QTableWidgetItem(pressure_str)
                pressure_item.setFlags(pressure_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.rate_table_widget.setItem(current_row, 1, pressure_item)

                # Column 2: Toggle Switch (滑动开关)
                if reaction_type in ['PLOG', 'TROE'] and len(pressures_to_use) > 1 and pressure_index > 0:
                    empty_item = QTableWidgetItem("")
                    empty_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self.rate_table_widget.setItem(current_row, 2, empty_item)
                else:
                    toggle_switch = ToggleSwitch()
                    toggle_switch.stateChanged.connect(
                        lambda state, rd=reaction_data, ri=reaction_index: self._on_calc_reverse_checkbox_changed(state, rd, ri)
                    )
                    # 创建一个容器widget来居中显示开关
                    container = QWidget()
                    container_layout = QHBoxLayout(container)
                    container_layout.addWidget(toggle_switch)
                    container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    container_layout.setContentsMargins(0, 0, 0, 0)
                    self.rate_table_widget.setCellWidget(current_row, 2, container)
                    
                reaction_data['table_rows'].append({
                    'row': current_row,
                    'pressure': pressure,
                    'pressure_index': pressure_index,
                    'is_main_row': pressure_index == 0
                })

                # Rate constants
                col_offset = 3
                for i, T_table in enumerate(self.table_temperatures):
                    kf_val = self._calculate_rate_for_reaction(reaction_data, T_table, pressure)
                    kf_str = "%.3E" % kf_val if kf_val is not None and kf_val > 0 else ("0.000E+00" if kf_val == 0 else "N/A")
                    kf_item = QTableWidgetItem(kf_str)
                    kf_item.setFlags(kf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.rate_table_widget.setItem(current_row, col_offset + i, kf_item)
                    
                    rate_key = f"row_{current_row}_col_{col_offset + i}"
                    reaction_data['original_forward_rates'][rate_key] = kf_str
                
                current_row += 1

        self.rate_table_widget.resizeColumnsToContents()

    def _on_calc_reverse_checkbox_changed(self, state, reaction_data, reaction_index):
        """Handle reverse rate calculation checkbox"""
        eq_string = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
        reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')

        if state == Qt.CheckState.Checked.value:
            self.statusBar.showMessage(f"Calculating reverse rates for: {eq_string[:30]}...")
            self._reload_thermo_data_if_changed()
            
            # 检查必要的函数和数据
            if not all([get_reaction_thermo_properties, calculate_equilibrium_constant_kp,
                        calculate_delta_n_gas, calculate_equilibrium_constant_kc,
                        calculate_reverse_rate_constant, self.thermo_data is not None,
                        is_valid_nasa_polynomial_string, append_nasa_polynomial_from_string]):
                self.statusBar.showMessage("Error: Missing critical functions or thermo data. Reverse calc disabled.")
                main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                if main_checkbox: main_checkbox.setChecked(False)
                return

            # 检查缺失的物种数据
            all_species_in_reaction = reaction_data.get('reactants', []) + reaction_data.get('products', [])
            current_missing_species = set()
            for coeff, spec_name in all_species_in_reaction:
                if not self.thermo_data.get(spec_name.upper()):
                    current_missing_species.add(spec_name)
            
            if current_missing_species:
                # 显示输入对话框让用户输入缺失的热力学数据
                missing_list = sorted(list(current_missing_species))

                dialog = NasaPolynomialInputDialog(missing_list, self.thermo_filepath, self)
                result = dialog.exec()

                if result == QDialog.DialogCode.Accepted:
                    # 用户输入了有效数据，重新加载热力学数据
                    self.statusBar.showMessage("重新加载热力学数据...")
                    try:
                        self.thermo_data = read_nasa_polynomials(self.thermo_filepath)
                        self.statusBar.showMessage(
                            f"热力学数据已更新，成功加载 {len(self.thermo_data)} 个物种"
                        )

                        # 重新检查是否还有缺失的物种
                        still_missing = set()
                        for coeff, spec_name in all_species_in_reaction:
                            if not self.thermo_data.get(spec_name.upper()):
                                still_missing.add(spec_name)

                        if still_missing:
                            # 仍然有缺失的物种
                            missing_str = ', '.join(sorted(still_missing))
                            QMessageBox.warning(
                                self,
                                "仍有缺失数据",
                                f"以下物种的热力学数据仍然缺失：\n{missing_str}\n\n"
                                "请检查输入的数据是否正确。"
                            )
                            main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                            if main_checkbox: main_checkbox.setChecked(False)
                            return

                        # 所有数据都有了，继续计算
                        # 不需要return，让代码继续执行到下面的计算逻辑

                    except Exception as e:
                        QMessageBox.critical(
                            self,
                            "加载失败",
                            f"重新加载热力学数据时发生错误：\n{str(e)}"
                        )
                        main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                        if main_checkbox: main_checkbox.setChecked(False)
                        return
                else:
                    # 用户取消了输入
                    missing_str = ', '.join(missing_list)
                    self.statusBar.showMessage(f"已取消：缺少热力学数据 ({missing_str})")
                    main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                    if main_checkbox: main_checkbox.setChecked(False)
                    return
            
            # 标记正在计算逆反应
            reaction_data['is_calculating_reverse'] = True
            
            # 为所有压力行计算逆反应速率
            any_thermo_calc_error = False
            for row_info in reaction_data['table_rows']:
                row = row_info['row']
                pressure = row_info['pressure']
                
                # 为当前行的所有温度计算逆反应速率
                for i, T_table in enumerate(self.table_temperatures):
                    col_index = 3 + i
                
                    delta_H, delta_S, delta_G, missing_at_T, err_thermo_calc = \
                        get_reaction_thermo_properties(reaction_data, T_table, self.thermo_data)

                    if err_thermo_calc:
                        any_thermo_calc_error = True
                        error_item = QTableWidgetItem("Thermo Err")
                        error_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        self.rate_table_widget.setItem(row, col_index, error_item)
                        continue

                    Kp = calculate_equilibrium_constant_kp(delta_G, T_table)
                    delta_n = calculate_delta_n_gas(reaction_data, self.thermo_data)
                    Kc = calculate_equilibrium_constant_kc(Kp, T_table, delta_n)

                    # 从原始保存的数据获取正反应速率
                    rate_key = f"row_{row}_col_{col_index}"
                    original_kf_str = reaction_data['original_forward_rates'].get(rate_key, "N/A")
                    
                    kf_val = None
                    if original_kf_str not in ["N/A", "Error", "kf Error", "Thermo Err"]:
                        try: 
                            kf_val = float(original_kf_str)
                        except ValueError: 
                            pass

                    # 计算逆反应速率常数
                    kr_str = "Error"
                    if kf_val is not None and Kc is not None:
                        kr = calculate_reverse_rate_constant(kf_val, Kc)
                        kr_str = "%.3E" % kr if kr is not None else "Error"
                    elif kf_val is None:
                        kr_str = "kf Error" 

                    # 更新表格中的数值
                    kr_item = QTableWidgetItem(kr_str)
                    kr_item.setFlags(kr_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.rate_table_widget.setItem(row, col_index, kr_item)
            
            # 更新绘图以显示逆反应速率
            self._update_plot_with_reverse_rates()
            
            if not any_thermo_calc_error:
                self.statusBar.showMessage(f"Reverse rates updated for: {eq_string[:30]} (all pressures)")
            else:
                self.statusBar.showMessage(f"Reverse rates updated for {eq_string[:30]} (some errors)")

        elif state == Qt.CheckState.Unchecked.value:
            self.statusBar.showMessage(f"Restoring forward rates for: {eq_string[:30]}...")
            
            # 恢复所有压力行的正反应速率
            reaction_data['is_calculating_reverse'] = False
            
            for row_info in reaction_data['table_rows']:
                row = row_info['row']
                
                # 恢复所有温度的正反应速率
                for i, T_table in enumerate(self.table_temperatures):
                    col_index = 3 + i
                    rate_key = f"row_{row}_col_{col_index}"
                    original_kf_str = reaction_data['original_forward_rates'].get(rate_key, "N/A")
                    
                    # 恢复原始正反应速率
                    kf_item = QTableWidgetItem(original_kf_str)
                    kf_item.setFlags(kf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.rate_table_widget.setItem(row, col_index, kf_item)
            
            # 更新绘图以显示正反应速率
            self._update_plot_with_reverse_rates()
            
            self.statusBar.showMessage(f"Forward rates restored for: {eq_string[:30]} (all pressures)")

    def _get_main_checkbox_for_reaction(self, reaction_data):
        """获取反应的主滑动开关（第一行）"""
        for row_info in reaction_data.get('table_rows', []):
            if row_info.get('is_main_row', False):
                # 获取容器widget
                container = self.rate_table_widget.cellWidget(row_info['row'], 2)
                if container:
                    # 从容器中获取滑动开关
                    layout = container.layout()
                    if layout and layout.count() > 0:
                        return layout.itemAt(0).widget()
                return None
        return None

    def _calculate_reverse_rate_for_plot(self, reaction_data, T, P_atm):
        """为绘图计算逆反应速率常数"""
        try:
            self._reload_thermo_data_if_changed()
            # 计算正反应速率常数
            kf = self._calculate_rate_for_reaction(reaction_data, T, P_atm)
            if kf is None or kf <= 0:
                return None
                
            # 计算热力学性质
            delta_H, delta_S, delta_G, missing_at_T, err_thermo_calc = \
                get_reaction_thermo_properties(reaction_data, T, self.thermo_data)
            
            if err_thermo_calc:
                return None
                
            # 计算平衡常数
            Kp = calculate_equilibrium_constant_kp(delta_G, T)
            delta_n = calculate_delta_n_gas(reaction_data, self.thermo_data)
            Kc = calculate_equilibrium_constant_kc(Kp, T, delta_n)
            
            if Kc is None:
                return None
                
            # 计算逆反应速率常数
            kr = calculate_reverse_rate_constant(kf, Kc)
            return kr
            
        except Exception as e:
            return None

    def _update_plot(self, parsed_reactions):
        """Update plot with reaction rates"""
        self.parsed_reactions = parsed_reactions
        self._update_plot_with_reverse_rates()

    def _update_plot_with_reverse_rates(self):
        """Update plot considering reverse rate settings"""
        if not hasattr(self, 'parsed_reactions'):
            return

        self.plot_axes.clear()
        try:
            T_min = float(self.temp_min_entry.text())
            T_max = float(self.temp_max_entry.text())
            if not (T_max > T_min and T_min > 0):
                self.statusBar.showMessage("Error: Invalid Temperature range.")
                self.figure_canvas.draw()
                return
            T_values = np.linspace(T_min, T_max, 100)

            # 功能3: 检查是否使用1000/T作为横坐标
            use_inverse_temp = self.use_inverse_temp_checkbox.isChecked()
            if use_inverse_temp:
                x_values = 1000.0 / T_values
                x_label = "1000/T (K⁻¹)"
            else:
                x_values = T_values
                x_label = "Temperature (K)"

            P_plot_values, use_pressure_range_plotting, P_default = self._get_plot_pressure_settings()

        except ValueError:
            self.statusBar.showMessage("Error: Temperature/Pressure values must be numeric.")
            self.figure_canvas.draw()
            return

        num_lines_plotted = 0
        for reaction_index, reaction_data in enumerate(self.parsed_reactions):
            reaction_label_base = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
            reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')
            is_calculating_reverse = reaction_data.get('is_calculating_reverse', False)

            if use_pressure_range_plotting and reaction_type in ['PLOG', 'TROE']:
                for pressure_index, P_val in enumerate(P_plot_values):
                    k_values_log10 = []
                    for T_val in T_values:
                        if is_calculating_reverse:
                            # 计算逆反应速率
                            k = self._calculate_reverse_rate_for_plot(reaction_data, T_val, P_val)
                        else:
                            # 计算正反应速率
                            k = self._calculate_rate_for_reaction(reaction_data, T_val, P_val)
                        
                        if k is not None and k > 1e-100:
                            k_values_log10.append(np.log10(k))
                        else:
                            k_values_log10.append(np.nan)
                    
                    if any(not np.isnan(val) for val in k_values_log10):
                        label_prefix = "kr" if is_calculating_reverse else "kf"
                        label = f"{label_prefix}: {reaction_label_base} @ {P_val:.2E} atm"
                        color, linestyle = self._get_reaction_style(reaction_index, pressure_index)
                        self.plot_axes.plot(x_values, k_values_log10, label=label, color=color, linestyle=linestyle)
                        num_lines_plotted += 1
            else:
                k_values_log10 = []
                current_P_for_plot = P_default
                for T_val in T_values:
                    if is_calculating_reverse:
                        k = self._calculate_reverse_rate_for_plot(reaction_data, T_val, current_P_for_plot)
                    else:
                        k = self._calculate_rate_for_reaction(reaction_data, T_val, current_P_for_plot)

                    if k is not None and k > 1e-100:
                        k_values_log10.append(np.log10(k))
                    else:
                        k_values_log10.append(np.nan)

                if any(not np.isnan(val) for val in k_values_log10):
                    label_prefix = "kr" if is_calculating_reverse else "kf"
                    label = f"{label_prefix}: {reaction_label_base}"
                    if reaction_type in ['PLOG', 'TROE']:
                        label += f" @ {current_P_for_plot:.2E} atm (Table P)"
                    color, linestyle = self._get_reaction_style(reaction_index, 0)
                    self.plot_axes.plot(x_values, k_values_log10, label=label, color=color, linestyle=linestyle)
                    num_lines_plotted += 1

        self.plot_axes.set_xlabel(x_label)
        self.plot_axes.set_ylabel("log10(k)")
        self.plot_axes.set_title("Reaction Rates vs Temperature")
        if num_lines_plotted > 0:
            self.plot_axes.legend(fontsize='small')
        self.plot_axes.grid(True)
        self.figure_canvas.draw()

    def _on_plot_button_clicked(self):
        chemkin_text = self.chemkin_input_text_edit.toPlainText()
        if not chemkin_text.strip():
            self.statusBar.showMessage("CHEMKIN input is empty.")
            self.rate_table_widget.setRowCount(0)
            self.plot_axes.clear()
            self.figure_canvas.draw()
            return

        self.statusBar.showMessage("Parsing CHEMKIN input...")
        QApplication.processEvents()

        # Parse temperature input first
        if not self._parse_temperature_input():
            return  # Error message already shown
        
        # Update table columns based on new temperatures
        self._setup_table_columns()
        self._setup_rate_table_widget_appearance()

        try:
            if not parse_chemkin_mechanism:
                self.statusBar.showMessage("Error: Parser module not loaded.")
                return
            
            parsed_reactions = parse_chemkin_mechanism(chemkin_text)
            
            if not parsed_reactions:
                self.statusBar.showMessage("No valid reactions parsed from input.")
                self.rate_table_widget.setRowCount(0)
                self.plot_axes.clear()
                self.figure_canvas.draw()
                return

            if merge_duplicate_reactions:
                original_count = len(parsed_reactions)
                parsed_reactions = merge_duplicate_reactions(parsed_reactions)
                merged_count = len(parsed_reactions)
                
                if original_count != merged_count:
                    dup_merged = original_count - merged_count
                    self.statusBar.showMessage(f"Merged {dup_merged} duplicate reactions. Processing {merged_count} unique reactions...")
                    QApplication.processEvents()

            self.statusBar.showMessage(f"Parsed {len(parsed_reactions)} reactions. Updating table and plot...")
            QApplication.processEvents()
            
            self._update_rate_constant_table(parsed_reactions)
            self._update_plot(parsed_reactions)

            if PANDAS_AVAILABLE:
                self.save_data_button.setEnabled(True)

            # 启用查看详细数据按钮
            self.view_detail_button.setEnabled(True)

            P_plot_values, use_pressure_range_plotting, P_default = self._get_plot_pressure_settings()
            if use_pressure_range_plotting and len(P_plot_values) > 1:
                pressure_info = f"P={P_plot_values[0]:.1f}-{P_plot_values[-1]:.1f} atm ({len(P_plot_values)} steps)"
            else:
                pressure_info = f"P={P_default:.1f} atm"
            self.statusBar.showMessage(f"Table and Plot updated ({pressure_info}). Data ready for export.")

        except Exception as e:
            self.statusBar.showMessage(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()

    def _get_plot_pressure_settings(self):
        """Get pressure settings for plotting"""
        try:
            P_min_plot = float(self.pressure_min_entry.text())
            P_max_plot = float(self.pressure_max_entry.text())
            P_log_steps = int(self.pressure_log_steps_entry.text())

            P_default = P_min_plot
            use_pressure_range_plotting = False
            P_plot_values = [P_default]

            if P_min_plot > 0 and P_max_plot >= P_min_plot and P_log_steps > 0:
                use_pressure_range_plotting = True
                if P_log_steps == 1:
                    P_plot_values = np.array([P_min_plot])
                elif P_min_plot == P_max_plot:
                    P_plot_values = np.array([P_min_plot])
                    use_pressure_range_plotting = True
                else:
                    P_plot_values = np.logspace(np.log10(P_min_plot), np.log10(P_max_plot), num=P_log_steps)
                    
            return P_plot_values, use_pressure_range_plotting, P_default
            
        except (ValueError, ZeroDivisionError):
            return [1.0], False, 1.0

    def _get_reaction_style(self, reaction_index, pressure_index=0):
        """Get color and linestyle for reaction plotting based on grouping"""
        try:
            group_count = max(1, int(self.group_count_entry.text()))
        except ValueError:
            group_count = 1

        # Available line styles
        line_styles = ['-', '--', '-.', ':']

        # Get colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        total_colors = len(colors)

        # Calculate group and position within group
        if hasattr(self, 'parsed_reactions') and self.parsed_reactions:
            total_reactions = len(self.parsed_reactions)
            reactions_per_group = max(1, total_reactions // group_count)
            group_index = reaction_index // reactions_per_group
            position_in_group = reaction_index % reactions_per_group
        else:
            group_index = 0
            position_in_group = reaction_index

        # 功能1: 检查是否启用"组内变化线型"模式
        intra_group_linestyle_mode = self.intra_group_linestyle_checkbox.isChecked()

        if intra_group_linestyle_mode:
            # 组内变化线型，组间变化颜色
            linestyle = line_styles[position_in_group % len(line_styles)]
            color_index = group_index % total_colors
        else:
            # 默认模式：组内变化颜色，组间变化线型
            linestyle = line_styles[group_index % len(line_styles)]
            # Select color based on position in group and pressure index
            if pressure_index > 0:
                base_color_index = position_in_group % total_colors
                color_index = (base_color_index + pressure_index) % total_colors
            else:
                color_index = position_in_group % total_colors

        color = colors[color_index]

        return color, linestyle

    def _get_reaction_color(self, reaction_index, pressure_index=0):
        """Get color for reaction plotting (backward compatibility)"""
        color, _ = self._get_reaction_style(reaction_index, pressure_index)
        return color

    def _on_style_mode_changed(self, state):
        """Handle style mode checkbox change"""
        if hasattr(self, 'parsed_reactions') and self.parsed_reactions:
            self._update_plot_with_reverse_rates()
            mode = "组内变化线型" if self.intra_group_linestyle_checkbox.isChecked() else "组内变化颜色"
            self.statusBar.showMessage(f"样式模式已切换为: {mode}")

    def _on_x_axis_mode_changed(self, state):
        """Handle x-axis mode checkbox change"""
        if hasattr(self, 'parsed_reactions') and self.parsed_reactions:
            self._update_plot_with_reverse_rates()
            mode = "1000/T" if self.use_inverse_temp_checkbox.isChecked() else "温度T"
            self.statusBar.showMessage(f"横坐标已切换为: {mode}")

    def _on_view_detail_button_clicked(self):
        """Show detailed rate constant data dialog"""
        if not hasattr(self, 'parsed_reactions') or not self.parsed_reactions:
            QMessageBox.warning(self, "无数据", "请先解析反应机理并计算速率常数。")
            return

        try:
            T_min = float(self.temp_min_entry.text())
            T_max = float(self.temp_max_entry.text())
            if not (T_max > T_min and T_min > 0):
                QMessageBox.warning(self, "参数错误", "温度范围无效，请检查最小和最大温度设置。")
                return
        except ValueError:
            QMessageBox.warning(self, "参数错误", "温度值必须是数字。")
            return

        # 创建并显示详细数据对话框
        dialog = RateConstantDetailDialog(
            self.parsed_reactions,
            T_min,
            T_max,
            self,
            self._calculate_rate_for_reaction,
            self._calculate_reverse_rate_for_plot,
            self._get_plot_pressure_settings
        )
        dialog.exec()

    def _on_save_data_button_clicked(self):
        """Export data to Excel"""
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(self, "导出失败", "Excel导出功能需要安装pandas库。\n请运行: pip install pandas openpyxl")
            return
            
        if not hasattr(self, 'parsed_reactions') or not self.parsed_reactions:
            QMessageBox.warning(self, "导出失败", "没有可导出的数据。请先解析反应机理。")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存速率常数数据", 
            "rate_constants_data.xlsx", 
            "Excel files (*.xlsx);;All files (*.*)"
        )
        
        if not file_path:
            return

        try:
            self.statusBar.showMessage("正在导出数据到Excel...")
            QApplication.processEvents()
            
            # Simple export - just the table data
            table_data = []
            for row in range(self.rate_table_widget.rowCount()):
                row_data = {}
                for col in range(self.rate_table_widget.columnCount()):
                    header = self.rate_table_widget.horizontalHeaderItem(col).text()
                    item = self.rate_table_widget.item(row, col)
                    if item:
                        row_data[header] = item.text()
                    else:
                        widget = self.rate_table_widget.cellWidget(row, col)
                        if isinstance(widget, QCheckBox):
                            row_data[header] = "Yes" if widget.isChecked() else "No"
                        else:
                            row_data[header] = ""
                table_data.append(row_data)
            
            df = pd.DataFrame(table_data)
            df.to_excel(file_path, index=False)
            
            self.statusBar.showMessage(f"数据已成功导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"速率常数数据已成功导出到:\n{file_path}")
            
        except Exception as e:
            error_msg = f"导出Excel文件时发生错误: {str(e)}"
            self.statusBar.showMessage(error_msg)
            QMessageBox.critical(self, "导出失败", error_msg)

    def _on_exp_reverse_rate_button_clicked(self):
        """功能5: 打开实验逆反应速率转换对话框"""
        # 创建并显示对话框
        self._reload_thermo_data_if_changed()
        dialog = ExperimentalReverseRateDialog(
            self,
            self.thermo_data if hasattr(self, 'thermo_data') else {},
            self.thermo_filepath,
        )
        dialog.exec()


class ExperimentalReverseRateDialog(QDialog):
    """功能5: 实验逆反应速率转换对话框"""
    def __init__(self, parent=None, thermo_data=None, thermo_filepath=None):
        super().__init__(parent)
        self.thermo_data = thermo_data if thermo_data else {}
        self.thermo_filepath = thermo_filepath
        self._thermo_file_signature = (
            get_thermo_file_signature(thermo_filepath) if thermo_filepath else None
        )

        self.setWindowTitle("实验逆反应速率转换工具")
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)

        layout = QVBoxLayout(self)

        # 说明文本
        info_text = """
        <b>功能说明：</b><br>
        根据实验测得的离散温度点的逆反应速率常数，通过平衡常数关系计算正反应速率常数。<br>
        <b>计算公式：</b>kf(推算) = kr(实验) × Kc<br>
        <b>应用场景：</b>将实验测得的逆反应速率数据转换为正反应速率，用于与理论计算结果对比验证。
        """
        info_label = QLabel(info_text)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 输入区域
        input_group = QGroupBox("输入数据")
        input_layout = QFormLayout()

        # 反应式输入
        self.reaction_input = QLineEdit()
        self.reaction_input.setPlaceholderText("例如: H+O2<=>O+OH")
        input_layout.addRow("反应式:", self.reaction_input)

        # 温度输入
        self.temperature_input = QLineEdit()
        self.temperature_input.setPlaceholderText("例如: 500, 800, 1000, 1500, 2000 或 500 800 1000 1500 2000")
        input_layout.addRow("温度 (K):", self.temperature_input)

        # 逆反应速率输入
        self.kr_input = QLineEdit()
        self.kr_input.setPlaceholderText("例如: 1.23E+05, 4.56E+06, 8.90E+07, 2.34E+09, 5.67E+10")
        input_layout.addRow("逆反应速率常数 kr:", self.kr_input)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 计算按钮
        calc_button = QPushButton("计算")
        calc_button.clicked.connect(self._calculate)
        layout.addWidget(calc_button)

        # 结果表格
        result_group = QGroupBox("计算结果")
        result_layout = QVBoxLayout()

        self.result_table = QTableWidget()
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

        # 添加右键菜单和快捷键
        self.result_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.result_table.customContextMenuRequested.connect(self._show_context_menu)
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.result_table)
        copy_shortcut.activated.connect(self._copy_selection)

        result_layout.addWidget(self.result_table)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 按钮区域
        button_layout = QHBoxLayout()

        if PANDAS_AVAILABLE:
            export_button = QPushButton("导出到Excel")
            export_button.clicked.connect(self._export_to_excel)
            button_layout.addWidget(export_button)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _parse_input_values(self, input_text):
        """解析输入的数值（支持逗号或空格分隔）"""
        # 替换逗号为空格，然后分割
        values_str = input_text.replace(',', ' ').split()
        values = []
        for v in values_str:
            try:
                values.append(float(v))
            except ValueError:
                raise ValueError(f"无法解析数值: {v}")
        return values

    def _validate_reaction(self, reaction_str):
        """验证反应式格式"""
        # 简单验证：检查是否包含<=>或=>
        if '<=>' not in reaction_str and '=>' not in reaction_str:
            raise ValueError("反应式格式错误，必须包含 <=> 或 =>")
        return reaction_str.strip()

    def _calculate(self):
        """执行计算"""
        try:
            # 验证和解析输入
            self._reload_thermo_data_if_changed()
            reaction_str = self._validate_reaction(self.reaction_input.text())
            temperatures = self._parse_input_values(self.temperature_input.text())
            kr_values = self._parse_input_values(self.kr_input.text())

            if len(temperatures) == 0:
                QMessageBox.warning(self, "输入错误", "请输入温度数据")
                return

            if len(kr_values) == 0:
                QMessageBox.warning(self, "输入错误", "请输入逆反应速率常数数据")
                return

            if len(temperatures) != len(kr_values):
                QMessageBox.warning(self, "输入错误",
                    f"温度点数量({len(temperatures)})与逆反应速率常数数量({len(kr_values)})不一致")
                return

            # 解析反应式
            from chemkin_parser import parse_chemkin_mechanism
            mechanism_text = f"UNITS CAL/MOLE\n{reaction_str}  1.0E+00  0.0  0.0"
            parsed = parse_chemkin_mechanism(mechanism_text)

            if not parsed or len(parsed) == 0:
                QMessageBox.warning(self, "解析错误", "无法解析反应式，请检查格式")
                return

            reaction_data = parsed[0]

            # 计算结果
            results = []
            for T, kr in zip(temperatures, kr_values):
                result = self._calculate_for_temperature(reaction_data, T, kr)
                results.append(result)

            # 显示结果
            self._display_results(results)

        except ValueError as e:
            QMessageBox.warning(self, "输入错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算过程中发生错误:\n{str(e)}")

    def _reload_thermo_data_if_changed(self):
        """Use externally saved persistent thermo entries on the next calculation."""
        if not self.thermo_filepath:
            return False
        current_signature = get_thermo_file_signature(self.thermo_filepath)
        if current_signature == self._thermo_file_signature:
            return False
        self.thermo_data = read_nasa_polynomials(self.thermo_filepath)
        self._thermo_file_signature = current_signature
        return True

    def _calculate_for_temperature(self, reaction_data, T, kr_exp):
        """计算单个温度点的结果"""
        result = {
            'T': T,
            'kr_exp': kr_exp,
            'Kc': None,
            'kf_calc': None,
            'error': None
        }

        try:
            # 计算热力学性质
            delta_H, delta_S, delta_G, missing_at_T, err_thermo_calc = \
                get_reaction_thermo_properties(reaction_data, T, self.thermo_data)

            if err_thermo_calc:
                result['error'] = "热力学数据缺失或计算错误"
                return result

            # 计算平衡常数
            Kp = calculate_equilibrium_constant_kp(delta_G, T)
            delta_n = calculate_delta_n_gas(reaction_data, self.thermo_data)
            Kc = calculate_equilibrium_constant_kc(Kp, T, delta_n)

            if Kc is None or Kc <= 0:
                result['error'] = "平衡常数计算失败"
                return result

            result['Kc'] = Kc

            # 计算正反应速率常数: kf = kr × Kc
            kf_calc = kr_exp * Kc
            result['kf_calc'] = kf_calc

        except Exception as e:
            result['error'] = str(e)

        return result

    def _display_results(self, results):
        """显示计算结果"""
        # 设置表格
        self.result_table.setRowCount(len(results))
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels([
            "温度 (K)",
            "kr(实验)",
            "Kc",
            "kf(推算)"
        ])

        # 填充数据
        for row_idx, result in enumerate(results):
            # 温度
            temp_item = QTableWidgetItem(f"{result['T']:.1f}")
            temp_item.setFlags(temp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.result_table.setItem(row_idx, 0, temp_item)

            # kr(实验)
            kr_item = QTableWidgetItem(f"{result['kr_exp']:.3E}")
            kr_item.setFlags(kr_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.result_table.setItem(row_idx, 1, kr_item)

            # Kc
            if result['Kc'] is not None:
                kc_str = f"{result['Kc']:.3E}"
            else:
                kc_str = "N/A"
            kc_item = QTableWidgetItem(kc_str)
            kc_item.setFlags(kc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.result_table.setItem(row_idx, 2, kc_item)

            # kf(推算)
            if result['kf_calc'] is not None:
                kf_str = f"{result['kf_calc']:.3E}"
            else:
                kf_str = "N/A"
            kf_item = QTableWidgetItem(kf_str)
            kf_item.setFlags(kf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.result_table.setItem(row_idx, 3, kf_item)

        # 调整列宽
        self.result_table.resizeColumnsToContents()

    def _show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu(self)

        copy_action = QAction("复制", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self._copy_selection)
        menu.addAction(copy_action)

        select_all_action = QAction("全选", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.result_table.selectAll)
        menu.addAction(select_all_action)

        menu.exec(self.result_table.viewport().mapToGlobal(position))

    def _copy_selection(self):
        """复制选中的表格数据到剪贴板（包含表头）"""
        selection = self.result_table.selectedRanges()
        if not selection:
            return

        # 获取选中区域的边界
        min_row = min(r.topRow() for r in selection)
        max_row = max(r.bottomRow() for r in selection)
        min_col = min(r.leftColumn() for r in selection)
        max_col = max(r.rightColumn() for r in selection)

        # 构建复制内容（包含表头）
        copied_data = []

        # 添加表头行
        header_row = []
        for col in range(min_col, max_col + 1):
            header_item = self.result_table.horizontalHeaderItem(col)
            header_text = header_item.text() if header_item else ""
            header_row.append(header_text)
        copied_data.append('\t'.join(header_row))

        # 添加数据行
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                item = self.result_table.item(row, col)
                cell_text = item.text() if item else ""
                row_data.append(cell_text)
            copied_data.append('\t'.join(row_data))

        # 复制到剪贴板
        clipboard_text = '\n'.join(copied_data)
        QApplication.clipboard().setText(clipboard_text)

    def _export_to_excel(self):
        """导出结果到Excel"""
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(self, "导出失败", "需要安装pandas库才能导出Excel文件。")
            return

        if self.result_table.rowCount() == 0:
            QMessageBox.warning(self, "导出失败", "没有可导出的数据，请先进行计算。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存实验逆反应速率转换结果",
            "experimental_reverse_rate_conversion.xlsx",
            "Excel files (*.xlsx);;All files (*.*)"
        )

        if not file_path:
            return

        try:
            # 提取表格数据
            table_data = []
            for row in range(self.result_table.rowCount()):
                row_data = {}
                for col in range(self.result_table.columnCount()):
                    header = self.result_table.horizontalHeaderItem(col).text()
                    item = self.result_table.item(row, col)
                    row_data[header] = item.text() if item else ""
                table_data.append(row_data)

            df = pd.DataFrame(table_data)
            df.to_excel(file_path, index=False)

            QMessageBox.information(self, "导出成功", f"数据已成功导出到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出Excel文件时发生错误:\n{str(e)}")


class RateConstantDetailDialog(QDialog):
    """Dialog to show detailed rate constant data for all reactions"""
    def __init__(self, parsed_reactions, T_min, T_max, parent=None,
                 calc_rate_func=None, calc_reverse_func=None, get_pressure_func=None):
        super().__init__(parent)
        self.parsed_reactions = parsed_reactions
        self.T_min = T_min
        self.T_max = T_max
        self.calc_rate_func = calc_rate_func
        self.calc_reverse_func = calc_reverse_func
        self.get_pressure_func = get_pressure_func

        self.setWindowTitle("速率常数详细数据")
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)

        layout = QVBoxLayout(self)

        # 说明文本
        info_text = f"<b>温度范围：</b>{T_min:.0f} K - {T_max:.0f} K，间隔：10 K<br>"
        info_text += f"<b>数据点数：</b>{int((T_max - T_min) / 10) + 1} 个温度点"
        info_label = QLabel(info_text)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(info_label)

        # 创建表格
        self.detail_table = QTableWidget()
        self.detail_table.setAlternatingRowColors(True)
        self.detail_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

        # 功能4: 设置右键菜单
        self.detail_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.detail_table.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.detail_table)

        # 功能4: 添加Ctrl+C快捷键支持
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.detail_table)
        copy_shortcut.activated.connect(self._copy_selection)

        # 填充数据
        self._populate_table()

        # 导出按钮
        button_layout = QHBoxLayout()

        if PANDAS_AVAILABLE:
            export_button = QPushButton("导出到Excel")
            export_button.clicked.connect(self._export_to_excel)
            button_layout.addWidget(export_button)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _populate_table(self):
        """Populate the table with rate constant data"""
        # 生成温度点（每10K一个点）
        T_values = np.arange(self.T_min, self.T_max + 10, 10)
        num_temps = len(T_values)

        # 获取压力设置
        if self.get_pressure_func:
            P_plot_values, use_pressure_range_plotting, P_default = self.get_pressure_func()
        else:
            P_default = 1.0

        # 准备列：温度 + 每个反应
        columns = ["温度 (K)"]
        reaction_pressure_pairs = []

        for reaction_data in self.parsed_reactions:
            reaction_label = reaction_data.get('equation_string_cleaned',
                                              reaction_data.get('equation_string', 'N/A'))
            reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')
            is_calculating_reverse = reaction_data.get('is_calculating_reverse', False)

            # 对于PLOG/TROE反应，使用默认压力
            pressure = P_default

            # 添加列标题
            prefix = "kr" if is_calculating_reverse else "kf"
            if reaction_type in ['PLOG', 'TROE']:
                col_name = f"{prefix}: {reaction_label} @ {pressure:.1f} atm"
            else:
                col_name = f"{prefix}: {reaction_label}"

            columns.append(col_name)
            reaction_pressure_pairs.append((reaction_data, pressure, is_calculating_reverse))

        # 设置表格大小
        self.detail_table.setRowCount(num_temps)
        self.detail_table.setColumnCount(len(columns))
        self.detail_table.setHorizontalHeaderLabels(columns)

        # 填充数据
        for row_idx, T in enumerate(T_values):
            # 第一列：温度
            temp_item = QTableWidgetItem(f"{T:.0f}")
            temp_item.setFlags(temp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.detail_table.setItem(row_idx, 0, temp_item)

            # 其余列：速率常数
            for col_idx, (reaction_data, pressure, is_reverse) in enumerate(reaction_pressure_pairs, start=1):
                if is_reverse and self.calc_reverse_func:
                    k = self.calc_reverse_func(reaction_data, T, pressure)
                elif self.calc_rate_func:
                    k = self.calc_rate_func(reaction_data, T, pressure)
                else:
                    k = None

                if k is not None and k > 0:
                    k_str = f"{k:.3E}"
                else:
                    k_str = "N/A"

                k_item = QTableWidgetItem(k_str)
                k_item.setFlags(k_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.detail_table.setItem(row_idx, col_idx, k_item)

        # 调整列宽
        self.detail_table.resizeColumnsToContents()
        header = self.detail_table.horizontalHeader()
        header.setStretchLastSection(False)

    def _show_context_menu(self, position):
        """功能4: 显示右键菜单"""
        menu = QMenu(self)

        copy_action = QAction("复制", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self._copy_selection)
        menu.addAction(copy_action)

        select_all_action = QAction("全选", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.detail_table.selectAll)
        menu.addAction(select_all_action)

        menu.exec(self.detail_table.viewport().mapToGlobal(position))

    def _copy_selection(self):
        """功能4: 复制选中的表格数据到剪贴板（包含表头）"""
        selection = self.detail_table.selectedRanges()
        if not selection:
            return

        # 获取选中区域的边界
        min_row = min(r.topRow() for r in selection)
        max_row = max(r.bottomRow() for r in selection)
        min_col = min(r.leftColumn() for r in selection)
        max_col = max(r.rightColumn() for r in selection)

        # 构建复制内容（包含表头）
        copied_data = []

        # 添加表头行
        header_row = []
        for col in range(min_col, max_col + 1):
            header_item = self.detail_table.horizontalHeaderItem(col)
            header_text = header_item.text() if header_item else ""
            header_row.append(header_text)
        copied_data.append('\t'.join(header_row))

        # 添加数据行
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                item = self.detail_table.item(row, col)
                cell_text = item.text() if item else ""
                row_data.append(cell_text)
            copied_data.append('\t'.join(row_data))

        # 复制到剪贴板
        clipboard_text = '\n'.join(copied_data)
        QApplication.clipboard().setText(clipboard_text)

    def _export_to_excel(self):
        """Export table data to Excel"""
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(self, "导出失败", "需要安装pandas库才能导出Excel文件。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存详细速率常数数据",
            "rate_constants_detail.xlsx",
            "Excel files (*.xlsx);;All files (*.*)"
        )

        if not file_path:
            return

        try:
            # 提取表格数据
            table_data = []
            for row in range(self.detail_table.rowCount()):
                row_data = {}
                for col in range(self.detail_table.columnCount()):
                    header = self.detail_table.horizontalHeaderItem(col).text()
                    item = self.detail_table.item(row, col)
                    row_data[header] = item.text() if item else ""
                table_data.append(row_data)

            df = pd.DataFrame(table_data)
            df.to_excel(file_path, index=False)

            QMessageBox.information(self, "导出成功", f"数据已成功导出到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出Excel文件时发生错误:\n{str(e)}")


def validate_nasa7_format(input_text):
    """
    验证NASA 7系数格式的输入数据
    只进行基本检查，尝试解析基本结构

    返回: (is_valid, error_messages, species_data_dict)
    - is_valid: bool, 是否验证通过
    - error_messages: list of str, 错误信息列表
    - species_data_dict: dict, 解析出的物种数据 {species_name: [line1, line2, line3, line4]}
    """
    errors = []
    species_data = {}

    lines = input_text.strip().split('\n')

    # 基本检查：输入不为空
    if len(lines) == 0:
        errors.append("❌ 输入为空，请输入NASA 7系数数据")
        return False, errors, species_data

    # 基本检查：行数是4的倍数
    if len(lines) % 4 != 0:
        errors.append(f"❌ 数据格式错误：输入了 {len(lines)} 行，NASA 7系数数据必须是4行的倍数（每个物种4行）")
        return False, errors, species_data

    # 尝试解析物种数据
    num_species = len(lines) // 4

    for species_idx in range(num_species):
        start_line = species_idx * 4
        species_lines = lines[start_line:start_line + 4]
        species_line_numbers = [start_line + i + 1 for i in range(4)]  # 1-based line numbers

        try:
            # 尝试提取物种名称（第1行前18个字符）
            species_name = species_lines[0][0:18].strip()
            if not species_name:
                errors.append(f"❌ 第 {species_line_numbers[0]} 行：无法提取物种名称（第1行前18个字符应包含物种名称）")
                return False, errors, {}

            # 尝试解析温度值（基本验证）
            line1 = species_lines[0]

            # 使用正则表达式提取温度值，更灵活地处理不同格式
            import re
            temp_pattern = r'[\d.]+(?:[eE][+-]?\d+)?'
            temp_matches = re.findall(temp_pattern, line1)

            if len(temp_matches) < 2:
                errors.append(f"❌ 第 {species_line_numbers[0]} 行：无法解析温度值（应至少包含2个温度值）")
                return False, errors, {}

            # 尝试转换温度值为浮点数
            try:
                float(temp_matches[0])
                float(temp_matches[1])
            except ValueError:
                errors.append(f"❌ 第 {species_line_numbers[0]} 行：温度值格式错误（应为浮点数）")
                return False, errors, {}

            # 尝试解析系数（基本验证）
            for i in range(1, 4):
                line = species_lines[i]
                if not line.strip():
                    errors.append(f"❌ 第 {species_line_numbers[i]} 行：行为空（应包含系数数据）")
                    return False, errors, {}

                # 尝试提取第一个系数
                coeff_pattern = r'[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.\d*|[+-]?\d+'
                coeff_matches = re.findall(coeff_pattern, line)

                if not coeff_matches:
                    errors.append(f"❌ 第 {species_line_numbers[i]} 行：无法解析系数（应包含浮点数）")
                    return False, errors, {}

                # 尝试转换第一个系数为浮点数
                try:
                    float(coeff_matches[0])
                except ValueError:
                    errors.append(f"❌ 第 {species_line_numbers[i]} 行：系数格式错误（应为浮点数）")
                    return False, errors, {}

            # 如果解析成功，将相态转换为大写并保存数据
            # 第1行的相态需要转换为大写
            line1 = species_lines[0]

            # 相态转换：找到相态字符（通常是单个字母G/g, L/l, S/s）并转换为大写
            # 相态应该在元素组成之后，温度值之前
            # 使用正则表达式找到相态：在数字之后，空格和数字之前
            line1_converted = re.sub(
                r'(\d)([a-z])(\s+\d)',  # 匹配：数字 + 小写字母 + 空格 + 数字
                lambda m: m.group(1) + m.group(2).upper() + m.group(3),
                line1
            )

            # 如果没有找到小写相态，尝试大写相态（保持不变）
            # 这样可以确保大写相态也被正确处理

            species_lines_converted = [line1_converted] + species_lines[1:]
            species_data[species_name.upper()] = species_lines_converted

        except Exception as e:
            errors.append(f"❌ 第 {species_line_numbers[0]} 行（物种 {species_name if 'species_name' in locals() else 'UNKNOWN'}）：解析失败 - {str(e)}")
            return False, errors, {}

    # 所有物种都解析成功
    return True, errors, species_data


class NasaPolynomialInputDialog(QDialog):
    def __init__(self, missing_species_list, thermo_filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入缺失的热力学数据")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.missing_species_list = missing_species_list
        self.thermo_filepath = thermo_filepath
        self.validated_data = {}

        layout = QVBoxLayout(self)

        # Important notice about therm.dat
        notice_text = """
<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
<b>重要提示：</b><br>
• 程序需要读取 <code>therm.dat</code> 文件来获取热力学数据<br>
• 用户新增数据永久保存在 <code>%LOCALAPPDATA%/CHEMKIN_RateViewer/therm.dat</code><br>
• 您可以直接编辑 <code>therm.dat</code> 文件添加缺失的物种数据<br>
• 或者在下方文本框中输入NASA 7系数数据，程序将验证格式并自动添加到文件中
</div>
        """
        notice_label = QLabel(notice_text)
        notice_label.setWordWrap(True)
        notice_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(notice_label)

        instruction_text = (
            f"以下物种缺少NASA 7系数数据: <b>{', '.join(missing_species_list)}</b><br><br>"
            "请在下方输入完整的NASA 7系数数据。每个物种必须包含4行数据（1行标题行 + 3行系数行）。"
        )
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.instruction_label)

        # 错误显示区域
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("""
            QLabel {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 10px;
                color: #721c24;
            }
        """)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        self.polynomial_input_text_edit = QTextEdit()
        # Set a monospaced font for easier editing of fixed-format text
        font = QFont("Courier New")
        font.setPointSize(9)
        self.polynomial_input_text_edit.setFont(font)

        example_text = """示例格式（每个物种必须是4行）：

CH4                 120186C   1H   4          G   300.000  5000.000 1084.126     1
 7.48514950E-02 1.33909467E-02-5.73285809E-06 1.22292535E-09-1.01815230E-13    2
-9.46834459E+03 1.84373180E+01 5.14987613E+00-1.36709788E-02 4.91800599E-05    3
-4.84743026E-08 1.66693956E-11-1.02466476E+04-4.64130376E+00                   4

提示：
• 每个物种必须是4行数据
• 可以从 therm.dat 文件中复制类似物种的数据作为参考
• 程序会自动验证数据格式是否正确"""

        self.polynomial_input_text_edit.setPlaceholderText(example_text)
        layout.addWidget(self.polynomial_input_text_edit)

        # Standard buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def validate_and_accept(self):
        """验证输入并接受对话框"""
        input_text = self.polynomial_input_text_edit.toPlainText()

        if not input_text.strip():
            self.show_error("请输入NASA 7系数数据")
            return

        # 验证格式
        is_valid, errors, species_data = validate_nasa7_format(input_text)

        if not is_valid:
            # 显示详细错误信息
            error_html = "<b>输入数据格式验证失败：</b><br><br>"
            error_html += "<br>".join(errors)
            error_html += "<br><br><b>请修正上述错误后重试。</b>"
            self.show_error(error_html)
            return

        # 验证是否包含所有缺失的物种
        input_species = set(species_data.keys())
        missing_species_upper = set(s.upper() for s in self.missing_species_list)

        if not missing_species_upper.issubset(input_species):
            still_missing = missing_species_upper - input_species
            self.show_error(
                f"<b>缺少以下物种的数据：</b><br>"
                f"{', '.join(sorted(still_missing))}<br><br>"
                f"请确保为所有缺失的物种提供数据。"
            )
            return

        # 验证通过，保存数据
        self.validated_data = species_data
        self.error_label.setVisible(False)

        # 从转换后的 species_data 重新构建文本
        # 这样可以确保相态等已被转换为大写
        converted_text = ""
        for species_name, lines in species_data.items():
            converted_text += "\n".join(lines) + "\n"

        # 使用 append_nasa_polynomial_from_string 函数写入文件
        # 这样可以确保新数据插入到 END 关键字之前
        try:
            success, message = append_nasa_polynomial_from_string(self.thermo_filepath, converted_text)

            if success:
                QMessageBox.information(
                    self,
                    "数据已保存",
                    f"已成功将 {len(species_data)} 个物种的热力学数据添加到 {self.thermo_filepath}\n\n"
                    "程序将重新加载热力学数据。"
                )
                super().accept()
            else:
                self.show_error(f"<b>保存数据失败：</b><br>{message}")
        except Exception as e:
            self.show_error(f"<b>保存数据失败：</b><br>{str(e)}")

    def show_error(self, error_html):
        """显示错误信息"""
        self.error_label.setText(error_html)
        self.error_label.setVisible(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 
