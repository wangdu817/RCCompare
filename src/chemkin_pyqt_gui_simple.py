import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit, 
    QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, 
    QTableWidget, QTableWidgetItem, QStatusBar, QSplitter, QGroupBox,
    QHeaderView, QCheckBox, QDialog, QDialogButtonBox, QSizePolicy,
    QScrollArea, QMenuBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QAction
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thermo_filepath = "therm.dat"
        self.setWindowTitle("CHEMKIN Rate Viewer (PyQt) - v1.0")
        self.setGeometry(100, 100, 1200, 800)

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
        
        self.left_panel_layout.addWidget(self.controls_groupbox)

        # Buttons
        self.plot_button = QPushButton("Parse, Calculate & Plot")
        self.plot_button.clicked.connect(self._on_plot_button_clicked)
        self.left_panel_layout.addWidget(self.plot_button)
        
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
        self._load_thermo_data()
        self.statusBar.showMessage("Ready")

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
        QMessageBox.about(self, "关于", "CHEMKIN Rate Viewer v1.0\n\n化学反应动力学机理分析工具\n\n开发者：王杜\n单位：中科院工程热物理研究所")

    def _setup_rate_table_widget_appearance(self):
        header = self.rate_table_widget.horizontalHeader()
        
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.rate_table_widget.setColumnWidth(0, 300)  # Set initial width for Reaction column
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.rate_table_widget.setColumnWidth(1, 80)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.rate_table_widget.setColumnWidth(2, 70)

        num_temp_points = len(self.table_temperatures)
        for i in range(num_temp_points):
            col_index = 3 + i
            header.setSectionResizeMode(col_index, QHeaderView.ResizeMode.Interactive)
            self.rate_table_widget.setColumnWidth(col_index, 120)

        self.rate_table_widget.setAlternatingRowColors(True)
        self.rate_table_widget.verticalHeader().setVisible(False)
        self.rate_table_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        header.setStretchLastSection(False)

    def _load_thermo_data(self):
        if read_nasa_polynomials is None:
            self.statusBar.showMessage("Error: Thermo parser not available. Reverse rates disabled.")
            self.thermo_data = {}
            return

        try:
            self.thermo_data = read_nasa_polynomials(self.thermo_filepath)
            if not self.thermo_data:
                self.statusBar.showMessage(f"Warning: No data read from {self.thermo_filepath}.")
                self.thermo_data = {}
            else:
                self.statusBar.showMessage(f"Successfully loaded {len(self.thermo_data)} species from {self.thermo_filepath}.")
        except FileNotFoundError:
            self.statusBar.showMessage(f"Error: {self.thermo_filepath} not found. Reverse rates disabled.")
            self.thermo_data = {}
        except Exception as e:
            self.statusBar.showMessage(f"Error parsing {self.thermo_filepath}: {e}. Reverse rates disabled.")
            self.thermo_data = {}

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

                # Column 2: Checkbox
                if reaction_type in ['PLOG', 'TROE'] and len(pressures_to_use) > 1 and pressure_index > 0:
                    empty_item = QTableWidgetItem("")
                    empty_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self.rate_table_widget.setItem(current_row, 2, empty_item)
                else:
                    checkbox = QCheckBox()
                    checkbox.stateChanged.connect(
                        lambda state, rd=reaction_data, ri=reaction_index: self._on_calc_reverse_checkbox_changed(state, rd, ri)
                    )
                    self.rate_table_widget.setCellWidget(current_row, 2, checkbox)
                    
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
                # 热力学数据输入对话框逻辑
                dialog = NasaPolynomialInputDialog(sorted(list(current_missing_species)), self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    entered_text = dialog.polynomial_input_text_edit.toPlainText()
                    if entered_text:
                        is_valid, _, validation_errors = is_valid_nasa_polynomial_string(entered_text)
                        if is_valid:
                            append_success, append_msg = append_nasa_polynomial_from_string(self.thermo_filepath, entered_text)
                            if append_success:
                                self._load_thermo_data()
                                # 重新检查数据
                                still_missing_after_update = set()
                                for coeff, spec_name in all_species_in_reaction:
                                    if not self.thermo_data.get(spec_name.upper()): 
                                        still_missing_after_update.add(spec_name)
                                if still_missing_after_update:
                                    self.statusBar.showMessage(f"Error: Thermo for {', '.join(still_missing_after_update)} still missing after update.")
                                    main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                                    if main_checkbox: main_checkbox.setChecked(False)
                                    return
                            else:
                                self.statusBar.showMessage(f"Error appending thermo: {append_msg}")
                                main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                                if main_checkbox: main_checkbox.setChecked(False)
                                return
                        else:
                            self.statusBar.showMessage("Invalid thermo input: " + "; ".join(validation_errors))
                            main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                            if main_checkbox: main_checkbox.setChecked(False)
                            return
                    else:
                        self.statusBar.showMessage("No thermo data submitted for missing species.")
                        main_checkbox = self._get_main_checkbox_for_reaction(reaction_data)
                        if main_checkbox: main_checkbox.setChecked(False)
                        return
                else:
                    self.statusBar.showMessage("Thermo input cancelled. Cannot calculate reverse rates.")
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
        """获取反应的主复选框（第一行）"""
        for row_info in reaction_data.get('table_rows', []):
            if row_info.get('is_main_row', False):
                return self.rate_table_widget.cellWidget(row_info['row'], 2)
        return None

    def _calculate_reverse_rate_for_plot(self, reaction_data, T, P_atm):
        """为绘图计算逆反应速率常数"""
        try:
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
                        self.plot_axes.plot(T_values, k_values_log10, label=label, color=color, linestyle=linestyle)
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
                    self.plot_axes.plot(T_values, k_values_log10, label=label, color=color, linestyle=linestyle)
                    num_lines_plotted += 1
        
        self.plot_axes.set_xlabel("Temperature (K)")
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
        
        # Select line style based on group
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


class NasaPolynomialInputDialog(QDialog):
    def __init__(self, missing_species_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入缺失的热力学数据")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)

        layout = QVBoxLayout(self)

        # Important notice about therm.dat
        notice_text = """
<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
<b>重要提示：</b><br>
• 程序需要读取 <code>therm.dat</code> 文件来获取热力学数据<br>
• 请确保 <code>therm.dat</code> 文件位于程序运行目录下<br>
• 您可以直接编辑 <code>therm.dat</code> 文件添加缺失的物种数据<br>
• 或者在下方文本框中输入NASA多项式数据，程序将自动添加到文件中
</div>
        """
        notice_label = QLabel(notice_text)
        notice_label.setWordWrap(True)
        notice_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(notice_label)

        instruction_text = (
            f"以下物种缺少NASA多项式数据: <b>{', '.join(missing_species_list)}</b><br><br>"
            "请在下方输入完整的NASA 7系数多项式数据。每个物种通常包含4行数据（1行标题行和3行系数行）。"
        )
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.instruction_label)

        self.polynomial_input_text_edit = QTextEdit()
        # Set a monospaced font for easier editing of fixed-format text
        font = QFont("Courier New")
        font.setPointSize(10)
        self.polynomial_input_text_edit.setFont(font)
        self.polynomial_input_text_edit.setPlaceholderText(
            "示例格式（每个物种4行）：\n"
            "SPECIES_NAME      DATE    EL1 C1 EL2 C2 ... G Tlow Thigh Tcommon     1\n"
            " A1_HIGH A2_HIGH A3_HIGH A4_HIGH A5_HIGH                             2\n"
            " A6_HIGH A7_HIGH A1_LOW  A2_LOW  A3_LOW                              3\n"
            " A4_LOW  A5_LOW  A6_LOW  A7_LOW                                      4\n\n"
            "注意：请严格按照NASA多项式的固定格式输入数据"
        )
        layout.addWidget(self.polynomial_input_text_edit)

        # Standard buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 