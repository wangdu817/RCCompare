import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit, 
    QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, 
    QTableWidget, QTableWidgetItem, QStatusBar, QSplitter, QGroupBox,
    QHeaderView 
)
from PyQt6.QtCore import Qt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Assuming chemkin_parser.py and rate_calculator.py are in the same directory or accessible in PYTHONPATH
try:
    from chemkin_parser import parse_chemkin_mechanism
    from rate_calculator import (
        calculate_arrhenius_rate,
        calculate_plog_rate,
        calculate_troe_rate,
        get_third_body_concentration,
        R_cal 
    )
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure chemkin_parser.py and rate_calculator.py are accessible.")
    parse_chemkin_mechanism = None
    calculate_arrhenius_rate = None
    calculate_plog_rate = None
    calculate_troe_rate = None
    get_third_body_concentration = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CHEMKIN Rate Viewer (PyQt)")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.main_splitter)

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
! PLOG example
CH4 + O2 = CH3 + HO2      0 0 0 PLOG /
    1.0   1.0E13  0.0  50000.0 / 
    10.0  1.0E14  0.0  55000.0 / 
! TROE example
H + O2 (+M) = HO2 (+M)    1.0E12 0.5 0.0   1.0E16 0.0 -1000.0 TROE / 0.6 100.0 1000.0 50.0 / 
        """
        self.chemkin_input_text_edit.setText(sample_chemkin_input.strip())
        self.left_panel_layout.addWidget(self.chemkin_input_text_edit, 1)

        self.controls_groupbox = QGroupBox("Controls")
        self.controls_layout = QFormLayout(self.controls_groupbox)

        self.temp_min_entry = QLineEdit("300")
        self.controls_layout.addRow("Min Temperature (K):", self.temp_min_entry)
        self.temp_max_entry = QLineEdit("2500")
        self.controls_layout.addRow("Max Temperature (K):", self.temp_max_entry)
        self.pressure_entry = QLineEdit("1.0")
        self.controls_layout.addRow("Pressure (atm):", self.pressure_entry)
        
        self.left_panel_layout.addWidget(self.controls_groupbox)

        self.plot_button = QPushButton("Parse, Calculate & Plot")
        self.plot_button.setObjectName("plot_button")
        self.plot_button.clicked.connect(self._on_plot_button_clicked)
        self.left_panel_layout.addWidget(self.plot_button)
        
        self.main_splitter.addWidget(self.left_panel_widget)

        self.right_panel_widget = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_panel_widget)
        self.right_panel_splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot Area Setup
        self.plot_area_widget = QWidget() # This is the QWidget for the plot
        self.plot_area_widget.setObjectName("plot_widget_placeholder") # Keep old name if referred
        plot_layout = QVBoxLayout(self.plot_area_widget) # Layout for the plot_area_widget
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure_canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.figure_canvas)
        self.plot_axes = self.figure.add_subplot(111)
        # self.plot_area_widget.setLayout(plot_layout) # Already set by passing to QVBoxLayout constructor
        self.right_panel_splitter.addWidget(self.plot_area_widget)


        self.rate_table_widget = QTableWidget()
        self.rate_table_widget.setObjectName("rate_table_widget")
        self.rate_table_widget.setColumnCount(5)
        self.rate_table_widget.setHorizontalHeaderLabels(["Reaction", "k @ 500K", "k @ 1000K", "k @ 1500K", "k @ 2000K"])
        self.rate_table_widget.horizontalHeader().setStretchLastSection(True)
        self.rate_table_widget.setAlternatingRowColors(True)
        self.rate_table_widget.setMinimumHeight(150)
        self.rate_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.right_panel_splitter.addWidget(self.rate_table_widget)
        self.right_panel_splitter.setSizes([400, 200])
        self.right_panel_layout.addWidget(self.right_panel_splitter)
        self.main_splitter.addWidget(self.right_panel_widget)
        self.main_splitter.setSizes([350, 850])

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def _calculate_rate_for_reaction(self, reaction_data, T, P_atm):
        k = None
        reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')
        if not (calculate_arrhenius_rate and calculate_plog_rate and calculate_troe_rate and get_third_body_concentration):
            self.statusBar.showMessage("Error: Rate calculation functions not loaded.")
            return None
        try:
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
        table_temperatures = [500.0, 1000.0, 1500.0, 2000.0]
        try:
            P_atm = float(self.pressure_entry.text())
            if P_atm <= 0: self.statusBar.showMessage("Error: Pressure must be positive."); return
        except ValueError: self.statusBar.showMessage("Error: Invalid pressure value."); return

        for row, reaction_data in enumerate(parsed_reactions):
            self.rate_table_widget.insertRow(row)
            eq_display_str = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
            eq_item = QTableWidgetItem(eq_display_str); eq_item.setFlags(eq_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.rate_table_widget.setItem(row, 0, eq_item)
            for col_idx, T_table in enumerate(table_temperatures, start=1):
                k_val = self._calculate_rate_for_reaction(reaction_data, T_table, P_atm)
                k_str = "%.3E" % k_val if k_val is not None and k_val > 0 else ("0.000E+00" if k_val == 0 else "N/A")
                k_item = QTableWidgetItem(k_str); k_item.setFlags(k_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.rate_table_widget.setItem(row, col_idx, k_item)
        self.rate_table_widget.resizeColumnsToContents()

    def _update_plot(self, parsed_reactions):
        self.plot_axes.clear()
        try:
            T_min = float(self.temp_min_entry.text())
            T_max = float(self.temp_max_entry.text())
            P_atm = float(self.pressure_entry.text())
            if not (T_max > T_min and T_min > 0 and P_atm >= 0): # P_atm can be 0 for some cases
                self.statusBar.showMessage("Error: Invalid T_min/T_max or Pressure. Ensure T_max > T_min > 0, P >= 0.")
                self.figure_canvas.draw() # Redraw cleared axes
                return
        except ValueError:
            self.statusBar.showMessage("Error: Temperature/Pressure values must be numeric.")
            self.figure_canvas.draw() # Redraw cleared axes
            return

        T_values = np.linspace(T_min, T_max, 100)
        num_lines_plotted = 0
        for reaction_data in parsed_reactions:
            k_values_log10 = []
            for T_val in T_values:
                k = self._calculate_rate_for_reaction(reaction_data, T_val, P_atm)
                if k is not None and k > 1e-100: # Avoid log10(0) or very small numbers
                    k_values_log10.append(np.log10(k))
                else:
                    k_values_log10.append(np.nan)
            
            if any(not np.isnan(val) for val in k_values_log10):
                self.plot_axes.plot(T_values, k_values_log10, label=reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A')))
                num_lines_plotted +=1

        self.plot_axes.set_xlabel("Temperature (K)")
        self.plot_axes.set_ylabel("log10(k)")
        self.plot_axes.set_title(f"Reaction Rates at {P_atm:.2f} atm")
        if num_lines_plotted > 0:
            self.plot_axes.legend(fontsize='small')
        self.plot_axes.grid(True)
        self.figure_canvas.draw()

    def _on_plot_button_clicked(self):
        chemkin_text = self.chemkin_input_text_edit.toPlainText()
        if not chemkin_text.strip():
            self.statusBar.showMessage("CHEMKIN input is empty.")
            self.rate_table_widget.setRowCount(0)
            self.plot_axes.clear(); self.figure_canvas.draw()
            return

        self.statusBar.showMessage("Parsing CHEMKIN input...")
        QApplication.processEvents() 

        try:
            if not parse_chemkin_mechanism:
                self.statusBar.showMessage("Error: Parser module not loaded.")
                return
            
            parsed_reactions = parse_chemkin_mechanism(chemkin_text)
            
            if not parsed_reactions:
                self.statusBar.showMessage("No valid reactions parsed from input.")
                self.rate_table_widget.setRowCount(0)
                self.plot_axes.clear(); self.figure_canvas.draw()
                return

            self.statusBar.showMessage(f"Parsed {len(parsed_reactions)} reactions. Updating table and plot...")
            QApplication.processEvents() 
            
            self._update_rate_constant_table(parsed_reactions)
            self._update_plot(parsed_reactions) # Call the new plot update method
            
            self.statusBar.showMessage("Table and Plot updated successfully.")

        except Exception as e:
            self.statusBar.showMessage(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
