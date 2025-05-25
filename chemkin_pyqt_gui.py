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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar # Use alias directly
from matplotlib.figure import Figure

# Assuming chemkin_parser.py and rate_calculator.py are in the same directory or accessible in PYTHONPATH
try:
    from chemkin_parser import parse_chemkin_mechanism
    from rate_calculator import (
        calculate_arrhenius_rate, calculate_plog_rate, calculate_troe_rate,
        get_third_body_concentration, R_cal,
        # For reverse rates
        get_reaction_thermo_properties, calculate_equilibrium_constant_kp,
        calculate_delta_n_gas, calculate_equilibrium_constant_kc,
        calculate_reverse_rate_constant
    )
    from thermo_parser import read_nasa_polynomials, append_nasa_polynomial_from_string, is_valid_nasa_polynomial_string

except ImportError as e:
    print(f"Error importing modules: {e}. Ensure chemkin_parser.py, rate_calculator.py, and thermo_parser.py are accessible.")
    # Set all potentially missing functions/modules to None
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thermo_filepath = "therm.dat" # Define path to thermo file, store as instance var

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
        
        self.pressure_entry = QLineEdit("1.0") # Existing single pressure entry
        self.controls_layout.addRow("Pressure for Table (atm):", self.pressure_entry) # Renamed label for clarity

        self.pressure_min_entry = QLineEdit("1.0")
        self.controls_layout.addRow("Min Plot Pressure (atm):", self.pressure_min_entry)
        self.pressure_max_entry = QLineEdit("10.0")
        self.controls_layout.addRow("Max Plot Pressure (atm):", self.pressure_max_entry)
        self.pressure_log_steps_entry = QLineEdit("5")
        self.controls_layout.addRow("Log Plot Pressure Steps:", self.pressure_log_steps_entry)
        
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
        
        # Create and add the Matplotlib Navigation Toolbar
        self.plot_toolbar = NavigationToolbar(self.figure_canvas, self.plot_area_widget) # Parent is plot_area_widget
        plot_layout.addWidget(self.plot_toolbar)
        
        plot_layout.addWidget(self.figure_canvas) # Add canvas after toolbar
        self.plot_axes = self.figure.add_subplot(111)
        # self.plot_area_widget.setLayout(plot_layout) # Already set by passing to QVBoxLayout constructor
        self.right_panel_splitter.addWidget(self.plot_area_widget)


        self.rate_table_widget = QTableWidget()
        self.rate_table_widget.setObjectName("rate_table_widget")
        
        self.table_temperatures = [500.0, 1000.0, 1500.0, 2000.0] # Store for reuse
        num_temp_points = len(self.table_temperatures)
        # Reaction | Calc Rev? | kf | kr | Kc | Kp (for each T)
        self.rate_table_widget.setColumnCount(1 + 1 + (4 * num_temp_points)) 
        
        header_labels = ["Reaction", "Calc Rev?"]
        for T_table in self.table_temperatures:
            header_labels.extend([
                f"kf @ {T_table:.0f}K", 
                f"kr @ {T_table:.0f}K", 
                f"Kc @ {T_table:.0f}K", 
                f"Kp @ {T_table:.0f}K"
            ])
        self.rate_table_widget.setHorizontalHeaderLabels(header_labels)
        
        self.rate_table_widget.horizontalHeader().setStretchLastSection(False) # Allow scrolling for many columns
        self.rate_table_widget.setAlternatingRowColors(True) # Keep this
        self.rate_table_widget.setMinimumHeight(150)
        # self.rate_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive) # Will be handled by new method
        self._setup_rate_table_widget_appearance() # Call the new method

        self.right_panel_splitter.addWidget(self.rate_table_widget)
        self.right_panel_splitter.setSizes([400, 200])
        self.right_panel_layout.addWidget(self.right_panel_splitter)
        self.main_splitter.addWidget(self.right_panel_widget)
        self.main_splitter.setSizes([350, 850])

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        self.thermo_data = None
        self._load_thermo_data() # Load thermo data on startup

        self.statusBar.showMessage("Ready")

    def _setup_rate_table_widget_appearance(self):
        header = self.rate_table_widget.horizontalHeader()
        
        # Column 0: Reaction string
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) 
        # Or Interactive, but Stretch often good for the main descriptive column
        # self.rate_table_widget.setColumnWidth(0, 300) # Example if using Interactive

        # Column 1: "Calc Rev?" Checkbox
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        # self.rate_table_widget.setColumnWidth(1, 80) # Or a fixed small width if ResizeToContents isn't perfect

        # kf, kr, Kc, Kp columns (indices 2 to 1 + 1 + (4 * num_temp_points) - 1)
        # These are columns 2, 3, 4, 5 (for T1), 6, 7, 8, 9 (for T2) etc.
        num_temp_points = len(self.table_temperatures)
        for i in range(num_temp_points * 4):
            col_index = 2 + i
            header.setSectionResizeMode(col_index, QHeaderView.ResizeMode.Interactive)
            self.rate_table_widget.setColumnWidth(col_index, 90) # Default width for rate constant columns

        self.rate_table_widget.setAlternatingRowColors(True) # Already set, but ensure it's here
        self.rate_table_widget.verticalHeader().setVisible(False) # Optional: Hide row numbers

    def _load_thermo_data(self):
        # self.thermo_filepath is defined in __init__
        if read_nasa_polynomials is None:
            self.statusBar.showMessage("Error: Thermo parser (read_nasa_polynomials) not available. Reverse rates disabled.")
            self.thermo_data = {} 
            return

        try:
            self.thermo_data = read_nasa_polynomials(self.thermo_filepath)
            if not self.thermo_data:
                self.statusBar.showMessage(f"Warning: No data read from {self.thermo_filepath}. Reverse rates may be affected.")
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
        # self.table_temperatures is defined in __init__
        try:
            P_atm = float(self.pressure_entry.text())
            if P_atm <= 0: self.statusBar.showMessage("Error: Pressure for Table must be positive."); return
        except ValueError: self.statusBar.showMessage("Error: Invalid Pressure for Table."); return

        for row, reaction_data in enumerate(parsed_reactions):
            self.rate_table_widget.insertRow(row)
            
            # Column 0: Reaction string
            eq_display_str = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
            eq_item = QTableWidgetItem(eq_display_str)
            eq_item.setFlags(eq_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.rate_table_widget.setItem(row, 0, eq_item)

            # Column 1: "Calc Rev?" Checkbox
            checkbox = QCheckBox()
            # Store reaction_data with the checkbox for the handler
            # Using a lambda that captures row and reaction_data
            checkbox.stateChanged.connect(
                lambda state, r=row, rd=reaction_data: self._on_calc_reverse_checkbox_changed(state, r, rd)
            )
            self.rate_table_widget.setCellWidget(row, 1, checkbox)
            
            # Store checkbox in reaction_data for potential future access if needed
            reaction_data['calc_rev_checkbox'] = checkbox 

            col_offset = 2 # Start populating kf, kr, etc. from column 2
            for T_table in self.table_temperatures:
                # kf
                kf_val = self._calculate_rate_for_reaction(reaction_data, T_table, P_atm)
                kf_str = "%.3E" % kf_val if kf_val is not None and kf_val > 0 else ("0.000E+00" if kf_val == 0 else "N/A")
                kf_item = QTableWidgetItem(kf_str)
                kf_item.setFlags(kf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.rate_table_widget.setItem(row, col_offset, kf_item)
                
                # kr, Kc, Kp - Placeholders for now
                na_item_flags = Qt.ItemFlag.ItemIsEnabled # Not editable, just enabled
                kr_item = QTableWidgetItem("N/A"); kr_item.setFlags(na_item_flags)
                kc_item = QTableWidgetItem("N/A"); kc_item.setFlags(na_item_flags)
                kp_item = QTableWidgetItem("N/A"); kp_item.setFlags(na_item_flags)
                
                self.rate_table_widget.setItem(row, col_offset + 1, kr_item)
                self.rate_table_widget.setItem(row, col_offset + 2, kc_item)
                self.rate_table_widget.setItem(row, col_offset + 3, kp_item)
                
                col_offset += 4 # Move to the next block of 4 columns for the next temperature

        self.rate_table_widget.resizeColumnsToContents()

        eq_string = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
        checkbox = self.rate_table_widget.cellWidget(row, 1) # Get the checkbox instance

        if state == Qt.CheckState.Checked.value:
            self.statusBar.showMessage(f"Calculating reverse rates for: {eq_string[:30]}...")
            
            if not all([get_reaction_thermo_properties, calculate_equilibrium_constant_kp,
                        calculate_delta_n_gas, calculate_equilibrium_constant_kc,
                        calculate_reverse_rate_constant, self.thermo_data is not None,
                        is_valid_nasa_polynomial_string, append_nasa_polynomial_from_string]):
                self.statusBar.showMessage("Error: Missing critical functions or thermo data. Reverse calc disabled.")
                if checkbox: checkbox.setChecked(False)
                return

            # Check for missing species before iterating through temperatures
            # This initial check uses the existing self.thermo_data
            all_species_in_reaction = reaction_data.get('reactants', []) + reaction_data.get('products', [])
            current_missing_species = set()
            for spec_name, _ in all_species_in_reaction:
                if not self.thermo_data.get(spec_name.upper()):
                    current_missing_species.add(spec_name)
            
            if current_missing_species:
                # print(f"Thermo data missing for: {current_missing_species}. Prompting user.")
                dialog = NasaPolynomialInputDialog(sorted(list(current_missing_species)), self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    entered_text = dialog.polynomial_input_text_edit.toPlainText()
                    if entered_text:
                        is_valid, _, validation_errors = is_valid_nasa_polynomial_string(entered_text)
                        if is_valid:
                            append_success, append_msg = append_nasa_polynomial_from_string(self.thermo_filepath, entered_text)
                            if append_success:
                                self._load_thermo_data() # Reload thermo data
                                self.statusBar.showMessage("Thermo data updated. Re-checking for current reaction...")
                                # Re-check if data is now available for this specific reaction
                                still_missing_after_update = set()
                                for spec_name, _ in all_species_in_reaction:
                                    if not self.thermo_data.get(spec_name.upper()): 
                                        still_missing_after_update.add(spec_name)
                                if still_missing_after_update:
                                    self.statusBar.showMessage(f"Error: Thermo for {', '.join(still_missing_after_update)} still missing after update.")
                                    self._clear_reverse_rate_cells(row); 
                                    if checkbox: checkbox.setChecked(False); 
                                    return
                                # If all good, proceed to calculations below
                            else: # Append failed
                                self.statusBar.showMessage(f"Error appending thermo: {append_msg}")
                                self._clear_reverse_rate_cells(row); 
                                if checkbox: checkbox.setChecked(False); 
                                return
                        else: # Invalid polynomial string
                            self.statusBar.showMessage("Invalid thermo input: " + "; ".join(validation_errors))
                            self._clear_reverse_rate_cells(row); 
                            if checkbox: checkbox.setChecked(False); 
                            return
                    else: # User submitted empty text
                        self.statusBar.showMessage("No thermo data submitted for missing species.")
                        self._clear_reverse_rate_cells(row); 
                        if checkbox: checkbox.setChecked(False); 
                        return
                else: # User cancelled dialog
                    self.statusBar.showMessage("Thermo input cancelled. Cannot calculate reverse rates.")
                    self._clear_reverse_rate_cells(row); 
                    if checkbox: checkbox.setChecked(False); 
                    return
            
            # If we've reached here, either thermo data was initially available, or it was successfully updated and confirmed.
            # Proceed to calculate and populate for each temperature point.
            col_offset = 2 # kf starts at column 2
            any_thermo_calc_error_for_reaction = False
            for T_table in self.table_temperatures:
                # Note: P_atm_table is not used for thermo calc, only for kf if it's P-dependent
                # P_atm_table = float(self.pressure_entry.text()) 
                
                delta_H, delta_S, delta_G, missing_at_T, err_thermo_calc = \
                    get_reaction_thermo_properties(reaction_data, T_table, self.thermo_data)

                if err_thermo_calc:
                    any_thermo_calc_error_for_reaction = True
                    status_msg = f"Thermo error for {eq_string[:20]} at {T_table}K"
                    if missing_at_T: status_msg += f": {', '.join(missing_at_T)}"
                    # print(status_msg) # For debugging
                    self._set_error_in_table_cells(row, col_offset, "Thermo Err")
                    col_offset += 4
                    continue 

                Kp = calculate_equilibrium_constant_kp(delta_G, T_table)
                self._set_table_item(row, col_offset + 3, Kp, "Error") # Kp

                delta_n = calculate_delta_n_gas(reaction_data, self.thermo_data)
                Kc = calculate_equilibrium_constant_kc(Kp, T_table, delta_n)
                self._set_table_item(row, col_offset + 2, Kc, "Error") # Kc

                kf_item = self.rate_table_widget.item(row, col_offset)
                kf_val = None
                if kf_item and kf_item.text() not in ["N/A", "Error", "kf Error", "Thermo Err"]: # Added "Thermo Err"
                    try: kf_val = float(kf_item.text())
                    except ValueError: pass # kf_val remains None
                
                kr_str = "Error" # Default if kf_val is None or Kc is None
                if kf_val is not None and Kc is not None : # Ensure Kc is also valid
                    kr = calculate_reverse_rate_constant(kf_val, Kc)
                    kr_str = "%.3E" % kr if kr is not None else "Error"
                elif kf_val is None:
                    kr_str = "kf Error" 
                # If Kc is None (error), kr_str remains "Error"
                
                self.rate_table_widget.setItem(row, col_offset + 1, QTableWidgetItem(kr_str)) # kr
                col_offset += 4
            
            if not any_thermo_calc_error_for_reaction:
                 self.statusBar.showMessage(f"Reverse rates updated for: {eq_string[:30]}...")
            else:
                 self.statusBar.showMessage(f"Reverse rates updated for {eq_string[:30]} (some errors).")


        elif state == Qt.CheckState.Unchecked.value:
            self.statusBar.showMessage(f"Reverse calc OFF for: {eq_string[:30]}...")
            self._clear_reverse_rate_cells(row)

    def _set_table_item(self, row, col, value, error_str="Error", precision="%.3E"):
        item_str = error_str
        if value is not None:
            try:
                item_str = precision % value
            except TypeError: # value might be np.inf or similar that doesn't format well with %
                item_str = str(value) 
        self.rate_table_widget.setItem(row, col, QTableWidgetItem(item_str))

    def _set_error_in_table_cells(self, row, start_col_offset, error_msg="Error"):
        self.rate_table_widget.setItem(row, start_col_offset + 1, QTableWidgetItem(error_msg)) # kr
        self.rate_table_widget.setItem(row, start_col_offset + 2, QTableWidgetItem(error_msg)) # Kc
        self.rate_table_widget.setItem(row, start_col_offset + 3, QTableWidgetItem(error_msg)) # Kp

    def _clear_reverse_rate_cells(self, row):
        col_offset = 2 # kf starts at column 2
        na_item_flags = Qt.ItemFlag.ItemIsEnabled
        for _ in self.table_temperatures: # Iterate 4 times for the 4 temperature points
            # kr is at col_offset + 1
            # Kc is at col_offset + 2
            # Kp is at col_offset + 3
            kr_item = QTableWidgetItem("N/A"); kr_item.setFlags(na_item_flags)
            kc_item = QTableWidgetItem("N/A"); kc_item.setFlags(na_item_flags)
            kp_item = QTableWidgetItem("N/A"); kp_item.setFlags(na_item_flags)
            self.rate_table_widget.setItem(row, col_offset + 1, kr_item)
            self.rate_table_widget.setItem(row, col_offset + 2, kc_item)
            self.rate_table_widget.setItem(row, col_offset + 3, kp_item)
            col_offset += 4

    def _clear_reverse_rate_cells(self, row):
        col_offset = 2 # kf starts at column 2
        na_item_flags = Qt.ItemFlag.ItemIsEnabled
        for _ in self.table_temperatures:
            kr_item = QTableWidgetItem("N/A"); kr_item.setFlags(na_item_flags)
            kc_item = QTableWidgetItem("N/A"); kc_item.setFlags(na_item_flags)
            kp_item = QTableWidgetItem("N/A"); kp_item.setFlags(na_item_flags)
            self.rate_table_widget.setItem(row, col_offset + 1, kr_item)
            self.rate_table_widget.setItem(row, col_offset + 2, kc_item)
            self.rate_table_widget.setItem(row, col_offset + 3, kp_item)
            col_offset += 4


    def _update_plot(self, parsed_reactions):
        self.plot_axes.clear()
        try:
            T_min = float(self.temp_min_entry.text())
            T_max = float(self.temp_max_entry.text())
            if not (T_max > T_min and T_min > 0):
                self.statusBar.showMessage("Error: Invalid Temperature range. Ensure T_max > T_min > 0.")
                self.figure_canvas.draw(); return
            T_values = np.linspace(T_min, T_max, 100)

            P_table_atm = float(self.pressure_entry.text()) # For table and non-P-dependent reactions
            if P_table_atm <= 0:
                self.statusBar.showMessage("Error: Pressure for Table (atm) must be positive.")
                self.figure_canvas.draw(); return

            P_min_plot = float(self.pressure_min_entry.text())
            P_max_plot = float(self.pressure_max_entry.text())
            P_log_steps = int(self.pressure_log_steps_entry.text())

            use_pressure_range_plotting = False
            P_plot_values = [P_table_atm] # Default to single pressure from table entry

            if P_min_plot > 0 and P_max_plot >= P_min_plot and P_log_steps > 0:
                use_pressure_range_plotting = True
                if P_log_steps == 1:
                    P_plot_values = np.array([P_min_plot]) # Use Min Plot P if steps is 1
                elif P_min_plot == P_max_plot : # handles steps > 1 but min=max
                    P_plot_values = np.array([P_min_plot])
                    use_pressure_range_plotting = True # Still treat as "range" for labeling consistency
                else: # P_max_plot > P_min_plot and P_log_steps > 1
                    P_plot_values = np.logspace(np.log10(P_min_plot), np.log10(P_max_plot), num=P_log_steps)
            # If P_min_plot, P_max_plot, P_log_steps are not set for a range, P_plot_values remains [P_table_atm]
            # and use_pressure_range_plotting remains False (unless min=max and steps=1, handled above)

        except ValueError:
            self.statusBar.showMessage("Error: Temperature/Pressure values must be numeric.")
            self.figure_canvas.draw(); return
        except Exception as e:
            self.statusBar.showMessage(f"Error processing plot parameters: {e}")
            self.figure_canvas.draw(); return

        num_lines_plotted = 0
        for reaction_data in parsed_reactions:
            reaction_label_base = reaction_data.get('equation_string_cleaned', reaction_data.get('equation_string', 'N/A'))
            reaction_type = reaction_data.get('reaction_type', 'ARRHENIUS')

            if use_pressure_range_plotting and reaction_type in ['PLOG', 'TROE']:
                for P_val in P_plot_values:
                    k_values_log10 = []
                    for T_val in T_values:
                        k = self._calculate_rate_for_reaction(reaction_data, T_val, P_val)
                        if k is not None and k > 1e-100:
                            k_values_log10.append(np.log10(k))
                        else:
                            k_values_log10.append(np.nan)
                    
                    if any(not np.isnan(val) for val in k_values_log10):
                        label = f"{reaction_label_base} @ {P_val:.2E} atm"
                        self.plot_axes.plot(T_values, k_values_log10, label=label)
                        num_lines_plotted +=1
            else: # Arrhenius or single pressure plot mode
                k_values_log10 = []
                # For Arrhenius, P_table_atm is irrelevant for calculation but used for consistency if needed elsewhere
                # For PLOG/TROE in single pressure mode, P_table_atm is used.
                current_P_for_plot = P_table_atm
                for T_val in T_values:
                    k = self._calculate_rate_for_reaction(reaction_data, T_val, current_P_for_plot)
                    if k is not None and k > 1e-100:
                        k_values_log10.append(np.log10(k))
                    else:
                        k_values_log10.append(np.nan)
                
                if any(not np.isnan(val) for val in k_values_log10):
                    label = reaction_label_base
                    if reaction_type in ['PLOG', 'TROE']: # Add pressure info for P-dependent reactions
                        label += f" @ {current_P_for_plot:.2E} atm (Table P)"
                    self.plot_axes.plot(T_values, k_values_log10, label=label)
                    num_lines_plotted +=1
        
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
            self._update_plot(parsed_reactions) 
            
            P_table_val_str = self.pressure_entry.text() # Get current table pressure
            self.statusBar.showMessage(f"Table (P={P_table_val_str} atm) and Plot updated.")

        except Exception as e:
            self.statusBar.showMessage(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtGui import QFont

class NasaPolynomialInputDialog(QDialog):
    def __init__(self, missing_species_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Missing Thermodynamic Data")
        self.setMinimumWidth(600) # Set a reasonable minimum width
        self.setMinimumHeight(400) # Set a reasonable minimum height

        layout = QVBoxLayout(self)

        instruction_text = (
            f"NASA polynomials are missing for the following species: <b>{', '.join(missing_species_list)}</b>.<br><br>"
            "Please paste complete NASA 7-coefficient polynomial strings below. "
            "Each species entry is typically 4 lines (1 header line and 3 coefficient lines)."
        )
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)

        self.polynomial_input_text_edit = QTextEdit()
        # Set a monospaced font for easier editing of fixed-format text
        font = QFont("Courier New") # Or "Monospace", "Consolas", etc.
        font.setPointSize(10) # Adjust size as needed
        self.polynomial_input_text_edit.setFont(font)
        self.polynomial_input_text_edit.setPlaceholderText(
            "Example for one species (4 lines):\n"
            "SPECIES_NAME      DATE    EL1 C1 EL2 C2 ... G Tlow Thigh Tcommon     1\n"
            " A1_HIGH A2_HIGH A3_HIGH A4_HIGH A5_HIGH                             2\n"
            " A6_HIGH A7_HIGH A1_LOW  A2_LOW  A3_LOW                              3\n"
            " A4_LOW  A5_LOW  A6_LOW  A7_LOW                                      4"
        )
        layout.addWidget(self.polynomial_input_text_edit)

        # Standard buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    # This method is part of how QDialog is typically used.
    # The actual text retrieval will be done by the caller after dialog.exec().
    # def get_polynomial_text(self): # This is not needed as a separate method on the dialog itself
    #     return self.polynomial_input_text_edit.toPlainText() if self.result() == QDialog.DialogCode.Accepted else None

# Example of how to call it (for testing, can be removed or integrated later)
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     # Create a dummy main window to act as parent for testing
#     # In real use, 'self' from MainWindow would be the parent.
#     # main_window_dummy = QMainWindow() 
    
#     missing_species = ["CH4_MISSING", "O2_MISSING", "H2O_VERY_LONG_SPECIES_NAME_EXAMPLE"]
#     dialog = NasaPolynomialInputDialog(missing_species, parent=None) # or parent=main_window_dummy
    
#     if dialog.exec() == QDialog.DialogCode.Accepted:
#         entered_text = dialog.polynomial_input_text_edit.toPlainText()
#         if entered_text:
#             print("Polynomials submitted:")
#             print(entered_text)
#         else:
#             print("No polynomial text submitted.")
#     else:
#         print("Polynomial input cancelled.")
    
#     # main_window_dummy.show() # If you want to see a parent window
#     # sys.exit(app.exec()) # This would run a new app loop. Only one app.exec() should run for the main app.