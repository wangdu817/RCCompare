import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext # For the input text area

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Module-level constant for gas constant (already in rate_calculator, but good for reference here too)
R_CAL = 1.987204  # cal/mol-K

# Assuming chemkin_parser.py and rate_calculator.py are in the same directory or accessible in PYTHONPATH
try:
    from chemkin_parser import parse_chemkin_input
    from rate_calculator import (
        calculate_arrhenius_rate,
        calculate_plog_rate,
        calculate_troe_rate,
        get_third_body_concentration,
        R_cal as R_CAL_from_module # Use the one from module if available
    )
    R_CAL = R_CAL_from_module # Prefer the one from the module
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure chemkin_parser.py and rate_calculator.py are accessible.")
    # Fallback for basic GUI functionality if imports fail, though plotting will not work.
    parse_chemkin_input = None
    calculate_arrhenius_rate = None
    calculate_plog_rate = None
    calculate_troe_rate = None
    get_third_body_concentration = None
    # R_CAL is already defined above as a fallback


def plot_rates(parsed_reactions, temp_min_str, temp_max_str, pressure_str, plot_frame_widget, gas_constant_R_cal=R_CAL):
    """
    Plots reaction rates for given parsed reactions, temperature range, and pressure.
    (This function was developed in a previous step and is assumed to be correct)
    """
    for widget in plot_frame_widget.winfo_children():
        widget.destroy()

    try:
        T_min = float(temp_min_str)
        T_max = float(temp_max_str)
        P_atm = float(pressure_str)
    except ValueError:
        error_label = ttk.Label(plot_frame_widget, text="Error: Temperature and Pressure must be valid numbers.")
        error_label.pack(pady=10)
        return

    if T_min <= 0 or T_max <= 0 or P_atm < 0: # Allow P_atm = 0 for certain cases if meaningful
        error_label = ttk.Label(plot_frame_widget, text="Error: Temperatures must be positive. Pressure must be non-negative.")
        error_label.pack(pady=10)
        return
    if T_max <= T_min:
        error_label = ttk.Label(plot_frame_widget, text="Error: Max temperature must be greater than Min temperature.")
        error_label.pack(pady=10)
        return
    
    if not (parse_chemkin_input and calculate_arrhenius_rate and calculate_plog_rate and calculate_troe_rate and get_third_body_concentration):
        error_label = ttk.Label(plot_frame_widget, text="Error: Calculation modules not loaded.")
        error_label.pack(pady=10)
        return

    T_values = np.linspace(T_min, T_max, 100)
    fig, ax = plt.subplots()

    num_reactions_plotted = 0
    for reaction_data in parsed_reactions:
        k_values_log10 = []
        equation = reaction_data.get('equation', 'Unknown Equation')
        reaction_type = reaction_data.get('type', 'UNKNOWN')
        params = reaction_data.get('params', [])

        for T in T_values:
            k = None
            if reaction_type == 'ARRHENIUS':
                k = calculate_arrhenius_rate(params, T)
            elif reaction_type == 'PLOG':
                plog_data = reaction_data.get('plog_data', [])
                if plog_data: k = calculate_plog_rate(plog_data, T, P_atm)
                else: print(f"Warning: PLOG reaction '{equation}' has no plog_data.")
            elif reaction_type == 'TROE':
                k_inf_params = params # Parser now stores only k_inf in 'params' for TROE
                k0_params = reaction_data.get('k0_params', [])
                troe_coeffs = reaction_data.get('troe_params', [])
                if len(k_inf_params) < 3: 
                    print(f"Warning: Insufficient k_inf params for TROE reaction '{equation}'. Skipping.")
                elif not k0_params: 
                    print(f"Warning: Missing k0_params for TROE reaction '{equation}'. Skipping for this T.")
                elif not troe_coeffs:
                     print(f"Warning: Missing troe_coeffs for TROE reaction '{equation}'. Skipping for this T.")
                else:
                    M_conc = get_third_body_concentration(P_atm, T)
                    k = calculate_troe_rate(k_inf_params, k0_params, troe_coeffs, T, P_atm, M_conc)
            
            if k is not None and k > 1e-100: # Avoid log10(0) or very small numbers leading to -inf
                k_values_log10.append(np.log10(k))
            else:
                k_values_log10.append(np.nan)
        
        if any(not np.isnan(val) for val in k_values_log10):
            ax.plot(T_values, k_values_log10, label=equation)
            num_reactions_plotted +=1
        else:
            print(f"Note: No valid rate constants calculated for reaction '{equation}' in the given range.")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("log10(k) (k in cm, mol, s units)")
    ax.set_title(f"Reaction Rates at {P_atm} atm")
    if num_reactions_plotted > 0 :
        ax.legend(fontsize='small', loc='best')
    else:
        no_data_label = ttk.Label(plot_frame_widget, text=f"No reactions plotted. Check input or console warnings.")
        no_data_label.pack(pady=10)

    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame_widget)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close(fig)


def main():
    root = tk.Tk()
    root.title("CHEMKIN Rate Viewer")
    root.geometry("1100x750") # Slightly larger for better layout

    # --- Main layout frames ---
    # Frame for input text area and controls (left side)
    input_controls_frame = ttk.Frame(root, padding=10)
    input_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    # Frame for displaying the plot (right side, expands)
    plot_display_frame = ttk.Frame(root, padding=10)
    plot_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # --- Populate Input Controls Frame ---
    # Using grid layout within input_controls_frame for alignment
    
    # CHEMKIN Input Area
    input_label = ttk.Label(input_controls_frame, text="CHEMKIN Input:")
    input_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0,2))
    
    chemkin_input_text = scrolledtext.ScrolledText(input_controls_frame, wrap=tk.WORD, width=60, height=20) # Increased height
    chemkin_input_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

    # Temperature Inputs
    temp_min_label = ttk.Label(input_controls_frame, text="Min Temperature (K):")
    temp_min_label.grid(row=2, column=0, sticky=tk.W, pady=2)
    temp_min_entry = ttk.Entry(input_controls_frame, width=15)
    temp_min_entry.grid(row=2, column=1, sticky=tk.E, pady=2)
    temp_min_entry.insert(0, "300")

    temp_max_label = ttk.Label(input_controls_frame, text="Max Temperature (K):")
    temp_max_label.grid(row=3, column=0, sticky=tk.W, pady=2)
    temp_max_entry = ttk.Entry(input_controls_frame, width=15)
    temp_max_entry.grid(row=3, column=1, sticky=tk.E, pady=2)
    temp_max_entry.insert(0, "2500")

    # Pressure Input
    pressure_label = ttk.Label(input_controls_frame, text="Pressure (atm):")
    pressure_label.grid(row=4, column=0, sticky=tk.W, pady=2)
    pressure_entry = ttk.Entry(input_controls_frame, width=15)
    pressure_entry.grid(row=4, column=1, sticky=tk.E, pady=2)
    pressure_entry.insert(0, "1.0")

    # Status Label (Optional, for feedback)
    status_label = ttk.Label(input_controls_frame, text="")
    status_label.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=5)


    # --- Process and Plot Button Command ---
    def process_and_plot():
        status_label.config(text="Processing...") # Basic feedback
        if not parse_chemkin_input: # Check if parser loaded
            status_label.config(text="Error: Parser not available.")
            print("Error: chemkin_parser.parse_chemkin_input is not available.")
            return

        chemkin_text = chemkin_input_text.get("1.0", tk.END)
        t_min_str = temp_min_entry.get()
        t_max_str = temp_max_entry.get()
        p_str = pressure_entry.get()

        try:
            parsed_data = parse_chemkin_input(chemkin_text)
            if not parsed_data:
                print("No reactions parsed or input was empty.")
                status_label.config(text="No reactions parsed from input.")
                # Clear plot area if no reactions
                for widget in plot_display_frame.winfo_children():
                    widget.destroy()
                no_data_label = ttk.Label(plot_display_frame, text=f"No reactions parsed.")
                no_data_label.pack(pady=10)
                return

            status_label.config(text=f"Parsed {len(parsed_data)} reaction(s). Plotting...")
            print(f"Parsed {len(parsed_data)} reaction(s).")
            
            # Call the existing plot_rates function
            plot_rates(parsed_data, t_min_str, t_max_str, p_str, plot_display_frame, R_CAL)
            status_label.config(text=f"Plotting complete for {len(parsed_data)} reaction(s).")

        except Exception as e:
            print(f"Error during processing or plotting: {e}")
            status_label.config(text=f"Error: {e}")
            # Display error in plot frame as well
            for widget in plot_display_frame.winfo_children():
                widget.destroy()
            error_label_plot = ttk.Label(plot_display_frame, text=f"Error during processing: {e}")
            error_label_plot.pack(pady=10)


    # Plot Button
    plot_button = ttk.Button(input_controls_frame, text="Parse and Plot", command=process_and_plot)
    plot_button.grid(row=5, column=0, columnspan=2, pady=10)

    # Example CHEMKIN text for easy testing
    sample_chemkin_input = """
! Arrhenius example
O + H2 <=> H + OH          1.0E10  0.5   1000.0

! PLOG example
CH4 + O2 = CH3 + HO2      PLOG /
    1.0   1.0E13  0.0  50000.0 /
    10.0  1.0E14  0.0  55000.0 /

! TROE example (embedded k0, TROE params on same line)
H + O2 (+M) = HO2 (+M)    1.0E12 0.5 0.0   1.0E16 0.0 -1000.0 TROE / 0.6 100.0 1000.0 50.0 / 

! TROE example (k_inf only, TROE params next line, LOW line)
N2O (+M) = N2 + O (+M)    1.0E15 0.0 55000.0 TROE /
    0.7 200.0 1200.0 /
LOW / 1.0E14 0.0 20000.0 / 
    """
    chemkin_input_text.insert("1.0", sample_chemkin_input.strip())

    root.mainloop()

if __name__ == '__main__':
    main()
