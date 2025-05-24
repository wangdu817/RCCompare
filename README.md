# CHEMKIN Rate Viewer

## Description
The CHEMKIN Rate Viewer is a Python application designed to parse CHEMKIN-formatted reaction mechanism files, calculate temperature-dependent rate constants (k) for the reactions, and visualize these rates. It provides a graphical user interface (GUI) for easy interaction, allowing users to input their CHEMKIN data, specify temperature ranges and pressure, and then view the results as plots of `log10(k)` vs. Temperature and in a tabular format.

## Key Features
*   **Graphical User Interface:** Built with PyQt6 for inputting CHEMKIN data and specifying calculation parameters (Temperature range, Pressure).
*   **CHEMKIN Parser:**
    *   Parses CHEMKIN-formatted reaction mechanisms.
    *   Supports standard Arrhenius reactions.
    *   Handles pressure-dependent PLOG reactions.
    *   Handles pressure-dependent TROE reactions, including support for the `LOW` keyword for `k0` parameters and embedded `k0` parameters.
    *   Recognizes and processes the `UNITS` keyword for activation energies, supporting 'CAL/MOLE', 'KCAL/MOLE', 'JOULES/MOLE', 'KJOULES/MOLE', 'KELVINS' ('K'), and 'EVOLTS'.
*   **Rate Constant Calculation:** Calculates rate constants (k) based on temperature and pressure using the parsed parameters.
*   **Data Visualization:**
    *   Displays `log10(k)` vs. Temperature plots using Matplotlib, embedded within the GUI.
    *   Shows a table of calculated rate constants at specific temperatures (500K, 1000K, 1500K, 2000K) for each parsed reaction.
*   **User Feedback:** Provides status messages and warnings through a status bar and console output.

## Dependencies
*   Python 3 (typically 3.7+).
*   Python package dependencies are listed in `requirements.txt` (`PyQt6`, `matplotlib`, `numpy`).
*   For detailed notes on system dependencies, especially for PyQt6/Qt on Linux, please refer to `dependencies.md`.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url> 
    cd <repository_directory>
    ```
    (Replace `<repository_url>` and `<repository_directory>` with actual values.)

2.  Create a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  Ensure system dependencies for PyQt6 are met (see `dependencies.md` for details, especially if on a minimal Linux environment). This might involve installing X11 client libraries and font configurations.

## How to Run
Execute the main GUI script from the project's root directory:
```bash
python chemkin_pyqt_gui.py
```

## Basic Usage Guide
1.  Launch the application using the command above.
2.  The main window will appear with an input area on the left and display areas (for plot and table) on the right.
3.  Paste your CHEMKIN-formatted reaction mechanism into the "CHEMKIN Input" text area. Sample input is provided by default.
4.  In the "Controls" section below the input area, specify the "Min Temperature (K)", "Max Temperature (K)", and "Pressure (atm)" for the calculations. Default values are provided.
5.  Click the "Parse, Calculate & Plot" button.
6.  The application will parse the input, calculate rate constants across the specified temperature range at the given pressure.
7.  The plot area will display `log10(k)` vs. Temperature graphs for the successfully parsed and calculated reactions.
8.  Below the plot, a table will show the calculated rate constants (k) at specific temperatures: 500K, 1000K, 1500K, and 2000K for each reaction.
9.  The status bar at the bottom of the window will provide feedback on the process. Console output may also contain additional details or warnings.

## Parser Scope & Limitations
*   **Supported Reaction Types:** The parser currently supports Arrhenius, PLOG, and TROE (with `LOW` keyword for `k0` or embedded `k0` parameters) reaction types, along with the `UNITS` keyword for activation energies.
*   **Equation Parsing:** The parser identifies reactants, products, stoichiometric coefficients (defaulting to 1 if not specified), and basic third bodies like `(+M)`. The `equation_string_cleaned` field in the parsed output reflects the parser's interpretation.
*   **Unsupported Keywords (Currently Ignored or Basic Parsing):**
    *   Keywords such as `REV` (for explicit reverse Arrhenius parameters), `DUP` (duplicate reaction flag), `FORD`/`RORD` (explicit forward/reverse reaction orders), and specific third-body efficiencies (e.g., `H2/2.0/ O2/3.0/`) might be captured in the `other_keywords` field or as part of the "rest of line" but their specific kinetic effects are not yet implemented in the `rate_calculator`.
    *   These represent areas for potential future enhancements. The parser structure is designed to accommodate these additional parameters when their calculation logic is added.
*   **Mechanism Validation:** The application assumes the CHEMKIN input primarily focused on reaction rate expressions. Full mechanism consistency checks (elements, species balance across all reactions) are not performed.
*   **Error Handling:** Basic error handling for invalid numerical inputs and parsing issues is included. Warnings are printed to the console and/or shown in the status bar.

## GUI Environment Notes
*   The PyQt6 GUI requires a functioning desktop environment or equivalent.
*   If running on a headless server or minimal container/environment:
    *   Ensure X11 forwarding is correctly set up if you intend to display the GUI remotely.
    *   Alternatively, a virtual framebuffer (like Xvfb) might be needed.
    *   Crucially, all necessary Qt platform plugin dependencies must be installed. Missing libraries (e.g., for the `xcb` plugin) can prevent the application from launching. Refer to `dependencies.md` for more notes on system dependencies for Qt on Linux.
```
