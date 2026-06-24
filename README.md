# CHEMKIN Rate Viewer v2.0

A PyQt6 desktop application for chemical kinetics mechanism analysis, built with QFluentWidgets Fluent Design UI.

## Author

**王杜 (Wang Du)** — wangdu@iet.cn

Institute of Engineering Thermophysics, Chinese Academy of Sciences (中科院工程热物理研究所)

## License

GNU General Public License v3

## Features

- **Rate Compare**: Parse CHEMKIN-format reaction mechanisms, calculate forward/reverse rate constants (Arrhenius, PLOG, TROE), interactive log10(k) vs. T plots, tabulated data with Excel export
- **Thermo Compare**: Compare NASA 7-coefficient polynomial thermodynamic properties (H/S/Cp) across species with interactive plots and detail dialogs
- **Dark/Light theme** toggle with Windows 11 Mica effect support
- **Persistent thermo database** at `%LOCALAPPDATA%\CHEMKIN_RateViewer\therm.dat`
- **Export** to Excel (pandas + openpyxl) and images (PNG/SVG/PDF)

## Running from Source

```bash
pip install -r requirements.txt
python chemkin_pyqt_gui_fluent.py
```

## Building

```bash
python build_package.py
```

Output: `dist\CHEMKIN_RateCalculator\CHEMKIN_RateCalculator.exe`
