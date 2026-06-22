# CHEMKIN Rate Viewer UI Beautification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the GUI with deep navy color scheme and pyqtgraph interactive charts, keeping matplotlib for export.

**Architecture:** Replace matplotlib FigureCanvas with pyqtgraph PlotWidget in the main plot area. Apply unified deep navy QSS using Qt C++ base class selectors with objectName targeting. Keep matplotlib as a static export engine only.

**Tech Stack:** PyQt6, QFluentWidgets 1.11.2, pyqtgraph 0.13.7, matplotlib (export only)

---

### Task 1: Apply Deep Navy Color Palette to QSS

**Files:**
- Modify: `d:\RCCompare\chemkin_pyqt_gui_fluent.py` (method `_apply_global_dark_qss`)

- [ ] **Step 1: Replace color variables in `_apply_global_dark_qss`**

Replace the dark theme colors:
```python
# OLD:
bg, card_bg, input_bg, fg, border, accent = (
    '#2b2b2b', '#353535', '#3c3c3c', '#e0e0e0', '#4a4a4a', '#4a90e2')
# NEW:
bg, card_bg, input_bg, fg, border, accent = (
    '#1a1d23', '#222630', '#2e3440', '#e8ecf4', '#3a3f4b', '#4a90e2')
```

Replace the light theme colors:
```python
# OLD:
bg, card_bg, input_bg, fg, border, accent = (
    '#f5f5f5', '#ffffff', '#ffffff', '#333333', '#e0e0e0', '#4a90e2')
# NEW:
bg, card_bg, input_bg, fg, border, accent = (
    '#f0f2f5', '#ffffff', '#f8f9fa', '#2c3e50', '#dce1e8', '#4a90e2')
```

Add secondary text color variable and update label styling:
```python
fg_secondary = '#b0b8c8' if dark else '#6c757d'
fg_muted = '#7a8599' if dark else '#95a5a6'
```

- [ ] **Step 2: Update caption/label colors to use secondary text**

In the QSS, add specificity for CaptionLabel-like elements (they use QLabel in Qt):
```python
# After the #leftPanelContent QLabel rule, add a more specific rule
# for muted/secondary labels by targeting the scroll area's direct children
```

- [ ] **Step 3: Run tests and visual verification**

Run: `C:\Users\17915\anaconda3\envs\gui-env\python.exe -m pytest tests/test_v13_regressions.py -q`
Expected: 10 passed

Run: `C:\Users\17915\anaconda3\envs\gui-env\python.exe -c "import sys; sys.argv=['t']; from PyQt6.QtWidgets import QApplication; app=QApplication(sys.argv); from chemkin_pyqt_gui_fluent import MainWindow; w=MainWindow(); w.show(); print('GUI launched - check navy colors')"`

- [ ] **Step 4: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "style: apply deep navy color palette (#1a1d23 base)"
```

---

### Task 2: Replace Matplotlib with pyqtgraph for Main Plot

**Files:**
- Modify: `d:\RCCompare\chemkin_pyqt_gui_fluent.py` (imports, MatplotlibCard → PyQtGraphCard, plot methods)

- [ ] **Step 1: Add pyqtgraph import**

At the top of the file, after the matplotlib imports, add:
```python
import pyqtgraph as pg
pg.setConfigOption('background', '#1a1d23')
pg.setConfigOption('foreground', '#b0b8c8')
pg.setConfigOption('antialias', True)
pg.setConfigOption('leftButtonPan', False)  # right-click drag to pan
```

- [ ] **Step 2: Replace MatplotlibCard with PyQtGraphCard**

Replace the entire `MatplotlibCard` class:
```python
class PyQtGraphCard(SimpleCardWidget):
    """Card containing a pyqtgraph PlotWidget for interactive charting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        self._title = SubtitleLabel("Reaction Rates vs Temperature")
        lay.addWidget(self._title)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Temperature', units='K')
        self.plot_widget.setLabel('left', 'log₁₀(k)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(60, 10))
        self.plot_widget.setMinimumHeight(300)

        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False,
            pen=pg.mkPen('#7a8599', style=Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False,
            pen=pg.mkPen('#7a8599', style=Qt.PenStyle.DashLine))
        self.vline.setVisible(False)
        self.hline.setVisible(False)
        self.plot_widget.addItem(self.vline)
        self.plot_widget.addItem(self.hline)

        self.label = pg.TextItem('', anchor=(0, 1), color='#e8ecf4',
            fill=pg.mkBrush('#222630cc'))
        self.label.setVisible(False)
        self.plot_widget.addItem(self.label)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        lay.addWidget(self.plot_widget, 1)

    def _on_mouse_moved(self, pos):
        vb = self.plot_widget.vb
        if not self.plot_widget.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        view_range = vb.viewRange()
        if (view_range[0][0] <= x <= view_range[0][1] and
                view_range[1][0] <= y <= view_range[1][1]):
            self.vline.setPos(x)
            self.hline.setPos(y)
            self.vline.setVisible(True)
            self.hline.setVisible(True)
            self.label.setText(f'T={x:.0f} K  log(k)={y:.2f}')
            self.label.setPos(x, y)
            self.label.setVisible(True)
        else:
            self.vline.setVisible(False)
            self.hline.setVisible(False)
            self.label.setVisible(False)

    def clear_plot(self):
        self.plot_widget.clear()
        # Re-add crosshair items after clear
        self.plot_widget.addItem(self.vline)
        self.plot_widget.addItem(self.hline)
        self.plot_widget.addItem(self.label)

    def set_title(self, t):
        self._title.setText(t)

    def apply_theme(self, dark=True):
        bg = '#1a1d23' if dark else '#f0f2f5'
        fg = '#b0b8c8' if dark else '#6c757d'
        grid_alpha = 0.3
        pg.setConfigOption('background', bg)
        pg.setConfigOption('foreground', fg)
        self.plot_widget.setBackground(bg)
        # Update axis and grid colors
        ax = self.plot_widget.getAxis('bottom')
        ax.setPen(pg.mkPen(fg))
        ax.setTextPen(pg.mkPen(fg))
        ax = self.plot_widget.getAxis('left')
        ax.setPen(pg.mkPen(fg))
        ax.setTextPen(pg.mkPen(fg))
        self.plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)
```

- [ ] **Step 3: Update MainWindow to use PyQtGraphCard**

In `_build_home`, replace:
```python
# OLD:
self.plot_card = MatplotlibCard()
# NEW:
self.plot_card = PyQtGraphCard()
```

- [ ] **Step 4: Rewrite `_update_plot_with_reverse_rates` for pyqtgraph**

Replace the entire method. Key changes:
- Use `self.plot_card.plot_widget.plot()` instead of `ax.plot()`
- Use `self.plot_card.clear_plot()` instead of `ax.clear()`
- Remove matplotlib-specific calls (set_xlabel, set_ylabel, legend, grid, canvas.draw)
- Add re-crosshair items after clear

```python
def _update_plot_with_reverse_rates(self):
    if not hasattr(self, 'parsed_reactions'):
        return
    pw = self.plot_card.plot_widget
    self.plot_card.clear_plot()
    try:
        T_min = float(self.temp_min.text())
        T_max = float(self.temp_max.text())
        if not (T_max > T_min and T_min > 0):
            self._info("Invalid temperature range.", "error")
            return
        T_vals = np.linspace(T_min, T_max, 100)
        inv = self.use_inv_temp_cb.isChecked()
        x_vals = 1000.0 / T_vals if inv else T_vals
        x_label = "1000/T (K⁻¹)" if inv else "Temperature (K)"
        P_vals, use_range, P_def = self._get_plot_pressure_settings()
    except ValueError:
        self._info("Invalid numeric input.", "error")
        return

    pw.setLabel('bottom', x_label)

    # Curve colors from tab10 palette, first color matches accent
    curve_colors = [
        '#4a90e2', '#e2764a', '#4ae28a', '#e24a90', '#904ae2',
        '#e2d04a', '#4ae2d0', '#e24a4a', '#8ae24a', '#4a6ee2',
    ]

    n_plotted = 0
    for ri, rd in enumerate(self.parsed_reactions):
        lbl_base = rd.get('equation_string_cleaned',
                          rd.get('equation_string', 'N/A'))
        rt = rd.get('reaction_type', 'ARRHENIUS')
        is_rev = rd.get('is_calculating_reverse', False)

        if use_range and rt in ('PLOG', 'TROE'):
            for pi, Pv in enumerate(P_vals):
                k_log = self._calc_k_log_array(rd, T_vals, Pv, is_rev)
                if k_log is not None and any(~np.isnan(k_log)):
                    pfx = "kr" if is_rev else "kf"
                    color = curve_colors[(ri + pi) % len(curve_colors)]
                    pen = pg.mkPen(color, width=2)
                    pw.plot(x_vals, k_log,
                            name=f"{pfx}: {lbl_base} @ {Pv:.1f} atm",
                            pen=pen)
                    n_plotted += 1
        else:
            P_use = P_def
            k_log = self._calc_k_log_array(rd, T_vals, P_use, is_rev)
            if k_log is not None and any(~np.isnan(k_log)):
                pfx = "kr" if is_rev else "kf"
                color = curve_colors[ri % len(curve_colors)]
                pen = pg.mkPen(color, width=2)
                lbl = f"{pfx}: {lbl_base}"
                if rt in ('PLOG', 'TROE'):
                    lbl += f" @ {P_use:.1f} atm"
                pw.plot(x_vals, k_log, name=lbl, pen=pen)
                n_plotted += 1

    # Re-add crosshair after plotting
    pw.addItem(self.plot_card.vline)
    pw.addItem(self.plot_card.hline)
    pw.addItem(self.plot_card.label)
```

- [ ] **Step 5: Add helper method `_calc_k_log_array`**

```python
def _calc_k_log_array(self, rd, T_vals, P, is_rev):
    """Calculate log10(k) array for plotting."""
    if is_rev:
        k_arr = np.array([
            (self._calculate_reverse_rate_for_plot(rd, t, P) or np.nan)
            for t in T_vals
        ])
    else:
        k_arr = self._calculate_rate_array(rd, T_vals, P)
    # Convert to log10, handling nan and zero
    with np.errstate(divide='ignore', invalid='ignore'):
        k_log = np.where(
            (k_arr != None) & ~np.isnan(k_arr) & (k_arr > 1e-100),
            np.log10(k_arr), np.nan)
    return k_log
```

- [ ] **Step 6: Remove matplotlib FigureCanvas/NavigationToolbar from imports**

Keep `from matplotlib.figure import Figure` and `import matplotlib.pyplot as plt` for export only. Remove:
```python
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
```

- [ ] **Step 7: Run tests and visual verification**

Run: `C:\Users\17915\anaconda3\envs\gui-env\python.exe -m pytest tests/test_v13_regressions.py -q`
Expected: 10 passed

Run: `C:\Users\17915\anaconda3\envs\gui-env\python.exe -c "import sys; sys.argv=['t']; from PyQt6.QtWidgets import QApplication; app=QApplication(sys.argv); from chemkin_pyqt_gui_fluent import MainWindow; w=MainWindow(); w._on_plot_clicked(); print(f'pyqtgraph plot: {len(w.plot_card.plot_widget.listDataItems())} curves'); w.plot_card.apply_theme(True); print('Dark theme applied'); print('ALL OK')"`

- [ ] **Step 8: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: replace matplotlib with pyqtgraph for interactive main plot

- PyQtGraphCard with PlotWidget, crosshair, and legend
- Mouse-tracking crosshair with coordinate label
- _calc_k_log_array helper for log10 conversion
- _update_plot_with_reverse_rates rewritten for pyqtgraph
- matplotlib kept for static export only"
```

---

### Task 3: Add matplotlib Static Export Method

**Files:**
- Modify: `d:\RCCompare\chemkin_pyqt_gui_fluent.py` (add `_export_plot_image` method)

- [ ] **Step 1: Add export method to MainWindow**

```python
def _export_plot_image(self, filepath):
    """Export current plot as high-quality static image using matplotlib."""
    if not hasattr(self, 'parsed_reactions') or not self.parsed_reactions:
        self._info("No data to export.", "warning")
        return

    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 6), dpi=300)
    fig.set_facecolor('#1a1d23')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#222630')
    ax.tick_params(colors='#b0b8c8')
    ax.title.set_color('#e8ecf4')
    ax.xaxis.label.set_color('#e8ecf4')
    ax.yaxis.label.set_color('#e8ecf4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3f4b')

    try:
        T_min = float(self.temp_min.text())
        T_max = float(self.temp_max.text())
        T_vals = np.linspace(T_min, T_max, 200)  # higher res for export
        inv = self.use_inv_temp_cb.isChecked()
        x_vals = 1000.0 / T_vals if inv else T_vals
        x_label = "1000/T (K⁻¹)" if inv else "Temperature (K)"
        P_vals, use_range, P_def = self._get_plot_pressure_settings()
    except ValueError:
        self._info("Invalid parameters for export.", "error")
        return

    curve_colors = [
        '#4a90e2', '#e2764a', '#4ae28a', '#e24a90', '#904ae2',
        '#e2d04a', '#4ae2d0', '#e24a4a', '#8ae24a', '#4a6ee2',
    ]

    for ri, rd in enumerate(self.parsed_reactions):
        lbl_base = rd.get('equation_string_cleaned',
                          rd.get('equation_string', 'N/A'))
        rt = rd.get('reaction_type', 'ARRHENIUS')
        is_rev = rd.get('is_calculating_reverse', False)
        pressures = P_vals if (use_range and rt in ('PLOG', 'TROE')) else [P_def]

        for pi, Pv in enumerate(pressures):
            k_log = self._calc_k_log_array(rd, T_vals, Pv, is_rev)
            if k_log is not None and any(~np.isnan(k_log)):
                pfx = "kr" if is_rev else "kf"
                color = curve_colors[(ri + pi) % len(curve_colors)]
                lbl = f"{pfx}: {lbl_base}"
                if rt in ('PLOG', 'TROE') and len(pressures) > 1:
                    lbl += f" @ {Pv:.1f} atm"
                ax.plot(x_vals, k_log, label=lbl, color=color, linewidth=2)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("log₁₀(k)", fontsize=12)
    ax.set_title("Reaction Rates vs Temperature", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#222630', edgecolor='#3a3f4b',
              labelcolor='#e8ecf4')
    ax.grid(True, color='#3a3f4b', alpha=0.5)

    fig.savefig(filepath, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    self._info(f"Exported: {filepath}")
```

- [ ] **Step 2: Add export button to plot card toolbar**

In `_build_home`, after creating the plot card, add a small toolbar:
```python
# Plot export toolbar (below the plot card)
toolbar_layout = QHBoxLayout()
toolbar_layout.setContentsMargins(0, 4, 0, 0)
export_img_btn = PushButton("Export Plot Image")
export_img_btn.clicked.connect(self._on_export_image_clicked)
toolbar_layout.addStretch()
toolbar_layout.addWidget(export_img_btn)
```
Add this toolbar_layout after `rl.addWidget(self.plot_card, 3)`.

Add the handler:
```python
def _on_export_image_clicked(self):
    fp, _ = QFileDialog.getSaveFileName(
        self, "Export Plot", "reaction_rates.svg",
        "SVG (*.svg);;PDF (*.pdf);;PNG (*.png)")
    if fp:
        self._export_plot_image(fp)
```

- [ ] **Step 3: Run tests**

Run: `C:\Users\17915\anaconda3\envs\gui-env\python.exe -m pytest tests/test_v13_regressions.py -q`
Expected: 10 passed

- [ ] **Step 4: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: add matplotlib static export for publication-quality images"
```

---

### Task 4: Update Requirements and Final Verification

**Files:**
- Modify: `d:\RCCompare\requirements.txt`

- [ ] **Step 1: Add pyqtgraph to requirements**

Add line after numpy:
```
pyqtgraph>=0.13.0
```

- [ ] **Step 2: Full verification**

Run regression tests:
`C:\Users\17915\anaconda3\envs\gui-env\python.exe -m pytest tests/test_v13_regressions.py -v`

Launch GUI and test full workflow:
```
C:\Users\17915\anaconda3\envs\gui-env\python.exe -c "
import sys; sys.argv=['t']
from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)
from chemkin_pyqt_gui_fluent import MainWindow
w = MainWindow()
w._on_plot_clicked()
print(f'[OK] pyqtgraph: {len(w.plot_card.plot_widget.listDataItems())} curves')
w._on_theme_toggled(False)
print('[OK] Light theme')
w._on_theme_toggled(True)
print('[OK] Dark theme')
print('FULL VERIFICATION PASSED')
"
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add pyqtgraph to requirements"
```

---

## Verification Checklist

- [ ] 10/10 regression tests pass
- [ ] GUI launches with deep navy color scheme (#1a1d23 background)
- [ ] pyqtgraph PlotWidget renders curves with crosshair interaction
- [ ] Mouse hover shows coordinate label
- [ ] Drag to zoom, right-click drag to pan
- [ ] Theme toggle updates pyqtgraph + Fluent + table + QSS
- [ ] Export Plot Image button generates SVG/PDF/PNG
- [ ] All dialogs (Detail, Reverse Rate, NASA Input) render with navy theme
