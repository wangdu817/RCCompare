# Thermo Compare Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Thermo Compare" navigation page where users paste NASA polynomial data to compare H, S, Cp across species using pyqtgraph plots and tables.

**Architecture:** New `_build_thermo_compare()` method on `MainWindow` follows the same skeleton as `_build_home()` — left `SmoothScrollArea`(420px) + right `QSplitter`(plot:table=2:3). A Pill TabBar switches between H/S/Cp parameters. Reuses `thermo_parser.read_nasa_polynomials` via tempfile, `thermo_calculator.calculate_cp_h_s` directly, and module-level `CURVE_COLORS`/`CURVE_LINE_STYLES`.

**Tech Stack:** PyQt6, QFluentWidgets, pyqtgraph, numpy

**Target file:** `chemkin_pyqt_gui_fluent.py` (~2104 lines → ~2400 lines)

---

### Task 1: Rename "Home" to "Rate" in navigation

**Files:**
- Modify: `chemkin_pyqt_gui_fluent.py:547`

- [ ] **Step 1: Change nav label**

Change `"Home"` to `"Rate"` in the `addSubInterface` call:

```python
self.addSubInterface(self._home, FluentIcon.HOME, "Rate",
                    position=NavigationItemPosition.TOP)
```

- [ ] **Step 2: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: rename Home nav to Rate"
```

---

### Task 2: Add thermo compare widget + navigation registration

**Files:**
- Modify: `chemkin_pyqt_gui_fluent.py:538-550`

- [ ] **Step 1: Create thermo compare widget and register nav**

Insert between `_build_home()` and `_settings` creation:

```python
        self._thermo_compare = QWidget()
        self._thermo_compare.setObjectName("thermoCompare")
        self._build_thermo_compare()

        self._settings = QWidget()
        self._settings.setObjectName("settings")
        self._build_settings()

        self.addSubInterface(self._home, FluentIcon.HOME, "Rate",
                            position=NavigationItemPosition.TOP)
        self.addSubInterface(self._thermo_compare, FluentIcon.LINK, "Thermo Compare",
                            position=NavigationItemPosition.TOP)
        self.addSubInterface(self._settings, FluentIcon.SETTING, "Settings",
                            position=NavigationItemPosition.BOTTOM)
```

- [ ] **Step 2: Add `_build_thermo_compare` stub**

Add a stub method after `_build_settings` to avoid NameError:

```python
    def _build_thermo_compare(self):
        pass
```

- [ ] **Step 3: Syntax check**

```bash
.\.venv\Scripts\python.exe -m py_compile chemkin_pyqt_gui_fluent.py
```
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: add Thermo Compare nav entry and stub page"
```

---

### Task 3: Build left panel — hero, tab bar, input, controls, button

**Files:**
- Modify: `chemkin_pyqt_gui_fluent.py` — replace the `_build_thermo_compare` stub
- New import: `SegmentedWidget` in qfluentwidgets import block

- [ ] **Step 1: Add SegmentedWidget import**

Add `SegmentedWidget` to the qfluentwidgets import block (line 48-61):

```python
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, FluentIcon,
    setTheme, Theme, isDarkTheme, toggleTheme,
    PrimaryPushButton, PushButton, LineEdit, PlainTextEdit,
    BodyLabel, TitleLabel, CaptionLabel, StrongBodyLabel, SubtitleLabel,
    SimpleCardWidget,
    SwitchButton, CheckBox, ComboBox,
    TableWidget, TableItemDelegate,
    InfoBar, InfoBarPosition,
    ScrollArea, SmoothScrollArea, SingleDirectionScrollArea,
    MessageBoxBase,
    MessageBox,
    SegmentedWidget,
    setFont,
)
```

- [ ] **Step 2: Add `tempfile` import**

Add `import tempfile` in the standard library import block (after line 10):

```python
import tempfile
```

- [ ] **Step 3: Replace `_build_thermo_compare` stub with full implementation**

```python
    def _build_thermo_compare(self):
        main_lay = QHBoxLayout(self._thermo_compare)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ── Left scroll panel ──
        scroll = SmoothScrollArea()
        scroll.setObjectName("leftPanel")
        scroll.setFixedWidth(420)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        lw = QWidget()
        lw.setObjectName("leftPanelContent")
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.setSpacing(12)

        # Hero header
        hero = QWidget()
        hero.setObjectName("heroHeader")
        hero_lay = QHBoxLayout(hero)
        hero_lay.setContentsMargins(20, 16, 20, 16)
        hero_lay.setSpacing(14)
        logo_lbl = QLabel()
        logo_path = self._resolve_icon_path()
        if os.path.exists(logo_path):
            px = QPixmap(logo_path).scaled(
                44, 44, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(px)
        logo_lbl.setFixedSize(44, 44)
        hero_lay.addWidget(logo_lbl)
        txt_col = QVBoxLayout()
        txt_col.setSpacing(2)
        tt = TitleLabel("Thermo Compare")
        txt_col.addWidget(tt)
        sub_row = QHBoxLayout()
        sub_row.setSpacing(10)
        sub_row.addWidget(CaptionLabel("NASA Polynomial Comparison"))
        badge = QLabel("v1.3")
        badge.setObjectName("versionBadge")
        badge.setFixedHeight(20)
        badge.setStyleSheet(
            "background-color: #4a90e2; color: #ffffff; padding: 1px 8px; "
            "border-radius: 10px; font-size: 10px; font-weight: bold;")
        sub_row.addWidget(badge)
        sub_row.addStretch()
        txt_col.addLayout(sub_row)
        hero_lay.addLayout(txt_col, 1)
        ll.addWidget(hero)

        # Pill TabBar: H | S | Cp
        self.thermo_tab = SegmentedWidget()
        self.thermo_tab.addItem(routeKey="H", text="H (kJ/mol)")
        self.thermo_tab.addItem(routeKey="S", text="S (J/mol·K)")
        self.thermo_tab.addItem(routeKey="Cp", text="Cp (J/mol·K)")
        self.thermo_tab.setCurrentItem("H")
        self.thermo_tab.setFixedHeight(40)
        self._thermo_tab_idx = 0
        self.thermo_tab.currentItemChanged.connect(
            lambda k: self._on_thermo_tab_changed({"H": 0, "S": 1, "Cp": 2}[k]))
        ll.addWidget(self.thermo_tab)

        # NASA input card
        ic = SimpleCardWidget()
        il = QVBoxLayout(ic)
        il.setContentsMargins(16, 16, 16, 16)
        il.addWidget(StrongBodyLabel("NASA Polynomial Input"))
        self.thermo_input = PlainTextEdit()
        self.thermo_input.setMinimumHeight(180)
        self.thermo_input.setPlaceholderText(
            "Paste NASA 7-coefficient polynomial data here...\n"
            "One species per 4-line block. Example:\n"
            "H2O    L 1/96H  2O  1   0   0G   200.000  3500.000  1000.000 1\n"
            " 3.386...e+00  3.474...e-03 -6.354...e-06  6.968...e-09 -2.506...e-12  2\n"
            " 2.590...e+03  1.195...e-03 -4.338...e-06  6.884...e-09 -3.696...e-12  3\n"
            " 3.042...e+03  1.203...e-03 -4.572...e-06  7.332...e-09 -3.975...e-12  4")
        il.addWidget(self.thermo_input)
        ll.addWidget(ic)

        # Controls card
        cc = SimpleCardWidget()
        cl = QVBoxLayout(cc)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.setSpacing(8)
        cl.addWidget(StrongBodyLabel("Controls"))
        self.thermo_tmin = LineEdit(); self.thermo_tmin.setText("300")
        self.thermo_tmax = LineEdit(); self.thermo_tmax.setText("3500")
        self.thermo_table_temps = LineEdit()
        self.thermo_table_temps.setText(
            "300, 500, 700, 1000, 1500, 2000, 2500, 3000")
        for lbl, w in [("T min (K)", self.thermo_tmin),
                        ("T max (K)", self.thermo_tmax),
                        ("Table temps (K)", self.thermo_table_temps)]:
            r = QHBoxLayout()
            r.addWidget(CaptionLabel(lbl)); r.addWidget(w); r.addStretch()
            cl.addLayout(r)
        ll.addWidget(cc)

        # Parse button
        self.thermo_parse_btn = PrimaryPushButton(FluentIcon.PLAY, "Parse & Plot")
        self.thermo_parse_btn.setMinimumHeight(40)
        self.thermo_parse_btn.clicked.connect(self._on_thermo_parse_clicked)
        ll.addWidget(self.thermo_parse_btn)
        ll.addStretch()
        scroll.setWidget(lw)
        main_lay.addWidget(scroll)

        # ── Right panel ──
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)

        # Plot
        self.thermo_plot_card = PyQtGraphCard()
        self.thermo_plot_card.plot_widget.setLabel("left", "H (kJ/mol)")
        self.thermo_plot_card.plot_widget.setLabel("bottom", "T (K)")
        rl.addWidget(self.thermo_plot_card, 2)

        # Export button inline in plot card header
        export_img_btn = PushButton(FluentIcon.PHOTO, "Export")
        export_img_btn.clicked.connect(self._on_thermo_export_image_clicked)
        self.thermo_plot_card.add_title_button(export_img_btn)

        # Empty state placeholder
        self.thermo_empty_placeholder = QWidget()
        self.thermo_empty_placeholder.setObjectName("emptyPlaceholder")
        ep_lay = QVBoxLayout(self.thermo_empty_placeholder)
        ep_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.setSpacing(12)
        ep_icon = QLabel("📊")
        ep_icon.setStyleSheet("font-size: 48px; background: transparent;")
        ep_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(ep_icon)
        ep_title = BodyLabel("No Data Yet")
        ep_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(ep_title)
        ep_hint = CaptionLabel(
            "Paste NASA polynomial data and click Parse to compare species")
        ep_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_hint.setStyleSheet("color: #95a5a6; background: transparent;")
        ep_lay.addWidget(ep_hint)
        rl.addWidget(self.thermo_empty_placeholder, 3)

        # Table
        self.thermo_table = QTableWidget()
        self.thermo_table.setAlternatingRowColors(True)
        self.thermo_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.thermo_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.thermo_table.horizontalHeader().setStretchLastSection(True)
        self.thermo_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.thermo_table.verticalHeader().setVisible(True)
        rl.addWidget(self.thermo_table, 3)
        main_lay.addWidget(rw, 1)
```

- [ ] **Step 4: Syntax check**

```bash
.\.venv\Scripts\python.exe -m py_compile chemkin_pyqt_gui_fluent.py
```
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: build Thermo Compare left panel layout"
```

---

### Task 4: Implement thermo data parsing logic

**Files:**
- Modify: `chemkin_pyqt_gui_fluent.py` — add new methods after `_load_thermo_data`

- [ ] **Step 1: Add `_on_thermo_parse_clicked` method**

Add after the `_load_thermo_data` method group (around line 1074):

```python
    # ── Thermo Compare: parsing ───────────────────────────────────────

    def _on_thermo_parse_clicked(self):
        text = self.thermo_input.toPlainText().strip()
        if not text:
            self._info("Paste NASA polynomial data first.", "warning")
            return
        try:
            # Validate basic format: lines must be multiple of 4
            lines = text.split('\n')
            if len(lines) % 4 != 0:
                self._info(
                    f"Expected multiple of 4 lines per species, got {len(lines)}.",
                    "error")
                return
            # Write to temp file for read_nasa_polynomials
            fd, tmp_path = tempfile.mkstemp(suffix='.dat', text=True)
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(text)
                self.thermo_species_data = read_nasa_polynomials(tmp_path)
            finally:
                os.unlink(tmp_path)
            if not self.thermo_species_data:
                self._info("No valid NASA polynomial entries found.", "error")
                return
            self._info(f"Loaded {len(self.thermo_species_data)} species.")
            self._update_thermo_tab_plot_and_table()
            self._show_thermo_empty_state(False)
        except Exception as e:
            self._info(f"Parse error: {e}", "error")

    def _show_thermo_empty_state(self, show=True):
        self.thermo_empty_placeholder.setVisible(show)
        self.thermo_plot_card.setVisible(not show)
        self.thermo_table.setVisible(not show)
```

- [ ] **Step 2: Syntax check**

```bash
.\.venv\Scripts\python.exe -m py_compile chemkin_pyqt_gui_fluent.py
```
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: implement Thermo Compare parsing logic"
```

---

### Task 5: Implement plot + table update with tab switching

**Files:**
- Modify: `chemkin_pyqt_gui_fluent.py` — add methods after the parse methods

- [ ] **Step 1: Add thermo_calculator import**

Add `calculate_cp_h_s` to the existing thermo-related imports. The `thermo_parser` functions are already imported. Add after line 85:

```python
    from thermo_calculator import calculate_cp_h_s
```

And add the fallback in the except block:

```python
    calculate_cp_h_s = None
```

- [ ] **Step 2: Add utility methods and update logic**

Append after `_show_thermo_empty_state`:

```python
    def _get_thermo_coeffs_for_temperature(self, species_data, T):
        """Select high-T or low-T coefficients based on T."""
        T_ranges = species_data.get('T_ranges', [])
        coeffs_list = species_data.get('coeffs', [])
        if len(T_ranges) != 2 or len(coeffs_list) != 2:
            return None
        # coeffs_list[0] = high-T, coeffs_list[1] = low-T
        # T_ranges[1] = (T_mid, T_high), T_ranges[0] = (T_low, T_mid)
        if T_ranges[1][0] <= T <= T_ranges[1][1]:
            return coeffs_list[0]
        elif T_ranges[0][0] <= T < T_ranges[1][0]:
            return coeffs_list[1]
        return None

    def _get_thermo_style(self, species_index):
        """Cycle (color, linestyle) per species — always groups=1 behavior."""
        n_colors = len(CURVE_COLORS)
        n_styles = len(CURVE_LINE_STYLES)
        color = CURVE_COLORS[species_index % n_colors]
        ls_idx = species_index % n_styles
        return color, CURVE_LINE_STYLES[ls_idx], ls_idx

    def _update_thermo_tab_plot_and_table(self):
        tab_idx = self._thermo_tab_idx
        data = getattr(self, 'thermo_species_data', None)
        if not data:
            return

        try:
            T_min = float(self.thermo_tmin.text())
            T_max = float(self.thermo_tmax.text())
            table_txt = self.thermo_table_temps.text().strip()
            table_temps = [
                float(s.strip()) for s in table_txt.split(',') if s.strip()
            ]
            if not table_temps:
                self._info("Enter at least one table temperature.", "error")
                return
        except (ValueError, AttributeError) as e:
            self._info(f"Invalid temperature input: {e}", "error")
            return

        if T_min <= 0 or T_max <= T_min:
            self._info("T min must be > 0 and < T max.", "error")
            return

        species_names = list(data.keys())
        if not species_names:
            return

        # Sweep temperatures for plot
        T_plot = np.linspace(T_min, T_max, 200)

        # Y-axis labels
        y_labels = ["H (kJ/mol)", "S (J/mol·K)", "Cp (J/mol·K)"]
        y_label = y_labels[tab_idx]

        # Compute data per species
        species_curves = {}
        for sn in species_names:
            sd = data[sn]
            vals = []
            for T in T_plot:
                coeffs = self._get_thermo_coeffs_for_temperature(sd, T)
                if coeffs is None:
                    vals.append(np.nan)
                else:
                    Cp, H, S = calculate_cp_h_s(T, coeffs)
                    if Cp is None:
                        vals.append(np.nan)
                    elif tab_idx == 0:
                        vals.append(H / 1000.0)  # kJ/mol
                    elif tab_idx == 1:
                        vals.append(S)  # J/mol·K
                    else:
                        vals.append(Cp)  # J/mol·K
            species_curves[sn] = np.array(vals)

        # Update plot
        pw = self.thermo_plot_card.plot_widget
        pw.clear()
        legend = pw.addLegend(offset=(10, 10),
            labelTextColor=self._thermo_legend_text_color(),
            brush=self._thermo_legend_brush())
        legend.setParentItem(pw.plotItem.vb)

        for si, sn in enumerate(species_names):
            y_vals = species_curves[sn]
            mask = ~np.isnan(y_vals)
            if not np.any(mask):
                continue
            color, pen_style, _ = self._get_thermo_style(si)
            pen = pg.mkPen(color=color, width=2, style=pen_style)
            pw.plot(T_plot[mask], y_vals[mask], pen=pen, name=sn)

        pw.setLabel("left", y_label)
        pw.setLabel("bottom", "T (K)")

        # Update table
        self._update_thermo_table(species_names, data, table_temps, tab_idx)

    def _thermo_legend_text_color(self):
        if isDarkTheme():
            return pg.mkColor('#e8ecf4')
        return pg.mkColor('#2c3e50')

    def _thermo_legend_brush(self):
        if isDarkTheme():
            return pg.mkBrush('#222630cc')
        return pg.mkBrush('#ffffffe6')

    def _update_thermo_table(self, species_names, data, table_temps, tab_idx):
        t = self.thermo_table
        t.setUpdatesEnabled(False)
        t.clear()
        n_rows = len(table_temps)
        n_cols = 2 + len(species_names)  # T(K) + Style + species value columns
        t.setRowCount(n_rows)
        t.setColumnCount(n_cols)
        t.setHorizontalHeaderLabels(
            ["T (K)"] + ["Style"] + species_names)
        t.verticalHeader().setVisible(False)

        for row_i, T in enumerate(table_temps):
            T_item = QTableWidgetItem(f"{T:.1f}")
            T_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setItem(row_i, 0, T_item)

            for si, sn in enumerate(species_names):
                sd = data[sn]
                coeffs = self._get_thermo_coeffs_for_temperature(sd, T)
                val = None
                if coeffs is not None:
                    Cp, H, S = calculate_cp_h_s(T, coeffs)
                    if Cp is not None:
                        if tab_idx == 0:
                            val = H / 1000.0
                        elif tab_idx == 1:
                            val = S
                        else:
                            val = Cp

                col = 2 + si  # offset for T + Style
                item = QTableWidgetItem(
                    f"{val:.2f}" if val is not None else "N/A")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
                if val is not None:
                    _, _, ls_idx = self._get_thermo_style(si)
                    color = QColor(CURVE_COLORS[si % len(CURVE_COLORS)])
                    item.setForeground(color)
                t.setItem(row_i, col, item)

        # Style column
        for si, sn in enumerate(species_names):
            _, _, ls_idx = self._get_thermo_style(si)
            style_item = QTableWidgetItem(LINESTYLE_LABELS.get(ls_idx, "—"))
            style_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            color = QColor(CURVE_COLORS[si % len(CURVE_COLORS)])
            style_item.setForeground(color)
            # Style col in row 0
            if si < n_rows:
                t.setItem(si, 1, style_item)

        # Resize: T col fixed, Style col fixed, value cols stretch
        header = t.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(2, n_cols):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        t.setUpdatesEnabled(True)

    def _on_thermo_tab_changed(self, idx):
        self._thermo_tab_idx = idx
        self._update_thermo_tab_plot_and_table()

    def _on_thermo_export_image_clicked(self):
        if not getattr(self, 'thermo_species_data', None):
            self._info("No data to export.", "warning")
            return
        y_labels = ["H (kJ/mol)", "S (J/mol·K)", "Cp (J/mol·K)"]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            pw = self.thermo_plot_card.plot_widget
            pw.setLabel("left", y_labels[self._thermo_tab_idx])
            ex = pg.exporters.ImageExporter(pw.plotItem)
            ex.parameters()['width'] = 1600
            ex.export(path)
            self._info(f"Exported to {path}")
        except Exception as e:
            self._info(f"Export error: {e}", "error")
```

- [ ] **Step 3: Syntax check**

```bash
.\.venv\Scripts\python.exe -m py_compile chemkin_pyqt_gui_fluent.py
```
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: implement Thermo Compare plot, table, tab switching and export"
```

---

### Task 6: Add theme-adaptive pyqtgraph settings for thermo page

**Files:**
- Modify: `chemkin_pyqt_gui_fluent.py` — update `_on_theme_toggled` and add thermo plot theme method

- [ ] **Step 1: Add `_apply_thermo_plot_theme` method**

Add after `_on_thermo_export_image_clicked`:

```python
    def _apply_thermo_plot_theme(self, dark):
        pw = self.thermo_plot_card.plot_widget
        if dark:
            pw.setBackground('#1a1d23')
            fg = '#e8ecf4'
        else:
            pw.setBackground('#f0f2f5')
            fg = '#2c3e50'
        for axis_name in ['left', 'bottom']:
            axis = pw.getAxis(axis_name)
            axis.setPen(pg.mkPen(color=fg))
            axis.setTextPen(pg.mkPen(color=fg))
        pw.getAxis('left').setGrid(200)
        pw.getAxis('bottom').setGrid(200)
```

- [ ] **Step 2: Update `_on_theme_toggled` to include thermo plot**

After the existing `self.table_card.apply_theme(dark)` line (line 1049), add:

```python
        if hasattr(self, 'thermo_plot_card'):
            self._apply_thermo_plot_theme(dark)
```

- [ ] **Step 3: Syntax check**

```bash
.\.venv\Scripts\python.exe -m py_compile chemkin_pyqt_gui_fluent.py
```
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "feat: add theme-adaptive pyqtgraph settings for Thermo Compare"
```

---

### Task 7: Verification — launch GUI and test manually

- [ ] **Step 1: Launch GUI**

```bash
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "chemkin_pyqt_gui_fluent.py" -WorkingDirectory "D:\RCCompare"
```

- [ ] **Step 2: Verify navigation**
  - Sidebar shows: Rate | Thermo Compare (top), Settings (bottom)
  - Click "Thermo Compare" → page loads with empty state "No Data Yet"

- [ ] **Step 3: Verify parsing**
  - Paste sample NASA data for 2 species
  - Click "Parse & Plot" → plot shows curves, table populates
  - Click H/S/Cp tabs → plot and table update correctly
  - Toggle dark theme → thermo plot adapts

- [ ] **Step 4: Commit final polish if needed**

```bash
git add chemkin_pyqt_gui_fluent.py
git commit -m "fix: final Thermo Compare polish"
```
