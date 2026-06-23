# Thermo Compare Page — Design Spec

**Date:** 2026-06-23
**Feature Branch:** `feature/fluent-gui`
**Target File:** `chemkin_pyqt_gui_fluent.py`

## Overview

Add a third navigation page "Thermo Compare" to the Fluent GUI sidebar. Users paste NASA-format thermodynamic polynomial data for one or more species, and compare enthalpy (H), entropy (S), or specific heat (Cp) vs temperature on interactive plots and tables.

## Navigation

- Rename existing "Home" nav item to **"Rate"** (`FluentIcon.HOME`)
- New item **"Thermo Compare"** (`FluentIcon.LINK`), inserted below Rate, `NavigationItemPosition.TOP`
- Settings icon unchanged

## Page Layout

Identical skeleton to Rate page:
- `QHBoxLayout` with 0 margin/spacing
- Left: `SmoothScrollArea` (fixed 420px) containing cards
- Right: `QSplitter` (vertical, `2:3` stretch) with pyqtgraph `PlotWidget` + `QTableWidget`

### Left Panel (Cards, top to bottom)

1. **Hero Header** — same gradient card as Rate page: icon, title "Thermo Compare", subtitle "NASA Polynomial Comparison", v1.3 badge

2. **Pill TabBar** — 3 horizontal pills using `qfluentwidgets.SegmentedWidget` or `PillPushButton[]`:
   | Pill | Parameter | Y-axis label | Unit |
   |------|-----------|-------------|------|
   | H | Enthalpy | H (kJ/mol) | kJ/mol (value / 1000) |
   | S | Entropy | S (J/mol·K) | J/mol·K |
   | Cp | Specific Heat | Cp (J/mol·K) | J/mol·K |
   - `currentChanged` signal triggers replot + retable

3. **NASA Polynomial Input Card** — `QTextEdit` with placeholder, multi-species paste area (standard NASA 7-coefficient format, one species per 4-line block)

4. **Controls Card** — `QLineEdit` fields:
   - T min (default 300)
   - T max (default 3500)
   - Table temperatures (comma-separated, default `300, 500, 700, 1000, 1500, 2000, 2500, 3000`)

5. **Parse & Plot button** — `PrimaryPushButton` with `FluentIcon.PLAY`, full width, repeatable (no disable after parse)

### Right Panel

- **PlotWidget** (pyqtgraph) — one curve per species, Y-axis label switches with tab
- **QTableWidget** — rows = T values, columns = species (one per species + Style column), stretch mode for value columns
- **Legend** — semi-transparent background `#222630cc` (dark) / white 85% (light), same styling as Rate page
- **Empty state** — "No Data Yet" placeholder shown before first parse, hidden after

### Color & Line Styles

- Reuse module-level `CURVE_COLORS` and `CURVE_LINE_STYLES` (already defined)
- Reuse `_get_reaction_style(index, group_index)` logic: cycle through (color, style) pairs for uniqueness per species. No grouping concept needed (always `groups=1` behavior).
- Table "Style" column shows line-style Unicode indicator (e.g., ── for solid) matching plot
- Plot legend entries colored to match curves

## Data Flow

```
User pastes NASA text in QTextEdit
  ↓ Parse button clicked
Write text to temp file → read_nasa_polynomials(temp_path) → delete temp file
  (read_nasa_polynomials accepts filepath only; tempfile module for cross-platform)
  → self.thermo_species_data: dict {species_name: {coeffs: [high_T_coeffs, low_T_coeffs], T_ranges: [...]}}
  ↓ _update_thermo_plot_and_table()
For each species, for each T in sweep range:
  thermo_calculator.calculate_cp_h_s(T, active_coeffs)
  → Cp, H, S
  ↓ Select H/S/Cp based on current tab index
Plot: enumerate species → style = _get_reaction_style(idx, 0) → pen color + dash pattern
Table: species as column headers, T values as rows
```

- No dependency on global `self.thermo_data` / `therm.dat` — user input is self-contained
- `thermo_parser.py` and `thermo_calculator.py` are zero-change — reused via temp-file wrapper

## Tab Switching

- `_on_tab_changed(index)`:
  1. Guard: if `self.thermo_species_data` is None, return
  2. Recompute all parameter arrays using `calculate_cp_h_s`
  3. Clear plot, redraw all curves with updated Y data (X unchanged)
  4. Update Y-axis label
  5. Clear table, repopulate with new parameter values
- No re-parse needed on tab switch — only the displayed parameter changes

## Theme Support

- Follows global `setTheme()` — light/dark toggle from Settings page applies automatically
- Pyqtgraph background, axis colors, legend styling adapt to `themeColor()`
- All `SimpleCardWidget` children inherit theme

## Dialogs

No new dialogs needed for this feature.

## Non-Goals

- No reverse rate / equilibrium calculation (thermo-only)
- No Excel export (can be added later)
- No pressure-dependent thermo
- No experimental data import

## API Dependencies

| Module | Function | Change |
|--------|----------|--------|
| `thermo_parser.py` | `read_nasa_polynomials()` | None — accepts file path or string-like |
| `thermo_calculator.py` | `calculate_cp_h_s()` | None |
| `chemkin_parser.py` | N/A | Not used |
| `qfluentwidgets` | `SegmentedWidget`, `PillPushButton` | New import |

## Implementation File Changes

Single file: `chemkin_pyqt_gui_fluent.py`
- New method: `_build_thermo_compare()` (~150 lines)
- New methods: `_parse_thermo_input()`, `_update_thermo_plot_and_table()`, `_on_thermo_tab_changed(idx)`, `_show_thermo_empty_state()`
- Modify `__init__`: add `self._thermo_compare` widget, nav registration
- Rename: `"Home"` → `"Rate"` in nav `addSubInterface` call
