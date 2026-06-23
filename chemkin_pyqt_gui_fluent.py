"""
CHEMKIN Rate Viewer - QFluentWidgets Edition
==============================================
Modern Fluent Design GUI for chemical kinetics mechanism analysis.
"""
import colorsys
import sys
import os
import re
import shutil
import tempfile
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidgetItem,
    QHeaderView, QSizePolicy, QFileDialog, QMenu, QColorDialog, QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QAction, QKeySequence, QShortcut, QPalette, QIcon, QPixmap

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pyqtgraph as pg
pg.setConfigOption('background', '#f0f2f5')
pg.setConfigOption('foreground', '#6c757d')
pg.setConfigOption('antialias', True)
pg.setConfigOption('leftButtonPan', False)

import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#f0f2f5'
matplotlib.rcParams['axes.facecolor'] = '#ffffff'
matplotlib.rcParams['axes.edgecolor'] = '#dce1e8'
matplotlib.rcParams['axes.labelcolor'] = '#2c3e50'
matplotlib.rcParams['xtick.color'] = '#6c757d'
matplotlib.rcParams['ytick.color'] = '#6c757d'
matplotlib.rcParams['text.color'] = '#2c3e50'
matplotlib.rcParams['grid.color'] = '#dce1e8'
matplotlib.rcParams['grid.alpha'] = 0.5
matplotlib.rcParams['legend.facecolor'] = '#ffffff'
matplotlib.rcParams['legend.edgecolor'] = '#dce1e8'
matplotlib.rcParams['legend.fontsize'] = 'small'
matplotlib.rcParams['legend.labelcolor'] = '#2c3e50'

# QFluentWidgets
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

# Excel export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Backend modules
try:
    from chemkin_parser import parse_chemkin_mechanism
    from rate_calculator import (
        calculate_arrhenius_rate, calculate_plog_rate, calculate_troe_rate,
        get_third_body_concentration, R_cal,
        get_reaction_thermo_properties, calculate_equilibrium_constant_kp,
        calculate_delta_n_gas, calculate_equilibrium_constant_kc,
        calculate_reverse_rate_constant,
        merge_duplicate_reactions, calculate_merged_duplicate_rate,
        _convert_ea_to_cal_per_mol,
    )
    from thermo_parser import (
        read_nasa_polynomials, append_nasa_polynomial_from_string,
        is_valid_nasa_polynomial_string, reload_thermo_if_changed,
    )
    from thermo_calculator import calculate_cp_h_s
except ImportError as e:
    print(f"Error importing modules: {e}")
    parse_chemkin_mechanism = None
    read_nasa_polynomials = None
    append_nasa_polynomial_from_string = None
    is_valid_nasa_polynomial_string = None
    reload_thermo_if_changed = None
    calculate_cp_h_s = None
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
    _convert_ea_to_cal_per_mol = None

THERMO_DATA_DIRNAME = "CHEMKIN_RateViewer"

# Shared color palette for plot curves and table rows
CURVE_COLORS = [
    '#4a90e2', '#e2764a', '#4ae28a', '#e24a90', '#904ae2',
    '#e2d04a', '#4ae2d0', '#e24a4a', '#8ae24a', '#4a6ee2',
]

CURVE_LINE_STYLES = [
    Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine,
    Qt.PenStyle.DashDotLine, Qt.PenStyle.DotLine,
]

LINESTYLE_LABELS = {0: "—", 1: "– –", 2: "–·–", 3: "···"}


# ── Thermo file utilities ────────────────────────────────────────────

def resolve_bundled_thermo_filepath():
    if getattr(sys, "frozen", False):
        app_dir = getattr(sys, "_MEIPASS",
                          os.path.dirname(os.path.abspath(sys.executable)))
        p = os.path.join(app_dir, "therm.dat")
        if os.path.exists(p):
            return p
        p = os.path.join(os.path.dirname(os.path.abspath(sys.executable)),
                         "therm.dat")
        if os.path.exists(p):
            return p
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, "therm.dat")


def resolve_thermo_filepath():
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        local = os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return os.path.join(local, THERMO_DATA_DIRNAME, "therm.dat")


def ensure_persistent_thermo_file(persistent_fp=None, baseline_fp=None):
    persistent_fp = persistent_fp or resolve_thermo_filepath()
    baseline_fp = baseline_fp or resolve_bundled_thermo_filepath()
    os.makedirs(os.path.dirname(persistent_fp), exist_ok=True)
    if not os.path.exists(persistent_fp):
        shutil.copy2(baseline_fp, persistent_fp)
    return persistent_fp


def get_thermo_file_signature(filepath):
    try:
        st = os.stat(filepath)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


# ── NASA 7-coefficient validation ────────────────────────────────────

def validate_nasa7_format(input_text):
    errors, species_data = [], {}
    lines = input_text.strip().split('\n')
    if not lines:
        return False, ["Input is empty."], species_data
    if len(lines) % 4 != 0:
        return False, [f"Expected multiple of 4 lines, got {len(lines)}."], species_data

    for si in range(len(lines) // 4):
        sl = lines[si * 4: si * 4 + 4]
        ln = [si * 4 + i + 1 for i in range(4)]
        try:
            name = sl[0][0:18].strip()
            if not name:
                return False, [f"Line {ln[0]}: no species name."], {}
            tp = r'[\d.]+(?:[eE][+-]?\d+)?'
            tm = re.findall(tp, sl[0])
            if len(tm) < 2:
                return False, [f"Line {ln[0]}: need >=2 temperature values."], {}
            float(tm[0]); float(tm[1])
            for i in range(1, 4):
                if not sl[i].strip():
                    return False, [f"Line {ln[i]}: empty."], {}
                cp = r'[+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.\d*|[+-]?\d+'
                cm = re.findall(cp, sl[i])
                if not cm:
                    return False, [f"Line {ln[i]}: no coefficients."], {}
                float(cm[0])
            l1 = re.sub(r'(\d)([a-z])(\s+\d)',
                        lambda m: m.group(1) + m.group(2).upper() + m.group(3),
                        sl[0])
            species_data[name.upper()] = [l1] + sl[1:]
        except Exception as e:
            return False, [f"Line {ln[0]}: {e}"], {}
    return True, errors, species_data


# ── PyQtGraph card ───────────────────────────────────────────────────

class PyQtGraphCard(SimpleCardWidget):
    """Card containing a pyqtgraph PlotWidget for interactive charting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        # Header row: title + export button
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        self._title = SubtitleLabel("Reaction Rates vs Temperature")
        hdr.addWidget(self._title)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Temperature', units='K')
        self.plot_widget.setLabel('left', 'log₁₀(k)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(60, 10),
            brush=pg.mkBrush('#222630cc'), pen=pg.mkPen('#3a3f4b'),
            labelTextColor='#e8ecf4', labelTextSize='9pt')
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

        self.coord_label = pg.TextItem('', anchor=(0, 1), color='#e8ecf4',
            fill=pg.mkBrush('#222630cc'))
        self.coord_label.setVisible(False)
        self.plot_widget.addItem(self.coord_label)

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
            self.coord_label.setText(f'T={x:.0f} K  log(k)={y:.2f}')
            self.coord_label.setPos(x, y)
            self.coord_label.setVisible(True)
        else:
            self.vline.setVisible(False)
            self.hline.setVisible(False)
            self.coord_label.setVisible(False)

    def clear_plot(self):
        self.plot_widget.clear()
        # Re-add crosshair items after clear
        self.plot_widget.addItem(self.vline)
        self.plot_widget.addItem(self.hline)
        self.plot_widget.addItem(self.coord_label)

    def set_title(self, t):
        self._title.setText(t)

    def add_title_button(self, btn):
        """Add a button inline with the title (right-aligned)."""
        self.layout().itemAt(0).layout().addWidget(btn)

    def apply_theme(self, dark=True):
        bg = '#1a1d23' if dark else '#f0f2f5'
        fg = '#b0b8c8' if dark else '#6c757d'
        pg.setConfigOption('background', bg)
        pg.setConfigOption('foreground', fg)
        self.plot_widget.setBackground(bg)
        for axis_name in ('bottom', 'left'):
            ax = self.plot_widget.getAxis(axis_name)
            ax.setPen(pg.mkPen(fg))
            ax.setTextPen(pg.mkPen(fg))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)


# ── Rate table card ──────────────────────────────────────────────────

class RateTableCard(SimpleCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(SubtitleLabel("Rate Constants"))
        self.table = TableWidget()
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setAlternatingRowColors(True)
        self.table.setItemDelegate(TableItemDelegate(self.table))
        self.table.setMinimumHeight(150)
        self.table.setSizePolicy(QSizePolicy.Policy.Preferred,
                                 QSizePolicy.Policy.MinimumExpanding)
        # Fluent-style table QSS
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #222630;
                alternate-background-color: #2e3440;
                color: #e8ecf4;
                gridline-color: #3a3f4b;
                border: 1px solid #3a3f4b;
                border-radius: 8px;
                selection-background-color: #4a90e2;
                selection-color: #ffffff;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 9pt;
            }
            QTableWidget::item {
                padding: 6px 10px;
                border-bottom: 1px solid #2e3440;
            }
            QTableWidget::item:hover {
                background-color: #3a3f4b;
            }
            QTableWidget::item:selected {
                background-color: #4a90e2;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1a1d23;
                color: #e8ecf4;
                padding: 8px 10px;
                border: none;
                border-bottom: 2px solid #4a90e2;
                border-right: 1px solid #2e3440;
                font-weight: bold;
                font-size: 9pt;
            }
            QHeaderView::section:last {
                border-right: none;
            }
            QScrollBar:vertical {
                background: #1a1d23; width: 12px; border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #3a3f4b; border-radius: 6px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a90e2;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: #1a1d23; height: 12px; border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #3a3f4b; border-radius: 6px; min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #4a90e2;
            }
        """)
        self.table_temperatures = [500.0, 1000.0, 1500.0, 2000.0]
        self._setup_columns(self.table_temperatures)
        lay.addWidget(self.table)

        # Copy/paste support
        self.table.setSelectionMode(
            TableWidget.SelectionMode.ExtendedSelection)
        sc = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        sc.activated.connect(self._copy_selection)
        self.table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            self._show_context_menu)

    def apply_theme(self, dark=True):
        """Update table theme."""
        if dark:
            bg, alt, fg, grid, hdr_bg = '#222630', '#2e3440', '#e8ecf4', '#3a3f4b', '#1a1d23'
        else:
            bg, alt, fg, grid, hdr_bg = '#ffffff', '#f8f9fa', '#2c3e50', '#dce1e8', '#f0f2f5'
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {bg}; alternate-background-color: {alt};
                color: {fg}; gridline-color: {grid};
                border: 1px solid {grid}; border-radius: 8px;
                selection-background-color: #4a90e2; selection-color: #ffffff;
                font-family: "Consolas", "Courier New", monospace; font-size: 9pt;
            }}
            QTableWidget::item {{ padding: 6px 10px; border-bottom: 1px solid {grid}; }}
            QTableWidget::item:hover {{ background-color: #e8e8e8; }}
            QTableWidget::item:selected {{ background-color: #4a90e2; color: #ffffff; }}
            QHeaderView::section {{
                background-color: {hdr_bg}; color: {fg};
                padding: 8px 10px; border: none;
                border-bottom: 2px solid #4a90e2; border-right: 1px solid {grid};
                font-weight: bold; font-size: 9pt;
            }}
        """)

    def _setup_columns(self, temps):
        self.table_temperatures = temps
        n = len(temps)
        self.table.setColumnCount(4 + n)
        hdr = ["Reaction", "Style", "Pressure (atm)", "Calc Rev?"] + \
              [f"k @ {t:.0f}K" for t in temps]
        self.table.setHorizontalHeaderLabels(hdr)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 300)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 50)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 100)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(3, 90)
        for i in range(n):
            h.setSectionResizeMode(4 + i, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.verticalHeader().setVisible(False)
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        h.setStretchLastSection(False)
        fnt = QFont("Consolas", 9)
        if not fnt.exactMatch():
            fnt = QFont("Courier New", 9)
        self.table.setFont(fnt)

    def _copy_selection(self):
        """Copy selected table cells to clipboard (with headers)."""
        sel = self.table.selectedRanges()
        if not sel:
            return
        mr = min(r.topRow() for r in sel)
        xr = max(r.bottomRow() for r in sel)
        mc = min(r.leftColumn() for r in sel)
        xc = max(r.rightColumn() for r in sel)
        rows = []
        # Header row
        hdr = []
        for c in range(mc, xc + 1):
            h = self.table.horizontalHeaderItem(c)
            hdr.append(h.text() if h else "")
        rows.append('\t'.join(hdr))
        # Data rows
        for r in range(mr, xr + 1):
            rd = []
            for c in range(mc, xc + 1):
                it = self.table.item(r, c)
                rd.append(it.text() if it else "")
            rows.append('\t'.join(rd))
        QApplication.clipboard().setText('\n'.join(rows))

    def _show_context_menu(self, pos):
        """Right-click context menu for the table."""
        menu = QMenu(self)
        copy_act = QAction("Copy", self)
        copy_act.setShortcut(QKeySequence.StandardKey.Copy)
        copy_act.triggered.connect(self._copy_selection)
        menu.addAction(copy_act)
        sel_all_act = QAction("Select All", self)
        sel_all_act.setShortcut(QKeySequence.StandardKey.SelectAll)
        sel_all_act.triggered.connect(self.table.selectAll)
        menu.addAction(sel_all_act)
        menu.exec(self.table.viewport().mapToGlobal(pos))


# ── Application palette ──────────────────────────────────────────────

def _apply_app_palette(dark=True):
    """Set QApplication palette to match the theme.

    QFluentWidgets setTheme() only manages its own widget styles.
    We must set the app palette for all other Qt widgets
    (QPlainTextEdit, QLineEdit, QCheckBox, QDialog, etc.)
    to prevent them from appearing in light/white colors.
    """
    pal = QPalette()
    if dark:
        pal.setColor(QPalette.ColorRole.Window, QColor('#1a1d23'))
        pal.setColor(QPalette.ColorRole.WindowText, QColor('#e8ecf4'))
        pal.setColor(QPalette.ColorRole.Base, QColor('#222630'))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor('#2e3440'))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor('#222630'))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor('#e8ecf4'))
        pal.setColor(QPalette.ColorRole.Text, QColor('#e8ecf4'))
        pal.setColor(QPalette.ColorRole.Button, QColor('#222630'))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor('#e8ecf4'))
        pal.setColor(QPalette.ColorRole.BrightText, QColor('#4a90e2'))
        pal.setColor(QPalette.ColorRole.Link, QColor('#4a90e2'))
        pal.setColor(QPalette.ColorRole.Highlight, QColor('#4a90e2'))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor('#7a8599'))
    else:
        pal.setColor(QPalette.ColorRole.Window, QColor('#f0f2f5'))
        pal.setColor(QPalette.ColorRole.WindowText, QColor('#2c3e50'))
        pal.setColor(QPalette.ColorRole.Base, QColor('#ffffff'))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor('#f8f9fa'))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor('#ffffff'))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor('#2c3e50'))
        pal.setColor(QPalette.ColorRole.Text, QColor('#2c3e50'))
        pal.setColor(QPalette.ColorRole.Button, QColor('#ffffff'))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor('#2c3e50'))
        pal.setColor(QPalette.ColorRole.BrightText, QColor('#4a90e2'))
        pal.setColor(QPalette.ColorRole.Link, QColor('#4a90e2'))
        pal.setColor(QPalette.ColorRole.Highlight, QColor('#4a90e2'))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor('#95a5a6'))
    QApplication.instance().setPalette(pal)


# ── Main FluentWindow ────────────────────────────────────────────────

class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CHEMKIN Rate Viewer - v1.3")
        self.setWindowIcon(QIcon(self._resolve_icon_path()))
        self.resize(1400, 900)

        # Thermo data init
        self.bundled_thermo_filepath = resolve_bundled_thermo_filepath()
        self.thermo_filepath = resolve_thermo_filepath()
        self._thermo_file_signature = None
        self.thermo_data = None

        # Build pages
        self._home = QWidget()
        self._home.setObjectName("home")
        self._build_home()

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

        setTheme(Theme.LIGHT)

        # Enable Windows 11 Mica effect (native frosted glass material)
        # Gracefully falls back on Windows 10 or unsupported systems
        try:
            self.setMicaEffectEnabled(True)
        except Exception:
            pass  # Mica not available on this system

        # Set application palette for all non-QFluentWidgets widgets
        _apply_app_palette(dark=False)

        # Apply theme-appropriate QSS
        self._apply_global_dark_qss(dark=False)

        # Load thermo data
        try:
            ensure_persistent_thermo_file(self.thermo_filepath,
                                          self.bundled_thermo_filepath)
            self._load_thermo_data()
        except Exception as e:
            self.thermo_data = {}
            self._info(f"Thermo init error: {e}", "error")

    def _resolve_icon_path(self):
        if getattr(sys, "frozen", False):
            app_dir = getattr(sys, "_MEIPASS",
                              os.path.dirname(os.path.abspath(sys.executable)))
            p = os.path.join(app_dir, "logo.png")
            if os.path.exists(p):
                return p
            p = os.path.join(os.path.dirname(os.path.abspath(sys.executable)),
                             "logo.png")
            if os.path.exists(p):
                return p
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logo.png")

    # ── Home page layout ─────────────────────────────────────────────

    def _build_home(self):
        main_lay = QHBoxLayout(self._home)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Left scroll panel
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

        # ── Hero header ──
        hero = QWidget()
        hero.setObjectName("heroHeader")
        hero_lay = QHBoxLayout(hero)
        hero_lay.setContentsMargins(20, 16, 20, 16)
        hero_lay.setSpacing(14)
        logo_lbl = QLabel()
        logo_path = self._resolve_icon_path()
        if os.path.exists(logo_path):
            px = QPixmap(logo_path).scaled(44, 44, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(px)
        logo_lbl.setFixedSize(44, 44)
        hero_lay.addWidget(logo_lbl)
        txt_col = QVBoxLayout()
        txt_col.setSpacing(2)
        tt = TitleLabel("CHEMKIN Rate Viewer")
        txt_col.addWidget(tt)
        sub_row = QHBoxLayout()
        sub_row.setSpacing(10)
        sub_row.addWidget(CaptionLabel("Chemical kinetics mechanism analysis"))
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

        # CHEMKIN input card
        ic = SimpleCardWidget()
        il = QVBoxLayout(ic)
        il.setContentsMargins(16, 16, 16, 16)
        il.addWidget(StrongBodyLabel("CHEMKIN Input"))
        self.chemkin_input = PlainTextEdit()
        self.chemkin_input.setMinimumHeight(180)
        self.chemkin_input.setPlaceholderText("Paste CHEMKIN mechanism here...")
        sample = """UNITS CAL/MOLE
O + H2 <=> H + OH          1.0E10  0.5   15000.0
CH4 + O2 = CH3 + HO2       1.0E12  0.0   50000.0
    PLOG / 0.1   1.0E11  0.0  48000.0 /
    PLOG / 1.0   1.0E12  0.0  50000.0 /
    PLOG / 10.0  1.0E13  0.0  52000.0 /
H + O2 (+M) = HO2 (+M)    1.475E12  0.6  0.0
    LOW / 6.366E20  -1.72  524.8 /
    TROE / 0.8  1E-30  1E30 /"""
        self.chemkin_input.setPlainText(sample)
        il.addWidget(self.chemkin_input)
        ll.addWidget(ic)

        # Controls card
        cc = SimpleCardWidget()
        cl = QVBoxLayout(cc)
        cl.setContentsMargins(16, 16, 16, 16)
        cl.setSpacing(8)
        cl.addWidget(StrongBodyLabel("Controls"))

        self.temp_min = LineEdit(); self.temp_min.setText("300")
        self.temp_max = LineEdit(); self.temp_max.setText("2500")
        self.pres_min = LineEdit(); self.pres_min.setText("1.0")
        self.pres_max = LineEdit(); self.pres_max.setText("10.0")
        self.pres_steps = LineEdit(); self.pres_steps.setText("2")
        self.table_temps = LineEdit(); self.table_temps.setText("500, 1000, 1500, 2000")
        self.group_count = LineEdit(); self.group_count.setText("1")
        self.intra_linestyle_cb = CheckBox("Intra-group linestyle variation")
        self.use_inv_temp_cb = CheckBox("Use 1000/T as X-axis")

        for lbl, w in [("T min (K)", self.temp_min), ("T max (K)", self.temp_max),
                        ("P min (atm)", self.pres_min), ("P max (atm)", self.pres_max),
                        ("Log P steps", self.pres_steps),
                        ("Table temps (K)", self.table_temps),
                        ("Groups", self.group_count)]:
            r = QHBoxLayout()
            r.addWidget(CaptionLabel(lbl)); r.addWidget(w); r.addStretch()
            cl.addLayout(r)
        cl.addWidget(self.intra_linestyle_cb)
        cl.addWidget(self.use_inv_temp_cb)
        ll.addWidget(cc)

        # Buttons
        self.plot_btn = PrimaryPushButton(FluentIcon.PLAY, "Parse, Calculate & Plot")
        self.plot_btn.setMinimumHeight(40)
        self.plot_btn.clicked.connect(self._on_plot_clicked)
        ll.addWidget(self.plot_btn)

        self.detail_btn = PushButton(FluentIcon.VIEW, "View Rate Constant Details")
        self.detail_btn.setMinimumHeight(36)
        self.detail_btn.clicked.connect(self._on_detail_clicked)
        self.detail_btn.setEnabled(False)
        ll.addWidget(self.detail_btn)

        self.rev_btn = PushButton(FluentIcon.UPDATE, "Experimental Reverse Rate")
        self.rev_btn.setMinimumHeight(36)
        self.rev_btn.clicked.connect(self._on_exp_reverse_clicked)
        ll.addWidget(self.rev_btn)

        self.save_btn = PushButton(FluentIcon.SAVE, "Save Data to Excel")
        self.save_btn.setMinimumHeight(36)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        if not PANDAS_AVAILABLE:
            self.save_btn.setToolTip("Requires pandas + openpyxl")
        ll.addWidget(self.save_btn)

        ll.addStretch()
        scroll.setWidget(lw)
        main_lay.addWidget(scroll)

        # Right panel
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)
        self.plot_card = PyQtGraphCard()
        rl.addWidget(self.plot_card, 2)

        # Empty state placeholder (hidden after first parse)
        self.empty_placeholder = QWidget()
        self.empty_placeholder.setObjectName("emptyPlaceholder")
        ep_lay = QVBoxLayout(self.empty_placeholder)
        ep_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.setSpacing(12)
        ep_icon = QLabel("📊")
        ep_icon.setStyleSheet("font-size: 48px; background: transparent;")
        ep_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(ep_icon)
        ep_title = BodyLabel("No Data Yet")
        ep_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_lay.addWidget(ep_title)
        ep_hint = CaptionLabel("Paste a CHEMKIN mechanism and click Parse to begin")
        ep_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ep_hint.setStyleSheet("color: #95a5a6; background: transparent;")
        ep_lay.addWidget(ep_hint)
        rl.addWidget(self.empty_placeholder, 3)

        # Export button inline in plot card header
        export_img_btn = PushButton(FluentIcon.PHOTO, "Export")
        export_img_btn.clicked.connect(self._on_export_image_clicked)
        self.plot_card.add_title_button(export_img_btn)

        self.table_card = RateTableCard()
        rl.addWidget(self.table_card, 3)
        main_lay.addWidget(rw, 1)

    # ── Settings page ────────────────────────────────────────────────

    def _build_settings(self):
        lay = QVBoxLayout(self._settings)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(16)
        lay.addWidget(TitleLabel("Settings"))

        tc = SimpleCardWidget()
        tl = QHBoxLayout(tc)
        tl.setContentsMargins(16, 16, 16, 16)
        tl.addWidget(BodyLabel("Dark Theme")); tl.addStretch()
        self.theme_sw = SwitchButton()
        self.theme_sw.setChecked(False)
        self.theme_sw.setOnText("Dark"); self.theme_sw.setOffText("Light")
        self.theme_sw.checkedChanged.connect(self._on_theme_toggled)
        tl.addWidget(self.theme_sw)
        lay.addWidget(tc)

        ac = SimpleCardWidget()
        al = QVBoxLayout(ac)
        al.setContentsMargins(16, 16, 16, 16)
        al.addWidget(StrongBodyLabel("About"))
        al.addWidget(BodyLabel("CHEMKIN Rate Viewer v1.3"))
        al.addWidget(CaptionLabel("QFluentWidgets Edition"))
        al.addWidget(CaptionLabel("中科院工程热物理研究所"))
        lay.addWidget(ac)
        lay.addStretch()

    # ── Thermo Compare page ──────────────────────────────────────────

    def _build_thermo_compare(self):
        main_lay = QHBoxLayout(self._thermo_compare)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Left scroll panel
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

        # Right panel
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(8)

        # Plot
        self.thermo_plot_card = PyQtGraphCard()
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
        ep_icon = QLabel("\U0001F4CA")
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

    # ── Global dark QSS ──────────────────────────────────────────────

    def _apply_global_dark_qss(self, dark=True):
        """Force dark/light theme on all standard Qt widgets that
        QFluentWidgets' setTheme doesn't fully cover.

        IMPORTANT: QSS type selectors match Qt C++ class names, NOT Python
        class names.  e.g. SimpleCardWidget's Qt class is QFrame, BodyLabel's
        is QLabel.  Always use Qt base class selectors.
        """
        if dark:
            bg, card_bg, input_bg, fg, border, accent = (
                '#1a1d23', '#222630', '#2e3440', '#e8ecf4', '#3a3f4b', '#4a90e2')
        else:
            bg, card_bg, input_bg, fg, border, accent = (
                '#f0f2f5', '#ffffff', '#f8f9fa', '#2c3e50', '#dce1e8', '#4a90e2')

        qss = f"""
            /* ── Left panel scroll area and content ── */
            QScrollArea#leftPanel {{
                background-color: {bg};
                border: none;
            }}
            QWidget#leftPanelContent {{
                background-color: {bg};
            }}

            /* ── Hero header ── */
            QWidget#heroHeader {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {card_bg}, stop:1 {bg});
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QWidget#heroHeader QLabel {{
                background: transparent;
            }}

            /* ── Card-like widgets (QFrame is the Qt base for SimpleCardWidget) ── */
            #leftPanelContent QFrame {{
                background-color: {card_bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}

            /* ── Labels ── */
            #leftPanelContent QLabel {{
                color: {fg};
                background-color: transparent;
            }}

            /* ── Line edits ── */
            QLineEdit {{
                background-color: {input_bg};
                color: {fg};
                border: 2px solid {border};
                border-radius: 6px;
                padding: 6px 10px;
                selection-background-color: {accent};
            }}
            QLineEdit:focus {{
                border: 2px solid {accent};
            }}

            /* ── Text edits (PlainTextEdit is QPlainTextEdit in Qt) ── */
            QPlainTextEdit {{
                background-color: {input_bg};
                color: {fg};
                border: 2px solid {border};
                border-radius: 6px;
                padding: 8px;
                selection-background-color: {accent};
                font-family: "Consolas", "Courier New", monospace;
            }}
            QPlainTextEdit:focus {{
                border: 2px solid {accent};
            }}

            /* ── Checkboxes ── */
            QCheckBox {{
                color: {fg};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px; height: 18px;
                border: 2px solid {border};
                border-radius: 4px;
                background-color: {input_bg};
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
                border: 2px solid {accent};
            }}

            /* ── Combo boxes ── */
            QComboBox {{
                background-color: {input_bg};
                color: {fg};
                border: 2px solid {border};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}

            /* ── Push buttons (non-primary) ── */
            #leftPanelContent QPushButton {{
                background-color: {card_bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            #leftPanelContent QPushButton:hover {{
                background-color: {input_bg};
                border: 1px solid {accent};
            }}
            #leftPanelContent QPushButton:pressed {{
                background-color: {accent};
                color: white;
            }}
            #leftPanelContent QPushButton:disabled {{
                color: #5a6070;
                border: 1px solid #3a3f4b;
            }}

            /* ── Navigation sidebar ── */
            NavigationInterface {{
                background-color: {bg};
                border-right: 1px solid {border};
            }}
            NavigationPanel {{
                background-color: {bg};
            }}
            NavigationTreeWidget {{
                background-color: transparent;
                border: none;
            }}
            NavigationPushButton {{
                color: {fg};
                background-color: transparent;
                border: none;
                border-radius: 6px;
                margin: 2px 4px;
            }}
            NavigationPushButton:hover {{
                background-color: {card_bg};
            }}
            NavigationPushButton[selected="true"] {{
                background-color: {accent};
                color: #ffffff;
            }}
            NavigationToolButton {{
                background-color: transparent;
                color: {fg};
                border: none;
                border-radius: 6px;
            }}
            NavigationToolButton:hover {{
                background-color: {card_bg};
            }}

            /* ── Scrollbars ── */
            QScrollBar:vertical {{
                background: {bg}; width: 12px; border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: #3a3f4b; border-radius: 6px; min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {accent};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background: {bg}; height: 12px; border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background: #3a3f4b; border-radius: 6px; min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {accent};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}

            /* ── Tooltips ── */
            QToolTip {{
                background-color: {card_bg};
                color: {fg};
                border: 1px solid {accent};
                border-radius: 4px;
                padding: 4px;
            }}

            /* ── Table header (global) ── */
            QHeaderView::section {{
                background-color: {bg};
                color: {fg};
                padding: 8px 10px;
                border: none;
                border-bottom: 2px solid {accent};
                font-weight: bold;
                font-size: 9pt;
            }}
            QHeaderView::section:hover {{
                background-color: {card_bg};
            }}

            /* ── Splitter handle ── */
            QSplitter::handle {{
                background-color: {border};
            }}
            QSplitter::handle:hover {{
                background-color: {accent};
            }}

            /* ── Table widget (global, for dialogs) ── */
            QTableWidget {{
                background-color: {card_bg};
                alternate-background-color: {input_bg};
                color: {fg};
                gridline-color: {border};
                border: 1px solid {border};
                border-radius: 8px;
                selection-background-color: {accent};
                selection-color: #ffffff;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 9pt;
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {border};
            }}
            QTableWidget::item:hover {{
                background-color: #3a3f4b;
            }}
            QTableWidget::item:selected {{
                background-color: {accent};
                color: #ffffff;
            }}
        """
        self.setStyleSheet(qss)

    # ── Notification helper ──────────────────────────────────────────

    def _info(self, msg, level="success", title=""):
        fn = getattr(InfoBar, level, InfoBar.info)
        fn(title=title or level.capitalize(), content=msg[:200],
           parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3000)

    # ── Theme toggle ─────────────────────────────────────────────────

    def _on_theme_toggled(self, dark):
        setTheme(Theme.DARK if dark else Theme.LIGHT)
        _apply_app_palette(dark)
        self._apply_global_dark_qss(dark)
        self.plot_card.apply_theme(dark)
        self.table_card.apply_theme(dark)
        if hasattr(self, 'parsed_reactions'):
            self._update_plot_with_reverse_rates()

    # ── Thermo data management ───────────────────────────────────────

    def _load_thermo_data(self):
        if read_nasa_polynomials is None:
            self._info("Thermo parser not available.", "error")
            self.thermo_data = {}
            self._thermo_file_signature = None
            return
        try:
            self.thermo_data = read_nasa_polynomials(self.thermo_filepath)
            self._thermo_file_signature = get_thermo_file_signature(
                self.thermo_filepath)
            if not self.thermo_data:
                self._info(f"No species loaded from {self.thermo_filepath}.",
                           "warning")
                self.thermo_data = {}
            else:
                self._info(f"Loaded {len(self.thermo_data)} species.")
        except Exception as e:
            self._info(f"Thermo load error: {e}", "error")
            self.thermo_data = {}

    def _reload_thermo_data_if_changed(self):
        if not reload_thermo_if_changed:
            return False
        new_data, new_sig, changed = reload_thermo_if_changed(
            self.thermo_filepath, self.thermo_data, self._thermo_file_signature)
        if changed:
            self.thermo_data = new_data
            self._thermo_file_signature = new_sig
            self._info(f"Thermo reloaded: {len(new_data)} species.")
        return changed

    # ── Thermo Compare: parsing ───────────────────────────────────────

    def _on_thermo_parse_clicked(self):
        text = self.thermo_input.toPlainText().strip()
        if not text:
            self._info("Paste NASA polynomial data first.", "warning")
            return
        try:
            lines = text.split('\n')
            if len(lines) % 4 != 0:
                self._info(
                    f"Expected multiple of 4 lines per species, got {len(lines)}.",
                    "error")
                return
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

    # ── Input parsing ────────────────────────────────────────────────

    def _parse_temperature_input(self):
        try:
            txt = self.table_temps.text().strip()
            if not txt:
                return [500.0, 1000.0, 1500.0, 2000.0]
            temps = []
            for s in txt.split(','):
                s = s.strip()
                if s:
                    v = float(s)
                    if v > 0:
                        temps.append(v)
                    else:
                        raise ValueError(f"Temperature must be positive: {v}")
            if not temps:
                raise ValueError("No valid temperatures.")
            return temps
        except ValueError as e:
            self._info(f"Invalid temperature: {e}", "error")
            return None

    def _get_plot_pressure_settings(self):
        try:
            p_min = float(self.pres_min.text())
            p_max = float(self.pres_max.text())
            p_steps = int(self.pres_steps.text())
            p_default = p_min
            use_range = False
            p_values = [p_default]
            if p_min > 0 and p_max >= p_min and p_steps > 0:
                use_range = True
                if p_steps == 1 or p_min == p_max:
                    p_values = np.array([p_min])
                else:
                    p_values = np.logspace(
                        np.log10(p_min), np.log10(p_max), num=p_steps)
            return p_values, use_range, p_default
        except (ValueError, ZeroDivisionError):
            return [1.0], False, 1.0

    # ── Rate calculation ─────────────────────────────────────────────

    def _calculate_rate_for_reaction(self, rd, T, P_atm):
        if not (calculate_arrhenius_rate and calculate_plog_rate
                and calculate_troe_rate and get_third_body_concentration):
            return None
        rt = rd.get('reaction_type', 'ARRHENIUS')
        try:
            if rd.get('duplicate_arrhenius_params') and calculate_merged_duplicate_rate:
                k = calculate_merged_duplicate_rate(rd, T, P_atm)
                if k is not None:
                    return k
            if rt == 'ARRHENIUS' and rd.get('arrhenius_params'):
                return calculate_arrhenius_rate(rd['arrhenius_params'], T)
            elif rt == 'PLOG' and rd.get('plog_data'):
                return calculate_plog_rate(rd['plog_data'], T, P_atm)
            elif rt == 'TROE' and rd.get('troe_data'):
                mc = get_third_body_concentration(P_atm, T)
                return calculate_troe_rate(rd['troe_data'], T, P_atm, M_conc=mc)
        except Exception:
            pass
        return None

    def _calculate_rate_array(self, rd, T_arr, P_atm):
        """Vectorized rate calc for Arrhenius; fallback loop for others."""
        rt = rd.get('reaction_type', 'ARRHENIUS')
        if rt == 'ARRHENIUS' and rd.get('arrhenius_params'):
            p = rd['arrhenius_params']
            A, n, Ea = p.get('A'), p.get('n'), p.get('Ea')
            if A is not None and n is not None and Ea is not None and _convert_ea_to_cal_per_mol:
                ea_cal = _convert_ea_to_cal_per_mol(Ea, p.get('units'))
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    return A * (T_arr ** n) * np.exp(-ea_cal / (R_cal * T_arr))
        return np.array([
            (k if k is not None and k > 1e-100 else np.nan)
            for k in (self._calculate_rate_for_reaction(rd, t, P_atm)
                      for t in T_arr)
        ])

    def _calculate_reverse_rate_for_plot(self, rd, T, P_atm):
        try:
            self._reload_thermo_data_if_changed()
            kf = self._calculate_rate_for_reaction(rd, T, P_atm)
            if kf is None or kf <= 0:
                return None
            dH, dS, dG, _, err = get_reaction_thermo_properties(
                rd, T, self.thermo_data)
            if err:
                return None
            Kp = calculate_equilibrium_constant_kp(dG, T)
            dn = calculate_delta_n_gas(rd, self.thermo_data)
            Kc = calculate_equilibrium_constant_kc(Kp, T, dn)
            if Kc is None:
                return None
            return calculate_reverse_rate_constant(kf, Kc)
        except Exception:
            return None

    def _calc_k_log_array(self, rd, T_vals, P, is_rev):
        """Calculate log10(k) array for plotting."""
        if is_rev:
            k_arr = np.array([
                (self._calculate_reverse_rate_for_plot(rd, t, P) or np.nan)
                for t in T_vals
            ])
        else:
            k_arr = self._calculate_rate_array(rd, T_vals, P)
        with np.errstate(divide='ignore', invalid='ignore'):
            k_log = np.where(
                (k_arr != None) & ~np.isnan(k_arr) & (k_arr > 1e-100),
                np.log10(k_arr), np.nan)
        return k_log

    # ── Table population ─────────────────────────────────────────────

    def _show_empty_state(self, show=True):
        self.empty_placeholder.setVisible(show)
        self.plot_card.setVisible(not show)
        self.table_card.setVisible(not show)

    def _update_rate_constant_table(self, parsed_reactions):
        self.table_card.table.setUpdatesEnabled(False)
        self.table_card.table.setRowCount(0)

        P_vals, use_range, P_def = self._get_plot_pressure_settings()
        self.parsed_reactions = parsed_reactions

        row = 0
        for ri, rd in enumerate(parsed_reactions):
            eq = rd.get('equation_string_cleaned',
                        rd.get('equation_string', 'N/A'))
            rt = rd.get('reaction_type', 'ARRHENIUS')
            pressures = P_vals if (use_range and rt in ('PLOG', 'TROE')) else [P_def]

            rd['table_rows'] = []
            rd['is_calculating_reverse'] = False
            rd['original_forward_rates'] = {}

            for pi, pres in enumerate(pressures):
                self.table_card.table.insertRow(row)

                # Col 0: reaction
                dtxt = eq if pi == 0 else f"  └─ {rt}"
                if rd.get('duplicate_arrhenius_params') and pi == 0:
                    dtxt += f" [DUP×{len(rd['duplicate_arrhenius_params'])}]"
                item = QTableWidgetItem(dtxt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                color_hex = self._get_reaction_color(ri, pi)
                c = QColor(color_hex); c.setAlpha(30)
                item.setBackground(c)
                self.table_card.table.setItem(row, 0, item)

                # Col 1: linestyle indicator
                _, _, ls_idx = self._get_reaction_style(ri, pi)
                ls_label = LINESTYLE_LABELS.get(ls_idx % len(LINESTYLE_LABELS), "—")
                ls_item = QTableWidgetItem(ls_label)
                ls_item.setFlags(ls_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                ls_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                ls_item.setForeground(QColor(color_hex))
                ls_item.setBackground(c)
                fnt_style = QFont("Consolas", 10)
                ls_item.setFont(fnt_style)
                self.table_card.table.setItem(row, 1, ls_item)

                # Col 2: pressure
                ptxt = f"{pres:.1f}"
                if rt in ('PLOG', 'TROE') and use_range:
                    ptxt = f"{pres:.1f}"
                pitem = QTableWidgetItem(ptxt)
                pitem.setFlags(pitem.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table_card.table.setItem(row, 2, pitem)

                # Col 3: SwitchButton for reverse rate
                if rt in ('PLOG', 'TROE') and len(pressures) > 1 and pi > 0:
                    self.table_card.table.setItem(row, 3, QTableWidgetItem(""))
                else:
                    sw = SwitchButton()
                    sw.setOnText("Yes"); sw.setOffText("No")
                    sw.checkedChanged.connect(
                        lambda state, r=rd, idx=ri:
                            self._on_reverse_toggled(state, r, idx))
                    ctr = QWidget()
                    cl_h = QHBoxLayout(ctr)
                    cl_h.addWidget(sw)
                    cl_h.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    cl_h.setContentsMargins(0, 0, 0, 0)
                    self.table_card.table.setCellWidget(row, 3, ctr)

                rd['table_rows'].append({
                    'row': row, 'pressure': pres,
                    'pressure_index': pi, 'is_main_row': pi == 0,
                    'color': color_hex, 'linestyle_index': ls_idx,
                })

                # Rate constants (col 4+)
                for i, T_val in enumerate(self.table_card.table_temperatures):
                    kf = self._calculate_rate_for_reaction(rd, T_val, pres)
                    kf_str = ("%.3E" % kf if kf is not None and kf > 0
                              else ("0.000E+00" if kf == 0 else "N/A"))
                    ki = QTableWidgetItem(kf_str)
                    ki.setFlags(ki.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table_card.table.setItem(row, 4 + i, ki)
                    rd['original_forward_rates'][f"row_{row}_col_{4+i}"] = kf_str

                row += 1

        self.table_card.table.setUpdatesEnabled(True)
        self.table_card.table.resizeColumnsToContents()

    # ── Reverse rate toggle ──────────────────────────────────────────

    def _on_reverse_toggled(self, state, rd, ri):
        eq = rd.get('equation_string_cleaned',
                    rd.get('equation_string', 'N/A'))
        if state:
            self._reload_thermo_data_if_changed()
            if not all([get_reaction_thermo_properties,
                        calculate_equilibrium_constant_kp,
                        calculate_delta_n_gas,
                        calculate_equilibrium_constant_kc,
                        calculate_reverse_rate_constant,
                        self.thermo_data is not None]):
                self._info("Missing functions or thermo data.", "error")
                self._uncheck_main_switch(rd)
                return

            # Check missing species
            all_sp = rd.get('reactants', []) + rd.get('products', [])
            missing = set()
            for _, sn in all_sp:
                if not self.thermo_data.get(sn.upper()):
                    missing.add(sn)
            if missing:
                dlg = NasaPolynomialInputDialog(
                    sorted(missing), self.thermo_filepath, self)
                if dlg.exec():
                    try:
                        self.thermo_data = read_nasa_polynomials(
                            self.thermo_filepath)
                    except Exception as e:
                        self._info(f"Reload error: {e}", "error")
                        self._uncheck_main_switch(rd)
                        return
                    still = {sn for _, sn in all_sp
                             if not self.thermo_data.get(sn.upper())}
                    if still:
                        self._info(f"Still missing: {', '.join(sorted(still))}",
                                   "warning")
                        self._uncheck_main_switch(rd)
                        return
                else:
                    self._uncheck_main_switch(rd)
                    return

            rd['is_calculating_reverse'] = True
            any_err = False
            for rinfo in rd['table_rows']:
                r, pres = rinfo['row'], rinfo['pressure']
                for i, T_val in enumerate(self.table_card.table_temperatures):
                    ci = 4 + i
                    dH, dS, dG, _, err = get_reaction_thermo_properties(
                        rd, T_val, self.thermo_data)
                    if err:
                        any_err = True
                        ei = QTableWidgetItem("Thermo Err")
                        ei.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        self.table_card.table.setItem(r, ci, ei)
                        continue
                    Kp = calculate_equilibrium_constant_kp(dG, T_val)
                    dn = calculate_delta_n_gas(rd, self.thermo_data)
                    Kc = calculate_equilibrium_constant_kc(Kp, T_val, dn)
                    rk = f"row_{r}_col_{ci}"
                    okf = rd['original_forward_rates'].get(rk, "N/A")
                    kf_val = None
                    if okf not in ("N/A", "Error", "kf Error", "Thermo Err"):
                        try: kf_val = float(okf)
                        except ValueError: pass
                    kr_str = "Error"
                    if kf_val is not None and Kc is not None:
                        kr = calculate_reverse_rate_constant(kf_val, Kc)
                        kr_str = "%.3E" % kr if kr is not None else "Error"
                    elif kf_val is None:
                        kr_str = "kf Error"
                    ki = QTableWidgetItem(kr_str)
                    ki.setFlags(ki.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table_card.table.setItem(r, ci, ki)
            self._update_plot_with_reverse_rates()
            self._info(f"Reverse rates updated: {eq[:30]}")
        else:
            rd['is_calculating_reverse'] = False
            for rinfo in rd['table_rows']:
                r = rinfo['row']
                for i in range(len(self.table_card.table_temperatures)):
                    ci = 4 + i
                    rk = f"row_{r}_col_{ci}"
                    okf = rd['original_forward_rates'].get(rk, "N/A")
                    ki = QTableWidgetItem(okf)
                    ki.setFlags(ki.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table_card.table.setItem(r, ci, ki)
            self._update_plot_with_reverse_rates()
            self._info(f"Forward rates restored: {eq[:30]}")

    def _uncheck_main_switch(self, rd):
        for rinfo in rd.get('table_rows', []):
            if rinfo.get('is_main_row'):
                ctr = self.table_card.table.cellWidget(rinfo['row'], 3)
                if ctr:
                    lay = ctr.layout()
                    if lay and lay.count() > 0:
                        sw = lay.itemAt(0).widget()
                        if isinstance(sw, SwitchButton):
                            sw.setChecked(False)
                return

    def _get_main_checkbox_for_reaction(self, rd):
        for rinfo in rd.get('table_rows', []):
            if rinfo.get('is_main_row'):
                ctr = self.table_card.table.cellWidget(rinfo['row'], 3)
                if ctr:
                    lay = ctr.layout()
                    if lay and lay.count() > 0:
                        return lay.itemAt(0).widget()
        return None

    # ── Plot styling ─────────────────────────────────────────────────

    def _get_reaction_style(self, ri, pi=0):
        """Return (color_hex, Qt_PenStyle, linestyle_index) for a reaction.
        When groups=1, each reaction gets a unique (color, style) pair.
        When groups>1, the intra-group checkbox determines whether groups
        share color (styles vary) or share style (colors vary)."""
        nls = len(CURVE_LINE_STYLES)

        try:
            gc = max(1, int(self.group_count.text()))
        except (AttributeError, TypeError, ValueError):
            gc = 1

        if gc <= 1:
            slot_count = self._get_style_pressure_slot_count()
            style_serial = ri * slot_count + pi
            color = self._get_curve_color(style_serial)
            ls_idx = style_serial % nls
        else:
            group_index, position_in_group, _ = self._get_reaction_group_position(ri)
            if self.intra_linestyle_cb.isChecked():
                # Intra-group variation: fixed group color, varying styles.
                color = self._get_curve_color(group_index)
                ls_idx = (position_in_group + pi) % nls
            else:
                # Intra-group variation: varying colors, fixed group style.
                color = self._get_curve_color(position_in_group + pi)
                ls_idx = group_index % nls
        return color, CURVE_LINE_STYLES[ls_idx], ls_idx

    def _get_curve_color(self, index):
        if index < len(CURVE_COLORS):
            return CURVE_COLORS[index]
        hue = (index * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.90)
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def _get_style_pressure_slot_count(self):
        try:
            pressures, use_range, _ = self._get_plot_pressure_settings()
        except Exception:
            return 1
        if not use_range:
            return 1
        try:
            return max(1, len(pressures))
        except TypeError:
            return 1

    def _get_reaction_group_position(self, ri):
        total = len(self.parsed_reactions) if getattr(self, 'parsed_reactions', None) else 1
        try:
            group_count = max(1, int(self.group_count.text()))
        except (AttributeError, TypeError, ValueError):
            group_count = 1

        group_count = min(group_count, max(1, total))
        if group_count <= 1:
            return 0, ri, total

        base_size = total // group_count
        remainder = total % group_count
        large_group_count = remainder
        large_group_size = base_size + 1
        large_group_limit = large_group_count * large_group_size

        if ri < large_group_limit:
            group_index = ri // large_group_size
            position_in_group = ri % large_group_size
            group_size = large_group_size
        else:
            offset = ri - large_group_limit
            group_index = large_group_count + offset // base_size
            position_in_group = offset % base_size
            group_size = base_size

        return group_index, position_in_group, group_size

    def _get_reaction_color(self, ri, pi=0):
        c, _, _ = self._get_reaction_style(ri, pi)
        return c

    def _get_reaction_linestyle_idx(self, ri, pi=0):
        _, _, ls = self._get_reaction_style(ri, pi)
        return ls

    # ── Plotting ─────────────────────────────────────────────────────

    def _update_plot(self, parsed_reactions):
        self.parsed_reactions = parsed_reactions
        self._update_plot_with_reverse_rates()

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
                        color, style, _ = self._get_reaction_style(ri, pi)
                        pen = pg.mkPen(color=color, width=2, style=style)
                        pw.plot(x_vals, k_log,
                                name=f"{pfx}: {lbl_base} @ {Pv:.1f} atm",
                                pen=pen)
                        n_plotted += 1
            else:
                P_use = P_def
                k_log = self._calc_k_log_array(rd, T_vals, P_use, is_rev)
                if k_log is not None and any(~np.isnan(k_log)):
                    pfx = "kr" if is_rev else "kf"
                    color, style, _ = self._get_reaction_style(ri)
                    pen = pg.mkPen(color=color, width=2, style=style)
                    lbl = f"{pfx}: {lbl_base}"
                    if rt in ('PLOG', 'TROE'):
                        lbl += f" @ {P_use:.1f} atm"
                    pw.plot(x_vals, k_log, name=lbl, pen=pen)
                    n_plotted += 1

    # ── Main pipeline ────────────────────────────────────────────────

    def _on_plot_clicked(self):
        txt = self.chemkin_input.toPlainText()
        if not txt.strip():
            self._info("CHEMKIN input is empty.", "warning")
            self.table_card.table.setRowCount(0)
            self.plot_card.clear_plot()
            self._show_empty_state(True)
            return

        temps = self._parse_temperature_input()
        if temps is None:
            return
        self.table_card._setup_columns(temps)

        if not parse_chemkin_mechanism:
            self._info("Parser not available.", "error")
            return

        try:
            reactions = parse_chemkin_mechanism(txt)
            if not reactions:
                self._info("No valid reactions found.", "warning")
                self._show_empty_state(True)
                return
            if merge_duplicate_reactions:
                old_n = len(reactions)
                reactions = merge_duplicate_reactions(reactions)
                if len(reactions) != old_n:
                    self._info(f"Merged {old_n - len(reactions)} duplicate reactions.")
            self._info(f"Parsed {len(reactions)} reactions.")
            self._update_rate_constant_table(reactions)
            self._update_plot(reactions)
            self._show_empty_state(False)
            if PANDAS_AVAILABLE:
                self.save_btn.setEnabled(True)
            self.detail_btn.setEnabled(True)
        except Exception as e:
            self._info(f"Error: {e}", "error")
            import traceback; traceback.print_exc()

    # ── Dialog launchers ─────────────────────────────────────────────

    def _on_detail_clicked(self):
        if not hasattr(self, 'parsed_reactions') or not self.parsed_reactions:
            self._info("No data. Parse a mechanism first.", "warning")
            return
        try:
            T_min = float(self.temp_min.text())
            T_max = float(self.temp_max.text())
        except ValueError:
            self._info("Invalid temperature.", "error")
            return
        dlg = RateConstantDetailDialog(
            self.parsed_reactions, T_min, T_max, self,
            self._calculate_rate_for_reaction,
            self._calculate_reverse_rate_for_plot,
            self._get_plot_pressure_settings,
        )
        dlg.exec()

    def _on_exp_reverse_clicked(self):
        self._reload_thermo_data_if_changed()
        dlg = ExperimentalReverseRateDialog(
            self, self.thermo_data or {}, self.thermo_filepath)
        dlg.exec()

    def _on_save_clicked(self):
        if not PANDAS_AVAILABLE:
            self._info("Requires pandas + openpyxl.", "warning")
            return
        if not hasattr(self, 'parsed_reactions') or not self.parsed_reactions:
            self._info("No data to export.", "warning")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Rate Constants", "rate_constants.xlsx",
            "Excel (*.xlsx);;All (*)")
        if not fp:
            return
        try:
            data = []
            tbl = self.table_card.table
            for r in range(tbl.rowCount()):
                rd = {}
                for c in range(tbl.columnCount()):
                    hdr = tbl.horizontalHeaderItem(c).text()
                    it = tbl.item(r, c)
                    if it:
                        rd[hdr] = it.text()
                    else:
                        w = tbl.cellWidget(r, c)
                        if isinstance(w, QWidget):
                            lay = w.layout()
                            if lay and lay.count() > 0:
                                sw = lay.itemAt(0).widget()
                                rd[hdr] = "Yes" if isinstance(sw, SwitchButton) and sw.isChecked() else "No"
                            else:
                                rd[hdr] = ""
                        else:
                            rd[hdr] = ""
                data.append(rd)
            pd.DataFrame(data).to_excel(fp, index=False)
            self._info(f"Exported to {fp}")
        except Exception as e:
            self._info(f"Export error: {e}", "error")

    def _on_export_image_clicked(self):
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "reaction_rates.svg",
            "SVG (*.svg);;PDF (*.pdf);;PNG (*.png)")
        if fp:
            self._export_plot_image(fp)

    def _export_plot_image(self, filepath):
        """Export current plot as high-quality static image using matplotlib."""
        if not hasattr(self, 'parsed_reactions') or not self.parsed_reactions:
            self._info("No data to export.", "warning")
            return

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure as MplFigure

        fig = MplFigure(figsize=(10, 6), dpi=300)
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
            T_vals = np.linspace(T_min, T_max, 200)
            inv = self.use_inv_temp_cb.isChecked()
            x_vals = 1000.0 / T_vals if inv else T_vals
            x_label = "1000/T (K\u207b\u00b9)" if inv else "Temperature (K)"
            P_vals, use_range, P_def = self._get_plot_pressure_settings()
        except ValueError:
            self._info("Invalid parameters for export.", "error")
            return

        ls_map = ['-', '--', '-.', ':']

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
                    color, _, ls_idx = self._get_reaction_style(ri, pi)
                    ls = ls_map[ls_idx % len(ls_map)]
                    lbl = f"{pfx}: {lbl_base}"
                    if rt in ('PLOG', 'TROE') and len(pressures) > 1:
                        lbl += f" @ {Pv:.1f} atm"
                    ax.plot(x_vals, k_log, label=lbl, color=color,
                            linewidth=2, linestyle=ls)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("log\u2081\u2080(k)", fontsize=12)
        ax.set_title("Reaction Rates vs Temperature", fontsize=14,
                     fontweight='bold')
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=9, facecolor='#222630', edgecolor='#3a3f4b',
                      labelcolor='#e8ecf4')
        ax.grid(True, color='#3a3f4b', alpha=0.5)

        fig.savefig(filepath, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        self._info(f"Exported: {filepath}")


# ── Dialogs ──────────────────────────────────────────────────────────

class RateConstantDetailDialog(MessageBoxBase):
    def __init__(self, reactions, T_min, T_max, parent=None,
                 calc_func=None, calc_rev_func=None, get_pres_func=None):
        super().__init__(parent)
        self.setWindowTitle("Rate Constant Details")
        self.resize(960, 680)
        self.setMinimumSize(600, 400)
        self.reactions = reactions
        self.calc_func = calc_func
        self.calc_rev_func = calc_rev_func
        self.get_pres_func = get_pres_func

        # Parameter input row
        params_row = QHBoxLayout()
        params_row.setSpacing(8)
        params_row.addWidget(CaptionLabel("T min (K):"))
        self.t_min_edit = LineEdit()
        self.t_min_edit.setText(str(int(T_min)))
        self.t_min_edit.setFixedWidth(80)
        params_row.addWidget(self.t_min_edit)
        params_row.addWidget(CaptionLabel("T max (K):"))
        self.t_max_edit = LineEdit()
        self.t_max_edit.setText(str(int(T_max)))
        self.t_max_edit.setFixedWidth(80)
        params_row.addWidget(self.t_max_edit)
        params_row.addWidget(CaptionLabel("Interval (K):"))
        self.interval_edit = LineEdit()
        self.interval_edit.setText("10")
        self.interval_edit.setFixedWidth(50)
        params_row.addWidget(self.interval_edit)
        calc_btn = PrimaryPushButton(FluentIcon.PLAY, "Calculate")
        calc_btn.clicked.connect(self._on_calc)
        params_row.addWidget(calc_btn)
        export_btn = PushButton(FluentIcon.SAVE, "Export Excel")
        export_btn.clicked.connect(self._export)
        params_row.addWidget(export_btn)
        params_row.addStretch()
        self.viewLayout.addLayout(params_row)

        self.table = TableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setItemDelegate(TableItemDelegate(self.table))
        self.viewLayout.addWidget(self.table)

        self._populate(int(T_min), int(T_max), 10)

        sc = QShortcut(QKeySequence.StandardKey.Copy, self.table)
        sc.activated.connect(self._copy)

    def _on_calc(self):
        try:
            t_min = float(self.t_min_edit.text())
            t_max = float(self.t_max_edit.text())
            interval = float(self.interval_edit.text())
            if t_min <= 0 or t_max <= t_min or interval <= 0:
                raise ValueError
        except ValueError:
            InfoBar.error("Invalid parameters", parent=self,
                          position=InfoBarPosition.TOP, duration=3000)
            return
        self._populate(t_min, t_max, interval)

    def _populate(self, T_min, T_max, interval):
        T_vals = np.arange(T_min, T_max + interval, interval)
        P_def = 1.0
        if self.get_pres_func:
            _, _, P_def = self.get_pres_func()

        cols = ["T (K)"]
        pairs = []
        for rd in self.reactions:
            lbl = rd.get('equation_string_cleaned',
                         rd.get('equation_string', 'N/A'))
            rt = rd.get('reaction_type', 'ARRHENIUS')
            is_rev = rd.get('is_calculating_reverse', False)
            pfx = "kr" if is_rev else "kf"
            cn = f"{pfx}: {lbl}"
            if rt in ('PLOG', 'TROE'):
                cn += f" @ {P_def:.1f} atm"
            cols.append(cn)
            pairs.append((rd, P_def, is_rev))

        self.table.setRowCount(len(T_vals))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        for ri, T in enumerate(T_vals):
            self.table.setItem(ri, 0, QTableWidgetItem(f"{T:.0f}"))
            for ci, (rd, P, is_rev) in enumerate(pairs, 1):
                if is_rev and self.calc_rev_func:
                    k = self.calc_rev_func(rd, T, P)
                elif self.calc_func:
                    k = self.calc_func(rd, T, P)
                else:
                    k = None
                ks = f"{k:.3E}" if k is not None and k > 0 else "N/A"
                self.table.setItem(ri, ci, QTableWidgetItem(ks))
        self.table.resizeColumnsToContents()

    def _export(self):
        if not PANDAS_AVAILABLE:
            InfoBar.warning("Requires pandas + openpyxl", parent=self,
                            position=InfoBarPosition.TOP, duration=3000)
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Rate Details", "rate_details.xlsx",
            "Excel (*.xlsx);;All (*)")
        if not fp:
            return
        try:
            data = []
            tbl = self.table
            for r in range(tbl.rowCount()):
                rd = {}
                for c in range(tbl.columnCount()):
                    hdr = tbl.horizontalHeaderItem(c).text()
                    it = tbl.item(r, c)
                    rd[hdr] = it.text() if it else ""
                data.append(rd)
            pd.DataFrame(data).to_excel(fp, index=False)
            InfoBar.success(f"Exported to {fp}", parent=self,
                            position=InfoBarPosition.TOP, duration=3000)
        except Exception as e:
            InfoBar.error(f"Export error: {e}", parent=self,
                          position=InfoBarPosition.TOP, duration=4000)

    def _copy(self):
        sel = self.table.selectedRanges()
        if not sel:
            return
        mr = min(r.topRow() for r in sel)
        xr = max(r.bottomRow() for r in sel)
        mc = min(r.leftColumn() for r in sel)
        xc = max(r.rightColumn() for r in sel)
        rows = []
        hdr = []
        for c in range(mc, xc + 1):
            h = self.table.horizontalHeaderItem(c)
            hdr.append(h.text() if h else "")
        rows.append('\t'.join(hdr))
        for r in range(mr, xr + 1):
            rd = []
            for c in range(mc, xc + 1):
                it = self.table.item(r, c)
                rd.append(it.text() if it else "")
            rows.append('\t'.join(rd))
        QApplication.clipboard().setText('\n'.join(rows))


class ExperimentalReverseRateDialog(MessageBoxBase):
    def __init__(self, parent=None, thermo_data=None, thermo_fp=None):
        super().__init__(parent)
        self.setWindowTitle("Experimental Reverse Rate Conversion")
        self.resize(700, 550)
        self.setMinimumSize(500, 400)
        self.thermo_data = thermo_data or {}
        self.thermo_filepath = thermo_fp
        self._thermo_file_signature = (
            get_thermo_file_signature(thermo_fp) if thermo_fp else None)

        self.viewLayout.addWidget(
            BodyLabel("Convert experimental reverse rate data to forward rates.\n"
                       "Formula: kf = kr × Kc"))

        self.rx_input = LineEdit()
        self.rx_input.setPlaceholderText("e.g. H+O2<=>O+OH")
        self.viewLayout.addWidget(self.rx_input)

        self.t_input = LineEdit()
        self.t_input.setPlaceholderText("Temperatures: 500, 1000, 1500, 2000")
        self.viewLayout.addWidget(self.t_input)

        self.kr_input = LineEdit()
        self.kr_input.setPlaceholderText("kr values: 1.23E+05, 4.56E+06, ...")
        self.viewLayout.addWidget(self.kr_input)

        calc_btn = PrimaryPushButton("Calculate")
        calc_btn.clicked.connect(self._calculate)
        self.viewLayout.addWidget(calc_btn)

        self.result_table = TableWidget()
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setItemDelegate(TableItemDelegate(self.result_table))
        self.result_table.setSelectionMode(
            TableWidget.SelectionMode.ExtendedSelection)
        self.viewLayout.addWidget(self.result_table)

        # Copy shortcut
        sc = QShortcut(QKeySequence.StandardKey.Copy, self.result_table)
        sc.activated.connect(self._copy_result)

    def _reload_thermo_if_changed(self):
        if not reload_thermo_if_changed or not self.thermo_filepath:
            return
        new_data, new_sig, changed = reload_thermo_if_changed(
            self.thermo_filepath, self.thermo_data, self._thermo_file_signature)
        if changed:
            self.thermo_data = new_data
            self._thermo_file_signature = new_sig

    def _calculate(self):
        try:
            self._reload_thermo_if_changed()
            rx = self.rx_input.text().strip()
            if '<=>' not in rx and '=>' not in rx:
                self._show_err("Reaction must contain <=> or =>")
                return
            temps = [float(v) for v in
                     self.t_input.text().replace(',', ' ').split()]
            krs = [float(v) for v in
                   self.kr_input.text().replace(',', ' ').split()]
            if len(temps) != len(krs):
                self._show_err(f"Temp count ({len(temps)}) != kr count ({len(krs)})")
                return

            mech = f"UNITS CAL/MOLE\n{rx}  1.0E+00  0.0  0.0"
            parsed = parse_chemkin_mechanism(mech)
            if not parsed:
                self._show_err("Cannot parse reaction.")
                return
            rd = parsed[0]

            self.result_table.setColumnCount(4)
            self.result_table.setHorizontalHeaderLabels(
                ["T (K)", "kr(exp)", "Kc", "kf(calc)"])
            self.result_table.setRowCount(len(temps))

            for i, (T, kr) in enumerate(zip(temps, krs)):
                self.result_table.setItem(i, 0, QTableWidgetItem(f"{T:.1f}"))
                self.result_table.setItem(i, 1, QTableWidgetItem(f"{kr:.3E}"))

                dH, dS, dG, _, err = get_reaction_thermo_properties(
                    rd, T, self.thermo_data)
                if err:
                    self.result_table.setItem(i, 2, QTableWidgetItem("Err"))
                    self.result_table.setItem(i, 3, QTableWidgetItem("Err"))
                    continue
                Kp = calculate_equilibrium_constant_kp(dG, T)
                dn = calculate_delta_n_gas(rd, self.thermo_data)
                Kc = calculate_equilibrium_constant_kc(Kp, T, dn)
                if Kc is None or Kc <= 0:
                    self.result_table.setItem(i, 2, QTableWidgetItem("N/A"))
                    self.result_table.setItem(i, 3, QTableWidgetItem("N/A"))
                    continue
                kf = kr * Kc
                self.result_table.setItem(i, 2, QTableWidgetItem(f"{Kc:.3E}"))
                self.result_table.setItem(i, 3, QTableWidgetItem(f"{kf:.3E}"))
            self.result_table.resizeColumnsToContents()
        except Exception as e:
            self._show_err(str(e))

    def _show_err(self, msg):
        InfoBar.error(title="Error", content=msg, parent=self,
                      position=InfoBarPosition.TOP, duration=4000)

    def _copy_result(self):
        """Copy selected result table cells to clipboard."""
        sel = self.result_table.selectedRanges()
        if not sel:
            return
        mr = min(r.topRow() for r in sel)
        xr = max(r.bottomRow() for r in sel)
        mc = min(r.leftColumn() for r in sel)
        xc = max(r.rightColumn() for r in sel)
        rows = []
        hdr = []
        for c in range(mc, xc + 1):
            h = self.result_table.horizontalHeaderItem(c)
            hdr.append(h.text() if h else "")
        rows.append('\t'.join(hdr))
        for r in range(mr, xr + 1):
            rd = []
            for c in range(mc, xc + 1):
                it = self.result_table.item(r, c)
                rd.append(it.text() if it else "")
            rows.append('\t'.join(rd))
        QApplication.clipboard().setText('\n'.join(rows))


class NasaPolynomialInputDialog(MessageBoxBase):
    def __init__(self, missing_species, thermo_fp, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add NASA Polynomial Data")
        self.resize(640, 550)
        self.setMinimumSize(500, 400)
        self.missing = missing_species
        self.thermo_filepath = thermo_fp

        self.viewLayout.addWidget(
            SubtitleLabel(f"Missing thermo data: {', '.join(missing_species)}"))
        self.viewLayout.addWidget(
            BodyLabel(f"New data saved to: {thermo_fp}"))

        self.text_edit = PlainTextEdit()
        self.text_edit.setFont(QFont("Courier New", 9))
        self.text_edit.setMinimumHeight(200)
        self.text_edit.setPlaceholderText(
            "Paste NASA 7-coefficient data (4 lines per species)...")
        self.viewLayout.addWidget(self.text_edit)

        self.err_label = BodyLabel("")
        self.err_label.setStyleSheet("color: #ff6b6b;")
        self.err_label.setVisible(False)
        self.viewLayout.addWidget(self.err_label)

        # Replace default buttons with custom
        self.ok_btn = PrimaryPushButton("Validate & Save")
        self.ok_btn.clicked.connect(self._validate_and_save)
        self.cancel_btn = PushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        btn_lay = QHBoxLayout()
        btn_lay.addWidget(self.ok_btn)
        btn_lay.addWidget(self.cancel_btn)
        self.viewLayout.addLayout(btn_lay)

    def _validate_and_save(self):
        txt = self.text_edit.toPlainText()
        if not txt.strip():
            self.err_label.setText("Input is empty.")
            self.err_label.setVisible(True)
            return
        ok, errs, sp_data = validate_nasa7_format(txt)
        if not ok:
            self.err_label.setText('\n'.join(errs))
            self.err_label.setVisible(True)
            return
        missing_upper = {s.upper() for s in self.missing}
        if not missing_upper.issubset(set(sp_data.keys())):
            still = missing_upper - set(sp_data.keys())
            self.err_label.setText(f"Still missing: {', '.join(sorted(still))}")
            self.err_label.setVisible(True)
            return

        rebuilt = ""
        for name, lines in sp_data.items():
            rebuilt += '\n'.join(lines) + '\n'

        try:
            success, msg = append_nasa_polynomial_from_string(
                self.thermo_filepath, rebuilt)
            if success:
                InfoBar.success(title="Saved",
                                content=f"{len(sp_data)} species added.",
                                parent=self,
                                position=InfoBarPosition.TOP, duration=3000)
                self.accept()
            else:
                self.err_label.setText(msg)
                self.err_label.setVisible(True)
        except Exception as e:
            self.err_label.setText(str(e))
            self.err_label.setVisible(True)


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_app_palette(dark=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
