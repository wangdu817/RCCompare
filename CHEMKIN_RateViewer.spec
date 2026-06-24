# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['chemkin_pyqt_gui_fluent.py'],
    pathex=[],
    binaries=[],
    datas=[('logo.png', '.'), ('therm.dat', '.'), ('modern_style.qss', '.')],
    hiddenimports=[
        'qfluentwidgets',
        'qfluentwidgets.components',
        'qfluentwidgets.gallery',
        'qfluentwidgets._rc',
        'qfluentwidgets._rc.icons',
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'PyQt6.QtSvgWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CHEMKIN_RateViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CHEMKIN_RateViewer',
)
