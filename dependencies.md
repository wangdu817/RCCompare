# Project Dependencies

## Python Libraries from PyPI
The following libraries are required and can be installed via pip:
```
pip install -r requirements.txt
```
The `requirements.txt` file lists:
- PyQt6 (for the graphical user interface)
- matplotlib (for plotting)
- numpy (for numerical operations)

## System Dependencies (for PyQt6)
PyQt6 relies on the Qt6 libraries. When you install PyQt6 using pip, it usually handles the Qt6 library components. However, on some Linux systems, you might need to ensure that basic X11 client libraries and related font configurations are available for Qt to function correctly (as encountered with errors like "could not load the Qt platform plugin 'xcb'").

For example, on a headless Linux server or minimal container, you might need to install packages like:
```bash
sudo apt-get update
sudo apt-get install -y libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libfontconfig1 libgl1-mesa-glx libegl1-mesa
```
(The exact list can vary. `libxcb-cursor0` was one specific library identified earlier, but more might be needed for a full desktop-like environment for Qt.)

For typical desktop Linux distributions, Windows, and macOS, these system-level dependencies for Qt are often already met or handled by the OS or Qt's own installation.
```
