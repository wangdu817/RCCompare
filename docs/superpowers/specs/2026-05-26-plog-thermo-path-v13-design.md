# PLOG And Thermo Path V1.3 Design

## Scope

Release `v1.3` addresses calculation reliability and persistent thermo-data reuse in the active root-level application entry point, `chemkin_pyqt_gui_simple.py`, and its imported calculation module, `rate_calculator.py`.

## Confirmed Findings

- The NASA polynomial append flow writes new entries before the final `END` record and reloads the file. The current user-added entries in `therm.dat` are parseable.
- Bundled `therm.dat` must remain a distribution baseline, not the writable user database. A one-file executable may unpack bundled data into a temporary directory, and replacing a distribution directory during an update can overwrite files beside the executable.
- The PLOG parser already preserves multiple Arrhenius entries at one pressure. `calculate_plog_rate()` discards later entries at duplicate pressures, although CHEMKIN PLOG semantics require their rates to be summed before pressure interpolation.
- The displayed application versions are inconsistent (`v1.2` in the window title and `v1.0` in the About dialog).

## Design

Add GUI-level path helpers with two distinct roles:

- The bundled baseline `therm.dat` resolves beside the source script while developing and through PyInstaller's runtime resource directory while packaged.
- The persistent writable database resolves to `%LOCALAPPDATA%\CHEMKIN_RateViewer\therm.dat`.

On first use, create the user-data directory and copy the bundled baseline into the persistent path only when no persistent database exists. All built-in NASA polynomial additions write to that persistent file, so later launches and application upgrades retain them.

Store a lightweight file signature for the loaded persistent database. Before any reverse-rate or thermo-dependent calculation action, compare the current signature with the loaded signature and reload when an external editor has saved a changed file. This makes saved edits effective on the next calculation without parsing a partially written file during the editing process.

Change PLOG evaluation so each parsed entry is evaluated at temperature `T`, positive rates sharing the same pressure are accumulated, and the resulting one-rate-per-pressure series is passed through the existing logarithmic pressure interpolation behavior.

Update user-visible version text to `v1.3`.

## Verification

- Add a regression test proving two PLOG expressions at the same pressure are summed at that pressure and participate in interpolation as their sum.
- Add regression tests proving the baseline resource and persistent user-data paths are distinct, initial creation copies the baseline, and later initialization never overwrites a modified persistent file.
- Add a regression test proving a changed persistent file is detected and reloaded before thermo use.
- Add a regression test proving the GUI source exposes `v1.3` consistently.
- Run these tests with `C:\Users\17915\anaconda3\envs\gui-env\python.exe`.
- Build the distributable with the existing directory-mode `CHEMKIN_RateCalculator.spec` and check that the packaged baseline is present for first-run initialization.
