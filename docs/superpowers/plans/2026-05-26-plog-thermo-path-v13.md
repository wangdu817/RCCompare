# PLOG And Thermo Path V1.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver `v1.3` with correct same-pressure PLOG summation, persistent user thermo data, automatic reload of saved edits, and a directory-mode package.

**Architecture:** Keep the parser unchanged because it already retains all PLOG records and reads appended NASA records. The GUI distinguishes a read-only bundled baseline from `%LOCALAPPDATA%\CHEMKIN_RateViewer\therm.dat`, copies the baseline only for first initialization, and reloads the persistent file before thermo-dependent calculations if its signature changes. Package with the existing PyInstaller `onedir` spec so the baseline resource remains available without treating it as user state.

**Tech Stack:** Python, `unittest`, NumPy, PyQt6, CHEMKIN/NASA text inputs.

---

### Task 1: Capture Regression Behavior

**Files:**
- Create: `tests/test_v13_regressions.py`

- [x] **Step 1: Write failing tests for same-pressure PLOG accumulation**

Create tests that calculate both Arrhenius terms independently, assert that a PLOG request at the shared pressure equals their sum, and assert an intermediate pressure interpolates between the summed low-pressure value and a single high-pressure value.

- [x] **Step 2: Write failing tests for application-local therm data and version presentation**

Create a test that imports the GUI module and verifies a path resolver points at `therm.dat` adjacent to an injected application path rather than the current directory. Read the GUI source and assert no stale `v1.0` or `v1.2` display strings remain.

- [x] **Step 3: Run tests and observe expected failures**

Run:

```powershell
& 'C:\Users\17915\anaconda3\envs\gui-env\python.exe' -m unittest -v tests.test_v13_regressions
```

Expected failures: duplicate-pressure PLOG returns only the first term; the GUI does not yet expose the path resolver; stale version strings remain.

### Task 2: Implement Correct PLOG Evaluation

**Files:**
- Modify: `rate_calculator.py`

- [x] **Step 1: Aggregate evaluated rates at identical pressures**

Evaluate each valid PLOG Arrhenius item at the requested temperature, sum `k` values sharing a pressure, then construct logarithmic interpolation points from the summed rates.

- [x] **Step 2: Run the PLOG regression tests**

Run the targeted test command from Task 1 and confirm the same-pressure and interpolated-value assertions pass.

### Task 3: Implement Persistent Therm Database, Live Refresh, And Version 1.3

**Files:**
- Modify: `chemkin_pyqt_gui_simple.py`

- [x] **Step 1: Write failing persistence and reload tests**

Extend `tests/test_v13_regressions.py` to require:

```python
user_path = gui.resolve_thermo_filepath()
baseline_path = gui.resolve_bundled_thermo_filepath()
created_path = gui.ensure_persistent_thermo_file(user_path, baseline_path)
```

The tests assert that `user_path` is under a patched `LOCALAPPDATA`, initialization copies baseline content once, a second initialization does not overwrite edited content, and `_reload_thermo_data_if_changed()` replaces cached parsed species after an external file write.

- [x] **Step 2: Run persistence tests and observe expected failures**

Run:

```powershell
& 'C:\Users\17915\anaconda3\envs\gui-env\python.exe' -m unittest discover -s 'D:\RCCompare\tests' -p 'test_v13_regressions.py' -v
```

Expected result: failures because the current path still resolves beside the executable/source and no baseline initialization or change-detection helper exists.

- [x] **Step 3: Add persistent user-data initialization and reload hooks**

Implement `resolve_bundled_thermo_filepath()`, `resolve_thermo_filepath()`, `ensure_persistent_thermo_file()`, and file-signature based reload support in `chemkin_pyqt_gui_simple.py`. Initialize the main window from the persistent path and refresh it at reverse-rate calculation entry points and before opening the experimental conversion dialog.

- [x] **Step 4: Set all visible version labels to v1.3**

Change the window title and About dialog string to `CHEMKIN Rate Viewer - v1.3` / `CHEMKIN Rate Viewer v1.3`.

- [x] **Step 5: Run all new regression tests**

Run:

```powershell
& 'C:\Users\17915\anaconda3\envs\gui-env\python.exe' -m unittest -v tests.test_v13_regressions
```

Expected result: all new tests pass.

### Task 4: Build And Verify The Working Runtime

**Files:**
- Verify: `therm.dat`
- Verify: `chemkin_parser.py`
- Verify: `rate_calculator.py`
- Verify: `chemkin_pyqt_gui_simple.py`
- Verify: `CHEMKIN_RateCalculator.spec`
- Verify: `dist/CHEMKIN_RateCalculator/`

- [x] **Step 1: Parse the current thermodynamic data and persistent-copy flow**

Run a short command under `gui-env` with a temporary `LOCALAPPDATA` that loads the bundled baseline into a new persistent database, appends a NASA record there, restarts initialization, and confirms that record remains present.

- [x] **Step 2: Evaluate the supplied PLOG example**

Parse the supplied PLOG reaction and confirm the reported rate at `1000 K` and `0.01 atm` equals the two individual evaluated rates summed together.

- [x] **Step 3: Compile-check the changed modules**

Run:

```powershell
& 'C:\Users\17915\anaconda3\envs\gui-env\python.exe' -m py_compile rate_calculator.py chemkin_pyqt_gui_simple.py tests\test_v13_regressions.py
```

Expected result: command succeeds with no syntax errors.

- [x] **Step 4: Create and verify the directory-mode distribution**

Run:

```powershell
& 'C:\Users\17915\anaconda3\envs\gui-env\python.exe' -m PyInstaller --clean --noconfirm 'D:\RCCompare\CHEMKIN_RateCalculator.spec'
```

Expected result: `D:\RCCompare\dist\CHEMKIN_RateCalculator\CHEMKIN_RateCalculator.exe` exists, the bundled baseline `therm.dat` is present in the distribution resources, and an isolated first-run check creates a reusable persistent user database.
