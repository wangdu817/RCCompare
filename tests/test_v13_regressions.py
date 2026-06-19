import math
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import chemkin_pyqt_gui_simple as gui
from rate_calculator import calculate_arrhenius_rate, calculate_plog_rate


class TestPlogSamePressureSums(unittest.TestCase):
    def setUp(self):
        self.temperature = 1000.0
        self.low_pressure_entries = [
            {
                "pressure": 1.0e-2,
                "A": 3.80273593e38,
                "n": -8.63938,
                "Ea": 4.87263e3,
                "units": "CAL/MOLE",
            },
            {
                "pressure": 1.0e-2,
                "A": 1.90285178e47,
                "n": -10.78882,
                "Ea": 1.3430745e4,
                "units": "CAL/MOLE",
            },
        ]

    def test_same_pressure_plog_entries_are_summed(self):
        expected = sum(
            calculate_arrhenius_rate(entry, self.temperature)
            for entry in self.low_pressure_entries
        )

        actual = calculate_plog_rate(
            self.low_pressure_entries, self.temperature, 1.0e-2
        )

        self.assertAlmostEqual(actual, expected, delta=expected * 1.0e-12)

    def test_interpolation_uses_sum_at_duplicate_pressure_endpoint(self):
        high_pressure_entry = {
            "pressure": 1.0,
            "A": 2.0e12,
            "n": 0.0,
            "Ea": 0.0,
            "units": "CAL/MOLE",
        }
        entries = self.low_pressure_entries + [high_pressure_entry]
        low_rate = sum(
            calculate_arrhenius_rate(entry, self.temperature)
            for entry in self.low_pressure_entries
        )
        high_rate = calculate_arrhenius_rate(high_pressure_entry, self.temperature)
        target_pressure = 1.0e-1
        fraction = (
            math.log(target_pressure) - math.log(1.0e-2)
        ) / (math.log(1.0) - math.log(1.0e-2))
        expected = math.exp(
            math.log(low_rate) + fraction * (math.log(high_rate) - math.log(low_rate))
        )

        actual = calculate_plog_rate(entries, self.temperature, target_pressure)

        self.assertAlmostEqual(actual, expected, delta=expected * 1.0e-12)


class TestThermoDataPath(unittest.TestCase):
    def test_source_run_resolves_bundled_baseline_beside_application_file(self):
        with tempfile.TemporaryDirectory() as application_dir:
            source_file = os.path.join(application_dir, "chemkin_pyqt_gui_simple.py")
            with mock.patch.object(gui, "__file__", source_file), \
                    mock.patch.object(gui.sys, "frozen", False, create=True):
                actual = gui.resolve_bundled_thermo_filepath()

        self.assertEqual(actual, os.path.join(application_dir, "therm.dat"))

    def test_frozen_run_resolves_bundled_baseline_from_pyinstaller_resources(self):
        with tempfile.TemporaryDirectory() as resource_dir:
            pathlib.Path(resource_dir, "therm.dat").write_text("BASELINE\n", encoding="utf-8")
            with mock.patch.object(gui.sys, "frozen", True, create=True), \
                    mock.patch.object(gui.sys, "_MEIPASS", resource_dir, create=True):
                actual = gui.resolve_bundled_thermo_filepath()

        self.assertEqual(actual, os.path.join(resource_dir, "therm.dat"))

    def test_onedir_run_falls_back_to_baseline_beside_executable(self):
        with tempfile.TemporaryDirectory() as resource_dir, tempfile.TemporaryDirectory() as exe_dir:
            executable = os.path.join(exe_dir, "CHEMKIN_RateCalculator.exe")
            adjacent_baseline = os.path.join(exe_dir, "therm.dat")
            pathlib.Path(adjacent_baseline).write_text("BASELINE\n", encoding="utf-8")
            with mock.patch.object(gui.sys, "frozen", True, create=True), \
                    mock.patch.object(gui.sys, "_MEIPASS", resource_dir, create=True), \
                    mock.patch.object(gui.sys, "executable", executable):
                actual = gui.resolve_bundled_thermo_filepath()

        self.assertEqual(actual, adjacent_baseline)

    def test_persistent_therm_database_is_stored_under_local_appdata(self):
        with tempfile.TemporaryDirectory() as local_appdata:
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": local_appdata}):
                actual = gui.resolve_thermo_filepath()

        self.assertEqual(
            actual,
            os.path.join(local_appdata, "CHEMKIN_RateViewer", "therm.dat"),
        )

    def test_initialization_copies_baseline_once_and_preserves_user_edits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = os.path.join(temp_dir, "baseline.dat")
            persistent_path = os.path.join(temp_dir, "userdata", "therm.dat")
            pathlib.Path(baseline_path).write_text("BASELINE\n", encoding="utf-8")

            actual = gui.ensure_persistent_thermo_file(persistent_path, baseline_path)
            self.assertEqual(actual, persistent_path)
            self.assertEqual(pathlib.Path(persistent_path).read_text(encoding="utf-8"), "BASELINE\n")

            pathlib.Path(persistent_path).write_text("USER EDIT\n", encoding="utf-8")
            pathlib.Path(baseline_path).write_text("NEW RELEASE\n", encoding="utf-8")
            gui.ensure_persistent_thermo_file(persistent_path, baseline_path)

            self.assertEqual(pathlib.Path(persistent_path).read_text(encoding="utf-8"), "USER EDIT\n")


class TestThermoAutoReload(unittest.TestCase):
    _THERMO_TEMPLATE = """THERMO ALL
   300.000  1000.000  5000.000
{name:<18}          C   1          G   300.000  5000.000 1000.000    1
 2.50000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
-7.45375000E+02 4.37969400E+00 2.50000000E+00 0.00000000E+00 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-7.45375000E+02 4.37969400E+00                   4
END
"""

    class _StatusBar:
        def showMessage(self, message):
            self.message = message

    def test_changed_persistent_file_is_reloaded_before_thermo_use(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            thermo_path = os.path.join(temp_dir, "therm.dat")
            pathlib.Path(thermo_path).write_text(
                self._THERMO_TEMPLATE.format(name="BASE"),
                encoding="utf-8",
            )
            window = type("WindowState", (), {})()
            window.thermo_filepath = thermo_path
            window.thermo_data = {}
            window._thermo_file_signature = None
            window.statusBar = self._StatusBar()
            window._load_thermo_data = types.MethodType(
                gui.MainWindow._load_thermo_data, window
            )

            gui.MainWindow._load_thermo_data(window)
            self.assertIn("BASE", window.thermo_data)

            pathlib.Path(thermo_path).write_text(
                self._THERMO_TEMPLATE.format(name="UPDATED"),
                encoding="utf-8",
            )
            modified = os.stat(thermo_path).st_mtime_ns + 1_000_000
            os.utime(thermo_path, ns=(modified, modified))

            self.assertTrue(gui.MainWindow._reload_thermo_data_if_changed(window))
            self.assertIn("UPDATED", window.thermo_data)
            self.assertNotIn("BASE", window.thermo_data)


class TestDisplayedVersion(unittest.TestCase):
    def test_gui_source_displays_only_v13_release_version(self):
        source_text = pathlib.Path(gui.__file__).read_text(encoding="utf-8")

        self.assertIn("CHEMKIN Rate Viewer - v1.3", source_text)
        self.assertIn("CHEMKIN Rate Viewer v1.3", source_text)
        self.assertNotIn("CHEMKIN Rate Viewer - v1.2", source_text)
        self.assertNotIn("CHEMKIN Rate Viewer v1.0", source_text)

    def test_launcher_displays_v13_release_version(self):
        launcher_path = pathlib.Path(gui.__file__).with_name("run_chemkin_viewer.bat")
        launcher_text = launcher_path.read_text(encoding="utf-8")

        self.assertIn("v1.3", launcher_text)
        self.assertNotIn("v1.1", launcher_text)


if __name__ == "__main__":
    unittest.main()
