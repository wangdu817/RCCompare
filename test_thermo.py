import unittest
import io
import os
from unittest import mock

# Assuming thermo_parser.py and thermo_calculator.py are in the same directory or accessible in PYTHONPATH
try:
    from thermo_parser import read_nasa_polynomials, is_valid_nasa_polynomial_string, append_nasa_polynomial_from_string
    from thermo_calculator import get_nasa_polynomial, calculate_cp_h_s, calculate_gibbs_g, get_thermo_properties, R_J_MOL_K
except ImportError as e:
    print(f"Error importing modules for testing: {e}. Ensure they are in PYTHONPATH.")
    # Define placeholders if imports fail, so tests can be discovered but will likely fail informatively
    read_nasa_polynomials = None
    is_valid_nasa_polynomial_string = None
    append_nasa_polynomial_from_string = None
    get_nasa_polynomial = None
    calculate_cp_h_s = None
    calculate_gibbs_g = None
    get_thermo_properties = None
    R_J_MOL_K = 8.314462618


# Fixture for valid thermo data
VALID_THERM_DAT_CONTENT = """THERMO ALL
   300.000  1000.000  5000.000
! Comment line 1
! species 1: H2O, two T ranges
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 0.26721056E+01 0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    2
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
-0.50379009E-08 0.18870803E-11-0.47553119E+05 0.80534405E+01                   4
! species 2: O2, two T ranges, different T_mid in header (but parser might default to 1000)
O2                G07/88O  2              G   200.000  3500.000 1000.000    1
 0.36740000E+01 0.75700000E-03-0.21100000E-06 0.25700000E-10-0.90000000E-15    2
-0.10400000E+04 0.29700000E+01 0.32000000E+01 0.50000000E-03 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-0.12345000E+04 0.60000000E+01                   4
! species 3: AR, T_common missing in header (should default or be handled)
AR                L03/87AR   1            G   300.000  6000.000             1
 0.25000000E+01 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
-0.74537500E+03 0.43796940E+01 0.25000000E+01 0.00000000E+00 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-0.74537500E+03 0.43796940E+01                   4
END
"""

class TestReadNasaPolynomials(unittest.TestCase):
    def test_parse_valid_therm_dat_content(self):
        if read_nasa_polynomials is None:
            self.skipTest("read_nasa_polynomials function not imported.")

        mock_file = io.StringIO(VALID_THERM_DAT_CONTENT)
        with mock.patch('builtins.open', return_value=mock_file):
            thermo_data = read_nasa_polynomials("dummy_path.dat")

        self.assertEqual(len(thermo_data), 3) # Expect 3 species

        # Test H2O
        self.assertIn("H2O", thermo_data)
        h2o_data = thermo_data["H2O"]
        self.assertEqual(h2o_data['species_name'], "H2O")
        self.assertEqual(h2o_data['file_T_low'], 200.000)
        self.assertEqual(h2o_data['file_T_high'], 3500.000)
        self.assertEqual(h2o_data['file_T_mid_header'], 1000.000) 
        # T_ranges: [(low_min, T_mid), (T_mid, high_max)]
        self.assertEqual(h2o_data['T_ranges'][0], (200.000, 1000.000))
        self.assertEqual(h2o_data['T_ranges'][1], (1000.000, 3500.000))
        self.assertEqual(len(h2o_data['coeffs']), 2) # High-T and Low-T sets
        self.assertEqual(len(h2o_data['coeffs'][0]), 7) # High-T coeffs
        self.assertEqual(len(h2o_data['coeffs'][1]), 7) # Low-T coeffs
        self.assertAlmostEqual(h2o_data['coeffs'][0][0], 0.26721056E+01) # a1_H
        self.assertAlmostEqual(h2o_data['coeffs'][0][6], 0.23700870E+01) # a7_H
        self.assertAlmostEqual(h2o_data['coeffs'][1][0], 0.33300075E+01) # a1_L
        self.assertAlmostEqual(h2o_data['coeffs'][1][6], 0.80534405E+01) # a7_L
        self.assertTrue(len(h2o_data['source_lines']) >= 4) # 4 main lines + comments

        # Test O2
        self.assertIn("O2", thermo_data)
        o2_data = thermo_data["O2"]
        self.assertEqual(o2_data['species_name'], "O2")
        self.assertEqual(o2_data['file_T_low'], 200.000)
        self.assertEqual(o2_data['file_T_high'], 3500.000)
        self.assertEqual(o2_data['file_T_mid_header'], 1000.000)
        self.assertEqual(o2_data['T_ranges'][0], (200.000, 1000.000))
        self.assertEqual(o2_data['T_ranges'][1], (1000.000, 3500.000))
        self.assertAlmostEqual(o2_data['coeffs'][0][0], 0.36740000E+01) # a1_H
        self.assertAlmostEqual(o2_data['coeffs'][1][6], 0.60000000E+01) # a7_L
        self.assertTrue(len(o2_data['source_lines']) >= 4)

        # Test AR (T_common missing in header, parser defaults to 1000.0 if T_low < 1000 < T_high)
        self.assertIn("AR", thermo_data)
        ar_data = thermo_data["AR"]
        self.assertEqual(ar_data['species_name'], "AR")
        self.assertEqual(ar_data['file_T_low'], 300.000)
        self.assertEqual(ar_data['file_T_high'], 6000.000)
        self.assertIsNone(ar_data['file_T_mid_header']) # Was missing in header
        # Default T_mid logic: if T_low < 1000 < T_high, T_mid = 1000
        self.assertEqual(ar_data['T_ranges'][0], (300.000, 1000.000)) 
        self.assertEqual(ar_data['T_ranges'][1], (1000.000, 6000.000))
        self.assertAlmostEqual(ar_data['coeffs'][0][0], 0.25000000E+01) # a1_H
        self.assertAlmostEqual(ar_data['coeffs'][1][6], 0.43796940E+01) # a7_L
        self.assertTrue(len(ar_data['source_lines']) >= 4)


    def test_handle_missing_file(self):
        if read_nasa_polynomials is None:
            self.skipTest("read_nasa_polynomials function not imported.")
        
        # Suppress print output from the function during this test
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            thermo_data = read_nasa_polynomials("non_existent_file.dat")
        self.assertEqual(thermo_data, {})

class TestNasaPolynomialValidation(unittest.TestCase):
    VALID_POLY_SINGLE_ENTRY = """
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 0.26721056E+01 0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    2
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
-0.50379009E-08 0.18870803E-11-0.47553119E+05 0.80534405E+01                   4
""".strip()

    VALID_POLY_MULTI_ENTRY = """
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 0.26721056E+01 0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    2
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
-0.50379009E-08 0.18870803E-11-0.47553119E+05 0.80534405E+01                   4
O2                G07/88O  2              G   200.000  3500.000 1000.000    1
 0.36740000E+01 0.75700000E-03-0.21100000E-06 0.25700000E-10-0.90000000E-15    2
-0.10400000E+04 0.29700000E+01 0.32000000E+01 0.50000000E-03 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-0.12345000E+04 0.60000000E+01                   4
""".strip()

    INVALID_POLY_LINE_COUNT = """
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 0.26721056E+01 0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    2
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
""".strip() # Missing 4th line

    INVALID_POLY_BAD_COEFF = """
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 BAD_COEFF    0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    2
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
-0.50379009E-08 0.18870803E-11-0.47553119E+05 0.80534405E+01                   4
""".strip()

    INVALID_POLY_BAD_INDICATOR = """
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 0.26721056E+01 0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    X
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
-0.50379009E-08 0.18870803E-11-0.47553119E+05 0.80534405E+01                   4
""".strip()

    def test_is_valid_nasa_polynomial_string_correct_single(self):
        if is_valid_nasa_polynomial_string is None:
            self.skipTest("is_valid_nasa_polynomial_string function not imported.")
        is_valid, parsed_species, errors = is_valid_nasa_polynomial_string(self.VALID_POLY_SINGLE_ENTRY)
        self.assertTrue(is_valid, f"Validation failed for a valid single entry: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(parsed_species), 1)
        self.assertEqual(parsed_species[0]['species_name'], "H2O")
        self.assertEqual(parsed_species[0]['raw_lines'], self.VALID_POLY_SINGLE_ENTRY)

    def test_is_valid_nasa_polynomial_string_correct_multi(self):
        if is_valid_nasa_polynomial_string is None:
            self.skipTest("is_valid_nasa_polynomial_string function not imported.")
        is_valid, parsed_species, errors = is_valid_nasa_polynomial_string(self.VALID_POLY_MULTI_ENTRY)
        self.assertTrue(is_valid, f"Validation failed for a valid multi-entry: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(parsed_species), 2)
        self.assertEqual(parsed_species[0]['species_name'], "H2O")
        self.assertEqual(parsed_species[1]['species_name'], "O2")

    def test_is_valid_nasa_polynomial_string_incorrect_line_count(self):
        if is_valid_nasa_polynomial_string is None:
            self.skipTest("is_valid_nasa_polynomial_string function not imported.")
        is_valid, _, errors = is_valid_nasa_polynomial_string(self.INVALID_POLY_LINE_COUNT)
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid number of lines" in err for err in errors))

    def test_is_valid_nasa_polynomial_string_bad_coeffs(self):
        if is_valid_nasa_polynomial_string is None:
            self.skipTest("is_valid_nasa_polynomial_string function not imported.")
        is_valid, _, errors = is_valid_nasa_polynomial_string(self.INVALID_POLY_BAD_COEFF)
        self.assertFalse(is_valid)
        self.assertTrue(any("not a valid float" in err for err in errors))

    def test_is_valid_nasa_polynomial_string_bad_line_indicators(self):
        if is_valid_nasa_polynomial_string is None:
            self.skipTest("is_valid_nasa_polynomial_string function not imported.")
        is_valid, _, errors = is_valid_nasa_polynomial_string(self.INVALID_POLY_BAD_INDICATOR)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing '2' at column 80" in err for err in errors))
        
    def test_is_valid_empty_string(self):
        if is_valid_nasa_polynomial_string is None:
            self.skipTest("is_valid_nasa_polynomial_string function not imported.")
        is_valid, _, errors = is_valid_nasa_polynomial_string("")
        self.assertFalse(is_valid)
        self.assertIn("Input string is empty.", errors)

class TestAppendNasaPolynomial(unittest.TestCase):
    TEST_FILE = "test_thermo_append.dat"
    VALID_POLY_N2 = """
N2                G03/87N  2              G   300.000  5000.000 1000.000    1
 0.02926640e+02 0.01487977e-01-0.05684761e-05 0.01009704e-08-0.06753351e-13    2
-0.09227977e+04 0.05905697e+01 0.03376172e+02-0.06317038e-02 0.01455836e-04    3
-0.01639043e-07 0.06547132e-11-0.01019103e+04 0.02449430e+01                   4
""".strip()

    VALID_POLY_AR = """
AR                L03/87AR   1            G   300.000  5000.000 1000.000    1
 2.50000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
-7.45375000E+02 4.37969400E+00 2.50000000E+00 0.00000000E+00 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-7.45375000E+02 4.37969400E+00                   4
""".strip()


    def setUp(self):
        # Ensure clean state before each test
        if os.path.exists(self.TEST_FILE):
            os.remove(self.TEST_FILE)

    def tearDown(self):
        # Clean up after each test
        if os.path.exists(self.TEST_FILE):
            os.remove(self.TEST_FILE)

    def test_append_to_existing_file_with_end(self):
        if append_nasa_polynomial_from_string is None:
            self.skipTest("append_nasa_polynomial_from_string function not imported.")
        
        initial_content = "THERMO ALL\n   300.000  1000.000  5000.000\n" + self.VALID_POLY_N2 + "\nEND\n"
        with open(self.TEST_FILE, 'w') as f:
            f.write(initial_content)

        success, msg = append_nasa_polynomial_from_string(self.TEST_FILE, self.VALID_POLY_AR)
        self.assertTrue(success, f"Append failed: {msg}")

        with open(self.TEST_FILE, 'r') as f:
            content = f.read()
        
        expected_content = "THERMO ALL\n   300.000  1000.000  5000.000\n" + \
                           self.VALID_POLY_N2 + "\n" + \
                           self.VALID_POLY_AR + "\n" + \
                           "END\n"
        self.assertEqual(content.strip(), expected_content.strip()) # Use strip for OS-agnostic newline compare

    def test_append_to_existing_file_without_end(self):
        if append_nasa_polynomial_from_string is None:
            self.skipTest("append_nasa_polynomial_from_string function not imported.")

        initial_content = "THERMO ALL\n   300.000  1000.000  5000.000\n" + self.VALID_POLY_N2 + "\n" 
        # No END line initially
        with open(self.TEST_FILE, 'w') as f:
            f.write(initial_content)

        success, msg = append_nasa_polynomial_from_string(self.TEST_FILE, self.VALID_POLY_AR)
        self.assertTrue(success, f"Append failed: {msg}")

        with open(self.TEST_FILE, 'r') as f:
            content = f.read()
        
        expected_content = initial_content.strip() + "\n" + \
                           self.VALID_POLY_AR + "\n" + \
                           "END\n"
        self.assertEqual(content.strip(), expected_content.strip())

    def test_create_new_therm_file_on_append(self):
        if append_nasa_polynomial_from_string is None:
            self.skipTest("append_nasa_polynomial_from_string function not imported.")

        self.assertFalse(os.path.exists(self.TEST_FILE)) # Ensure file doesn't exist

        success, msg = append_nasa_polynomial_from_string(self.TEST_FILE, self.VALID_POLY_AR)
        self.assertTrue(success, f"Append failed for new file: {msg}")
        self.assertTrue(os.path.exists(self.TEST_FILE))

        with open(self.TEST_FILE, 'r') as f:
            content = f.read()

        expected_content = "THERMO ALL\n" + \
                           "   300.000  1000.000  5000.000\n" + \
                           self.VALID_POLY_AR + "\n" + \
                           "END\n"
        self.assertEqual(content.strip(), expected_content.strip())
        
    def test_append_empty_string(self):
        if append_nasa_polynomial_from_string is None:
            self.skipTest("append_nasa_polynomial_from_string function not imported.")
        
        initial_content = "THERMO ALL\nEND\n"
        with open(self.TEST_FILE, 'w') as f:
            f.write(initial_content)
            
        success, msg = append_nasa_polynomial_from_string(self.TEST_FILE, "  \n  ") # Empty or whitespace only
        self.assertFalse(success)
        self.assertIn("Polynomial string is empty", msg)
        
        with open(self.TEST_FILE, 'r') as f:
            content_after = f.read()
        self.assertEqual(initial_content.strip(), content_after.strip()) # File should not change

class TestGetNasaPolynomial(unittest.TestCase):
    def setUp(self):
        self.thermo_data_fixture = {
            "H2O": {
                'species_name': "H2O", 
                'coeffs': [[1.0]*7, [2.0]*7], # Dummy coeffs
                'T_ranges': [(200.0, 1000.0), (1000.0, 3500.0)]
            },
            "O2": {
                'species_name': "O2",
                'coeffs': [[3.0]*7, [4.0]*7], 
                'T_ranges': [(200.0, 1000.0), (1000.0, 3500.0)]
            }
        }
        if get_nasa_polynomial is None:
            self.skipTest("get_nasa_polynomial function not imported.")

    def test_get_existing_species(self):
        species_data = get_nasa_polynomial("H2O", self.thermo_data_fixture)
        self.assertIsNotNone(species_data)
        self.assertEqual(species_data['species_name'], "H2O")

    def test_get_existing_species_case_insensitive(self):
        species_data = get_nasa_polynomial("h2o", self.thermo_data_fixture)
        self.assertIsNotNone(species_data)
        self.assertEqual(species_data['species_name'], "H2O")
        
        species_data_o2 = get_nasa_polynomial("o2", self.thermo_data_fixture)
        self.assertIsNotNone(species_data_o2)
        self.assertEqual(species_data_o2['species_name'], "O2")

    def test_get_non_existing_species(self):
        species_data = get_nasa_polynomial("N2", self.thermo_data_fixture)
        self.assertIsNone(species_data)

class TestCalculateCpHS(unittest.TestCase):
    # Using coefficients for H2O from a standard source (e.g., GRI-Mech 3.0 for H2O)
    # High-T coeffs for H2O (1000-3500K range in GRI30)
    H2O_HIGH_T_COEFFS = [
        2.67210560E+00, 3.05630070E-03, -8.73028300E-07, 
        1.20099590E-10, -6.40076790E-15, -4.70210030E+04, 
        2.37008700E+00
    ]
    # Low-T coeffs for H2O (200-1000K range in GRI30)
    # Using slightly modified a3-a5 for more rigorous calculation testing, not actual GRI H2O low-T.
    H2O_LOW_T_COEFFS_TEST = [ 
        3.33000750E+00, 2.10628580E-03, -1.67197690E-07, 
        -5.03790090E-08, 1.88708030E-11, -4.75531190E+04, 
        8.05344050E+00
    ]

    def test_calculate_values_at_temp_high_T(self):
        if calculate_cp_h_s is None or R_J_MOL_K is None:
            self.skipTest("calculate_cp_h_s or R_J_MOL_K not imported.")
        
        T = 1500.0 # Kelvin
        a = self.H2O_HIGH_T_COEFFS
        
        cp_R_expected = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        h_RT_expected = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T
        s_R_expected = a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]

        expected_Cp = cp_R_expected * R_J_MOL_K
        expected_H = h_RT_expected * R_J_MOL_K * T
        expected_S = s_R_expected * R_J_MOL_K

        Cp, H, S = calculate_cp_h_s(T, self.H2O_HIGH_T_COEFFS)
        self.assertIsNotNone(Cp); self.assertIsNotNone(H); self.assertIsNotNone(S)
        self.assertAlmostEqual(Cp, expected_Cp, places=1) # Relaxed precision for Cp
        self.assertAlmostEqual(H, expected_H, places=0) # Relaxed precision for H
        self.assertAlmostEqual(S, expected_S, places=1) # Relaxed precision for S

    def test_calculate_values_at_temp_low_T(self):
        if calculate_cp_h_s is None or R_J_MOL_K is None:
            self.skipTest("calculate_cp_h_s or R_J_MOL_K not imported.")
        
        T = 300.0 # Kelvin
        a = self.H2O_LOW_T_COEFFS_TEST
        
        cp_R_expected = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        h_RT_expected = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T
        s_R_expected = a[0]*np.log(T) + a[1]*T + a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6]

        expected_Cp = cp_R_expected * R_J_MOL_K
        expected_H = h_RT_expected * R_J_MOL_K * T
        expected_S = s_R_expected * R_J_MOL_K
        
        Cp, H, S = calculate_cp_h_s(T, self.H2O_LOW_T_COEFFS_TEST)
        self.assertIsNotNone(Cp); self.assertIsNotNone(H); self.assertIsNotNone(S)
        self.assertAlmostEqual(Cp, expected_Cp, places=1)
        self.assertAlmostEqual(H, expected_H, places=0)
        self.assertAlmostEqual(S, expected_S, places=1)

    def test_calculate_at_T_zero_or_negative(self):
        if calculate_cp_h_s is None: self.skipTest("calculate_cp_h_s not imported.")
        Cp, H, S = calculate_cp_h_s(0.0, self.H2O_HIGH_T_COEFFS)
        self.assertIsNone(Cp); self.assertIsNone(H); self.assertIsNone(S)
        Cp, H, S = calculate_cp_h_s(-100.0, self.H2O_HIGH_T_COEFFS)
        self.assertIsNone(Cp); self.assertIsNone(H); self.assertIsNone(S)
        
    def test_invalid_coeffs(self):
        if calculate_cp_h_s is None: self.skipTest("calculate_cp_h_s not imported.")
        Cp, H, S = calculate_cp_h_s(300.0, [1.0, 2.0]) # Too few
        self.assertIsNone(Cp); self.assertIsNone(H); self.assertIsNone(S)
        Cp, H, S = calculate_cp_h_s(300.0, None) # None coeffs
        self.assertIsNone(Cp); self.assertIsNone(H); self.assertIsNone(S)

class TestGetThermoProperties(unittest.TestCase):
    def setUp(self):
        self.thermo_data_fixture = {
            "H2O": {
                'species_name': "H2O",
                'coeffs': [TestCalculateCpHS.H2O_HIGH_T_COEFFS, TestCalculateCpHS.H2O_LOW_T_COEFFS_TEST],
                'T_ranges': [(200.0, 1000.0), (1000.0, 3500.0)], 
            },
            "O2_NOCFG": {
                'species_name': "O2_NOCFG", 'T_ranges': [], 'coeffs': [] 
            }
        }
        if get_thermo_properties is None or calculate_gibbs_g is None:
            self.skipTest("get_thermo_properties or calculate_gibbs_g not imported.")

    def test_properties_low_T_range(self):
        T = 300.0
        H, S, G, Cp = get_thermo_properties("H2O", T, self.thermo_data_fixture)
        self.assertIsNotNone(H) # Check if calculation succeeded
        # Exact value check can be done if H2O_LOW_T_COEFFS_TEST were actual H2O values
        # For now, just ensuring it ran and used some coeffs.
        # Re-use calculation from TestCalculateCpHS as a proxy
        a = TestCalculateCpHS.H2O_LOW_T_COEFFS_TEST
        cp_R_exp = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        exp_Cp = cp_R_exp * R_J_MOL_K
        self.assertAlmostEqual(Cp, exp_Cp, places=1)

    def test_properties_high_T_range(self):
        T = 1500.0
        H, S, G, Cp = get_thermo_properties("H2O", T, self.thermo_data_fixture)
        self.assertIsNotNone(H)
        a = TestCalculateCpHS.H2O_HIGH_T_COEFFS
        cp_R_exp = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        exp_Cp = cp_R_exp * R_J_MOL_K
        self.assertAlmostEqual(Cp, exp_Cp, places=1)

    def test_properties_at_T_mid(self): # T_mid should use high-T coeffs
        T = 1000.0
        H, S, G, Cp = get_thermo_properties("H2O", T, self.thermo_data_fixture)
        self.assertIsNotNone(H)
        a = TestCalculateCpHS.H2O_HIGH_T_COEFFS 
        cp_R_exp = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        exp_Cp = cp_R_exp * R_J_MOL_K
        self.assertAlmostEqual(Cp, exp_Cp, places=1)

    def test_properties_T_out_of_all_ranges(self):
        H, S, G, Cp = get_thermo_properties("H2O", 50.0, self.thermo_data_fixture)
        self.assertIsNone(H); self.assertIsNone(S); self.assertIsNone(G); self.assertIsNone(Cp)
        H, S, G, Cp = get_thermo_properties("H2O", 4000.0, self.thermo_data_fixture)
        self.assertIsNone(H); self.assertIsNone(S); self.assertIsNone(G); self.assertIsNone(Cp)

    def test_properties_for_unknown_species(self):
        H, S, G, Cp = get_thermo_properties("UNKNOWN", 1000.0, self.thermo_data_fixture)
        self.assertIsNone(H); self.assertIsNone(S); self.assertIsNone(G); self.assertIsNone(Cp)
        
    def test_properties_species_with_no_coeffs_or_ranges(self):
        H, S, G, Cp = get_thermo_properties("O2_NOCFG", 1000.0, self.thermo_data_fixture)
        self.assertIsNone(H); self.assertIsNone(S); self.assertIsNone(G); self.assertIsNone(Cp)

if __name__ == '__main__':
    unittest.main()
