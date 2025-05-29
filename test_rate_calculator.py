import unittest
import numpy as np
from unittest import mock # Make sure mock is imported

# Existing imports from rate_calculator
from rate_calculator import (
    _convert_ea_to_cal_per_mol, 
    get_third_body_concentration,
    calculate_arrhenius_rate,
    calculate_plog_rate,
    calculate_troe_rate,
    R_cal, 
    R_atm_cm3
)

# Imports for new reverse rate calculation tests
try:
    from rate_calculator import (
        get_reaction_thermo_properties, calculate_equilibrium_constant_kp,
        calculate_delta_n_gas, calculate_equilibrium_constant_kc,
        calculate_reverse_rate_constant, R_J_MOL_K, R_L_ATM_MOL_K
    )
except ImportError:
    get_reaction_thermo_properties = None
    calculate_equilibrium_constant_kp = None
    calculate_delta_n_gas = None
    calculate_equilibrium_constant_kc = None
    calculate_reverse_rate_constant = None
    R_J_MOL_K = 8.314462618 
    R_L_ATM_MOL_K = 0.08205736608

class TestRateCalculatorNew(unittest.TestCase):

    def test_ea_conversion(self):
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(1000, 'CAL/MOLE'), 1000.0)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(1.0, 'KCAL/MOLE'), 1000.0)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(10.0, 'KCAL/MOLE'), 10000.0)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(418.4, 'J/MOL'), 100.0, places=5)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(4184.0, 'JOULES/MOLE'), 1000.0, places=5)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(4.184, 'KJ/MOL'), 1000.0, places=5)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(41.84, 'KJOULES/MOLE'), 10000.0, places=5)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(1000.0, 'KELVINS'), 1000.0 * R_cal, places=5)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(1000.0, 'K'), 1000.0 * R_cal, places=5)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(1.0, 'EVOLTS'), 23060.5, places=5)
        
        # Test unrecognized unit (should return original value and print warning)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(500.0, 'UNKNOWN_UNIT'), 500.0)
        # Test None unit (should default to CAL/MOLE)
        self.assertAlmostEqual(_convert_ea_to_cal_per_mol(500.0, None), 500.0)


    def test_get_third_body_concentration(self):
        expected_M = 1 / (R_atm_cm3 * 298.0) 
        self.assertAlmostEqual(get_third_body_concentration(1.0, 298.0), expected_M, places=8)
        self.assertEqual(get_third_body_concentration(1.0, 0.0), 0.0) # T=0 should return 0
        self.assertEqual(get_third_body_concentration(-1.0, 298.0), 0.0) # P<0 should return 0
        self.assertEqual(get_third_body_concentration(0.0, 298.0), 0.0) # P=0 should return 0

    def test_calculate_arrhenius_rate_new_structure(self):
        T1 = 1000.0
        # Case 1: CAL/MOLE
        params1 = {'A': 1e13, 'n': 0, 'Ea': 10000, 'units': 'CAL/MOLE'}
        expected_k1 = 1e13 * np.exp(-10000 / (R_cal * T1))
        self.assertAlmostEqual(calculate_arrhenius_rate(params1, T1), expected_k1, places=5)

        # Case 2: KCAL/MOLE
        params2 = {'A': 1e13, 'n': 0, 'Ea': 10.0, 'units': 'KCAL/MOLE'} # Ea = 10 kcal/mol
        expected_k2 = 1e13 * np.exp(-10000 / (R_cal * T1)) # Same as above
        self.assertAlmostEqual(calculate_arrhenius_rate(params2, T1), expected_k2, places=5)

        # Case 3: JOULES/MOLE
        params3 = {'A': 1e13, 'n': 0, 'Ea': 10000 * 4.184, 'units': 'JOULES/MOLE'} # Ea = 10000 cal/mol in Joules
        self.assertAlmostEqual(calculate_arrhenius_rate(params3, T1), expected_k1, places=5)
        
        # Case 4: KELVINS
        params4 = {'A': 1e13, 'n': 0, 'Ea': 10000 / R_cal, 'units': 'KELVINS'} # Ea/R = 5027.1 K
        self.assertAlmostEqual(calculate_arrhenius_rate(params4, T1), expected_k1, places=5)

        # Test missing params
        self.assertIsNone(calculate_arrhenius_rate({'A': 1e13, 'n': 0, 'units': 'CAL/MOLE'}, T1)) # Missing Ea
        self.assertIsNone(calculate_arrhenius_rate({}, T1))


    def test_calculate_plog_rate_new_structure(self):
        T_test = 500.0  # K
        plog_entries = [
            {'pressure': 1.0, 'A': 1e10, 'n': 0, 'Ea': 1.0, 'units': 'KCAL/MOLE'}, # 1000 cal/mol
            {'pressure': 10.0, 'A': 1e12, 'n': 0, 'Ea': 2000.0, 'units': 'CAL/MOLE'} # 2000 cal/mol
        ]
        # k1 at P=1, T=500: 1e10 * exp(-1000 / (R_cal * 500))
        k1_val = 1e10 * np.exp(-1000 / (R_cal * T_test))
        # k2 at P=10, T=500: 1e12 * exp(-2000 / (R_cal * 500))
        k2_val = 1e12 * np.exp(-2000 / (R_cal * T_test))

        self.assertAlmostEqual(calculate_plog_rate(plog_entries, T_test, 1.0), k1_val, delta=k1_val*1e-5)
        self.assertAlmostEqual(calculate_plog_rate(plog_entries, T_test, 10.0), k2_val, delta=k2_val*1e-5)
        
        log_k1 = np.log(k1_val); log_k2 = np.log(k2_val)
        log_P1 = np.log(1.0); log_P2 = np.log(10.0)
        log_P_target_interp = np.log(5.0)
        log_k_expected_interp = log_k1 + (log_k2 - log_k1) * (log_P_target_interp - log_P1) / (log_P2 - log_P1)
        expected_k_interp = np.exp(log_k_expected_interp)
        self.assertAlmostEqual(calculate_plog_rate(plog_entries, T_test, 5.0), expected_k_interp, delta=expected_k_interp*1e-5)

        # Test extrapolation (P_target below min pressure)
        self.assertAlmostEqual(calculate_plog_rate(plog_entries, T_test, 0.1), k1_val, delta=k1_val*1e-5)
        # Test extrapolation (P_target above max pressure)
        self.assertAlmostEqual(calculate_plog_rate(plog_entries, T_test, 100.0), k2_val, delta=k2_val*1e-5)

        # Test P_target <= 0
        self.assertIsNone(calculate_plog_rate(plog_entries, T_test, 0.0))
        self.assertIsNone(calculate_plog_rate(plog_entries, T_test, -1.0))
        
        # Test T <= 0
        self.assertIsNone(calculate_plog_rate(plog_entries, 0.0, 5.0))
        self.assertIsNone(calculate_plog_rate(plog_entries, -100.0, 5.0))

        # Test with invalid pressure in PLOG entries
        plog_invalid_pressure = [
            {'pressure': -1.0, 'A': 1e10, 'n': 0, 'Ea': 1.0, 'units': 'KCAL/MOLE'},
            plog_entries[1] # Valid entry
        ]
        # Should use only the valid entry (k2_val at P=10) if P_target is closer to it or exact
        self.assertAlmostEqual(calculate_plog_rate(plog_invalid_pressure, T_test, 10.0), k2_val, delta=k2_val*1e-5)
        
        plog_zero_pressure = [
            {'pressure': 0.0, 'A': 1e10, 'n': 0, 'Ea': 1.0, 'units': 'KCAL/MOLE'},
            plog_entries[1]
        ]
        self.assertAlmostEqual(calculate_plog_rate(plog_zero_pressure, T_test, 10.0), k2_val, delta=k2_val*1e-5)


        # Test with a single PLOG entry
        single_plog_entry = [plog_entries[0]]
        self.assertAlmostEqual(calculate_plog_rate(single_plog_entry, T_test, 1.0), k1_val, delta=k1_val*1e-5)
        # Extrapolation with single entry should return that entry's rate
        self.assertAlmostEqual(calculate_plog_rate(single_plog_entry, T_test, 0.1), k1_val, delta=k1_val*1e-5)
        self.assertAlmostEqual(calculate_plog_rate(single_plog_entry, T_test, 10.0), k1_val, delta=k1_val*1e-5)

        # Test with duplicate pressure points in PLOG entries
        # Current behavior: if pressures are identical, first one encountered after sort is used.
        # If log_P are identical, unique_rates logic keeps the first one.
        plog_duplicate_pressure = [
            {'pressure': 1.0, 'A': 1e10, 'n': 0, 'Ea': 1.0, 'units': 'KCAL/MOLE'}, # k1_val
            {'pressure': 1.0, 'A': 1e11, 'n': 0, 'Ea': 1.5, 'units': 'KCAL/MOLE'}, # Different rate params
            {'pressure': 10.0, 'A': 1e12, 'n': 0, 'Ea': 2000.0, 'units': 'CAL/MOLE'} # k2_val
        ]
        # Rate at P=1.0 should be k1_val because it's the first one for P=1.0
        self.assertAlmostEqual(calculate_plog_rate(plog_duplicate_pressure, T_test, 1.0), k1_val, delta=k1_val*1e-5)
        
        # Test with empty PLOG entries
        self.assertIsNone(calculate_plog_rate([], T_test, 1.0))
        
        # Test with PLOG entries missing required keys
        plog_missing_keys = [{'pressure': 1.0, 'A': 1e10}] # Missing n, Ea
        self.assertIsNone(calculate_plog_rate(plog_missing_keys, T_test, 1.0))

    def test_calculate_troe_rate_new_structure(self):
        T = 1000.0  # K
        P_target = 1.0  # atm
        
        troe_data1 = {
            'k_inf': {'A': 1.0E14, 'n': 0.0, 'Ea': 2.0, 'units': 'KCAL/MOLE'}, # 2000 cal/mol
            'k0': {'A': 1.0E16, 'n': 0.0, 'Ea': 500.0, 'units': 'CAL/MOLE'},   # 500 cal/mol
            'coeffs': [0.6, 200.0, 1200.0, 5000.0] 
        }
        
        # Calculate expected k_inf and k0_val with correct units
        k_inf_val_exp = 1.0E14 * np.exp(-2000.0 / (R_cal * T))
        k0_val_exp = 1.0E16 * np.exp(-500.0 / (R_cal * T))
        M_conc = get_third_body_concentration(P_target, T)
        k0_eff_exp = k0_val_exp * M_conc
        Pr_exp = k0_eff_exp / k_inf_val_exp

        alpha, T3s, T1s, T2s = troe_data1['coeffs']
        F_cent_exp = (1 - alpha) * np.exp(-T / T3s) + alpha * np.exp(-T / T1s) + np.exp(-T / T2s)
        
        log10_Pr_exp = np.log10(Pr_exp) if Pr_exp > 0 else -np.inf # Avoid log(0)
        c_exp = -0.4 - 0.67 * log10_Pr_exp
        n_exp = 0.75 - 1.27 * log10_Pr_exp
        val_exp = log10_Pr_exp - c_exp
        denom_exp = n_exp - 0.14 * val_exp
        F_exp = 10**(np.log10(F_cent_exp) / (1.0 + (val_exp / denom_exp)**2)) if F_cent_exp > 0 and denom_exp != 0 else 1.0
        k_expected = k_inf_val_exp * (Pr_exp / (1 + Pr_exp)) * F_exp
        
        k_calc = calculate_troe_rate(troe_data1, T, P_target, M_conc=None)
        self.assertAlmostEqual(k_calc, k_expected, delta=k_expected * 1e-5)

        # Test with missing k0
        troe_data_no_k0 = {'k_inf': troe_data1['k_inf'], 'k0': {}, 'coeffs': troe_data1['coeffs']}
        self.assertIsNone(calculate_troe_rate(troe_data_no_k0, T, P_target, None))

        # Test with missing k_inf
        troe_data_no_kinf = {'k_inf': {}, 'k0': troe_data1['k0'], 'coeffs': troe_data1['coeffs']}
        self.assertIsNone(calculate_troe_rate(troe_data_no_kinf, T, P_target, None))

        # Test with missing troe_coeffs
        troe_data_no_coeffs = {'k_inf': troe_data1['k_inf'], 'k0': troe_data1['k0'], 'coeffs': [] }
        self.assertIsNone(calculate_troe_rate(troe_data_no_coeffs, T, P_target, None))

        # Test with M_conc provided directly
        M_conc_direct = 0.05 # mol/cm^3, arbitrary value
        k0_eff_exp_direct = k0_val_exp * M_conc_direct # Use k0_val_exp from the top of the method
        Pr_exp_direct = k0_eff_exp_direct / k_inf_val_exp # Use k_inf_val_exp from the top
        
        F_cent_exp_direct = F_cent_exp # Coeffs are the same
        log10_Pr_exp_direct = np.log10(Pr_exp_direct) if Pr_exp_direct > 0 else -np.inf
        c_exp_direct = -0.4 - 0.67 * log10_Pr_exp_direct
        n_exp_direct = 0.75 - 1.27 * log10_Pr_exp_direct
        val_exp_direct = log10_Pr_exp_direct - c_exp_direct
        denom_exp_direct = n_exp_direct - 0.14 * val_exp_direct
        F_exp_direct = 10**(np.log10(F_cent_exp_direct) / (1.0 + (val_exp_direct / denom_exp_direct)**2)) if F_cent_exp_direct > 0 and denom_exp_direct != 0 else 1.0
        
        k_expected_direct_M = k_inf_val_exp * (Pr_exp_direct / (1 + Pr_exp_direct)) * F_exp_direct
        k_calc_direct_M = calculate_troe_rate(troe_data1, T, P_target=999, M_conc=M_conc_direct) 
        self.assertAlmostEqual(k_calc_direct_M, k_expected_direct_M, delta=k_expected_direct_M * 1e-5)

        self.assertAlmostEqual(calculate_troe_rate(troe_data1, T, P_target=0, M_conc=None), 0.0, places=8)
        self.assertIsNone(calculate_troe_rate(troe_data1, T, P_target=-1.0, M_conc=None))
        self.assertIsNone(calculate_troe_rate(troe_data1, 0, P_target, M_conc=None))
        self.assertIsNone(calculate_troe_rate(troe_data1, -100, P_target, M_conc=None))

        troe_invalid_coeffs1 = {**troe_data1, 'coeffs': [0.6, 0.0, 1200.0]}
        self.assertIsNone(calculate_troe_rate(troe_invalid_coeffs1, T, P_target, None))
        
        troe_invalid_coeffs2 = {**troe_data1, 'coeffs': [0.6, 200.0, -100.0]}
        self.assertIsNone(calculate_troe_rate(troe_invalid_coeffs2, T, P_target, None))

        troe_non_positive_T2s = {**troe_data1, 'coeffs': [0.6, 200.0, 1200.0, -50.0]}
        alpha_noT2s, T3s_noT2s, T1s_noT2s, _ = troe_data1['coeffs']
        F_cent_exp_noT2s = (1 - alpha_noT2s) * np.exp(-T / T3s_noT2s) + alpha_noT2s * np.exp(-T / T1s_noT2s)
        log10_Pr_exp_noT2s = np.log10(Pr_exp) if Pr_exp > 0 else -np.inf
        c_exp_noT2s = -0.4 - 0.67 * log10_Pr_exp_noT2s
        n_exp_noT2s = 0.75 - 1.27 * log10_Pr_exp_noT2s
        val_exp_noT2s = log10_Pr_exp_noT2s - c_exp_noT2s
        denom_exp_noT2s = n_exp_noT2s - 0.14 * val_exp_noT2s
        F_exp_noT2s = 10**(np.log10(F_cent_exp_noT2s) / (1.0 + (val_exp_noT2s / denom_exp_noT2s)**2)) if F_cent_exp_noT2s > 0 and denom_exp_noT2s != 0 else 1.0
        k_expected_noT2s = k_inf_val_exp * (Pr_exp / (1 + Pr_exp)) * F_exp_noT2s
        self.assertAlmostEqual(calculate_troe_rate(troe_non_positive_T2s, T, P_target, None), k_expected_noT2s, delta=k_expected_noT2s*1e-5)
        
        troe_with_None_T2s = {**troe_data1, 'coeffs': [0.6, 200.0, 1200.0, None]}
        self.assertAlmostEqual(calculate_troe_rate(troe_with_None_T2s, T, P_target, None), k_expected_noT2s, delta=k_expected_noT2s*1e-5)

# ---- New Test Classes for Reverse Rate Calculations ----
MOCK_THERMO_SPECIES_PROPERTIES = {
    "A": (10000.0, 200.0, -190000.0, 50.0), 
    "B": (15000.0, 220.0, -205000.0, 60.0),
    "C": (5000.0, 180.0, -175000.0, 40.0),
    "D_SOLID": (2000.0, 50.0, -48000.0, 30.0),
    "X_TRANGE": (None, None, None, None) 
}

MOCK_THERMO_DATA_DICT = {
    "A": {'species_name': "A", 'phase': 'G'}, 
    "B": {'species_name': "B", 'phase': 'G'},
    "C": {'species_name': "C", 'phase': 'G'},
    "D_SOLID": {'species_name': "D_SOLID", 'phase': 'S'},
    "X_TRANGE": {'species_name': "X_TRANGE", 'phase': 'G'}
}

def mock_get_thermo_properties_for_rate_calc_tests(species_name, T_kelvin, thermo_data_dict_ignored):
    if T_kelvin == 1000.0: # Mock specific behavior for T=1000K
        return MOCK_THERMO_SPECIES_PROPERTIES.get(species_name.upper(), (None, None, None, None))
    # Add behavior for other temperatures if needed by other tests, or keep it simple
    return (None, None, None, None) # Default for unmocked T or species

@mock.patch('rate_calculator.get_thermo_properties', mock_get_thermo_properties_for_rate_calc_tests)
class TestGetReactionThermoProperties(unittest.TestCase):
    def setUp(self):
        if get_reaction_thermo_properties is None:
            self.skipTest("get_reaction_thermo_properties not imported.")
        self.reaction_AB_C = {'reactants': [('A',1), ('B',1)], 'products': [('C',1)]}
        self.reaction_missing_from_dict = {'reactants': [('Z_NOT_IN_DICT',1)], 'products': [('A',1)]}
        self.reaction_calc_error_for_species = {'reactants': [('A',1)], 'products': [('X_TRANGE',1)]}

    def test_calc_delta_hsg_simple_reaction_AB_C(self):
        dH, dS, dG, missing, err = get_reaction_thermo_properties(self.reaction_AB_C, 1000.0, MOCK_THERMO_DATA_DICT)
        self.assertFalse(err)
        self.assertEqual(len(missing), 0)
        self.assertAlmostEqual(dH, -20000.0) # 5000 - (10000+15000)
        self.assertAlmostEqual(dS, -240.0)  # 180 - (200+220)
        self.assertAlmostEqual(dG, 220000.0) # -175000 - (-190000 + -205000)

    def test_species_missing_from_thermo_data_dict(self):
        dH, dS, dG, missing, err = get_reaction_thermo_properties(self.reaction_missing_from_dict, 1000.0, MOCK_THERMO_DATA_DICT)
        self.assertTrue(err)
        self.assertIn("Z_NOT_IN_DICT", missing)
        self.assertIsNone(dH)

    def test_thermo_calculation_error_for_species(self): # e.g. T out of range
        dH, dS, dG, missing, err = get_reaction_thermo_properties(self.reaction_trange_err, 1000.0, MOCK_THERMO_DATA_DICT)
        self.assertTrue(err) 
        self.assertTrue(any("X_TRANGE (T_range @1000.0K)" in m for m in missing))
        self.assertIsNone(dH)

class TestCalculateEquilibriumConstantKp(unittest.TestCase):
    def test_kp_calc_valid_input(self):
        if calculate_equilibrium_constant_kp is None: self.skipTest("Kp function not imported.")
        expected_Kp = np.exp(5000.0 / (R_J_MOL_K * 1000.0)) # dG = -5000 J/mol
        self.assertAlmostEqual(calculate_equilibrium_constant_kp(-5000.0, 1000.0), expected_Kp)

    def test_kp_calc_T_zero_or_negative(self):
        if calculate_equilibrium_constant_kp is None: self.skipTest("Kp function not imported.")
        self.assertIsNone(calculate_equilibrium_constant_kp(-5000.0, 0.0))
        self.assertIsNone(calculate_equilibrium_constant_kp(-5000.0, -100.0))
        
    def test_kp_calc_dG_None(self):
        if calculate_equilibrium_constant_kp is None: self.skipTest("Kp function not imported.")
        self.assertIsNone(calculate_equilibrium_constant_kp(None, 1000.0))

    def test_kp_overflow_and_underflow(self): 
        if calculate_equilibrium_constant_kp is None: self.skipTest("Kp function not imported.")
        # Test overflow: -dG / RT is very large positive
        overflow_neg_dG = -710 * R_J_MOL_K * 1000.0 # exp(710) overflows float64
        self.assertEqual(calculate_equilibrium_constant_kp(overflow_neg_dG, 1000.0), np.inf)
        # Test underflow: -dG / RT is very large negative (dG is very large positive)
        large_pos_dG = 710 * R_J_MOL_K * 1000.0 # exp(-710) is ~0
        self.assertAlmostEqual(calculate_equilibrium_constant_kp(large_pos_dG, 1000.0), 0.0)

class TestCalculateDeltaNGas(unittest.TestCase):
    def setUp(self):
        if calculate_delta_n_gas is None: self.skipTest("delta_n_gas function not imported.")
        self.mock_thermo = MOCK_THERMO_DATA_DICT 

    def test_delta_n_all_gas(self):
        reaction = {'reactants': [('A',1), ('B',1)], 'products': [('C',1)]} 
        self.assertEqual(calculate_delta_n_gas(reaction, self.mock_thermo), -1.0)

    def test_delta_n_mixed_phase(self):
        reaction = {'reactants': [('A',1), ('D_SOLID',1)], 'products': [('C',1)]}
        self.assertEqual(calculate_delta_n_gas(reaction, self.mock_thermo), 0.0)
        
    def test_delta_n_species_not_in_thermo_defaults_to_gas(self):
        reaction = {'reactants': [('X_UNKNOWN',1)], 'products': [('Y_UNKNOWN',1), ('Z_UNKNOWN',1)]} 
        self.assertEqual(calculate_delta_n_gas(reaction, self.mock_thermo), 1.0) # Z+Y - X = 1+1-1 = 1

class TestCalculateEquilibriumConstantKc(unittest.TestCase):
    def test_kc_calc_valid_input(self):
        if calculate_equilibrium_constant_kc is None: self.skipTest("Kc function not imported.")
        R_L = R_L_ATM_MOL_K 
        Kp, T, dn = 100.0, 1000.0, -1.0
        expected_Kc = Kp * ( (1.0 / (R_L * T)) ** dn )
        self.assertAlmostEqual(calculate_equilibrium_constant_kc(Kp, T, dn), expected_Kc)

    def test_kc_calc_T_zero_or_negative(self):
        if calculate_equilibrium_constant_kc is None: self.skipTest("Kc function not imported.")
        self.assertIsNone(calculate_equilibrium_constant_kc(100.0, 0.0, 1.0))
        self.assertIsNone(calculate_equilibrium_constant_kc(100.0, -100.0, 1.0))

    def test_kc_with_invalid_inputs(self):
        if calculate_equilibrium_constant_kc is None: self.skipTest("Kc function not imported.")
        self.assertIsNone(calculate_equilibrium_constant_kc(None, 1000.0, 1.0)) # Kp is None
        self.assertIsNone(calculate_equilibrium_constant_kc(100.0, 1000.0, None)) # delta_n_gas is None

class TestCalculateReverseRateConstant(unittest.TestCase):
    def test_kr_calc_valid_input(self):
        if calculate_reverse_rate_constant is None: self.skipTest("kr function not imported.")
        self.assertAlmostEqual(calculate_reverse_rate_constant(kf=100.0, Kc=10.0), 10.0)

    def test_kr_with_Kc_zero(self):
        if calculate_reverse_rate_constant is None: self.skipTest("kr function not imported.")
        self.assertIsNone(calculate_reverse_rate_constant(kf=100.0, Kc=0.0))

    def test_kr_with_invalid_inputs(self):
        if calculate_reverse_rate_constant is None: self.skipTest("kr function not imported.")
        self.assertIsNone(calculate_reverse_rate_constant(kf=100.0, Kc=None))
        self.assertIsNone(calculate_reverse_rate_constant(kf=None, Kc=10.0))

if __name__ == '__main__':
    unittest.main()
