import unittest
import numpy as np
from rate_calculator import (
    _convert_ea_to_cal_per_mol, # Make sure to test this if it's directly callable, or test via main functions
    get_third_body_concentration,
    calculate_arrhenius_rate,
    calculate_plog_rate,
    calculate_troe_rate,
    R_cal, 
    R_atm_cm3
)

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
        self.assertEqual(get_third_body_concentration(1.0, 0.0), 0.0)
        self.assertEqual(get_third_body_concentration(-1.0, 298.0), 0.0)

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
        
        log10_Pr_exp = np.log10(Pr_exp)
        c_exp = -0.4 - 0.67 * log10_Pr_exp
        n_exp = 0.75 - 1.27 * log10_Pr_exp
        val_exp = log10_Pr_exp - c_exp
        denom_exp = n_exp - 0.14 * val_exp
        F_exp = 10**(np.log10(F_cent_exp) / (1.0 + (val_exp / denom_exp)**2))
        
        k_expected = k_inf_val_exp * (Pr_exp / (1 + Pr_exp)) * F_exp
        
        k_calc = calculate_troe_rate(troe_data1, T, P_target, M_conc=None)
        self.assertAlmostEqual(k_calc, k_expected, delta=k_expected * 1e-5)

        # Test with missing k0
        troe_data_no_k0 = {
            'k_inf': {'A': 1.0E14, 'n': 0.0, 'Ea': 2.0, 'units': 'KCAL/MOLE'},
            'k0': {}, # Missing A, n, Ea
            'coeffs': [0.6, 200.0, 1200.0] 
        }
        self.assertIsNone(calculate_troe_rate(troe_data_no_k0, T, P_target, None))

        # Test with missing k_inf
        troe_data_no_kinf = {
            'k_inf': {},
            'k0': {'A': 1.0E16, 'n': 0.0, 'Ea': 500.0, 'units': 'CAL/MOLE'},
            'coeffs': [0.6, 200.0, 1200.0] 
        }
        self.assertIsNone(calculate_troe_rate(troe_data_no_kinf, T, P_target, None))

        # Test with missing troe_coeffs
        troe_data_no_coeffs = {
            'k_inf': {'A': 1.0E14, 'n': 0.0, 'Ea': 2.0, 'units': 'KCAL/MOLE'},
            'k0': {'A': 1.0E16, 'n': 0.0, 'Ea': 500.0, 'units': 'CAL/MOLE'},
            'coeffs': [] 
        }
        self.assertIsNone(calculate_troe_rate(troe_data_no_coeffs, T, P_target, None))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
