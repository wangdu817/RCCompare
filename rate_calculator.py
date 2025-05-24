import numpy as np

# Global constants
R_cal = 1.987204  # Gas constant in cal/mol-K
R_atm_cm3 = 82.057 # Gas constant in cm^3*atm/mol-K

def _convert_ea_to_cal_per_mol(Ea_original, units_str):
    """
    Converts activation energy to cal/mol based on its original units.
    """
    if not units_str: # Default to CAL/MOLE if units not specified
        # print(f"Warning: Units not specified for Ea={Ea_original}. Assuming CAL/MOLE.")
        return Ea_original

    units_str_norm = units_str.upper().replace(" ", "").replace("_", "") # Normalize

    if units_str_norm == 'CAL/MOLE':
        return Ea_original
    elif units_str_norm == 'KCAL/MOLE':
        return Ea_original * 1000.0
    elif units_str_norm in ['J/MOL', 'JOULES/MOL', 'JOULE/MOLE']: # Added JOULE/MOLE
        return Ea_original / 4.184
    elif units_str_norm in ['KJ/MOL', 'KJOULES/MOL', 'KJOULE/MOLE']: # Added KJOULE/MOLE
        return (Ea_original * 1000.0) / 4.184
    elif units_str_norm == 'KELVINS' or units_str_norm == 'K':
        return Ea_original * R_cal
    elif units_str_norm == 'EVOLTS':
        # 1 eV = 23.0605 kcal/mol = 23060.5 cal/mol
        return Ea_original * 23060.5
    else:
        print(f"Warning: Unrecognized Ea units '{units_str}'. Ea value {Ea_original} used as is (assumed cal/mol).")
        return Ea_original

def get_third_body_concentration(P_atm, T_K):
    """
    Calculates third-body concentration [M] in mol/cm^3.
    """
    if T_K <= 0: return 0.0
    if P_atm < 0: return 0.0
    return P_atm / (R_atm_cm3 * T_K)

def calculate_arrhenius_rate(arrhenius_params_dict, T):
    """
    Calculates Arrhenius rate constant.
    arrhenius_params_dict: Dict {'A': val, 'n': val, 'Ea': val, 'units': unit_str}
    T: Temperature in Kelvin
    """
    if not arrhenius_params_dict or \
       arrhenius_params_dict.get('A') is None or \
       arrhenius_params_dict.get('n') is None or \
       arrhenius_params_dict.get('Ea') is None:
        # print("Warning: Arrhenius parameters A, n, Ea are required.")
        return None
    
    A = arrhenius_params_dict['A']
    n = arrhenius_params_dict['n']
    Ea_orig = arrhenius_params_dict['Ea']
    original_units = arrhenius_params_dict.get('units') # units might be missing

    Ea_cal_mol = _convert_ea_to_cal_per_mol(Ea_orig, original_units)

    if T <= 0:
        if Ea_cal_mol > 0 : return 0.0
        elif Ea_cal_mol == 0 and n > 0: return 0.0
        elif Ea_cal_mol == 0 and n == 0: return A
        else: return None

    try:
        rate_constant = A * (T**n) * np.exp(-Ea_cal_mol / (R_cal * T))
    except OverflowError:
        rate_constant = np.inf 
    return rate_constant


def calculate_plog_rate(plog_entries_list, T, P_target):
    """
    Calculates rate constant using PLOG interpolation.
    plog_entries_list: list of dicts, each like {'pressure': P, 'A': A, 'n': n, 'Ea': Ea, 'units': unit_str}
    """
    if T <= 0: return None
    if P_target <= 0: return None
    if not plog_entries_list: return None

    rates_at_T = []
    for entry in plog_entries_list:
        if not all(k in entry for k in ['pressure', 'A', 'n', 'Ea']):
            # print(f"Warning: Invalid PLOG entry, missing keys: {entry}")
            continue
        
        pressure_val = entry['pressure']
        if pressure_val <= 0: continue

        # Each PLOG entry is an arrhenius_params_dict
        ki = calculate_arrhenius_rate(entry, T) 
        
        if ki is None or ki < 0: continue

        if ki == 0.0 and pressure_val > 0 :
             rates_at_T.append({'log_P': np.log(pressure_val), 'log_k': -np.inf, 'k': 0.0})
        elif ki > 0 :
            rates_at_T.append({'log_P': np.log(pressure_val), 'log_k': np.log(ki), 'k': ki})

    if not rates_at_T: return None
    rates_at_T.sort(key=lambda x: x['log_P'])

    if len(rates_at_T) == 1: return rates_at_T[0]['k']

    unique_rates = []
    if rates_at_T:
        unique_rates.append(rates_at_T[0])
        for i in range(1, len(rates_at_T)):
            if not np.isclose(rates_at_T[i]['log_P'], rates_at_T[i-1]['log_P']):
                unique_rates.append(rates_at_T[i])
    rates_at_T = unique_rates

    if not rates_at_T: return None
    if len(rates_at_T) == 1: return rates_at_T[0]['k']

    log_P_target = np.log(P_target)

    for entry_u in rates_at_T: # Use 'entry_u' to avoid clash
        if np.isclose(log_P_target, entry_u['log_P']):
            return entry_u['k']

    if log_P_target < rates_at_T[0]['log_P']: return rates_at_T[0]['k']
    if log_P_target > rates_at_T[-1]['log_P']: return rates_at_T[-1]['k']

    P_low_entry, P_high_entry = None, None
    for i in range(len(rates_at_T) - 1):
        if rates_at_T[i]['log_P'] < log_P_target < rates_at_T[i+1]['log_P']:
            P_low_entry = rates_at_T[i]
            P_high_entry = rates_at_T[i+1]
            break
    
    if P_low_entry is None or P_high_entry is None :
        min_dist = float('inf'); closest_k = None
        for entry_u in rates_at_T:
            dist = abs(log_P_target - entry_u['log_P'])
            if dist < min_dist: min_dist = dist; closest_k = entry_u['k']
        return closest_k

    log_P1, log_k1 = P_low_entry['log_P'], P_low_entry['log_k']
    log_P2, log_k2 = P_high_entry['log_P'], P_high_entry['log_k']

    if log_P1 == log_P2: return P_low_entry['k']
    
    log_k_target = np.interp(log_P_target, [log_P1, log_P2], [log_k1, log_k2])

    if log_k_target == -np.inf: return 0.0
    try: return np.exp(log_k_target)
    except OverflowError: return np.inf


def calculate_troe_rate(troe_data_dict, T, P_target, M_conc=None):
    """
    Calculates Troe fall-off rate constant.
    troe_data_dict: Dict {'k_inf': k_inf_dict, 'k0': k0_dict, 'coeffs': troe_coeffs_list}
    """
    if T <= 0: return None

    k_inf_dict = troe_data_dict.get('k_inf', {})
    k0_dict = troe_data_dict.get('k0', {})
    troe_coeffs = troe_data_dict.get('coeffs', [])

    if not k_inf_dict or not k_inf_dict.get('A') is not None: # Check if A is present and not None
        # print("Warning: k_inf params (A,n,Ea) are required for TROE.")
        return None
    if not k0_dict or not k0_dict.get('A') is not None: # Check if A is present and not None
        # print("Warning: k0 params (A0,n0,Ea0) are required for TROE.")
        return None
    if not troe_coeffs or len(troe_coeffs) < 3:
        # print("Warning: troe_coeffs [alpha, T***, T*] are required.")
        return None

    if M_conc is None:
        if P_target is None or P_target < 0: return None
        M_conc = get_third_body_concentration(P_target, T)
        if M_conc < 0 : M_conc = 0.0
    if M_conc < 0: return None

    k_inf = calculate_arrhenius_rate(k_inf_dict, T)
    k0_val = calculate_arrhenius_rate(k0_dict, T)

    if k_inf is None or k0_val is None: return None 
    if k_inf < 0 or k0_val < 0: return 0.0 
    if k_inf == 0.0: return 0.0

    k0_eff = k0_val * M_conc
    if k0_eff < 0: k0_eff = 0.0 

    if k_inf <= 1e-100: return 0.0

    Pr = k0_eff / k_inf

    alpha = troe_coeffs[0]
    T3star = troe_coeffs[1]
    T1star = troe_coeffs[2]

    if T3star <= 0 or T1star <= 0: return None 

    F_cent = (1 - alpha) * np.exp(-T / T3star) + alpha * np.exp(-T / T1star)
    
    if len(troe_coeffs) >= 4 and troe_coeffs[3] is not None:
        T2star = troe_coeffs[3] 
        if T2star > 0: 
            F_cent += np.exp(-T / T2star) 

    if Pr <= 1e-30: F = 1.0 
    elif F_cent <= 1e-30: F = 0.0
    else:
        log10_Pr = np.log10(Pr)
        c = -0.4 - 0.67 * log10_Pr
        n_troe = 0.75 - 1.27 * log10_Pr
        val = log10_Pr - c
        denom_of_frac = n_troe - 0.14 * val
        if np.abs(denom_of_frac) < 1e-30: log10_F = 0.0
        else:
            inner_frac_squared = (val / denom_of_frac)**2
            log10_F_num = np.log10(F_cent)
            log10_F = log10_F_num / (1.0 + inner_frac_squared)
        try: F = 10**log10_F
        except OverflowError: F = np.inf

    if (1 + Pr) <= 1e-30 : k = 0.0 
    else: k = k_inf * (Pr / (1 + Pr)) * F
    return k

if __name__ == '__main__':
    print("--- Unit Conversion Tests ---")
    print(f"1000 CAL/MOLE -> {_convert_ea_to_cal_per_mol(1000, 'CAL/MOLE')} cal/mol")
    print(f"1 KCAL/MOLE -> {_convert_ea_to_cal_per_mol(1, 'KCAL/MOLE')} cal/mol")
    print(f"418.4 J/MOL -> {_convert_ea_to_cal_per_mol(418.4, 'J/MOL')} cal/mol (expect ~100)")
    print(f"4.184 KJ/MOL -> {_convert_ea_to_cal_per_mol(4.184, 'KJ/MOL')} cal/mol (expect ~1000)")
    print(f"1000 KELVINS -> {_convert_ea_to_cal_per_mol(1000, 'KELVINS')} cal/mol (expect ~1987.2)")
    print(f"1 EVOLTS -> {_convert_ea_to_cal_per_mol(1, 'EVOLTS')} cal/mol (expect ~23060.5)")


    print("\n--- Arrhenius Tests (New Structure) ---")
    arrh_params1 = {'A': 1e13, 'n': 0, 'Ea': 10000, 'units': 'CAL/MOLE'}
    print(f"Arrhenius (CAL/MOLE): {calculate_arrhenius_rate(arrh_params1, 1000)}")
    arrh_params2 = {'A': 1e13, 'n': 0, 'Ea': 10, 'units': 'KCAL/MOLE'} # Ea = 10 kcal/mol
    print(f"Arrhenius (KCAL/MOLE): {calculate_arrhenius_rate(arrh_params2, 1000)}")
    # Expected value for arrh_params2 manually: 1e13 * exp(-10000 / (1.987204 * 1000)) = 7.386e10

    print("\n--- PLOG Tests (New Structure) ---")
    plog_data_new = [
        {'pressure': 1.0, 'A': 1e10, 'n': 0, 'Ea': 1, 'units': 'KCAL/MOLE'},
        {'pressure': 10.0, 'A': 1e12, 'n': 0, 'Ea': 2000, 'units': 'CAL/MOLE'} 
    ]
    print(f"PLOG (T=500K, P=5atm): {calculate_plog_rate(plog_data_new, 500, 5.0)}")

    print("\n--- TROE Tests (New Structure) ---")
    troe_data_example = {
        'k_inf': {'A': 1.0E14, 'n': 0.0, 'Ea': 2.0, 'units': 'KCAL/MOLE'}, # 2000 cal/mol
        'k0': {'A': 1.0E16, 'n': 0.0, 'Ea': 500.0, 'units': 'CAL/MOLE'},   # 500 cal/mol
        'coeffs': [0.5, 100.0, 1000.0, 5000.0] 
    }
    T_test = 1000; P_test = 1.0
    print(f"TROE (T={T_test}K, P={P_test}atm): {calculate_troe_rate(troe_data_example, T_test, P_test, M_conc=None)}")

    troe_data_no_k0 = {
        'k_inf': {'A': 1.0E14, 'n': 0.0, 'Ea': 2.0, 'units': 'KCAL/MOLE'},
        'k0': {}, # Missing k0 params
        'coeffs': [0.5, 100.0, 1000.0] 
    }
    print(f"TROE (No k0 A, T={T_test}K, P={P_test}atm): {calculate_troe_rate(troe_data_no_k0, T_test, P_test, M_conc=None)}")

```
