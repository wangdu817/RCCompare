import numpy as np
import math # For np.exp, math.log if needed, though np usually handles it.

# Assuming thermo_calculator.py is accessible
try:
    from thermo_calculator import get_thermo_properties
except ImportError:
    print("Warning: thermo_calculator.py not found or get_thermo_properties could not be imported.")
    get_thermo_properties = None 

# Global constants
R_cal = 1.987204  # Gas constant in cal/mol-K
R_atm_cm3 = 82.057 # Gas constant in cm^3*atm/mol-K

# Constants for reverse rate calculation
R_J_MOL_K = 8.314462618       # Gas constant in J/mol-K
P_ATM_PA = 101325.0           # Standard pressure in Pascals (1 atm)
R_L_ATM_MOL_K = 0.08205736608 # Gas constant in L*atm/mol-K


def get_reaction_thermo_properties(reaction_data, T_kelvin, thermo_data_dict):
    """
    Calculates the change in standard enthalpy, entropy, and Gibbs free energy
    for a reaction at a given temperature.

    Args:
        reaction_data (dict): Dictionary for a single reaction, including
                              'reactants' and 'products' lists. Each item in
                              these lists is expected to be a tuple or list
                              like (species_name, stoichiometric_coeff).
        T_kelvin (float): Temperature in Kelvin.
        thermo_data_dict (dict): Dictionary of parsed thermodynamic data from
                                 thermo_parser.read_nasa_polynomials.

    Returns:
        tuple: (delta_H_reaction_J_mol, delta_S_reaction_J_mol_K,
                delta_G_reaction_J_mol, missing_species_list (list), error_occurred (bool))
               Returns (None, None, None, missing_species_list, True) if any species' thermo data
               is not found or T is out of range for any species.
    """
    if get_thermo_properties is None:
        print("Error: get_thermo_properties function not available from thermo_calculator.")
        return None, None, None, [], True # Return empty list for missing_species

    delta_H_reaction = 0.0
    delta_S_reaction = 0.0
    delta_G_reaction = 0.0
    missing_species_set = set() # Use a set to store unique missing species
    error_occurred_flag = False

    all_species_in_reaction = reaction_data.get('reactants', []) + reaction_data.get('products', [])

    for species_name, _ in all_species_in_reaction:
        # Check if species exists in thermo_data_dict first to identify missing ones
        if not thermo_data_dict.get(species_name.upper()):
            missing_species_set.add(species_name)
            error_occurred_flag = True # Mark error if any species is missing from dict

    if error_occurred_flag: # If any species are outright missing from the thermo_data
        return None, None, None, sorted(list(missing_species_set)), True

    # If all species are in thermo_data_dict, proceed to get properties
    # Products
    for species_name, coeff in reaction_data.get('products', []):
        H, S, G, Cp = get_thermo_properties(species_name, T_kelvin, thermo_data_dict)
        if H is None or S is None or G is None or Cp is None : # Check all relevant properties
            # This implies T is out of range for an existing species, or other calculation error
            # print(f"Error: Thermo properties could not be calculated for product {species_name} at {T_kelvin}K (likely T out of range).")
            missing_species_set.add(species_name + f" (T_range @{T_kelvin}K)") # Indicate T-range issue
            error_occurred_flag = True
        else:
            delta_H_reaction += coeff * H
            delta_S_reaction += coeff * S
            delta_G_reaction += coeff * G

    # Reactants
    for species_name, coeff in reaction_data.get('reactants', []):
        H, S, G, Cp = get_thermo_properties(species_name, T_kelvin, thermo_data_dict)
        if H is None or S is None or G is None or Cp is None:
            # print(f"Error: Thermo properties could not be calculated for reactant {species_name} at {T_kelvin}K (likely T out of range).")
            missing_species_set.add(species_name + f" (T_range @{T_kelvin}K)")
            error_occurred_flag = True
        else:
            delta_H_reaction -= coeff * H
            delta_S_reaction -= coeff * S
            delta_G_reaction -= coeff * G
    
    if error_occurred_flag:
        return None, None, None, sorted(list(missing_species_set)), True
        
    return delta_H_reaction, delta_S_reaction, delta_G_reaction, [], False


def calculate_equilibrium_constant_kp(delta_G_reaction_J_mol, T_kelvin):
    """
    Calculates the equilibrium constant Kp from delta_G_reaction.

    Args:
        delta_G_reaction_J_mol (float): Standard Gibbs free energy change of reaction in J/mol.
        T_kelvin (float): Temperature in Kelvin.

    Returns:
        float: The equilibrium constant Kp (unitless, based on activities/partial pressures
               referenced to a standard state of 1 atm or 1 bar, effectively unitless).
               Returns None if inputs are invalid or calculation fails.
    """
    if delta_G_reaction_J_mol is None or T_kelvin is None or T_kelvin <= 0:
        return None
    try:
        # Kp = exp(-deltaG_0 / RT)
        Kp = np.exp(-delta_G_reaction_J_mol / (R_J_MOL_K * T_kelvin))
        return Kp
    except OverflowError:
        # print(f"OverflowError calculating Kp for deltaG={delta_G_reaction_J_mol} at T={T_kelvin}K.")
        return np.inf # Or some other indicator of a very large number if appropriate
    except Exception as e:
        # print(f"Error calculating Kp: {e}")
        return None

def calculate_delta_n_gas(reaction_data, thermo_data_dict):
    """
    Calculates the change in the number of moles of gas-phase species in a reaction.

    Args:
        reaction_data (dict): Dictionary for a single reaction.
        thermo_data_dict (dict): Dictionary of parsed thermodynamic data.
                                 Used to check phase if available.

    Returns:
        float: The change in moles of gas, delta_n_gas.
               Returns 0 if phase information is not available and cannot be determined.
    """
    delta_n = 0.0
    
    # Helper to check if a species is gas phase
    def is_gas(species_name, thermo_db):
        if get_thermo_properties is None: # Relies on thermo_calculator being imported
            return True # Assume gas if thermo lookup isn't possible

        species_thermo_data = thermo_db.get(species_name.upper())
        if species_thermo_data:
            phase = species_thermo_data.get('phase')
            if phase and isinstance(phase, str) and phase.upper() == 'G':
                return True
            elif phase is None: # If phase is not specified, assume gas (common in some CHEMKIN files)
                return True 
            return False # If phase is specified and not 'G'
        return True # Assume gas if species not in thermo_db (could be an error, but delta_n proceeds)

    # Products
    for species_name, coeff in reaction_data.get('products', []):
        if is_gas(species_name, thermo_data_dict):
            delta_n += coeff
            
    # Reactants
    for species_name, coeff in reaction_data.get('reactants', []):
        if is_gas(species_name, thermo_data_dict):
            delta_n -= coeff
            
    return delta_n


def calculate_equilibrium_constant_kc(Kp, T_kelvin, delta_n_gas):
    """
    Calculates the equilibrium constant Kc from Kp.
    Kc = Kp * (1/(R_L_ATM_MOL_K * T)) ^ delta_n_gas
    Assumes Kp is unitless (activities/partial pressures referenced to 1 atm standard state)
    and Kc is in (mol/L)^delta_n_gas.

    Args:
        Kp (float): The equilibrium constant Kp.
        T_kelvin (float): Temperature in Kelvin.
        delta_n_gas (float): Change in the number of moles of gas-phase species.

    Returns:
        float: The equilibrium constant Kc.
               Returns None if inputs are invalid or calculation fails.
    """
    if Kp is None or T_kelvin is None or T_kelvin <= 0 or delta_n_gas is None:
        return None
    
    try:
        # Factor = (1 / (R_L_ATM_MOL_K * T_kelvin))
        # Kc = Kp * (Factor ^ delta_n_gas)
        # Using R_L_ATM_MOL_K (0.082057 L*atm/mol*K)
        # If Kp is unitless (standard state P_std = 1 atm), then
        # Kc = Kp * (P_std / (R_L_ATM_MOL_K * T_kelvin)) ^ delta_n_gas
        # Since P_std = 1 atm, this simplifies to Kp * (1 / (R_L_ATM_MOL_K * T_kelvin)) ^ delta_n_gas
        
        RT = R_L_ATM_MOL_K * T_kelvin
        if RT == 0: # Avoid division by zero if T_kelvin was not strictly > 0
            return None 
            
        conversion_factor = 1.0 / RT
        Kc = Kp * (conversion_factor ** delta_n_gas)
        return Kc
    except OverflowError:
        # print(f"OverflowError calculating Kc for Kp={Kp}, T={T_kelvin}K, delta_n={delta_n_gas}.")
        return np.inf # Or some other indicator
    except Exception as e:
        # print(f"Error calculating Kc: {e}")
        return None

def calculate_reverse_rate_constant(kf, Kc):
    """
    Calculates the reverse rate constant (kr) from the forward rate
    constant (kf) and the equilibrium constant in concentration units (Kc).

    Args:
        kf (float): Forward rate constant. Units depend on reaction order,
                    but must be consistent with Kc's implied concentration units.
        Kc (float): Equilibrium constant in concentration units (e.g., (mol/L)^delta_n).

    Returns:
        float: Reverse rate constant kr.
               Returns None if inputs are invalid or Kc is zero.
    """
    if kf is None or Kc is None:
        return None
    if Kc == 0: # Avoid division by zero
        # print("Warning: Kc is zero, reverse rate constant cannot be determined (or is infinite).")
        return None # Or np.inf if that's preferred for an "infinitely fast" reverse reaction
    
    try:
        kr = kf / Kc
        return kr
    except OverflowError:
        # print(f"OverflowError calculating kr for kf={kf}, Kc={Kc}.")
        # This case is less common if kf and Kc are typical floats unless Kc is extremely small.
        return np.inf 
    except Exception as e:
        # print(f"Error calculating kr: {e}")
        return None


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

    # Explicitly check for the failing test cases' exact strings first
    if units_str_norm == 'JOULES/MOLE':
        return Ea_original / 4.184
    if units_str_norm == 'KJOULES/MOLE':
        return (Ea_original * 1000.0) / 4.184

    # Then, try the general list approach (retaining the previous fix attempt for broader compatibility)
    joules_options = ['J/MOL', 'JOULES/MOL', 'JOULE/MOLE'] # 'JOULES/MOLE' is repeated here, but it's fine
    kjoules_options = ['KJ/MOL', 'KJOULES/MOL', 'KJOULE/MOLE'] # 'KJOULES/MOLE' is repeated, fine

    if units_str_norm in joules_options:
        return Ea_original / 4.184
    elif units_str_norm in kjoules_options:
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

