import math
import numpy as np # For T^2, T^3, T^4 in polynomial calculations

# Gas constant in J/mol-K
R_J_MOL_K = 8.31446261815324

def get_nasa_polynomial(species_name, thermo_data_dict):
    """
    Retrieves the polynomial data for a given species.

    Args:
        species_name (str): Name of the species (case-insensitive).
        thermo_data_dict (dict): Dictionary of thermo data from read_nasa_polynomials.

    Returns:
        dict: The specific species data dictionary containing 'T_ranges', 
              'coeffs', etc., or None if species not found.
    """
    return thermo_data_dict.get(species_name.upper())

def calculate_cp_h_s(T_kelvin, species_coeffs):
    """
    Calculates Cp/R, H/RT, S/R and subsequently Cp, H, S for a given
    temperature and set of 7 NASA polynomial coefficients.

    Args:
        T_kelvin (float): Temperature in Kelvin.
        species_coeffs (list): A list of 7 coefficients [a1, a2, a3, a4, a5, a6, a7].

    Returns:
        tuple: (Cp_J_mol_K, H_J_mol, S_J_mol_K)
               Returns (None, None, None) if species_coeffs is invalid.
    """
    if not species_coeffs or len(species_coeffs) != 7:
        # print("Error: Invalid species_coeffs provided.")
        return None, None, None

    a1, a2, a3, a4, a5, a6, a7 = species_coeffs
    T = float(T_kelvin) # Ensure T is float

    # Avoid issues with T=0 for H/RT term (a6/T) and S/R term (a1*ln(T))
    if T <= 0:
        # print("Error: Temperature must be positive for Cp/H/S calculation.")
        return None, None, None

    # Calculate Cp/R, H/RT, S/R
    # Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    cp_R = a1 + a2*T + a3*(T**2) + a4*(T**3) + a5*(T**4)

    # H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
    h_RT = a1 + (a2*T)/2 + (a3*(T**2))/3 + (a4*(T**3))/4 + (a5*(T**4))/5 + a6/T

    # S/R  = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
    s_R = a1*math.log(T) + a2*T + (a3*(T**2))/2 + (a4*(T**3))/3 + (a5*(T**4))/4 + a7

    # Convert to absolute values
    Cp_J_mol_K = cp_R * R_J_MOL_K
    H_J_mol = h_RT * R_J_MOL_K * T
    S_J_mol_K = s_R * R_J_MOL_K
    
    return Cp_J_mol_K, H_J_mol, S_J_mol_K

def calculate_gibbs_g(H_J_mol, S_J_mol_K, T_kelvin):
    """
    Calculates Gibbs free energy G = H - T*S.

    Args:
        H_J_mol (float): Enthalpy in J/mol.
        S_J_mol_K (float): Entropy in J/mol-K.
        T_kelvin (float): Temperature in Kelvin.

    Returns:
        float: Gibbs free energy G in J/mol, or None if inputs are invalid.
    """
    if H_J_mol is None or S_J_mol_K is None or T_kelvin is None or T_kelvin < 0:
        return None
    return H_J_mol - (T_kelvin * S_J_mol_K)

def get_thermo_properties(species_name, T_kelvin, thermo_data_dict):
    """
    Retrieves thermodynamic properties (H, S, G, Cp) for a species at a given temperature.

    Args:
        species_name (str): Name of the species.
        T_kelvin (float): Temperature in Kelvin.
        thermo_data_dict (dict): Dictionary of thermo data from read_nasa_polynomials.

    Returns:
        tuple: (H_J_mol, S_J_mol_K, G_J_mol, Cp_J_mol_K)
               Returns (None, None, None, None) if species not found or T is out of range.
    """
    species_data = get_nasa_polynomial(species_name, thermo_data_dict)
    if not species_data:
        # print(f"Warning: Species '{species_name}' not found in thermo data.")
        return None, None, None, None

    T_ranges = species_data.get('T_ranges', [])
    coeffs_list = species_data.get('coeffs', [])

    if not T_ranges or not coeffs_list or len(T_ranges) != len(coeffs_list):
        # print(f"Error: Invalid T_ranges or coeffs for species '{species_name}'.")
        return None, None, None, None

    active_coeffs = None
    # T_ranges should be [(low_min, low_max), (high_min, high_max)]
    # where low_max is T_mid and high_min is T_mid
    
    # First set of coeffs is usually high-T, second is low-T in the parsed data
    # from `read_nasa_polynomials` which stores [coeffs_high_T, coeffs_low_T]
    # T_ranges from `read_nasa_polynomials` is [(file_T_low, t_mid_resolved), (t_mid_resolved, file_T_high)]
    
    # Check high-T range (second entry in T_ranges, first in coeffs_list)
    if T_ranges[1][0] <= T_kelvin <= T_ranges[1][1]: # T_mid <= T <= T_high
        active_coeffs = coeffs_list[0] # High-T coefficients
    # Check low-T range (first entry in T_ranges, second in coeffs_list)
    elif T_ranges[0][0] <= T_kelvin < T_ranges[1][0]: # T_low <= T < T_mid
        active_coeffs = coeffs_list[1] # Low-T coefficients
    else:
        # print(f"Warning: Temperature {T_kelvin}K is outside the defined ranges "
        #       f"({T_ranges[0][0]}-{T_ranges[1][1]}K) for species '{species_name}'.")
        return None, None, None, None

    if not active_coeffs: # Should be covered by T range check, but as a safeguard
        # print(f"Error: No active coefficients found for {species_name} at {T_kelvin}K.")
        return None, None, None, None

    Cp_J_mol_K, H_J_mol, S_J_mol_K = calculate_cp_h_s(T_kelvin, active_coeffs)
    if Cp_J_mol_K is None: # Indicates an error in calculate_cp_h_s
        return None, None, None, None
        
    G_J_mol = calculate_gibbs_g(H_J_mol, S_J_mol_K, T_kelvin)

    return H_J_mol, S_J_mol_K, G_J_mol, Cp_J_mol_K

if __name__ == '__main__':
    # Example usage, requires a thermo_data_dict (e.g., from thermo_parser.py)
    # For demonstration, let's create a dummy thermo_data_dict.
    # This would typically be populated by read_nasa_polynomials('therm.dat')
    
    # Dummy data for H2O (coeffs are illustrative, not real)
    # High T: 1000K - 3500K
    # Low T: 200K - 1000K
    dummy_H2O_data = {
        'species_name': 'H2O',
        'T_ranges': [(200.0, 1000.0), (1000.0, 3500.0)], # [(low_min, low_max/T_mid), (high_min/T_mid, high_max)]
        'coeffs': [ # [coeffs_high_T, coeffs_low_T]
            [2.5, 1.0e-3, 0.5e-6, -0.1e-9, 0.02e-12, -45000.0, 5.0], # High-T coeffs (a1-a7)
            [3.0, 2.0e-3, 1.0e-6, -0.2e-9, 0.04e-12, -47000.0, 3.0]  # Low-T coeffs (a1-a7)
        ],
        'source_lines': ["H2O ... 1", "...", "...", "..."]
    }
    dummy_thermo_data = {'H2O': dummy_H2O_data}

    print("--- Testing thermo_calculator ---")
    
    species = "H2O"
    T_test_low = 300.0   # Should use low-T coeffs
    T_test_high = 1200.0 # Should use high-T coeffs
    T_test_mid = 1000.0  # Should use high-T coeffs (as per typical NASA format T_mid belongs to high-T range)
    T_test_out_of_range = 50.0

    print(f"\nProperties for {species} at {T_test_low}K (Low-T range):")
    H, S, G, Cp = get_thermo_properties(species, T_test_low, dummy_thermo_data)
    if H is not None:
        print(f"  H (J/mol): {H:.2f}, S (J/mol-K): {S:.2f}, G (J/mol): {G:.2f}, Cp (J/mol-K): {Cp:.2f}")

    print(f"\nProperties for {species} at {T_test_high}K (High-T range):")
    H, S, G, Cp = get_thermo_properties(species, T_test_high, dummy_thermo_data)
    if H is not None:
        print(f"  H (J/mol): {H:.2f}, S (J/mol-K): {S:.2f}, G (J/mol): {G:.2f}, Cp (J/mol-K): {Cp:.2f}")

    print(f"\nProperties for {species} at {T_test_mid}K (Mid-T point, typically uses High-T coeffs):")
    # Test with T_mid. Standard NASA format usually means T_common (T_mid) is the lower bound for high-T coeffs
    # and upper bound for low-T coeffs. My get_thermo_properties logic:
    # High-T: T_mid <= T <= T_high
    # Low-T:  T_low <= T < T_mid
    # So, 1000.0K should use high-T coeffs.
    H, S, G, Cp = get_thermo_properties(species, T_test_mid, dummy_thermo_data)
    if H is not None:
        print(f"  H (J/mol): {H:.2f}, S (J/mol-K): {S:.2f}, G (J/mol): {G:.2f}, Cp (J/mol-K): {Cp:.2f}")

    print(f"\nProperties for {species} at {T_test_out_of_range}K (Out of range):")
    H, S, G, Cp = get_thermo_properties(species, T_test_out_of_range, dummy_thermo_data)
    if H is None:
        print(f"  Correctly returned None for out-of-range temperature.")

    print(f"\nProperties for non_existent_species at {T_test_low}K:")
    H, S, G, Cp = get_thermo_properties("XYZ", T_test_low, dummy_thermo_data)
    if H is None:
        print(f"  Correctly returned None for non-existent species.")

    # Test calculate_cp_h_s directly
    print("\nDirectly testing calculate_cp_h_s with dummy low-T H2O coeffs at 300K:")
    Cp_direct, H_direct, S_direct = calculate_cp_h_s(300.0, dummy_H2O_data['coeffs'][1])
    if Cp_direct is not None:
        print(f"  Cp: {Cp_direct:.2f} J/mol-K, H: {H_direct:.2f} J/mol, S: {S_direct:.2f} J/mol-K")
        G_direct = calculate_gibbs_g(H_direct, S_direct, 300.0)
        print(f"  G (calculated from H,S): {G_direct:.2f} J/mol")

    # Test with T=0
    print("\nDirectly testing calculate_cp_h_s with T=0 (should fail):")
    Cp_direct, H_direct, S_direct = calculate_cp_h_s(0.0, dummy_H2O_data['coeffs'][1])
    if Cp_direct is None:
        print("  Correctly returned None for T=0.")

"""
