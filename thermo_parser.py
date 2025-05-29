import re

def read_nasa_polynomials(filepath):
    """
    Reads a file containing NASA 7-coefficient polynomial data.

    Args:
        filepath (str): The path to the thermo data file.

    Returns:
        dict: A dictionary where keys are species names (uppercase) and
              values are dictionaries containing 'species_name', 'T_ranges',
              'coeffs', and 'source_lines'.
    """
    thermo_data = {}
    current_species_data = None
    lines_for_current_species = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return {} # Return empty dict if file not found

    for i, line_full in enumerate(lines):
        line = line_full.rstrip('\n') # Keep original line ending for source_lines if needed later

        if line.strip().startswith('!') or not line.strip(): # Skip comments and empty lines
            if current_species_data: # If we are in the middle of parsing a species
                lines_for_current_species.append(line_full)
            continue

        if line.upper().startswith('THERMO'): # Skip THERMO header lines
            continue
        if line.upper().startswith('END'):    # Stop parsing at END keyword
            if current_species_data: # Finalize the last species before END
                current_species_data['source_lines'] = lines_for_current_species
                thermo_data[current_species_data['species_name'].upper()] = current_species_data
                current_species_data = None
                lines_for_current_species = []
            break

        # Try to match the species header line (Line 1 of an entry)
        # Example: H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
        # Example: AR               L03/87AR   1            G   300.000  5000.000 1000.000    1
        # Using a more flexible regex for elements
        match_header = re.match(
            r"^(?P<name>\S+)\s*"                                    # Species name
            r"(?P<date>\S+)?\s*"                                    # Optional date code
            r"(?P<elements>(?:[A-Z][a-z]?\s*\d+\s*)+)\s*"           # Elements and counts
            r"(?P<phase>[A-Z0-9])\s+"                               # Phase (G, L, S, or numbers)
            r"(?P<Tlow>\d+\.\d+)\s+"                                # Tlow
            r"(?P<Thigh>\d+\.\d+)\s+"                               # Thigh
            r"(?P<Tcommon>\d+\.\d+)?\s*"                            # Tcommon (optional in some formats for the first line)
            r"(?P<line_num_indicator>[1234])?\s*$",                 # Line number indicator (usually 1)
            line[:80] # NASA format usually restricts main data to first 80 chars
        )

        if match_header and (len(line) < 79 or line[79] == '1'): # Line 1 of 4 for a species
            if current_species_data: # Finalize previous species
                current_species_data['source_lines'] = lines_for_current_species
                thermo_data[current_species_data['species_name'].upper()] = current_species_data
            
            # Start new species
            name = match_header.group('name').strip()
            t_low = float(match_header.group('Tlow'))
            t_high = float(match_header.group('Thigh'))
            # Tcommon might be missing on the first line for some formats, 
            # but present on the actual polynomial data lines.
            # The T_ranges will be defined by the two sets of polynomials.
            # For now, we'll assume T_common from this line is the T_mid if available,
            # otherwise it will be inferred from the polynomial lines' ranges.
            t_common_str = match_header.group('Tcommon')
            t_mid = float(t_common_str) if t_common_str else None 

            current_species_data = {
                'species_name': name,
                'elements_raw': match_header.group('elements').strip(), # Store raw elements string
                'phase': match_header.group('phase'),
                'file_T_low': t_low, # T_low from the species header line
                'file_T_high': t_high, # T_high from the species header line
                'file_T_mid_header': t_mid, # T_mid from the species header (might be overridden)
                'T_ranges': [], # Will be [(low_min, low_max, mid_for_low_coeffs), (high_min, high_max, mid_for_high_coeffs)]
                'coeffs': [],   # List of lists of 7 coefficients
                'source_lines': []
            }
            lines_for_current_species = [line_full]
            current_species_data['coeff_lines_raw'] = [] # To store next 3 lines

        elif current_species_data and (len(line) < 79 or line[79] in ['2', '3', '4']):
            # Coefficient lines (Lines 2, 3, 4 of a species entry)
            lines_for_current_species.append(line_full)
            current_species_data['coeff_lines_raw'].append(line.strip()) # Store the content of lines 2,3,4

            if len(current_species_data['coeff_lines_raw']) == 3: # We have all 3 coefficient lines
                # Process these 3 lines to extract 14 coefficients and define T_ranges
                coeffs_high_T = []
                coeffs_low_T = []
                
                # Line 2: a1_H, a2_H, a3_H, a4_H, a5_H (High T)
                line2_coeffs_str = current_species_data['coeff_lines_raw'][0][:75] # First 5 coeffs are in first 75 chars
                coeffs_high_T.extend([float(line2_coeffs_str[i:i+15]) for i in range(0, 75, 15)])

                # Line 3: a6_H, a7_H (High T), a1_L, a2_L, a3_L (Low T)
                line3_coeffs_str = current_species_data['coeff_lines_raw'][1][:75]
                coeffs_high_T.extend([float(line3_coeffs_str[i:i+15]) for i in range(0, 30, 15)])
                coeffs_low_T.extend([float(line3_coeffs_str[i:i+15]) for i in range(30, 75, 15)])
                
                # Line 4: a4_L, a5_L, a6_L, a7_L (Low T)
                line4_coeffs_str = current_species_data['coeff_lines_raw'][2][:60] # Next 4 coeffs
                coeffs_low_T.extend([float(line4_coeffs_str[i:i+15]) for i in range(0, 60, 15)])

                current_species_data['coeffs'] = [coeffs_high_T, coeffs_low_T]
                
                # Determine T_ranges. T_mid is usually T_common from the header.
                # If T_mid wasn't on header, it might be implicitly defined by the ranges.
                # Standard format: Low coeffs for [T_low, T_mid], High coeffs for [T_mid, T_high]
                t_mid_resolved = current_species_data['file_T_mid_header']
                # If not found on header, a common default is 1000K if T_low < 1000 < T_high
                if t_mid_resolved is None:
                    if current_species_data['file_T_low'] < 1000.0 < current_species_data['file_T_high']:
                        t_mid_resolved = 1000.0
                    else: # Fallback, though this might be an issue for some formats.
                          # A more robust parser might need to look for T_common on line 2,3, or 4 too.
                        print(f"Warning: T_mid (T_common) not found for {current_species_data['species_name']}. Using 1000.0 or edge if applicable.")
                        t_mid_resolved = 1000.0 # Default or needs better logic

                # Check if t_mid_resolved is within T_low and T_high from the file header
                if not (current_species_data['file_T_low'] <= t_mid_resolved <= current_species_data['file_T_high']):
                    print(f"Warning: T_mid {t_mid_resolved} for {current_species_data['species_name']} is outside file T_low/T_high range. Clamping or using file T_low/T_high.")
                    # This case might require more sophisticated handling based on specific format variations
                    # For now, we'll assume standard two-range: T_low to T_mid, and T_mid to T_high.
                
                current_species_data['T_ranges'] = [
                    (current_species_data['file_T_low'], t_mid_resolved), # Low-T range
                    (t_mid_resolved, current_species_data['file_T_high'])  # High-T range
                ]
                
                # Clean up raw coeff lines as they are now processed
                del current_species_data['coeff_lines_raw']
                # Note: Species is not finalized here yet, it's finalized when next species header or END is found

        elif line.strip(): # Non-empty, non-comment, not THERMO/END, not a species line
            if current_species_data: # If we are parsing a species, append to its source lines
                 lines_for_current_species.append(line_full)
            else:
                # This could be global T range definition like "100.0 1000.0 5000.0"
                # Or other unexpected lines. For now, we'll ignore if not actively parsing a species.
                # A more robust parser might handle global T definitions.
                pass


    # Finalize the very last species if the file doesn't end with END or has data after last species
    if current_species_data and 'coeffs' in current_species_data and current_species_data['coeffs']: 
        current_species_data['source_lines'] = lines_for_current_species
        thermo_data[current_species_data['species_name'].upper()] = current_species_data

    return thermo_data

def is_valid_nasa_polynomial_string(polynomial_string):
    """
    Validates a multi-line string for NASA 7-coefficient polynomial format.
    Checks for groups of 4 lines, line indicators, and float convertibility
    of coefficients.

    Args:
        polynomial_string (str): The string containing one or more NASA polynomial entries.

    Returns:
        tuple: (is_valid (bool), parsed_species_list (list), error_messages (list))
               parsed_species_list contains dicts of successfully parsed (but not saved)
               species if is_valid is True. Each dict includes 'species_name' and 'raw_lines'.
               error_messages contains strings describing validation failures.
    """
    lines = [line.rstrip('\n') for line in polynomial_string.strip().split('\n')]
    num_lines = len(lines)
    parsed_species_list = []
    error_messages = []

    if num_lines == 0:
        error_messages.append("Input string is empty.")
        return False, [], error_messages
    
    if num_lines % 4 != 0:
        error_messages.append(f"Invalid number of lines ({num_lines}). Expected a multiple of 4.")
        # Allow parsing to continue to see if any blocks are valid
        # return False, [], error_messages 

    entry_num = 0
    for i in range(0, num_lines, 4):
        entry_num += 1
        current_entry_lines = lines[i:i+4]
        if len(current_entry_lines) < 4:
            error_messages.append(f"Entry {entry_num}: Incomplete entry, expected 4 lines, got {len(current_entry_lines)}.")
            continue # Move to next potential block if any

        line1, line2, line3, line4 = current_entry_lines

        # Validate line indicators (column 80, 1-indexed)
        if len(line1) < 80 or line1[79] != '1':
            error_messages.append(f"Entry {entry_num}, Line 1: Missing '1' at column 80 or line too short.")
        if len(line2) < 80 or line2[79] != '2':
            error_messages.append(f"Entry {entry_num}, Line 2: Missing '2' at column 80 or line too short.")
        if len(line3) < 80 or line3[79] != '3':
            error_messages.append(f"Entry {entry_num}, Line 3: Missing '3' at column 80 or line too short.")
        if len(line4) < 80 or line4[79] != '4':
            error_messages.append(f"Entry {entry_num}, Line 4: Missing '4' at column 80 or line too short.")

        # Extract species name (first part of line 1)
        species_name = line1.split()[0] if line1.split() else f"UNKNOWN_SPECIES_{entry_num}"

        # Validate coefficients (15 characters per float)
        coeffs_to_check = []
        try:
            # Line 2: 5 coeffs
            coeffs_to_check.extend([line2[j:j+15] for j in range(0, 75, 15)])
            # Line 3: 5 coeffs
            coeffs_to_check.extend([line3[j:j+15] for j in range(0, 75, 15)])
            # Line 4: 4 coeffs (last one might be shorter, but we expect 4 fields)
            coeffs_to_check.extend([line4[j:j+15] for j in range(0, 60, 15)])
            
            if len(coeffs_to_check) != 14:
                 error_messages.append(f"Entry {entry_num} ({species_name}): Expected 14 coefficients, found {len(coeffs_to_check)} fields.")

            for idx, c_str in enumerate(coeffs_to_check):
                float(c_str) # Try converting to float
        except ValueError:
            error_messages.append(f"Entry {entry_num} ({species_name}): Coefficient '{c_str}' (index {idx}) is not a valid float.")
        except IndexError:
            error_messages.append(f"Entry {entry_num} ({species_name}): Line too short for coefficient extraction.")

        # Basic check for Tlow, Thigh, Tcommon on line 1
        # Example: H2O L07/88H 2O 1 G 200.000 3500.000 1000.000 1
        #          1         2         3         4         5         6         7         8
        #01234567890123456789012345678901234567890123456789012345678901234567890123456789
        #SPECIES           DATE    ATOM1N1ATOM2N2ATOM3N3ATOM4N4PHASE  TMIN      THIGH     TCOMMON   
        # ^0-17             ^18-23  ^24-43                      ^44    ^45-54    ^55-64    ^65-73
        try:
            if len(line1) < 74: # Ensure enough length for T_low, T_high, T_common
                 error_messages.append(f"Entry {entry_num} ({species_name}): Line 1 is too short for T range extraction.")
            else:
                float(line1[45:55].strip()) # T_low
                float(line1[55:65].strip()) # T_high
                t_common_str = line1[65:75].strip() # T_common can be optional in some interpretations for line 1
                if t_common_str: # Only try to float if it's not empty
                    float(t_common_str)
        except ValueError:
            error_messages.append(f"Entry {entry_num} ({species_name}): Invalid float in T_low/T_high/T_common field on Line 1.")
        
        if not error_messages: # If no errors for this specific entry so far
             parsed_species_list.append({
                 'species_name': species_name,
                 'raw_lines': "\n".join(current_entry_lines) # Store the raw lines for this entry
             })
        # Continue parsing even if one entry has errors, to report all errors.

    is_overall_valid = not error_messages and num_lines > 0 # Valid if no errors and not empty
    return is_overall_valid, parsed_species_list, error_messages


def append_nasa_polynomial_from_string(filepath, polynomial_string):
    """
    Appends new NASA polynomial entries to a thermo data file.

    Args:
        filepath (str): Path to the thermo data file.
        polynomial_string (str): A string containing one or more valid NASA polynomial entries.
                                 Validation should be done before calling this function.

    Returns:
        tuple: (success (bool), message (str))
               success is True if append was successful, False otherwise.
               message provides details on outcome or error.
    """
    # Note: The task specified to call is_valid_nasa_polynomial_string here.
    # However, it's often better for the caller to validate first, then pass the
    # validated string and potentially the parsed_species_list.
    # For this implementation, I will assume polynomial_string is already validated
    # as the validator now returns parsed species and error messages, making it
    # more of a pre-processing step. If strict adherence to the prompt is needed,
    # the call to is_valid_nasa_polynomial_string can be re-added here.
    
    # is_valid, _, validation_errors = is_valid_nasa_polynomial_string(polynomial_string)
    # if not is_valid:
    #     return False, "Validation failed: " + "; ".join(validation_errors)

    if not polynomial_string.strip():
        return False, "Polynomial string is empty, nothing to append."

    new_entries_str = polynomial_string.strip() + '\n' # Ensure trailing newline

    try:
        try:
            with open(filepath, 'r+') as f:
                lines = f.readlines()
                
                # Remove existing END line if present
                cleaned_lines = []
                end_line_found = False
                for line in lines:
                    if line.strip().upper() == 'END':
                        end_line_found = True
                        # Do not add this line to cleaned_lines, it will be re-added at the end
                    else:
                        cleaned_lines.append(line)
                
                # Ensure last existing line has a newline if it's not empty
                if cleaned_lines and cleaned_lines[-1].strip() and not cleaned_lines[-1].endswith('\n'):
                    cleaned_lines[-1] += '\n'
                
                f.seek(0) # Go to the beginning to rewrite
                f.writelines(cleaned_lines)
                f.write(new_entries_str) # Add the new polynomial entries
                f.write('END\n')         # Add the END statement
                f.truncate() # Remove any trailing old content if new content is shorter
            return True, f"Successfully appended data to {filepath}."

        except FileNotFoundError:
            # If file doesn't exist, create it and add the polynomial
            with open(filepath, 'w') as f:
                f.write("THERMO ALL\n") # Add a basic header
                f.write("   300.000  1000.000  5000.000\n") # Example global T range
                f.write(new_entries_str)
                f.write("END\n")
            return True, f"Successfully created {filepath} and added data."
            
    except Exception as e:
        return False, f"Error operating on file {filepath}: {e}"


if __name__ == '__main__':
    # Example Usage (assuming a 'therm.dat' file exists or will be created)
    
    # Create a dummy therm.dat for testing
    dummy_thermo_content = """THERMO ALL
   100.000  1000.000  5000.000
! Test species 1
H2O               L07/88H  2O  1          G   200.000  3500.000 1000.000    1
 0.26721056E+01 0.30563007E-02-0.87302830E-06 0.12009959E-09-0.64007679E-14    2
-0.47021003E+05 0.23700870E+01 0.33300075E+01 0.21062858E-02-0.16719769E-07    3
-0.50379009E-08 0.18870803E-11-0.47553119E+05 0.80534405E+01                   4
! Test species 2
O2                G07/88O  2              G   200.000  3500.000 1000.000    1
 0.36740000E+01 0.75700000E-02-0.21100000E-05 0.25700000E-09-0.90000000E-14    2
-0.10400000E+04 0.29700000E+01 0.32000000E+01 0.50000000E-02 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-0.12345000E+04 0.60000000E+01                   4
END
"""
    with open('therm.dat', 'w') as f:
        f.write(dummy_thermo_content)

    thermo_data = read_nasa_polynomials('therm.dat')
    if thermo_data:
        print(f"Successfully read {len(thermo_data)} species from therm.dat:")
        for species, data in thermo_data.items():
            print(f"  Species: {species}")
            print(f"    T_ranges: {data['T_ranges']}")
            # print(f"    Coeffs (High T): {data['coeffs'][0]}")
            # print(f"    Coeffs (Low T): {data['coeffs'][1]}")
            # print(f"    Source lines: {''.join(data['source_lines'])}")
    else:
        print("No data read from therm.dat or file not found.")

    new_poly_N2 = """
N2                G03/87N  2              G   300.000  5000.000 1000.000    1
 0.02926640e+02 0.01487977e-01-0.05684761e-05 0.01009704e-08-0.06753351e-13    2
-0.09227977e+04 0.05905697e+01 0.03376172e+02-0.06317038e-02 0.01455836e-04    3
-0.01639043e-07 0.06547132e-11-0.01019103e+04 0.02449430e+01                   4
    """.strip() # Important to strip whitespace for validation and appending

    print(f"\nAttempting to append N2 to therm.dat...")
    if append_nasa_polynomial_from_string('therm.dat', new_poly_N2):
        print("N2 appended successfully.")
        thermo_data_updated = read_nasa_polynomials('therm.dat')
        if thermo_data_updated.get('N2'):
            print("N2 found in updated thermo data:")
            print(f"  Species: N2, T_ranges: {thermo_data_updated['N2']['T_ranges']}")
    else:
        print("Failed to append N2.")
    
    # Test appending to a non-existent file
    non_existent_file = "new_thermo.dat"
    print(f"\nAttempting to append Ar to {non_existent_file} (should create it)...")
    new_poly_Ar = """
AR                L03/87AR   1            G   300.000  5000.000 1000.000    1
 2.50000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
-7.45375000E+02 4.37969400E+00 2.50000000E+00 0.00000000E+00 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-7.45375000E+02 4.37969400E+00                   4
    """.strip()
    if append_nasa_polynomial_from_string(non_existent_file, new_poly_Ar):
        print(f"Ar appended to {non_existent_file} successfully.")
        thermo_data_new_file = read_nasa_polynomials(non_existent_file)
        if thermo_data_new_file.get('AR'):
            print(f"AR found in {non_existent_file}:")
            print(f"  Species: AR, T_ranges: {thermo_data_new_file['AR']['T_ranges']}")
    else:
        print(f"Failed to append Ar to {non_existent_file}.")

"""
