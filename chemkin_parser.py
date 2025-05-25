import re

# --- New Structures and Context ---

class ChemkinParserContext:
    """
    Context class to store parsing state, like current units.
    """
    def __init__(self):
        self.current_ea_units = 'CAL/MOLE'  # Default EA units

def _get_default_reaction_data():
    """
    Returns a new dictionary with the default structure for a reaction.
    """
    return {
        'equation_string': None, 
        'equation_string_cleaned': None, 
        'reactants': [],       
        'products': [],        
        'is_reversible': False,
        'reaction_type': 'ARRHENIUS', 
        'arrhenius_params': {'A': None, 'n': None, 'Ea': None, 'units': None},
        'troe_data': { 
            'k_inf': {}, 
            'k0': {},    
            'coeffs': [] 
        },
        'plog_data': [], 
        'rev_params': [],
        'is_duplicate': False,
        'third_body_efficiencies': {},
        'third_body_species': None, 
        'other_keywords': [], 
        'raw_lines': [],
        'warnings': []
    }

def _parse_species_list(s_list_str_original):
    """
    Parses a species string like "2H + O2 (+M)" into a list of (coeff, species) tuples 
    and identifies an explicit third body like (+M).
    Returns (species_tuples_list, third_body_explicit_str or None, species_string_part_for_reconstruction)
    """
    species_list = []
    third_body_explicit = None 
    s_list_str = s_list_str_original.strip()
    
    species_string_for_reconstruction = s_list_str 

    explicit_tb_regex = r'\s*\(\s*\+\s*([A-Za-z0-9_]+)\s*\)\s*'
    tb_find = re.search(explicit_tb_regex, s_list_str)
    
    third_body_name_inside_parens = None

    if tb_find:
        third_body_name_inside_parens = tb_find.group(1).strip()
        third_body_explicit = f"(+{third_body_name_inside_parens})"
        species_string_for_reconstruction = s_list_str.replace(tb_find.group(0), " ").strip()

    raw_parts = [p.strip() for p in species_string_for_reconstruction.split('+') if p.strip()]

    for part_idx, part in enumerate(raw_parts):
        if not part: continue

        # Regex to capture coefficient (optional, defaults to 1) and species name.
        coeff_match = re.match(r'(\d*)\s*([A-Za-z0-9_()\[\]\-\+]+(?:\[\d+\])?)', part.strip())
        if coeff_match:
            coeff_str = coeff_match.group(1)
            coeff = int(coeff_str) if coeff_str else 1 
            species_name = coeff_match.group(2).strip()
        else: 
            coeff = 1
            species_name = part.strip()
        
        if third_body_name_inside_parens and species_name.upper() == third_body_name_inside_parens.upper():
            continue 
        
        if species_name: 
            species_list.append((coeff, species_name))
            
    cleaned_species_str_from_list = " + ".join([f"{c} {s}" if c > 1 else s for c, s in species_list])
            
    return species_list, third_body_explicit, cleaned_species_str_from_list


def _parse_equation_string(eq_str):
    parsed_eq_output = {
        'reactants': [], 'products': [], 'is_reversible': False, 
        'third_body_species': None, 'equation_string_cleaned': eq_str,
        'warnings': [], 'equation_string': eq_str 
    }

    op_match = re.search(r'(<=>|=>|=)', eq_str)
    if not op_match:
        parsed_eq_output['warnings'].append(f"No valid reaction operator (=, <=>, =>) found in '{eq_str}'")
        return parsed_eq_output

    op_str = op_match.group(1)
    eq_parts = eq_str.split(op_str, 1)
    parsed_eq_output['is_reversible'] = (op_str == '<=>') 
    
    reactant_str_raw = eq_parts[0].strip()
    product_str_raw = eq_parts[1].strip()
    
    reactants_list, r_third_body, r_cleaned_species_str = _parse_species_list(reactant_str_raw)
    products_list, p_third_body, p_cleaned_species_str = _parse_species_list(product_str_raw)

    parsed_eq_output['reactants'] = reactants_list
    parsed_eq_output['products'] = products_list
    
    final_third_body = None
    if r_third_body and p_third_body:
        if r_third_body == p_third_body:
            final_third_body = r_third_body
        else:
            warning_msg = f"Conflicting third bodies: {r_third_body} (reactants) vs {p_third_body} (products). Using reactant side's ({r_third_body})."
            parsed_eq_output['warnings'].append(warning_msg)
            final_third_body = r_third_body
    elif r_third_body:
        final_third_body = r_third_body
    elif p_third_body:
        final_third_body = p_third_body
    
    parsed_eq_output['third_body_species'] = final_third_body
    
    lhs_display = r_cleaned_species_str
    rhs_display = p_cleaned_species_str

    if final_third_body:
        temp_lhs = r_cleaned_species_str
        if r_third_body : 
            temp_lhs = f"{temp_lhs} {r_third_body}" if temp_lhs else r_third_body
        elif final_third_body and not p_third_body: 
            temp_lhs = f"{temp_lhs} {final_third_body}" if temp_lhs else final_third_body

        temp_rhs = p_cleaned_species_str
        if p_third_body : 
            temp_rhs = f"{temp_rhs} {p_third_body}" if temp_rhs else p_third_body
        elif final_third_body and not r_third_body:  
            temp_rhs = f"{temp_rhs} {final_third_body}" if temp_rhs else final_third_body
        
        if final_third_body and (not r_third_body or not p_third_body) and r_third_body != p_third_body :
             pass
        elif final_third_body and r_third_body == p_third_body: 
            temp_lhs = f"{r_cleaned_species_str} {r_third_body}" if r_cleaned_species_str else r_third_body
            temp_rhs = f"{p_cleaned_species_str} {p_third_body}" if p_cleaned_species_str else p_third_body
        parsed_eq_output['equation_string_cleaned'] = f"{temp_lhs} {op_str} {temp_rhs}"
    else: 
        parsed_eq_output['equation_string_cleaned'] = f"{lhs_display} {op_str} {rhs_display}"
            
    return parsed_eq_output

def parse_chemkin_mechanism(text_input):
    reactions_output = []
    context = ChemkinParserContext()
    lines = text_input.splitlines()
    num_lines = len(lines)
    current_line_idx = 0

    units_regex = re.compile(r"^\s*UNITS\s+(.+)\s*$", re.IGNORECASE)
    reaction_line_regex = re.compile(
        r"^\s*(.+?)\s+"                                                  
        r"([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)"        
        r"(?:\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+))?" 
        r"\s*(.*?)\s*$",                                                  
        re.IGNORECASE)
    plog_data_regex = re.compile(
        r"^\s*(?:/\s*)?([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s*(?:/\s*)?(?:!.*)?\s*$", re.IGNORECASE)
    troe_params_regex = re.compile(
        r"^\s*(?:/\s*)?([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)(?:\s+([\d\.\+\-Eef]+))?\s*(?:/\s*)?(?:!.*)?\s*$", re.IGNORECASE)
    low_line_regex = re.compile(
        r"^\s*LOW\s*(?:/\s*)?([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s+([\d\.\+\-Eef]+)\s*(?:/\s*)?(?:!.*)?\s*$", re.IGNORECASE)

    while current_line_idx < num_lines:
        raw_line = lines[current_line_idx]
        line_stripped = raw_line.strip()
        next_iteration_line_idx = current_line_idx + 1

        if not line_stripped or line_stripped.startswith('!'):
            current_line_idx = next_iteration_line_idx; continue

        units_match = units_regex.match(line_stripped)
        if units_match:
            units_content = units_match.group(1).strip()
            if '!' in units_content: units_content = units_content.split('!', 1)[0].strip()
            context.current_ea_units = units_content.upper().replace(" ", "")
            current_line_idx = next_iteration_line_idx; continue

        reaction_match = reaction_line_regex.match(line_stripped)
        if reaction_match:
            reaction_data = _get_default_reaction_data() 
            reaction_data['raw_lines'].append(raw_line)
            reaction_data['equation_string'] = reaction_match.group(1).strip() 

            parsed_eq_details = _parse_equation_string(reaction_data['equation_string'])
            for key, value in parsed_eq_details.items():
                if key == 'warnings' and value: 
                    reaction_data['warnings'].extend(value if isinstance(value, list) else [value])
                elif key not in ['raw_lines', 'arrhenius_params', 'troe_data', 'plog_data', 'reaction_type', 'equation_string']: 
                    reaction_data[key] = value
            
            if any('No valid reaction operator' in w for w in reaction_data.get('warnings', [])):
                 current_line_idx = next_iteration_line_idx; continue

            all_params_str_list = [reaction_match.group(i) for i in range(2, 8) if reaction_match.group(i)]
            all_params_float = []
            try:
                for p_str in all_params_str_list: all_params_float.append(float(p_str))
            except ValueError:
                reaction_data['warnings'].append(f"Failed to convert Arrhenius/k0 params: {all_params_str_list} on line {raw_line}")
                current_line_idx = next_iteration_line_idx; continue 
            
            reaction_data['arrhenius_params'] = {'A': all_params_float[0], 'n': all_params_float[1], 'Ea': all_params_float[2], 'units': context.current_ea_units}
            tentative_k0_params = {}
            if len(all_params_float) == 6:
                tentative_k0_params = {'A': all_params_float[3], 'n': all_params_float[4], 'Ea': all_params_float[5], 'units': context.current_ea_units}

            rest_of_line_full = reaction_match.group(8) if reaction_match.group(8) else ""
            rest_of_line_upper = rest_of_line_full.upper()

            if 'PLOG' in rest_of_line_upper:
                reaction_data['reaction_type'] = 'PLOG'
                plog_scan_idx = current_line_idx + 1
                while plog_scan_idx < num_lines:
                    plog_line_raw = lines[plog_scan_idx]; plog_line_stripped = plog_line_raw.strip()
                    if not plog_line_stripped or plog_line_stripped.startswith('!'):
                        reaction_data['raw_lines'].append(plog_line_raw); plog_scan_idx += 1; break
                    plog_m = plog_data_regex.match(plog_line_stripped)
                    if plog_m:
                        try:
                            reaction_data['plog_data'].append({'pressure': float(plog_m.group(1)), 'A': float(plog_m.group(2)), 'n': float(plog_m.group(3)), 'Ea': float(plog_m.group(4)), 'units': context.current_ea_units})
                            reaction_data['raw_lines'].append(plog_line_raw); plog_scan_idx += 1
                        except ValueError: reaction_data['warnings'].append(f"PLOG data conversion error: {plog_line_stripped}"); break
                    else: break
                next_iteration_line_idx = plog_scan_idx
            
            elif 'TROE' in rest_of_line_upper:
                reaction_data['reaction_type'] = 'TROE'
                reaction_data['troe_data']['k_inf'] = reaction_data['arrhenius_params'].copy()
                reaction_data['arrhenius_params'] = {} 
                if tentative_k0_params: reaction_data['troe_data']['k0'] = tentative_k0_params.copy()

                troe_keyword_m = re.search(r'TROE', rest_of_line_full, re.IGNORECASE)
                content_after_troe = rest_of_line_full[troe_keyword_m.end():].strip()
                troe_coeffs_m = troe_params_regex.match(content_after_troe)
                if troe_coeffs_m:
                    for i in range(1, 5):
                        if troe_coeffs_m.group(i): reaction_data['troe_data']['coeffs'].append(float(troe_coeffs_m.group(i)))
                
                if not reaction_data['troe_data']['coeffs'] or (content_after_troe.startswith('/') and not content_after_troe[1:].strip()):
                    troe_coeff_scan_idx = current_line_idx + 1
                    if troe_coeff_scan_idx < num_lines:
                        next_l_raw = lines[troe_coeff_scan_idx]; next_l_strip = next_l_raw.strip()
                        if not next_l_strip.startswith('!'):
                            troe_coeffs_m_next = troe_params_regex.match(next_l_strip)
                            if troe_coeffs_m_next:
                                reaction_data['troe_data']['coeffs'] = [] 
                                for i in range(1, 5):
                                    if troe_coeffs_m_next.group(i): reaction_data['troe_data']['coeffs'].append(float(troe_coeffs_m_next.group(i)))
                                if reaction_data['troe_data']['coeffs']:
                                    reaction_data['raw_lines'].append(next_l_raw); next_iteration_line_idx = troe_coeff_scan_idx + 1
                
                if not reaction_data['troe_data']['coeffs']: reaction_data['warnings'].append(f"TROE reaction no TROE coeffs: {reaction_data['equation_string']}")

                low_scan_idx = next_iteration_line_idx
                temp_low_idx = low_scan_idx
                while temp_low_idx < num_lines:
                    low_raw = lines[temp_low_idx]; low_strip = low_raw.strip()
                    if not low_strip or low_strip.startswith('!'): temp_low_idx += 1; continue
                    low_m = low_line_regex.match(low_strip)
                    if low_m:
                        reaction_data['troe_data']['k0'] = {'A': float(low_m.group(1)), 'n': float(low_m.group(2)), 'Ea': float(low_m.group(3)), 'units': context.current_ea_units}
                        for i_raw in range(low_scan_idx, temp_low_idx + 1):
                            if lines[i_raw] not in reaction_data['raw_lines']: reaction_data['raw_lines'].append(lines[i_raw])
                        next_iteration_line_idx = temp_low_idx + 1
                    break
            
            else: 
                reaction_data['reaction_type'] = 'ARRHENIUS'
                if len(all_params_float) != 3: reaction_data['warnings'].append(f"Arrhenius expected 3 params, got {len(all_params_float)}")

            # Simplified Cleanup of optional fields
            if reaction_data['reaction_type'] != 'PLOG':
                if 'plog_data' in reaction_data and not reaction_data['plog_data']: del reaction_data['plog_data']
            
            if reaction_data['reaction_type'] != 'TROE':
                # Check if troe_data contains any actual data beyond the initialized empty structures
                td = reaction_data.get('troe_data', {})
                if not td.get('k_inf', {}).get('A') and \
                   not td.get('k0', {}).get('A') and \
                   not td.get('coeffs'):
                    if 'troe_data' in reaction_data: del reaction_data['troe_data']
            else: 
                if not reaction_data.get('troe_data',{}).get('k_inf',{}).get('A'):
                    reaction_data['warnings'].append("TROE reaction missing k_inf details.")
            
            if reaction_data['reaction_type'] == 'TROE' and 'arrhenius_params' in reaction_data and not reaction_data.get('arrhenius_params',{}):
                del reaction_data['arrhenius_params']
            elif reaction_data['reaction_type'] == 'PLOG':
                ap = reaction_data.get('arrhenius_params', {})
                if ap.get('A') is None: 
                     reaction_data['warnings'].append("PLOG reaction is missing/has invalid reference Arrhenius parameters.")
            
            reactions_output.append(reaction_data)
            current_line_idx = next_iteration_line_idx; continue
        
        current_line_idx = next_iteration_line_idx

    return reactions_output

if __name__ == '__main__':
    test_mech_input_full = """
    ! Test mechanism for new parser
    UNITS KCAL/MOLE

    ! Arrhenius Reaction
    H + O2 = OH + O            1.0E10  0.5   15.0    ! Ea in kcal/mol

    ! PLOG Reaction
    CH4 + O2 = CH3 + HO2       0.0 0.0 0.0 PLOG /  ! Params for ref pressure
        1.0   1.0E13  0.0  50.0 /  ! P, A, n, Ea (kcal/mol)
        10.0  1.0E14  0.0  55.0 /
    ! Comment after PLOG

    ! TROE Reaction with embedded k0 and TROE params on same line
    H + O2 (+M) = HO2 (+M)     1.0E12 0.5 0.0   1.0E16 0.0 -1.0 TROE / 0.6 100.0 1000.0 50.0 / 

    ! TROE Reaction with k_inf, TROE params next line, then LOW line
    UNITS CAL/MOLE ! Switch units for this reaction's Ea
    N2O (+M) = N2 + O (+M)     1.0E15 0.0 55000.0 TROE / ! Ea in cal/mol for k_inf
        0.7 200.0 1200.0 /  ! TROE centering factors
    LOW / 1.0E14 0.0 20000.0 /    ! Ea in cal/mol for k0

    END
    """
    parsed_reactions = parse_chemkin_mechanism(test_mech_input_full)
    
    print(f"--- Parsed {len(parsed_reactions)} Reactions ---")
    for i, r_data in enumerate(parsed_reactions):
        print(f"\nReaction {i+1}: {r_data.get('equation_string_cleaned', r_data.get('equation_string', 'N/A'))}")
        print(f"  Type: {r_data['reaction_type']}")
        print(f"  Reversible: {r_data['is_reversible']}")
        print(f"  Reactants: {r_data['reactants']}")
        print(f"  Products: {r_data['products']}")
        if r_data.get('third_body_species'):
            print(f"  Third Body: {r_data['third_body_species']}")
        
        if r_data.get('arrhenius_params') and r_data['arrhenius_params'].get('A') is not None : 
            print(f"  Arrhenius Params: {r_data['arrhenius_params']}")
        if r_data.get('plog_data'): 
            print(f"  PLOG Data:")
            for entry in r_data['plog_data']: print(f"    {entry}")
        if r_data.get('troe_data'): 
            td = r_data['troe_data']
            if td.get('k_inf') and td['k_inf'].get('A') is not None: 
                print(f"  k_inf Params: {td['k_inf']}")
            if td.get('k0') and td['k0'].get('A') is not None: 
                print(f"  k0 Params: {td['k0']}")
            if td.get('coeffs'): 
                print(f"  TROE Coeffs: {td['coeffs']}")
        
        if r_data['warnings']: print(f"  Warnings: {r_data['warnings']}")
