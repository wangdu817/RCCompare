import unittest
from chemkin_parser import parse_chemkin_mechanism # Updated import

class TestChemkinParserDetailed(unittest.TestCase):

    def test_units_parsing(self):
        test_input_base = "H+O2=OH+O  1.0E10  0.0  1000.0"
        
        units_to_test = {
            'CAL/MOLE': 1000.0,
            'KCAL/MOLE': 1.0, # 1 kcal = 1000 cal
            'JOULES/MOLE': 4184.0, # ~1000 cal
            'KJOULES/MOLE': 4.184, # ~1000 cal
            'KELVINS': 502.7 , # 502.7 K * 1.9872 cal/mol-K = ~1000 cal
            # 'EVOLTS': 0.04336 # 0.04336 eV * 23060.5 cal/mol-eV = ~1000 cal
        }

        for unit_str_in_file, ea_val_in_file in units_to_test.items():
            chemkin_text = f"UNITS {unit_str_in_file}\n{test_input_base.replace('1000.0', str(ea_val_in_file))}"
            parsed = parse_chemkin_mechanism(chemkin_text)
            self.assertEqual(len(parsed), 1, f"Failed for unit: {unit_str_in_file}")
            reaction = parsed[0]
            self.assertEqual(reaction['arrhenius_params']['units'], unit_str_in_file.upper().replace(" ", ""), f"Unit mismatch for {unit_str_in_file}")
            self.assertEqual(reaction['arrhenius_params']['Ea'], ea_val_in_file, f"Ea value mismatch for {unit_str_in_file}")

        # Test that units persist
        chemkin_text_persistent = f"UNITS KCAL/MOLE\n{test_input_base}\n{test_input_base.replace('H+O2','H2+O')}"
        parsed_persistent = parse_chemkin_mechanism(chemkin_text_persistent)
        self.assertEqual(len(parsed_persistent), 2)
        self.assertEqual(parsed_persistent[0]['arrhenius_params']['units'], 'KCAL/MOLE')
        self.assertEqual(parsed_persistent[1]['arrhenius_params']['units'], 'KCAL/MOLE')


    def test_arrhenius_reaction_detailed(self):
        test_input = "2H + O2 (+M) <=> H + OH + M  1.0E10  0.5  1000.0"
        parsed = parse_chemkin_mechanism(f"UNITS CAL/MOLE\n{test_input}") # Ensure default units
        self.assertEqual(len(parsed), 1)
        r = parsed[0]
        
        self.assertEqual(r['equation_string'], "2H + O2 (+M) <=> H + OH + M")
        self.assertEqual(r['equation_string_cleaned'], "2 H + O2 (+M) <=> H + OH + M (+M)") # Based on current _parse_equation_string
        self.assertListEqual(r['reactants'], [(2, 'H'), (1, 'O2')])
        self.assertListEqual(r['products'], [(1, 'H'), (1, 'OH'), (1,'M')]) # M is kept if not matching specific (+X)
        self.assertTrue(r['is_reversible'])
        self.assertEqual(r['third_body_species'], '(+M)')
        self.assertEqual(r['reaction_type'], 'ARRHENIUS')
        self.assertDictEqual(r['arrhenius_params'], {'A': 1.0E10, 'n': 0.5, 'Ea': 1000.0, 'units': 'CAL/MOLE'})
        
        # Ensure other complex fields are default/empty for simple Arrhenius
        self.assertFalse(r.get('plog_data')) # Should be deleted by cleanup
        self.assertFalse(r.get('troe_data')) # Should be deleted by cleanup
        self.assertEqual(r['raw_lines'], [test_input])

    def test_plog_reaction_detailed(self):
        test_input = """
        UNITS KCAL/MOLE
        CH4 + O2 = CH3 + HO2  1.0 0.0 5.0 PLOG /  ! Ref Ea in KCAL
            1.0   1.0E13  0.0  50.0 /  ! PLOG Ea in KCAL
            10.0  1.0E14  0.0  55.0 /
        """
        parsed = parse_chemkin_mechanism(test_input)
        self.assertEqual(len(parsed), 1)
        r = parsed[0]

        self.assertEqual(r['reaction_type'], 'PLOG')
        self.assertEqual(r['equation_string'], 'CH4 + O2 = CH3 + HO2')
        self.assertFalse(r['is_reversible'])
        self.assertListEqual(r['reactants'], [(1, 'CH4'), (1, 'O2')])
        self.assertListEqual(r['products'], [(1, 'CH3'), (1, 'HO2')])
        self.assertDictEqual(r['arrhenius_params'], {'A': 1.0, 'n': 0.0, 'Ea': 5.0, 'units': 'KCAL/MOLE'})
        
        expected_plog_data = [
            {'pressure': 1.0, 'A': 1.0E13, 'n': 0.0, 'Ea': 50.0, 'units': 'KCAL/MOLE'},
            {'pressure': 10.0, 'A': 1.0E14, 'n': 0.0, 'Ea': 55.0, 'units': 'KCAL/MOLE'}
        ]
        self.assertListEqual(r['plog_data'], expected_plog_data)
        self.assertFalse(r.get('troe_data')) # Should be deleted
        self.assertEqual(len(r['raw_lines']), 3) # Main line + 2 PLOG lines


    def test_troe_reaction_detailed(self):
        # Test 1: Embedded k0, TROE coeffs on same line
        test_input_embedded_k0 = """
        UNITS CAL/MOLE
        H + O2 (+M) = HO2 (+M)  1.0E12 0.5 100.0   1.0E16 0.0 -500.0 TROE / 0.6 150.0 1000.0 500.0 / 
        """
        parsed = parse_chemkin_mechanism(test_input_embedded_k0)
        self.assertEqual(len(parsed), 1)
        r1 = parsed[0]
        self.assertEqual(r1['reaction_type'], 'TROE')
        self.assertEqual(r1['equation_string'], 'H + O2 (+M) = HO2 (+M)')
        self.assertEqual(r1['third_body_species'], '(+M)')
        self.assertFalse(r1.get('arrhenius_params')) # Should be deleted for TROE
        self.assertIsNotNone(r1.get('troe_data'))
        self.assertDictEqual(r1['troe_data']['k_inf'], {'A': 1.0E12, 'n': 0.5, 'Ea': 100.0, 'units': 'CAL/MOLE'})
        self.assertDictEqual(r1['troe_data']['k0'], {'A': 1.0E16, 'n': 0.0, 'Ea': -500.0, 'units': 'CAL/MOLE'})
        self.assertListEqual(r1['troe_data']['coeffs'], [0.6, 150.0, 1000.0, 500.0])
        self.assertFalse(r1.get('plog_data'))

        # Test 2: k_inf only on main line, TROE coeffs on next, LOW line after comment
        test_input_low_keyword = """
        UNITS KCAL/MOLE
        N2O (+AR) = N2 + O (+AR)  1.0E15 0.0 55.0 TROE / 
            0.7 200.0 1200.0 /  ! TROE coeffs
        ! Comment
        LOW / 1.0E14 0.0 20.0 /    ! k0 params
        """
        parsed = parse_chemkin_mechanism(test_input_low_keyword)
        self.assertEqual(len(parsed), 1)
        r2 = parsed[0]
        self.assertEqual(r2['reaction_type'], 'TROE')
        self.assertEqual(r2['equation_string'], 'N2O (+AR) = N2 + O (+AR)')
        self.assertEqual(r2['third_body_species'], '(+AR)')
        self.assertFalse(r2.get('arrhenius_params'))
        self.assertIsNotNone(r2.get('troe_data'))
        self.assertDictEqual(r2['troe_data']['k_inf'], {'A': 1.0E15, 'n': 0.0, 'Ea': 55.0, 'units': 'KCAL/MOLE'})
        self.assertDictEqual(r2['troe_data']['k0'], {'A': 1.0E14, 'n': 0.0, 'Ea': 20.0, 'units': 'KCAL/MOLE'})
        self.assertListEqual(r2['troe_data']['coeffs'], [0.7, 200.0, 1200.0])
        self.assertEqual(len(r2['raw_lines']), 4) # Main, TROE coeffs, comment, LOW line

        # Test 3: Embedded k0 overridden by LOW
        test_input_override = """
        UNITS JOULES/MOLE
        X+Y(+M)=Z(+M) 1e10 0 1000  1e12 0 500 TROE / 0.5 100 200 /
        LOW / 1e18 0.5 800 /
        """
        parsed = parse_chemkin_mechanism(test_input_override)
        self.assertEqual(len(parsed),1)
        r3 = parsed[0]
        self.assertEqual(r3['reaction_type'], 'TROE')
        self.assertDictEqual(r3['troe_data']['k_inf'], {'A':1e10, 'n':0, 'Ea':1000, 'units':'JOULES/MOLE'})
        self.assertDictEqual(r3['troe_data']['k0'], {'A':1e18, 'n':0.5, 'Ea':800, 'units':'JOULES/MOLE'}) # Overridden
        self.assertListEqual(r3['troe_data']['coeffs'], [0.5, 100, 200])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
