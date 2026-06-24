import unittest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import chemkin_pyqt_gui_fluent as gui


class _TextValue:
    def __init__(self, value):
        self._value = str(value)

    def text(self):
        return self._value


class _CheckedValue:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


def make_window(reaction_count, group_count=1, intra_linestyle=False):
    window = gui.MainWindow.__new__(gui.MainWindow)
    window.parsed_reactions = [{} for _ in range(reaction_count)]
    window.group_count = _TextValue(group_count)
    window.intra_linestyle_cb = _CheckedValue(intra_linestyle)
    window._get_plot_pressure_settings = lambda: ([1.0], False, 1.0)
    return window


class TestFluentStyleMapping(unittest.TestCase):
    def test_single_group_assigns_no_duplicate_styles_for_many_reactions(self):
        window = make_window(25, group_count=1)

        styles = [window._get_reaction_style(i, 0) for i in range(25)]

        self.assertEqual(len(styles), len(set(styles)))

    def test_table_linestyle_helper_uses_pressure_index_like_plotting(self):
        window = make_window(3, group_count=1)
        window._get_plot_pressure_settings = lambda: ([0.1, 1.0, 10.0], True, 0.1)

        _, _, plot_ls_idx = window._get_reaction_style(1, 2)
        table_ls_idx = window._get_reaction_linestyle_idx(1, 2)

        self.assertEqual(table_ls_idx, plot_ls_idx)

    def test_multi_group_checked_keeps_group_color_and_varies_linestyle(self):
        window = make_window(6, group_count=2, intra_linestyle=True)

        group0 = [window._get_reaction_style(i, 0) for i in range(3)]
        group1 = [window._get_reaction_style(i, 0) for i in range(3, 6)]

        self.assertEqual({style[0] for style in group0}, {group0[0][0]})
        self.assertEqual({style[0] for style in group1}, {group1[0][0]})
        self.assertNotEqual(group0[0][0], group1[0][0])
        self.assertGreater(len({style[2] for style in group0}), 1)
        self.assertGreater(len({style[2] for style in group1}), 1)

    def test_multi_group_unchecked_keeps_group_linestyle_and_varies_color(self):
        window = make_window(6, group_count=2, intra_linestyle=False)

        group0 = [window._get_reaction_style(i, 0) for i in range(3)]
        group1 = [window._get_reaction_style(i, 0) for i in range(3, 6)]

        self.assertEqual({style[2] for style in group0}, {group0[0][2]})
        self.assertEqual({style[2] for style in group1}, {group1[0][2]})
        self.assertNotEqual(group0[0][2], group1[0][2])
        self.assertGreater(len({style[0] for style in group0}), 1)
        self.assertGreater(len({style[0] for style in group1}), 1)

    def test_remainder_reactions_do_not_create_extra_groups(self):
        window = make_window(5, group_count=2, intra_linestyle=True)

        group_indices = [window._get_reaction_group_position(i)[0] for i in range(5)]

        self.assertEqual(sorted(set(group_indices)), [0, 1])


class TestFluentStylePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_table_metadata_matches_plotted_curve_pen(self):
        window = gui.MainWindow()
        window.group_count.setText("1")
        window.intra_linestyle_cb.setChecked(False)
        window.table_card._setup_columns([1000.0])
        reactions = [
            {
                "equation_string_cleaned": f"R{i}",
                "reaction_type": "ARRHENIUS",
                "arrhenius_params": {
                    "A": 1.0e10 + i,
                    "n": 0.0,
                    "Ea": 0.0,
                    "units": "CAL/MOLE",
                },
            }
            for i in range(4)
        ]

        window._update_rate_constant_table(reactions)
        window._update_plot(reactions)

        plotted_items = window.plot_card.plot_widget.listDataItems()
        self.assertEqual(len(plotted_items), len(reactions))
        for ri, reaction in enumerate(reactions):
            expected_color, expected_style, expected_style_index = (
                window._get_reaction_style(ri, 0)
            )
            row_style = reaction["table_rows"][0]
            self.assertEqual(row_style["color"], expected_color)
            self.assertEqual(row_style["linestyle_index"], expected_style_index)

            pen = plotted_items[ri].opts["pen"]
            self.assertEqual(pen.color().name().lower(), expected_color.lower())
            self.assertEqual(pen.style(), expected_style)


if __name__ == "__main__":
    unittest.main()
