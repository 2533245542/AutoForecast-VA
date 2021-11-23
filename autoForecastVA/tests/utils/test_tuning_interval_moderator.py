import unittest
import numpy as np

from autoForecastVA.src.utils.tuning_interval_moderator import OperationIntervalModerator


class TestTuningIntervalModerator(unittest.TestCase):
    def test_toy(self):
        days = ['c', 'b', 'a', 'i', 'f', 'e', 'j', 'd', 'g', 'h']

        tuning_interval = 3  # split to 3,3,3,1
        tuningIntervalModerator = OperationIntervalModerator(days=days, operation_interval=tuning_interval)

        self.assertEqual(str(tuningIntervalModerator.list_of_operation_idle_day_index_list), '[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]')

        self.assertDictEqual(tuningIntervalModerator.operation_day_to_idle_day_list_dictionary, {'a': ['b', 'c'], 'd': ['e', 'f'],
                                                                                  'g': ['h', 'i'], 'j': []})
        self.assertDictEqual(tuningIntervalModerator.day_to_operation_day_dictionary, {'a': 'a', 'b': 'a', 'c': 'a',
                                                                        'd': 'd', 'e': 'd', 'f': 'd',
                                                                        'g': 'g', 'h': 'g', 'i': 'g',
                                                                        'j': 'j'})

        self.assertEqual(tuningIntervalModerator.day_is_operation_day('a'), True)
        self.assertEqual(tuningIntervalModerator.day_is_operation_day('c'), False)
        self.assertEqual(tuningIntervalModerator.day_is_operation_day('d'), True)
        self.assertEqual(tuningIntervalModerator.get_operation_day_of_a_day('a'), 'a')
        self.assertEqual(tuningIntervalModerator.get_operation_day_of_a_day('c'), 'a')
        self.assertEqual(tuningIntervalModerator.get_operation_day_of_a_day('d'), 'd')
        self.assertEqual(tuningIntervalModerator.day_is_operation_day('j'), True)
        self.assertEqual(tuningIntervalModerator.get_operation_day_of_a_day('j'), 'j')





