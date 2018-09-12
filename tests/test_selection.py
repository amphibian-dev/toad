import unittest
import numpy as np
import pandas as pd

from toad.selection import drop_empty, drop_corr

frame = pd.read_csv('tests/test_data.csv')

class TestSelection(unittest.TestCase):
    def setUp(self):
        pass

    def test_drop_empty(self):
        df = drop_empty(frame, threshold = 0.8)
        self.assertNotIn('E', df)

    def test_drop_corr(self):
        df = drop_corr(frame, target = 'target')
        self.assertListEqual(['B', 'D', 'E', 'target'], df.columns.tolist())
