import unittest
import numpy as np
import pandas as pd

from toad.selection import drop_empty, drop_corr, drop_iv, drop_vif, select, stepwise

from tests.generate_data import frame

class TestSelection(unittest.TestCase):
    def setUp(self):
        pass

    def test_drop_empty(self):
        df = drop_empty(frame, threshold = 0.8)
        self.assertNotIn('E', df)

    def test_drop_corr(self):
        df = drop_corr(frame, target = 'target')
        self.assertListEqual(['B', 'D', 'E', 'target'], df.columns.tolist())

    def test_drop_iv(self):
        df = drop_iv(frame, target = 'target', threshold = 0.42)
        self.assertNotIn('C', df)

    def test_select(self):
        df = select(frame, target = 'target', empty = 0.8, iv = 0.42, corr = 0.7)
        self.assertListEqual(['B', 'D', 'target'], df.columns.tolist())

    def test_select_exclude(self):
        df = select(frame, target = 'target', empty = 0.8, iv = 0.42, corr = 0.7, exclude = ['A'])
        self.assertListEqual(['A', 'B', 'D', 'target'], df.columns.tolist())

    def test_stepwise(self):
        df = stepwise(frame.fillna(-1), target = 'target')
        self.assertListEqual(['E', 'C', 'F', 'target'], df.columns.tolist())

    def test_drop_vif(self):
        df = drop_vif(frame.fillna(-1), exclude = 'target')
        self.assertListEqual(['C', 'E', 'F', 'target'], df.columns.tolist())
