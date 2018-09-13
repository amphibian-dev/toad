import unittest
import numpy as np
import pandas as pd

from toad.transform import woe_transform

np.random.seed(1)

feature = np.random.randint(10, size = 500)
target = np.random.randint(2, size = 500)


class TestTransform(unittest.TestCase):
    def setUp(self):
        pass

    def test_woe_transform(self):
        f = woe_transform(feature, target)
        self.assertEqual(len(np.unique(f)), 10)
