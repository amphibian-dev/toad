import unittest
import numpy as np
import pyximport
pyximport.install()

from toad import ChiMerge

np.random.seed(1)
feature = np.random.rand(500)
target = np.random.randint(2, size = 500)

class TestMerge(unittest.TestCase):
    def setUp(self):
        pass

    def test_chimerge(self):
        splits = ChiMerge(feature, target, n_bins = 10)
        self.assertEqual(len(splits), 9)

    def test_get_unknow_value(self):
        # unknow = self.config.get('unknow')
        # self.assertEqual(unknow, None)
        pass
