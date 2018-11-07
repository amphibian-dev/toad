import unittest
import numpy as np
import pandas as pd

from toad import merge, ChiMerge, DTMerge, QuantileMerge, StepMerge, KMeansMerge

np.random.seed(1)
feature = np.random.rand(500)
target = np.random.randint(2, size = 500)
A = np.random.randint(100, size = 500)
B = np.random.randint(3, size = 500)

df = pd.DataFrame({
    'feature': feature,
    'target': target,
    'A': A,
})



class TestMerge(unittest.TestCase):
    def setUp(self):
        pass

    def test_chimerge(self):
        splits = ChiMerge(feature, target, n_bins = 10)
        self.assertEqual(len(splits), 9)

    def test_chimerge_bins_not_enough(self):
        splits = ChiMerge(B, target, n_bins = 10)
        self.assertEqual(len(splits), 2)

    def test_chimerge_bins_with_min_samples(self):
        splits = ChiMerge(feature, target, min_samples = 0.02)
        self.assertEqual(len(splits), 10)

    def test_dtmerge(self):
        splits = DTMerge(feature, target, n_bins = 10)
        self.assertEqual(len(splits), 9)

    def test_quantilemerge(self):
        splits = QuantileMerge(feature, n_bins = 10)
        self.assertEqual(len(splits), 9)

    def test_stepmerge(self):
        splits = StepMerge(feature, n_bins = 10)
        self.assertEqual(len(splits), 9)

    def test_kmeansmerge(self):
        splits = KMeansMerge(feature, n_bins = 10)
        self.assertEqual(len(splits), 9)

    def test_merge(self):
        res = merge(feature, target = target, method = 'chi')
        self.assertEqual(len(np.unique(res)), 20)

    def test_merge_frame(self):
        res = merge(df, target = 'target', method = 'chi')
        self.assertEqual(len(np.unique(res['A'])), 20)
