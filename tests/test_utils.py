import unittest
import numpy as np

from toad.utils import clip

np.random.seed(1)
feature = np.random.rand(500)
target = np.random.randint(2, size = 500)

class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_clip(self):
        res1 = clip(feature, quantile = (.05, .95))
        res2 = clip(feature, quantile = 0.05)
        self.assertIsNone(np.testing.assert_array_equal(res1, res2))
