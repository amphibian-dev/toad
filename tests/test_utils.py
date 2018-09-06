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
        res = clip(feature, quantile = (None, .98))
        self.assertEqual(len(res), 500)
