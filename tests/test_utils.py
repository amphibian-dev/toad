import unittest
import numpy as np
import pandas as pd
import pyximport
pyximport.install()

from detector import IV, WOE, gini
from detector.utils import _IV

np.random.seed(1)

feature = np.random.rand(500)
target = np.random.randint(2, size = 500)
df = pd.DataFrame({
    'feature': feature,
    'target': target,
})

class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_woe(self):
        value = WOE(0.2, 0.3)
        self.assertEqual(value, -0.4054651081081643)

    def test_iv_priv(self):
        value = _IV(df['feature'], df['target'])
        self.assertEqual(value, 0.010385942643745353)

    def test_iv(self):
        value = IV(df['feature'], df['target'])
        self.assertEqual(value, 1.3752313490741406)

    def test_gini(self):
        value = gini(df['target'])
        self.assertEqual(value, 0.499352)

    def test_get_unknow_value(self):
        # unknow = self.config.get('unknow')
        # self.assertEqual(unknow, None)
        pass
