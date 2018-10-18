import unittest
import numpy as np
import pandas as pd

from toad.transform import WOETransformer, Combiner

np.random.seed(1)

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size = 500)
target = np.random.randint(2, size = 500)
str_feat = ab[np.random.choice(7, 500)]

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
})


class TestTransform(unittest.TestCase):
    def setUp(self):
        pass

    def test_woe_transformer(self):
        f = WOETransformer().fit_transform(feature, target)
        self.assertEqual(f[451], -0.17061154127869285)

    def test_woe_transformer_with_str(self):
        f = WOETransformer().fit_transform(str_feat, target)
        self.assertEqual(f[451], -0.2198594761130199)

    def test_woe_transformer_frame(self):
        res = WOETransformer().fit_transform(df, target)
        self.assertEqual(res.iloc[451, 1], -0.2198594761130199)

    def test_combiner(self):
        f = Combiner().fit_transform(feature, target, method = 'chi')
        self.assertEqual(f[451], 3)

    def test_combiner_with_str(self):
        f = Combiner().fit_transform(str_feat, target, method = 'chi')
        self.assertEqual(f[451], 0)

    def test_combiner_frame(self):
        res = Combiner().fit_transform(df, target)
        self.assertEqual(res.iloc[404, 1], 1)
