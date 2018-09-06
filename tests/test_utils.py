import unittest
import numpy as np
import pandas as pd

from toad.utils import clip, diff_time_frame, bin_to_number

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

    def test_diff_time_frame(self):
        time_data = [
            {
                'base': '2018-01',
                'time1': '2018-04',
                'time2': '2018-04-02',
            },
            {
                'base': '2018-01',
                'time1': '2018-05',
                'time2': '2018-04-05',
            },
            {
                'base': '2018-02',
                'time1': '2018-04',
                'time2': '2018-04-10',
            },
        ]

        frame = pd.DataFrame(time_data)
        res = diff_time_frame(frame['base'], frame[['time1', 'time2']], format='%Y-%m-%d')
        self.assertEqual(res.iloc[0, 1].days, 91)

    def test_bin_to_number(self):
        s = pd.Series([
            '1',
            '1-100',
            '-',
            '100-200',
            '200-300',
            '300',
            '100-200',
        ])

        res = s.apply(bin_to_number())
        self.assertEqual(res[3], 150)
