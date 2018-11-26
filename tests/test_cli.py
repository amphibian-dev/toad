import unittest
import numpy as np
import pandas as pd

from toad.cli import get_parser

def disable_stdout(fn):

    def wrapper(*args):
        import os
        import sys

        with open(os.devnull, 'w') as f:
            so = sys.stdout
            sys.stdout = f

            fn(*args)

            sys.stdout = so

    return wrapper

# np.random.seed(1)
#
# ab = np.array(list('ABCDEFG'))
# feature = np.random.randint(10, size = 500)
# target = np.random.randint(2, size = 500)
# str_feat = ab[np.random.choice(7, 500)]


parser = get_parser()


class TestTransform(unittest.TestCase):
    def setUp(self):
        pass

    @disable_stdout
    def test_detect(self):
        args = parser.parse_args(['detect', '-i', 'tests/test_data.csv'])
        rep = args.func(args)
        self.assertEqual(rep.loc['E', 'unique'], 20)

    @disable_stdout
    def test_tree(self):
        args = parser.parse_args(['tree', '-i', 'tests/test_data.csv'])
        args.func(args)
        pass
