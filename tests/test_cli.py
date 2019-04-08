import pytest
import numpy as np
import pandas as pd

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

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


parser = get_parser()



@disable_stdout
def test_detect():
    args = parser.parse_args(['detect', '-i', 'tests/test_data.csv'])
    rep = args.func(args)
    assert rep.loc['E', 'unique'] == 20

@pytest.mark.skip("tree command will generate a pic in travis-ci log")
@disable_stdout
def test_tree():
    args = parser.parse_args(['tree', '-i', 'tests/test_data.csv'])
    args.func(args)
    pass
