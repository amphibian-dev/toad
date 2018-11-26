import sys
import argparse
import pandas as pd

def func(args):
    """detect csv data

    Examples:

        toad tree -i xxx.csv
    """
    import toad
    from .tree import split_data, dtree
    args = vars(args)

    # remove func attribute
    args.pop('func')
    
    input = args.pop('input')
    target = args.pop('target')
    include = args.pop('include')
    exclude = args.pop('exclude')

    sys.stdout.write('reading data....\n')
    data = pd.read_csv(input)

    X, *tars = split_data(data, target = target)

    if include is not None:
        X = X[include]

    if exclude is not None:
        X = X.drop(columns = exclude)

    X = toad.utils.get_dummies(X)


    for t in tars:
        sys.stdout.write('analyse '+ t.name +' ...\n')
        dtree(X, t, **args)


ARGS = {
    'info': {
        'name': 'tree',
        'description': 'analyse bad rate from a csv file',
    },
    'defaults': {
        'func': func,
    },
    'args': [
        {
            'flag': ('-i', '--input'),
            'type': argparse.FileType('r', encoding='utf-8'),
            'help': 'the csv file which will be analysed',
            'required': True,
        },
        {
            'flag': ('-t', '--target'),
            'nargs': '+',
            'help': 'the target(s) will be analysed',
            'default': 'target',
        },
        {
            'flag': ('-c', '--criterion'),
            'type': str,
            'help': 'criterion to measure the quality of a split. Support "gini" (default), "entropy"',
            'default': 'gini',
        },
        {
            'flag': ('-d', '--depth'),
            'type': int,
            'help': 'the maximum depth of the tree',
            'default': None,
        },
        {
            'flag': ('-s', '--sample'),
            'type': float,
            'help': 'minimum number of sample in each node',
            'default': 0.01,
        },
        {
            'flag': ('-r', '--ratio'),
            'type': float,
            'help': 'threshold of ratio that will be highlighted',
            'default': 0.15,
        },
        {
            'flag': ('--exclude',),
            'nargs': '+',
            'help': 'feature names that will not use to analyse',
            'default': None,
        },
        {
            'flag': ('--include',),
            'nargs': '+',
            'help': 'feature names that will be used to analyse',
            'default': None,
        },
    ]
}
