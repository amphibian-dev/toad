import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score

from .merge import merge
from .transform import Combiner

from .utils import (
    feature_splits,
    iter_df,
    unpack_tuple,
)


def KS(score, target):
    """calculate ks value

    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target

    Returns:
        float: the max KS value
    """
    df = pd.DataFrame({
        'score': score,
        'target': target,
    })
    df = df.sort_values(by='score', ascending=False)
    df['good'] = 1 - df['target']
    df['bad_rate'] = df['target'].cumsum() / df['target'].sum()
    df['good_rate'] = df['good'].cumsum() / df['good'].sum()
    df['ks'] = df['bad_rate'] - df['good_rate']
    return max(abs(df['ks']))


def KS_bucket(score, target, bucket = 10, method = 'quantile', **kwargs):
    """calculate ks value by bucket

    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target
        bucket (int): n groups that will bin into
        method (str): method to bin score. `quantile` (default), `step`

    Returns:
        DataFrame
    """
    df = pd.DataFrame({
        'score': score,
        'bad': target,
    })

    df['good'] = 1 - df['bad']

    bad_total = df['bad'].sum()
    good_total = df['good'].sum()

    df['bucket'] = 0
    if bucket is False:
        df['bucket'] = score
    elif isinstance(bucket, (list, np.ndarray, pd.Series)):
        df['bucket'] = bucket
    elif isinstance(bucket, int):
        df['bucket'] = merge(score, n_bins = bucket, method = method, **kwargs)

    grouped = df.groupby('bucket', as_index = False)

    agg1 = pd.DataFrame()
    agg1['min'] = grouped.min()['score']
    agg1['max'] = grouped.max()['score']
    agg1['bads'] = grouped.sum()['bad']
    agg1['goods'] = grouped.sum()['good']
    agg1['total'] = agg1['bads'] + agg1['goods']

    agg2 = (agg1.sort_values(by = 'min')).reset_index(drop = True)

    agg2['bad_rate'] = agg2['bads'] / agg2['total']
    agg2['good_rate'] = agg2['goods'] / agg2['total']

    agg2['odds'] = agg2['bads'] / agg2['goods']

    agg2['bad_prop'] = agg2['bads'] / bad_total
    agg2['good_prop'] = agg2['goods'] / good_total

    agg2['cum_bads'] = agg2['bads'].cumsum()
    agg2['cum_goods'] = agg2['goods'].cumsum()

    agg2['cum_bads_prop'] = agg2['cum_bads'] / bad_total
    agg2['cum_goods_prop'] = agg2['cum_goods'] / good_total


    agg2['ks'] = agg2['cum_bads_prop'] - agg2['cum_goods_prop']

    return agg2

def KS_by_col(df, by='feature', score='score', target='target'):
    """
    """

    pass


def SSE(y_pred, y):
    """sum of squares due to error
    """
    return np.sum((y_pred - y) ** 2)


def MSE(y_pred, y):
    """mean of squares due to error
    """
    return np.mean((y_pred - y) ** 2)


def AIC(y_pred, y, k, llf = None):
    """Akaike Information Criterion

    Args:
        y_pred (array-like)
        y (array-like)
        k (int): number of featuers
        llf (float): result of log-likelihood function
    """
    if llf is None:
        llf = np.log(SSE(y_pred, y))

    return 2 * k - 2 * llf


def BIC(y_pred, y, k, llf = None):
    """Bayesian Information Criterion

    Args:
        y_pred (array-like)
        y (array-like)
        k (int): number of featuers
        llf (float): result of log-likelihood function
    """
    n = len(y)
    if llf is None:
        llf = np.log(SSE(y_pred, y))

    return np.log(n) * k - 2 * llf


def F1(score, target, split = 'best', return_split = False):
    """calculate f1 value

    Args:
        score (array-like)
        target (array-like)

    Returns:
        float: best f1 score
        float: best spliter
    """
    dataframe = pd.DataFrame({
        'score': score,
        'target': target,
    })

    if split == 'best':
        # find best split for score
        splits = feature_splits(dataframe['score'], dataframe['target'])
    else:
        splits = [split]

    best = 0
    sp = None
    for df, pointer in iter_df(dataframe, 'score', 'target', splits):
        v = f1_score(df['target'], df['score'])

        if v > best:
            best = v
            sp = pointer

    if return_split:
        return best, sp

    return best


def AUC(score, target):
    """AUC Score

    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target

    Returns:
        float: auc score
    """

    return roc_auc_score(target, score)


def _PSI(test, base):
    test_prop = pd.Series(test).value_counts(normalize = True, dropna = False)
    base_prop = pd.Series(base).value_counts(normalize = True, dropna = False)

    psi = np.sum((test_prop - base_prop) * np.log(test_prop / base_prop))

    frame = pd.DataFrame({
        'test': test_prop,
        'base': base_prop,
    })
    frame.index.name = 'value'

    return psi, frame.reset_index()



def PSI(test, base, combiner = None, return_frame = False):
    """calculate PSI

    Args:
        test (array-like): data to test PSI
        base (array-like): base data for calculate PSI
        combiner (Combiner|list|dict): combiner to combine data
        return_frame (bool): if need to return frame of proportion

    Returns:
        float|Series
    """

    if combiner is not None:
        if isinstance(combiner, (dict, list)):
            combiner = Combiner().set_rules(combiner)

        test = combiner.transform(test, labels = True)
        base = combiner.transform(base, labels = True)

    psi = list()
    frame = list()

    if isinstance(test, pd.DataFrame):
        for col in test:
            p, f = _PSI(test[col], base[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index = test.columns)

        frame = pd.concat(
            frame,
            keys = test.columns,
            names = ['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns = 'id')
    else:
        psi, frame = _PSI(test, base)


    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)
