from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import f1_score
from .merge import merge

cimport cython

cdef double FEATURE_THRESHOLD = 1e-7

def _count(arr, value, default = None):
    c = (arr == value).sum()

    if default is not None and c == 0:
        return default

    return c

def _to_ndarray(s):
    if isinstance(s, np.ndarray):
        return s

    if isinstance(s, pd.core.base.PandasObject):
        return s.values

    return s

def KS(score, target):
    """calculate ks value
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


def KS_bucket(score, target, int bucket = 10):
    """calculate ks value by bucket
    """
    df = pd.DataFrame({
        'score': score,
        'bad': target,
    })

    df['good'] = 1 - df['bad']
    df['bucket'] = pd.qcut(df['score'], bucket, duplicates = 'drop')
    grouped = df.groupby('bucket', as_index = False)

    agg1 = pd.DataFrame()
    agg1['min'] = grouped.min()['score']
    agg1['max'] = grouped.max()['score']
    agg1['bads'] = grouped.sum()['bad']
    agg1['goods'] = grouped.sum()['good']
    agg1['total'] = agg1['bads'] + agg1['goods']

    agg2 = (agg1.sort_values(by = 'min')).reset_index(drop = True)
    agg2['bad_rate'] = (agg2['bads'] / agg2['total']).apply('{0:.2%}'.format)

    agg2['ks'] = np.round((agg2['bads'] / agg2['bads'].sum()).cumsum() - (agg2['goods'] / agg2['goods'].sum()).cumsum(), 4) * 100

    return agg2

def KS_by_col(df, by='feature', score='score', target='target'):
    """
    """

    pass


def feature_splits(feature, target):
    """find posibility spilt points
    """
    feature = _to_ndarray(feature)
    target = _to_ndarray(target)

    matrix = np.vstack([feature, target])
    matrix = matrix[:, matrix[0,:].argsort()]

    splits_values = []
    for i in range(1, len(matrix[0])):
        # if feature value is almost same, then skip
        if matrix[0,i] <= matrix[0, i-1] + FEATURE_THRESHOLD:
            continue

        # if target value is not same, calculate split
        if matrix[1, i] != matrix[1, i-1]:
            v = (matrix[0, i] + matrix[0, i-1]) / 2.0
            splits_values.append(v)

    return np.unique(splits_values)


def iter_df(dataframe, feature, target, splits):
    """iterate dataframe by split points

    Returns:
        iterator (df, splitter)
    """
    splits.sort()
    df = pd.DataFrame()
    df['source'] = dataframe[feature]
    df[target] = dataframe[target]
    df[feature] = 0

    for v in splits:
        df.loc[df['source'] < v, feature] = 1
        yield df, v

def inter_feature(feature, splits):
    splits.sort()
    bin = np.zeros(len(feature))

    for v in splits:
        bin[feature < v] = 1
        yield bin


def is_continuous(series):
    series = _to_ndarray(series)
    if not np.issubdtype(series.dtype, np.number):
        return False

    n = len(np.unique(series))
    return n > 20 or n / series.size > 0.5
    # return n / series.size > 0.5


def gini(target):
    """get gini index of a feature
    """
    target = _to_ndarray(target)
    v, c = np.unique(target, return_counts = True)

    return 1 - ((c / target.size) ** 2).sum()

def _gini_cond(feature, target):
    """private conditional gini function
    """
    size = feature.size
    matrix = np.vstack([feature, target])

    cdef double value = 0
    for v, c in zip(*np.unique(feature, return_counts = True)):
        target_series = matrix[:, matrix[0,:] == v][1,:]
        value += c / size * gini(target_series)

    return value

def gini_cond(feature, target):
    """get conditional gini index of a feature
    """
    if not is_continuous(feature):
        return _gini_cond(feature, target)

    # find best split for continuous data
    splits = feature_splits(feature, target)
    cdef double best = 999
    cdef double v

    for f in inter_feature(feature, splits):
        v = _gini_cond(f, target)
        if v < best:
            best = v
    return best

def entropy(target):
    """get infomation entropy of a feature
    """
    target = _to_ndarray(target)
    uni, counts = np.unique(target, return_counts = True)
    prob = counts / len(target)
    cdef double entropy = stats.entropy(prob)
    return entropy

def _entropy_cond(feature, target):
    """private conditional entropy func
    """
    size = len(feature)

    value = 0
    for v, c in zip(*np.unique(feature, return_counts = True)):
        target_series = target[feature == v]
        value += c/size * entropy(target_series)

    return value

def entropy_cond(feature, target):
    """get conditional entropy of a feature
    """
    feature = _to_ndarray(feature)
    target = _to_ndarray(target)

    if not is_continuous(feature):
        return _entropy_cond(feature, target)

    # find best split for continuous data
    splits = feature_splits(feature, target)
    best = 0
    for f in inter_feature(feature, splits):
        v = _entropy_cond(f, target)
        if v > best:
            best = v
    return best


def WOE(y_prob, n_prob):
    """get WOE of a group

    Args:
        y_prob: the probability of grouped y in total y
        n_prob: the probability of grouped n in total n
    """
    return np.log(y_prob / n_prob)


def _IV(feature, target):
    """
    """
    feature = _to_ndarray(feature)
    target = _to_ndarray(target)

    cdef double t_counts_0 = _count(target, 0, default = 1)
    cdef double t_counts_1 = _count(target, 1, default = 1)

    cdef double value = 0

    for v in np.unique(feature):
        sub_target = target[feature == v]

        sub_0 = _count(sub_target, 0, default = 1)
        sub_1 = _count(sub_target, 1, default = 1)

        y_prob = sub_1 / t_counts_1
        n_prob = sub_0 / t_counts_0

        value += (y_prob - n_prob) * WOE(y_prob, n_prob)

    return value


def IV(feature, target, **kwargs):
    """get IV of a feature
    """
    if not is_continuous(feature):
        return _IV(feature, target)

    # df = pd.DataFrame({
    #     feature: pd.cut(dataframe[feature], 20),
    #     target: dataframe[target],
    # })

    feature = merge(feature, target, **kwargs)

    return _IV(feature, target)



def F1(score, target):
    """

    Returns:
        float: best f1 score
        float: best spliter
    """
    dataframe = pd.DataFrame({
        'score': score,
        'target': target,
    })

    # find best split for score
    splits = feature_splits(dataframe['score'], dataframe['target'])
    best = 0
    split = None
    for df, pointer in iter_df(dataframe, 'score', 'target', splits):
        v = f1_score(df['target'], df['score'])

        if v > best:
            best = v
            split = pointer

    return best, split


def column_quality(feature, target, name = 'feature'):
    c = len(np.unique(feature))
    iv = g = e = '--'

    # skip when unique is too much
    if is_continuous(feature) or c / len(feature) < 0.5:
        iv = IV(feature, target)
        g = gini_cond(feature, target)
        e = entropy_cond(feature, target)

    row = pd.Series(
        index = ['iv', 'gini', 'entropy', 'unique'],
        data = [iv, g, e, c],
    )

    row.name = name
    return row


def quality(dataframe, target = 'target'):
    """get quality of features in data

    Returns:
        dataframe
    """
    res = []
    pool = Pool(cpu_count())
    for column in dataframe:
        r = pool.apply_async(column_quality, args = (dataframe[column].values, dataframe[target].values, column))
        res.append(r)

    pool.close()
    pool.join()

    rows = [r.get() for r in res]

    return pd.DataFrame(rows)
