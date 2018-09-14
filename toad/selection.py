import numpy as np
import pandas as pd
from .stats import IV
from .utils import split_target, unpack_tuple

def drop_empty(frame, threshold = 0.9, nan = None, return_drop = False):
    """drop columns by empty
    """
    if nan is not None:
        df = frame.replace(nan, np.nan)
    else:
        df = frame.copy()

    if threshold < 1:
        threshold = len(df) * threshold

    drop_list = []
    for col in df:
        n = df[col].isnull().sum()
        if n > threshold:
            drop_list.append(col)

    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (np.array(drop_list),)

    return unpack_tuple(res)

def drop_corr(frame, target = None, threshold = 0.7, by = 'IV', return_drop = False):
    """drop columns by corr
    """
    f, t = split_target(frame, target)

    corr = f.corr()

    # get the graph of relationship
    ix, cn = np.where(np.triu(corr.values, 1) > threshold)
    graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])

    uni, counts = np.unique(graph, return_counts = True)

    # calc weights for nodes
    weights = np.zeros(len(corr.index))


    if isinstance(by, np.ndarray):
        weights = by
    elif isinstance(by, pd.Series):
        weights = by.values
    elif isinstance(by, list):
        weights = np.array(by)
    elif by.upper() == 'IV':
        for ix in uni:
            weights[ix] = IV(frame[corr.index[ix]], target = t)

    drops = []
    while(True):
        # TODO deal with circle

        # get nodes with the most relationship
        nodes = uni[np.argwhere(counts == np.amax(counts))].flatten()

        # get node who has the min weights
        n = nodes[np.argsort(weights[nodes])[0]]

        # get nodes of 1 degree relationship of n
        i, c = np.where(graph == n)
        pairs = graph[(i, 1-c)]

        # if sum of 1 degree nodes greater than n
        # then delete n self
        # else delete all 1 degree nodes
        if weights[pairs].sum() > weights[n]:
            dro = [n]
        else:
            dro = pairs.tolist()

        # add nodes to drops list
        drops += dro

        # delete nodes from graph
        di, _ = np.where(np.isin(graph, dro))
        graph = np.delete(graph, di, axis = 0)

        # if graph is empty
        if len(graph) <= 0:
            break

        # update nodes and counts
        uni, counts = np.unique(graph, return_counts = True)


    drop_list = corr.index[drops].values
    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


def drop_iv(frame, target = 'target', threshold = 0.02, return_drop = False, return_iv = False):
    """
    """
    f, t = split_target(frame, target)

    l = len(f.columns)
    iv = np.zeros(l)

    for i in range(l):
        iv[i] = IV(frame.iloc[:,i], target = t)

    drop_ix = np.where(iv < threshold)

    drop_list = f.columns[drop_ix].values
    df = frame.drop(columns = drop_list)

    res = (df,)
    if return_drop:
        res += (drop_list,)

    if return_iv:
        res += (pd.Series(iv, index = f.columns),)

    return unpack_tuple(res)


def select(frame, target = 'target', empty = 0.9, iv = 0.02, corr = 0.7, return_drop = False):
    """
    """
    empty_drop = iv_drop = corr_drop = None

    if empty is not False:
        frame, empty_drop = drop_empty(frame, threshold = empty, return_drop = True)

    if iv is not False:
        frame, iv_drop, iv_list = drop_iv(frame, target = target, threshold = iv, return_drop = True, return_iv = True)

    if corr is not False:
        weights = 'IV'

        if iv is not False:
            ix = frame.columns.tolist()
            ix.remove(target)
            weights = iv_list[ix]

        frame, corr_drop = drop_corr(frame, target = target, threshold = corr, by = weights, return_drop = True)

    res = (frame,)
    if return_drop:
        d = {
            'empty': empty_drop,
            'iv': iv_drop,
            'corr': corr_drop,
        }
        res += (d,)

    return unpack_tuple(res)
