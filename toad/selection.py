import numpy as np
import pandas as pd
from .stats import IV
from .utils import split_target, unpack_tuple, to_ndarray

def drop_empty(frame, threshold = 0.9, nan = None, return_drop = False,
            exclude = None):
    """drop columns by empty
    """
    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    if nan is not None:
        df = df.replace(nan, np.nan)

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

def drop_corr(frame, target = None, threshold = 0.7, by = 'IV',
            return_drop = False, exclude = None):
    """drop columns by corr
    """
    if not isinstance(by, str):
        by = to_ndarray(by)

    df = frame.copy()

    if exclude is not None:
        exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        drop_ix = np.argwhere(df.columns.isin(exclude)).flatten()
        df = df.drop(columns = exclude)

        # drop exclude weight
        if not isinstance(by, str):
            by = np.delete(by, drop_ix)

    f, t = split_target(df, target)

    corr = f.corr()

    drops = []

    # get position who's corr greater than threshold
    ix, cn = np.where(np.triu(corr.values, 1) > threshold)

    # if has position
    if len(ix):
        # get the graph of relationship
        graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])

        uni, counts = np.unique(graph, return_counts = True)

        # calc weights for nodes
        weights = np.zeros(len(corr.index))


        if isinstance(by, np.ndarray):
            weights = by
        elif by.upper() == 'IV':
            for ix in uni:
                weights[ix] = IV(df[corr.index[ix]], target = t)


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


def drop_iv(frame, target = 'target', threshold = 0.02, return_drop = False,
            return_iv = False, exclude = None):
    """
    """
    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    f, t = split_target(df, target)

    l = len(f.columns)
    iv = np.zeros(l)

    for i in range(l):
        iv[i] = IV(f[f.columns[i]], target = t)

    drop_ix = np.where(iv < threshold)

    drop_list = f.columns[drop_ix].values
    df = frame.drop(columns = drop_list)

    res = (df,)
    if return_drop:
        res += (drop_list,)

    if return_iv:
        res += (pd.Series(iv, index = f.columns),)

    return unpack_tuple(res)


def select(frame, target = 'target', empty = 0.9, iv = 0.02, corr = 0.7,
            return_drop = False, exclude = None):
    """
    """
    empty_drop = iv_drop = corr_drop = None

    if empty is not False:
        frame, empty_drop = drop_empty(frame, threshold = empty, return_drop = True, exclude = exclude)

    if iv is not False:
        frame, iv_drop, iv_list = drop_iv(frame, target = target, threshold = iv, return_drop = True, return_iv = True, exclude = exclude)

    if corr is not False:
        weights = 'IV'

        if iv is not False:
            ix = frame.columns.tolist()
            ix.remove(target)
            weights = iv_list.reindex(ix)

        frame, corr_drop = drop_corr(frame, target = target, threshold = corr, by = weights, return_drop = True, exclude = exclude)

    res = (frame,)
    if return_drop:
        d = {
            'empty': empty_drop,
            'iv': iv_drop,
            'corr': corr_drop,
        }
        res += (d,)

    return unpack_tuple(res)
