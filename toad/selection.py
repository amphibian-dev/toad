import numpy as np
import pandas as pd

from .stats import IV
from .utils import split_target, unpack_tuple, to_ndarray


def stats_features(X, y, intercept = False):
    """
    """
    import statsmodels.api as sm

    if intercept:
        X = sm.add_constant(X)

    res = sm.OLS(y, X).fit()

    return res


def stepwise(frame, target = 'target', direction = 'both', criterion = 'aic', p_enter = 0,
            p_remove = 0.01, intercept = False, p_value_enter = 0.2, max_iter = None):
    """stepwise to select features

    Args:
        frame (DataFrame): dataframe that will be use to select
        target (str): target name in frame
        direction (str): direction of stepwise, support 'forward', 'backward' and 'both'
        criterion (str): criterion to statistic model, support 'aic', 'bic'
        p_enter (float): threshold that will be used in 'forward' and 'both' to keep features
        p_remove (float): threshold that will be used in 'backward' to remove features
        intercept (bool): if have intercept
        p_value_enter (float): threshold that will be used in 'both' to remove features
        max_iter (int): maximum number of iterate

    Returns:
        DataFrame:
    """
    df, y = split_target(frame, target)

    remaining = df.columns.tolist()

    if direction is 'backward':
        selected = remaining
    else:
        selected = []

    best_res = stats_features(
        df[remaining[0]],
        y,
        intercept = intercept,
    )
    best_score = getattr(best_res, criterion)

    iter = -1
    while remaining:
        iter += 1
        if max_iter and iter > max_iter:
            break

        l = len(remaining)
        test_score = np.zeros(l)
        test_res = np.empty(l, dtype = np.object)

        if direction is 'backward':
            for i in range(l):
                test_res[i] = stats_features(
                    df[ remaining[:i] + remaining[i+1:] ],
                    y,
                    intercept = intercept,
                )
                test_score[i] = getattr(test_res[i], criterion)

            curr_ix = np.argmin(test_score)
            curr_score = test_score[curr_ix]

            name = remaining.pop(curr_ix)

            if best_score - curr_score < p_remove:
                break

            best_score = curr_score

        # forward and both
        else:
            for i in range(l):
                test_res[i] = stats_features(
                    df[ selected + [remaining[i]] ],
                    y,
                    intercept = intercept,
                )
                test_score[i] = getattr(test_res[i], criterion)

            curr_ix = np.argmin(test_score)
            curr_score = test_score[curr_ix]

            name = remaining.pop(curr_ix)
            if best_score - curr_score < p_enter:
                # early stop
                if selected: break
                continue

            selected.append(name)
            best_score = curr_score

            if direction is 'both':
                p_values = getattr(test_res[curr_ix], 'pvalues')

                max_name = p_values.idxmax()
                if p_values[max_name] > p_value_enter:
                    selected.remove(max_name)

    if isinstance(target, str):
        selected += [target]

    return frame[selected]


def drop_empty(frame, threshold = 0.9, nan = None, return_drop = False,
            exclude = None):
    """drop columns by empty

    Args:
        frame (DataFrame): dataframe that will be used
        threshold (number): drop the features whose empty num is greater than threshold. if threshold is float, it will be use as percentage
        nan (any): values will be look like empty
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): ist of feature name that will not drop

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
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
    """drop columns by correlation

    Args:
        frame (DataFrame): dataframe that will be used
        target (str): target name in dataframe
        threshold (float): drop features that has the smallest weight in each groups whose correlation is greater than threshold
        by (array-like): weight of features that will be used to drop the features
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): ist of feature name that will not drop

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    if not isinstance(by, (str, pd.Series)):
        by = pd.Series(by, index = frame.columns)

    df = frame.copy()

    if exclude is not None:
        exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        df = df.drop(columns = exclude)


    f, t = split_target(df, target)

    corr = f.corr().abs()

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


        if isinstance(by, pd.Series):
            weights = by[corr.index].values
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
    """drop columns by IV

    Args:
        frame (DataFrame): dataframe that will be used
        target (str): target name in dataframe
        threshold (float): drop the features whose IV is less than threshold
        return_drop (bool): if need to return features' name who has been dropped
        return_iv (bool): if need to return features' IV
        exclude (array-like): ist of feature name that will not drop

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
        Series: list of features' IV
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


def drop_vif(frame, threshold = 6, return_drop = False, exclude = None):
    """variance inflation factor

    Args:
        frame (DataFrame)
        threshold (float): drop features until all vif is less than threshold
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): ist of feature name that will not drop

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    drop_list = []
    while(True):
        l = len(df.columns)
        vif = np.zeros(l)
        for i in range(l):
            vif[i] = variance_inflation_factor(df.values, i)

        ix = np.argmax(vif)
        max = vif[ix]

        if max < threshold:
            break

        col = df.columns[ix]
        df = df.drop(columns = col)
        drop_list.append(col)


    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


def select(frame, target = 'target', empty = 0.9, iv = 0.02, corr = 0.7,
            return_drop = False, exclude = None):
    """select features by rate of empty, iv and correlation

    Args:
        frame (DataFrame)
        target (str): target's name in dataframe
        empty (number): drop the features which empty num is greater than threshold. if threshold is float, it will be use as percentage
        iv (float): drop the features whose IV is less than threshold
        corr (float): drop features that has the smallest IV in each groups which correlation is greater than threshold
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature name that will not drop

    Returns:
        DataFrame: selected dataframe
        dict: list of dropped feature names in each step
    """
    empty_drop = iv_drop = corr_drop = None

    if empty is not False:
        frame, empty_drop = drop_empty(frame, threshold = empty, return_drop = True, exclude = exclude)

    if iv is not False:
        frame, iv_drop, iv_list = drop_iv(frame, target = target, threshold = iv, return_drop = True, return_iv = True, exclude = exclude)

    if corr is not False:
        weights = 'IV'

        if iv is not False:
            weights = iv_list

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
