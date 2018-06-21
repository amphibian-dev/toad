import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, _tree


def _fillna(feature, by = -1):
    feature[np.isnan(feature)] = by
    return feature

def _bin(feature, splits):
    """Bin feature by split points
    """
    feature = _fillna(feature)
    return np.digitize(feature, splits)


def DTMerge(feature, target, nan = -1, n_bins = None, min_samples = 1):
    """Merge continue

    Returns:
        array: array of split points
    """
    feature = _fillna(feature, by = nan)

    tree = DecisionTreeClassifier(
        min_samples_leaf = min_samples,
        max_leaf_nodes = n_bins,
    )
    tree.fit(np.reshape(feature, (-1, 1)), target)

    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
    return np.sort(thresholds)


def ChiMerge(feature, target, n_bins = 2, min_threshold = np.inf, nan = -1):
    """Chi-Merge

    Returns:
        array: array of split points
    """
    feature = _fillna(feature, by = nan)

    df = pd.DataFrame({
        'feature': feature,
        'target': target,
    })

    df = pd.get_dummies(df, columns = ['target'])
    grouped = df.groupby('feature').sum()

    while(True):
        # Calc chi square for each group
        chi_list = []
        for i in range(len(grouped) - 1):
            couple = grouped.values[i:i+2,:]
            total = np.sum(couple)
            cols = np.sum(couple, axis = 0)
            rows = np.sum(couple, axis = 1)

            e = np.zeros(couple.shape)
            for i in range(couple.shape[0]):
                for j in range(couple.shape[1]):
                    e[i,j] = rows[i] * cols[j] / total

            chi = np.sum(np.nan_to_num((couple - e) ** 2 / e))
            chi_list.append(chi)

        chi_list = np.array(chi_list)
        chi_min = chi_list.min()

        # break loop when the minimun chi greater the threshold
        if chi_min > min_threshold:
            break

        # get indexes of the groups who has the minimun chi
        min_ix = np.where(chi_list == chi_min)[0]
        mask = min_ix - np.arange(min.size)

        # bin groups by indexes
        for n in np.unique(mask):
            ix = min_ix[np.where(mask == n)]
            grouped.iloc[ix[0]] = np.sum(grouped.iloc[ix[0] : ix[0]+1+ix.size], axis = 0)

        # drop binned groups
        grouped = grouped.drop(index = grouped.index[min_ix + 1])

        # break loop
        if len(grouped) <= n_bins:
            break

    return grouped.index.values[1:]


def merge(feature, target, method = 'dt', **kwargs):
    """merge feature into groups

    Params:
        feature (array-like)
        target (array-like)
        method (str): 'dt', 'chi' - the strategy to be used to merge feature

    Returns:
        array: a array of merged label with the same size of feature
    """
    if method is 'dt':
        splits = DTMerge(feature, target, **kwargs)
    elif method is 'chi':
        splits = ChiMerge(feature, target, **kwargs)

    return _bin(feature, splits)
