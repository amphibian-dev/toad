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


def ChiMerge(feature, target, n_bins = None, min_samples = None, min_threshold = None, nan = -1):
    """Chi-Merge

    Args:
        feature (array-like): feature to be merged
        target (array-like): a array of target classes
        n_bins (int): n bins will be merged into
        min_samples (number): min sample in each group, if float, it will be the percentage of samples
        min_threshold (number): min threshold of chi-square

    Returns:
        array: array of split points
    """

    # set default break condition
    if n_bins is None and min_samples is None and min_threshold is None:
        n_bins = 20

    if min_samples and min_samples < 1:
        min_samples = len(feature) * min_samples

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
        if min_threshold and chi_min > min_threshold:
            break

        # get indexes of the groups who has the minimun chi
        min_ix = np.where(chi_list == chi_min)[0]
        mask = min_ix - np.arange(min_ix.size)

        # bin groups by indexes
        for n in np.unique(mask):
            ix = min_ix[np.where(mask == n)]
            grouped.iloc[ix[0]] = np.sum(grouped.iloc[ix[0] : ix[0]+1+ix.size], axis = 0)

        # drop binned groups
        grouped = grouped.drop(index = grouped.index[min_ix + 1])

        # break loop when reach n_bins
        if n_bins and len(grouped) <= n_bins:
            break

        # break loop if min samples of groups is greater than threshold
        if min_samples and np.sum(grouped.values, axis = 1).min() > min_samples:
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

    # print(splits)
    return _bin(feature, splits)
