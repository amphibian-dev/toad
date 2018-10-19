import numpy as np
import pandas as pd


from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.cluster import KMeans

from .utils import fillna, bin_by_splits, to_ndarray, support_dataframe

DEFAULT_BINS = 20


def StepMerge(feature, nan = None, n_bins = None):
    if n_bins is None:
        n_bins = DEFAULT_BINS

    if nan is not None:
        feature = fillna(feature, by = nan)

    max = np.nanmax(feature)
    min = np.nanmin(feature)

    step = (max - min) / n_bins
    return np.arange(min, max, step)[1:]

def QuantileMerge(feature, nan = -1, n_bins = None, q = None):
    """Merge by quantile
    """
    if n_bins is None and quantile is None:
        n_bins = DEFAULT_BINS

    if q is None:
        step = 1 / n_bins
        q = np.arange(0, 1, step)[1:]

    feature = fillna(feature, by = nan)

    return np.quantile(feature, q)


def KMeansMerge(feature, target = None, nan = -1, n_bins = None, random_state = 1):
    """Merge by KMeans
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by = nan)

    model = KMeans(
        n_clusters = n_bins,
        random_state = random_state
    )
    model.fit(feature.reshape((-1 ,1)), target)

    centers = np.sort(model.cluster_centers_.reshape(-1))

    l = len(centers) - 1
    splits = np.zeros(l)
    for i in range(l):
        splits[i] = (centers[i] + centers[i+1]) / 2

    return splits



def DTMerge(feature, target, nan = -1, n_bins = None, min_samples = 1):
    """Merge continue

    Returns:
        array: array of split points
    """
    if n_bins is None and min_samples == 1:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by = nan)

    tree = DecisionTreeClassifier(
        min_samples_leaf = min_samples,
        max_leaf_nodes = n_bins,
    )
    tree.fit(feature.reshape((-1, 1)), target)

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
        n_bins = DEFAULT_BINS

    if min_samples and min_samples < 1:
        min_samples = len(feature) * min_samples

    feature = fillna(feature, by = nan)
    target = to_ndarray(target)

    target_unique = np.unique(target)
    feature_unique = np.unique(feature)
    len_f = len(feature_unique)
    len_t = len(target_unique)
    grouped = np.zeros((len_f, len_t))
    # grouped[:,1] = feature_unique
    for i in range(len_f):
        tmp = target[feature == feature_unique[i]]
        for j in range(len_t):
            grouped[i,j] = (tmp == target_unique[j]).sum()


    while(True):
        # Calc chi square for each group
        l = len(grouped) - 1
        chi_list = np.zeros(l)
        chi_min = np.inf
        chi_ix = []
        for i in range(l):
            chi = 0
            couple = grouped[i:i+2,:]
            total = np.sum(couple)
            cols = np.sum(couple, axis = 0)
            rows = np.sum(couple, axis = 1)

            for j in range(couple.shape[0]):
                for k in range(couple.shape[1]):
                    e = rows[j] * cols[k] / total
                    if e != 0:
                        chi += (couple[j, k] - e) ** 2 / e
            
            chi_list[i] = chi

            if chi == chi_min:
                chi_ix.append(i)
                continue

            if chi < chi_min:
                chi_min = chi
                chi_ix = [i]

        # break loop when the minimun chi greater the threshold
        if min_threshold and chi_min > min_threshold:
            break

        # get indexes of the groups who has the minimun chi
        min_ix = np.array(chi_ix)

        # get the indexes witch needs to drop
        drop_ix = min_ix + 1


        # combine groups by indexes
        retain_ix = min_ix[0]
        last_ix = retain_ix
        for ix in min_ix:
            # set a new group
            if ix - last_ix > 1:
                retain_ix = ix

            # combine all contiguous indexes into one group
            grouped[retain_ix] = grouped[retain_ix] + grouped[ix + 1]
            last_ix = ix


        # drop binned groups
        grouped = np.delete(grouped, drop_ix, axis = 0)
        feature_unique = np.delete(feature_unique, drop_ix)

        # break loop when reach n_bins
        if n_bins and len(grouped) <= n_bins:
            break

        # break loop if min samples of groups is greater than threshold
        if min_samples and np.sum(grouped.values, axis = 1).min() > min_samples:
            break

    return feature_unique[1:]

@support_dataframe(require_target = False)
def merge(feature, target = None, method = 'dt', return_splits = False, **kwargs):
    """merge feature into groups

    Params:
        feature (array-like)
        target (array-like)
        method (str): 'dt', 'chi' - the strategy to be used to merge feature

    Returns:
        array: a array of merged label with the same size of feature
    """
    feature = to_ndarray(feature)

    if method is 'dt':
        splits = DTMerge(feature, target, **kwargs)
    elif method is 'chi':
        splits = ChiMerge(feature, target, **kwargs)
    elif method is 'quantile':
        splits = QuantileMerge(feature, **kwargs)
    elif method is 'step':
        splits = StepMerge(feature, **kwargs)
    elif method is 'kmeans':
        splits = KMeaMerge(feature, target = target, **kwargs)


    if len(splits):
        bins = bin_by_splits(feature, splits)
    else:
        bins = np.zeros(len(feature))

    if return_splits:
        return bins, splits

    return bins
