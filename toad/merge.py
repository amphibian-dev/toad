"""
Merge module - binning algorithms for feature engineering

This module provides various binning methods including:
- StepMerge: Equal-width binning
- QuantileMerge: Quantile-based binning
- KMeansMerge: KMeans clustering based binning
- DTMerge: Decision tree based binning
- ChiMerge: Chi-square statistic based binning (Rust implementation)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.cluster import KMeans

from .utils import fillna, bin_by_splits, to_ndarray, clip
from .utils.decorator import support_dataframe

DEFAULT_BINS = 10


def StepMerge(feature, nan=None, n_bins=None, clip_v=None, clip_std=None, clip_q=None):
    """Merge by step

    Args:
        feature (array-like)
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        clip_v (number | tuple): min/max value of clipping
        clip_std (number | tuple): min/max std of clipping
        clip_q (number | tuple): min/max quantile of clipping
    Returns:
        array: split points of feature
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    if nan is not None:
        feature = fillna(feature, by=nan)

    feature = clip(feature, value=clip_v, std=clip_std, quantile=clip_q)

    max_val = np.nanmax(feature)
    min_val = np.nanmin(feature)

    step = (max_val - min_val) / n_bins
    return np.arange(min_val, max_val, step)[1:]


def QuantileMerge(feature, nan=-1, n_bins=None, q=None):
    """Merge by quantile

    Args:
        feature (array-like)
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        q (array-like): list of percentage split points

    Returns:
        array: split points of feature
    """
    if n_bins is None and q is None:
        n_bins = DEFAULT_BINS

    if q is None:
        step = 1 / n_bins
        q = np.arange(0, 1, step)

    feature = fillna(feature, by=nan)

    splits = np.quantile(feature, q)
    return np.unique(splits)[1:]


def KMeansMerge(feature, target=None, nan=-1, n_bins=None, random_state=1):
    """Merge by KMeans

    Args:
        feature (array-like)
        target (array-like): target will be used to fit kmeans model
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        random_state (int): random state will be used for kmeans model

    Returns:
        array: split points of feature
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by=nan)

    model = KMeans(
        n_clusters=n_bins,
        random_state=random_state
    )
    model.fit(feature.reshape((-1, 1)), target)

    centers = np.sort(model.cluster_centers_.reshape(-1))

    l = len(centers) - 1
    splits = np.zeros(l)
    for i in range(l):
        splits[i] = (centers[i] + centers[i+1]) / 2

    return splits


def DTMerge(feature, target, nan=-1, n_bins=None, min_samples=1, **kwargs):
    """Merge by Decision Tree

    Args:
        feature (array-like)
        target (array-like): target will be used to fit decision tree
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        min_samples (int): min number of samples in each leaf nodes

    Returns:
        array: array of split points
    """
    if n_bins is None and min_samples == 1:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by=nan)

    tree = DecisionTreeClassifier(
        min_samples_leaf=min_samples,
        max_leaf_nodes=n_bins,
        **kwargs,
    )
    tree.fit(feature.reshape((-1, 1)), target)

    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]
    return np.sort(thresholds)


def ChiMerge(feature, target, n_bins=None, min_samples=None,
            min_threshold=None, nan=-1, balance=True):
    """Chi-Merge using Rust implementation

    Args:
        feature (array-like): feature to be merged
        target (array-like): a array of target classes
        n_bins (int): n bins will be merged into
        min_samples (number): min sample in each group, if float, it will be the percentage of samples
        min_threshold (number): min threshold of chi-square
        nan (number): value to replace NaN
        balance (bool): whether to balance chi-square by group size

    Returns:
        array: array of split points
    """
    # Lazy import of Rust implementation
    try:
        import toad_core
        chi_merge_rust = toad_core.merge.chi_merge
    except ImportError as e:
        raise NotImplementedError(
            f"ChiMerge Rust implementation not available: {e}. "
            "Please build the Rust extension with: maturin develop"
        )

    # Convert to numpy arrays
    feature_arr = to_ndarray(feature)
    target_arr = to_ndarray(target).astype(np.int32)

    # Call unified Rust chi_merge - it handles type dispatching automatically
    # Supports f64, i32, i64 with automatic dtype detection
    return chi_merge_rust(
        feature_arr, target_arr,
        n_bins=n_bins,
        min_samples=min_samples,
        min_threshold=min_threshold,
        nan=nan,
        balance=balance
    )


@support_dataframe(require_target=False)
def merge(feature, target=None, method='dt', return_splits=False, **kwargs):
    """merge feature into groups

    Args:
        feature (array-like)
        target (array-like)
        method (str): 'dt', 'chi', 'quantile', 'step', 'kmeans' - the strategy to be used to merge feature
        return_splits (bool): if needs to return splits
        n_bins (int): n groups that will be merged into


    Returns:
        array: a array of merged label with the same size of feature
        array: list of split points
    """
    method = method.lower()
    assert method in ['dt', 'chi', 'quantile', 'step', 'kmeans'], \
        "`method` must be in ['dt', 'chi', 'quantile', 'step', 'kmeans']"

    feature = to_ndarray(feature)
    method = method.lower()

    if method == 'dt':
        splits = DTMerge(feature, target, **kwargs)
    elif method == 'chi':
        splits = ChiMerge(feature, target, **kwargs)
    elif method == 'quantile':
        splits = QuantileMerge(feature, **kwargs)
    elif method == 'step':
        splits = StepMerge(feature, **kwargs)
    elif method == 'kmeans':
        splits = KMeansMerge(feature, target=target, **kwargs)
    else:
        splits = np.empty(shape=(0,))

    if len(splits):
        bins = bin_by_splits(feature, splits)
    else:
        bins = np.zeros(len(feature))

    if return_splits:
        return bins, splits

    return bins
