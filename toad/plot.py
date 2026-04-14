import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from .stats import IV, feature_bin_stats
from .metrics import AUC
from .tadpole import tadpole
from .tadpole.utils import HEATMAP_CMAP, MAX_STYLE, add_annotate, add_text, reset_ylim
from .utils import unpack_tuple, generate_str

def badrate_plot(frame, x = None, target = 'target', by = None,
                freq = None, format = None, return_counts = False,
                return_proportion = False, return_frame = False):
    """plot for badrate

    Args:
        frame (DataFrame)
        x (str): column in frame that will be used as x axis
        target (str): target column in frame
        by (str): column in frame that will be calculated badrate by it
        freq (str): offset aliases string by pandas
                    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        format (str): format string for time
        return_counts (bool): if need return counts plot
        return_frame (bool): if need return frame

    Returns:
        Axes: badrate plot
        Axes: counts plot
        Axes: proportion plot
        Dataframe: grouping detail data
    """
    frame = frame.copy()
    markers = True

    if not isinstance(target, str):
        temp_name = generate_str()
        frame[temp_name] = target
        target = temp_name

    grouper = x
    if freq is not None:
        frame.loc[:, x] = pd.to_datetime(frame[x], format = format)
        grouper = pd.Grouper(key = x, freq = freq)

    if by is not None:
        grouper = [by, grouper]

        styles_count = frame[by].nunique()
        if styles_count > MAX_STYLE:
            markers = ['o'] * styles_count

    group = frame.groupby(grouper)
    table = group[target].agg(['sum', 'count']).reset_index()
    table['badrate'] = table['sum'] / table['count']

    # set number dtype to object
    if np.issubdtype(table[x].dtype, np.number):
        table[x] = table[x].astype(str)


    rate_plot = tadpole.lineplot(
        x = x,
        y = 'badrate',
        hue = by,
        style = by,
        data = table,
        legend = 'full',
        markers = markers,
        dashes = False,
    )

    # set y axis start with 0
    rate_plot.set_ylim(0, None)

    res = (rate_plot,)

    if return_counts:
        count_plot = tadpole.barplot(
            x = x,
            y = 'count',
            hue = by,
            data = table,
        )
        res += (count_plot,)


    if return_proportion:
        table['prop'] = 0
        for v in table[x].unique():
            mask = (table[x] == v)
            table.loc[mask, 'prop'] = table[mask]['count'] / table[mask]['count'].sum()

        prop_plot = tadpole.barplot(
            x = x,
            y = 'prop',
            hue = by,
            data = table,
        )
        res += (prop_plot,)


    if return_frame:
        res += (table,)

    return unpack_tuple(res)


def corr_plot(frame, figure_size = (20, 15), ax = None):
    """plot for correlation

    Args:
        frame (DataFrame): frame to draw plot
    Returns:
        Axes
    """
    corr = frame.corr()

    mask = np.zeros_like(corr, dtype = bool)
    mask[np.triu_indices_from(mask)] = True

    map_plot = tadpole.heatmap(
        corr,
        mask = mask,
        cmap = HEATMAP_CMAP,
        vmax = 1,
        vmin = -1,
        center = 0,
        square = True,
        cbar_kws = {"shrink": .5},
        linewidths = .5,
        annot = True,
        fmt = '.2f',
        figure_size = figure_size,
        ax = ax,
    )

    return map_plot


def proportion_plot(x = None, keys = None, ax = None):
    """plot for comparing proportion in different dataset

    Args:
        x (Series|list): series or list of series data for plot
        keys (str|list): keys for each data

    Returns:
        Axes
    """
    if not isinstance(x, list):
        x = [x]

    if keys is None:
        keys = [
            x[ix].name
            if hasattr(x[ix], 'name') and x[ix].name is not None
            else ix
            for ix in range(len(x))
        ]
    elif isinstance(keys, str):
        keys = [keys]

    x = map(pd.Series, x)
    data = pd.concat(x, keys = keys, names = ['keys']).reset_index()
    data = data.rename(columns = {data.columns[2]: 'value'})

    prop_data = data.groupby('keys')['value'].value_counts(
        normalize = True,
        dropna = False,
    ).rename('proportion').reset_index()

    prop_plot = tadpole.barplot(
        x = 'value',
        y = 'proportion',
        hue = 'keys',
        data = prop_data,
        ax = ax,
    )

    return prop_plot


def roc_plot(score, target, compare = None, figsize = (14, 10), ax = None):
    """plot for roc

    Args:
        score (array-like): predicted score
        target (array-like): true target
        compare (array-like): another score for comparing with score

    Returns:
        Axes
    """
    auc, fpr, tpr, thresholds = AUC(score, target, return_curve = True)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
    
    ax.plot(fpr, tpr, label = 'ROC curve (area = %0.5f)' % auc)
    ax.fill_between(fpr, tpr, alpha = 0.3)
    if compare is not None:
        c_aux, c_fpr, c_tpr, _ = AUC(compare, target, return_curve = True)
        ax.plot(c_fpr, c_tpr,label = 'ROC compare (area = %0.5f)' % c_aux)
        ax.fill_between(c_fpr, c_tpr, alpha = 0.3)

    ax.plot([0, 1], [0, 1], color = 'red', linestyle = '--')
    plt.legend(loc = "lower right")

    return ax

def ks_plot(score, target, figsize = (14, 10), ax = None):
    """plot for ks

    Args:
        score (array-like): predicted score
        target (array-like): true target
        compare (array-like): another score for comparing with score

    Returns:
        Axes
    """
    fpr, tpr, thresholds = roc_curve(target, score)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
    
    ax.plot(thresholds[1 : ], tpr[1 : ], label = 'tpr')
    ax.plot(thresholds[1 : ], fpr[1 : ], label = 'fpr')
    ax.plot(thresholds[1 : ], (tpr - fpr)[1 : ], label = 'ks')

    ax.invert_xaxis()
    ax.legend()

    ks_value = max(tpr - fpr)
    x = np.argwhere(abs(fpr - tpr) == ks_value)[0, 0]
    thred_value = thresholds[x]
    ax.axvline(thred_value, color = 'r', linestyle = '--')
    plt.title(f'ks:{ks_value:.5f}    threshold:{thred_value:.5f}')

    return ax

def bin_plot(frame, x = None, target = 'target', iv = True, annotate_format = ".2f", 
            return_frame = False, figsize = (12, 6), ax = None):
    """plot for bins

    Args:
        frame (DataFrame)
        x (str): column in frame that will be used as x axis
        target (str): target column in frame
        iv (bool): if need to show iv in plot
        annotate_format (str): format str for axis annotation of chart
        return_frame (bool): if need return bin frame
        figsize (tuple): size of the figure (width, height)

    Returns:
        Dataframe: contains good, bad, badrate, prop, y_prop, n_prop, woe, iv
    """
    frame = frame.copy()

    if not isinstance(target, str):
        temp_name = generate_str()
        frame[temp_name] = target
        target = temp_name
    
    table = feature_bin_stats(frame, x, target)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax = tadpole.barplot(
        x = x,
        y = 'prop',
        data = table,
        color = '#82C6E2',
        ax = ax,
    )

    ax = add_annotate(ax, format = annotate_format)

    badrate_ax = ax.twinx()
    badrate_ax.grid(False)

    badrate_ax = tadpole.lineplot(
        x = x,
        y = 'badrate',
        data = table,
        color = '#D65F5F',
        ax = badrate_ax,
    )

    badrate_ax.set_ylim([0, None])
    badrate_ax = add_annotate(badrate_ax, format = annotate_format)

    if iv:
        ax = reset_ylim(ax)
        ax = add_text(ax, 'IV: {:.5f}'.format(table['iv'].sum()))

    res = (ax,)
    
    if return_frame:
        res += (table,)

    return unpack_tuple(res)
