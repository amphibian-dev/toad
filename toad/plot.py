import numpy as np
import pandas as pd

from .stats import IV
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


def corr_plot(frame, figure_size = (20, 15)):
    """plot for correlation

    Args:
        frame (DataFrame): frame to draw plot
    Returns:
        Axes
    """
    corr = frame.corr()

    mask = np.zeros_like(corr, dtype = np.bool)
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
    )

    return map_plot


def proportion_plot(x = None, keys = None):
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
    )

    return prop_plot


def roc_plot(score, target, compare = None):
    """plot for roc

    Args:
        score (array-like): predicted score
        target (array-like): true target
        compare (array-like): another score for comparing with score

    Returns:
        Axes
    """
    auc, fpr, tpr, thresholds = AUC(score, target, return_curve = True)

    ax = tadpole.lineplot(
        x = fpr,
        y = tpr,
    )
    ax.fill_between(fpr, tpr, alpha = 0.3)
    ax = add_text(ax, 'AUC: {:.5f}'.format(auc))

    if compare is not None:
        c_aux, c_fpr, c_tpr, _ = AUC(compare, target, return_curve = True)
        ax.plot(c_fpr, c_tpr)
        ax.fill_between(c_fpr, c_tpr, alpha = 0.3)

    ax.plot([0, 1], [0, 1], color = 'red', linestyle = '--')

    return ax


def bin_plot(frame, x = None, target = 'target', iv = True, annotate_format = ".2f"):
    """plot for bins

    Args:
        frame (DataFrame)
        x (str): column in frame that will be used as x axis
        target (str): target column in frame
        iv (bool): if need to show iv in plot
        annotate_format (str): format str for axis annotation of chart

    Returns:
        Axes: bins' proportion and badrate plot
    """
    frame = frame.copy()

    if not isinstance(target, str):
        temp_name = generate_str()
        frame[temp_name] = target
        target = temp_name
    
    
    group = frame.groupby(x)

    table = group[target].agg(['sum', 'count']).reset_index()
    table['badrate'] = table['sum'] / table['count']
    table['prop'] = table['count'] / table['count'].sum()

    prop_ax = tadpole.barplot(
        x = x,
        y = 'prop',
        data = table,
        color = '#82C6E2',
    )

    prop_ax = add_annotate(prop_ax, format = annotate_format)

    badrate_ax = prop_ax.twinx()
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
        prop_ax = reset_ylim(prop_ax)
        prop_ax = add_text(prop_ax, 'IV: {:.5f}'.format(IV(frame[x],frame[target])))

    return prop_ax
