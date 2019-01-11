import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from .utils import unpack_tuple

sns.set_palette('muted')

FONTS_PATH = os.path.join(os.path.abspath(__file__), '../../fonts/pingfang.ttf')
myfont = FontProperties(fname = os.path.abspath(FONTS_PATH))
sns.set(font = myfont.get_family())

HEATMAP_CMAP = sns.diverging_palette(240, 10, as_cmap = True)


def badrate_plot(frame, x = None, target = 'target', by = None,
                freq = None, format = None, return_counts = False, return_frame = False):
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

    """
    frame = frame.copy()

    grouper = x
    if freq is not None:
        frame.loc[:, x] = pd.to_datetime(frame[x], format = format)
        grouper = pd.Grouper(key = x, freq = freq)

    if by is not None:
        grouper = [by, grouper]

    group = frame.groupby(grouper)
    table = group[target].agg(['sum', 'count']).reset_index()
    table['badrate'] = table['sum'] / table['count']

    rate_plot = sns.relplot(
        x = x,
        y = 'badrate',
        hue = by,
        markers = True,
        kind = 'line',
        data = table,
        aspect = 2,
    )

    res = (rate_plot,)

    if return_counts:
        count_plot = sns.catplot(
            x = x,
            y = 'count',
            kind = 'bar',
            hue = by,
            data = table,
            aspect = 2,
        )
        res += (count_plot,)

    if return_frame:
        res += (table,)

    return unpack_tuple(res)


def corr_plot(frame):
    """plot for correlation

    Args:
        frame (DataFrame): frame to draw plot
    """
    corr = frame.corr()

    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    _, ax = plt.subplots(figsize = (20, 15))

    sns.heatmap(
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
        ax = ax,
    )

    return ax
