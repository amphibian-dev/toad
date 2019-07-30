import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from .utils import unpack_tuple, generate_str

sns.set_palette('muted')

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
FONT_FILE = 'NotoSansCJKsc-Regular.otf'
FONTS_PATH = os.path.join(CURRENT_PATH, 'fonts', FONT_FILE)
myfont = FontProperties(fname = os.path.abspath(FONTS_PATH))
sns.set(font = myfont.get_family())

HEATMAP_CMAP = sns.diverging_palette(240, 10, as_cmap = True)
MAX_STYLE = 6
FIG_SIZE = (12, 6)

def get_axes(size = FIG_SIZE):
    _, ax = plt.subplots(figsize = size)
    return ax

def reset_legend(axes):
    axes.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        framealpha = 0,
        prop = myfont,
    )

    return axes

def reset_ticklabels(axes):
    labels = []
    if axes.get_xticklabels():
        labels += axes.get_xticklabels()

    if axes.get_yticklabels():
        labels += axes.get_yticklabels()

    for label in labels:
        label.set_fontproperties(myfont)

    return axes

def fix_axes(axes):
    functions = [reset_ticklabels, reset_legend]

    for func in functions:
        func(axes)
    return axes


class Tadpole:
    def __getattr__(self, name):
        t = getattr(sns, name)
        if callable(t):
            return self.wrapsns(t)

        return t

    def wrapsns(self, f):
        def wrapper(*args, figure_size = FIG_SIZE, **kwargs):
            kw = kwargs.copy()
            if 'ax' not in kw:
                kw['ax'] = get_axes(size = figure_size)

            try:
                a = f(*args, **kw)
                a = fix_axes(a)
                return a
            except:
                return f(*args, **kwargs)

        return wrapper


tpl = Tadpole()


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


    rate_plot = tpl.lineplot(
        x = x,
        y = 'badrate',
        hue = by,
        style = by,
        data = table,
        legend = 'full',
        markers = markers,
        dashes = False,
    )
    res = (rate_plot,)

    if return_counts:
        count_plot = tpl.barplot(
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

        prop_plot = tpl.barplot(
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
    """
    corr = frame.corr()

    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    map_plot = tpl.heatmap(
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
