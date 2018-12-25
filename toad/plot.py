import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties

from .utils import unpack_tuple

FONTS_PATH = os.path.join(os.path.abspath(__file__), '../../fonts/pingfang.ttf')
myfont = FontProperties(fname = os.path.abspath(FONTS_PATH))
sns.set(font = myfont.get_family())


def badrate_plot(frame, x = None, target = 'target', by = None,
                freq = None, format = None, return_counts = False, return_frame = False):
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
