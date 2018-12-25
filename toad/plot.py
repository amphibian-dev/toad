import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties

FONTS_PATH = os.path.join(os.path.abspath(__file__), '../../fonts/pingfang.ttf')
myfont = FontProperties(fname = os.path.abspath(FONTS_PATH))
sns.set(font = myfont.get_family())


def badrate_plot(frame, x = None, target = 'target', by = None, freq = None, format = None):
    frame = frame.copy()

    if freq is not None:
        frame.loc[:, x] = pd.to_datetime(frame[x], format = format)
        grouper = pd.Grouper(key = x, freq = freq)

    if by is not None:
        grouper = [by, grouper]

    group = frame.groupby(grouper)
    table = group[target].agg(['sum', 'count']).reset_index()
    table['badrate'] = table['sum'] / table['count']

    return sns.relplot(
        x = x,
        y = 'badrate',
        hue = by,
        markers = True,
        kind = 'line',
        data = table,
    )
