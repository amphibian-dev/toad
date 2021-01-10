import os
import seaborn as sns
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

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
    if axes.get_legend() is not None:
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


def reset_title(axes):
    title = axes.get_title()
    
    if title:
        axes.set_title(title, fontproperties = myfont)
    
    return axes


def reset_xylabels(axes):
    y_label = axes.get_ylabel()
    if y_label:
        axes.set_ylabel(y_label, fontproperties = myfont)
    
    x_label = axes.get_xlabel()
    if x_label:
        axes.set_xlabel(x_label, fontproperties = myfont)
    
    return axes


def reset_ylim(axes):
    # for axes and twins
    for ax in axes.figure.axes:
        if ax.bbox.bounds == axes.bbox.bounds:
            bottom, top = ax.get_ylim()
            top += (top - bottom) * 0.1
            ax.set_ylim(bottom, top)

    return axes


def fix_axes(axes):
    if not isinstance(axes, Axes):
        return axes

    functions = [reset_title, reset_xylabels, reset_ticklabels, reset_legend]

    for func in functions:
        func(axes)
    return axes

def tadpole_axes(fn):
    @wraps(fn)
    def func(*args, **kwargs):
        res = fn(*args, **kwargs)

        if not isinstance(res, tuple):
            return fix_axes(res)

        r = tuple()
        for i in res:
            r += (fix_axes(i),)

        return r

    return func



def annotate(ax, x, y, space = 5, format = ".2f"):
    """
    """
    va = 'bottom'

    if y < 0:
        space *= -1
        va = 'top'

    ax.annotate(
        ("{:"+ format +"}").format(y),
        (x, y),
        xytext = (0, space),
        textcoords = "offset points",
        ha = 'center',
        va = va,
    )



def add_bar_annotate(ax, **kwargs):
    """
    """
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        annotate(ax, x_value, y_value, **kwargs)

    return ax


def add_line_annotate(ax, **kwargs):
    """
    """
    for line in ax.lines:
        points = line.get_xydata()

        for point in points:
            annotate(ax, point[0], point[1], **kwargs)

    return ax


def add_annotate(ax, **kwargs):
    if len(ax.lines) > 0:
        add_line_annotate(ax, **kwargs)

    if len(ax.patches) > 0:
        add_bar_annotate(ax, **kwargs)

    return ax


def add_text(ax, text, loc = 'top left', offset = (0.01, 0.04)):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_offset = (x_max - x_min) * offset[0]
    y_offset = (y_max - y_min) * offset[1]

    if loc == 'top left':
        loc = (x_min + x_offset, y_max - y_offset)
    elif loc == 'top right':
        loc = (x_max - x_offset, y_max - y_offset)
    elif loc == 'bottom left':
        loc = (x_min + x_offset, y_min + y_offset)
    elif loc == 'bottom right':
        loc = (x_max - x_offset, y_min + y_offset)

    ax.text(*loc, text, fontsize = 'x-large')

    return ax
