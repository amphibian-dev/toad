#!/usr/bin/python

"""Command line tools for detecting csv data

Team: ESC

Examples:

    python detector.py -i xxx.csv -o report.csv

"""

import pandas as pd

def getTopValues(series, top = 5, reverse = False):
    """Get top/bottom n values

    Args:
        series (Series): data series
        top (number): number of top/bottom n values
        reverse (bool): it will return bottom n values if True is given

    Returns:
        Series: Series of top/bottom n values and percentage. ['value:percent', None]
    """
    itype = 'top'
    counts = series.value_counts()
    counts = list(zip(counts.index, counts, counts.divide(series.size)))

    if reverse:
        counts.reverse()
        itype = 'bottom'

    template = "{0[0]}:{0[2]:.2%}"
    indexs = [itype + str(i + 1) for i in range(top)]
    values = [template.format(counts[i]) if i < len(counts) else None for i in range(top)]

    return pd.Series(values, index = indexs)


def getDescribe(series, percentiles = [.25, .5, .75]):
    """Get describe of series

    Args:
        series (Series): data series
        percentiles: the percentiles to include in the output

    Returns:
        Series: the describe of data include mean, std, min, max and percentiles
    """
    d = series.describe(percentiles)
    return d.drop('count')


def countBlank(series, blanks = [None]):
    """Count number and percentage of blank values in series

    Args:
        series (Series): data series
        blanks (list): list of blank values

    Returns:
        number: number of blanks
        str: the percentage of blank values
    """
    # n = 0
    # counts = series.value_counts()
    # for blank in blanks:
    #     if blank in counts.keys():
    #         n += counts[blank]

    n = series.isnull().sum()

    return (n, "{0:.2%}".format(n / series.size))


def isNumeric(series):
    """Check if the series's type is numeric

    Args:
        series (Series): data series

    Returns:
        bool
    """
    return series.dtype.kind in 'ifc'


def detect(dataframe):
    """ Detect data

    Args:
        dataframe (DataFrame): data that will be detected

    Returns:
        DataFrame: report of detecting
    """

    rows = []
    for name, series in dataframe.items():
        numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
        discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2', 'bottom1']

        details_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
        details = []

        if isNumeric(series):
            desc = getDescribe(
                series,
                percentiles = [.01, .1, .5, .75, .9, .99]
            )
            details = desc.tolist()
        else:
            top5 = getTopValues(series)
            bottom5 = getTopValues(series, reverse = True)
            details = top5.tolist() + bottom5[::-1].tolist()

        # print(details_index)
        nblank, pblank = countBlank(series)

        row = pd.Series(
            index = ['type', 'size', 'missing', 'unique'] + details_index,
            data = [series.dtype, series.size, pblank, series.nunique()] + details
        )

        row.name = name
        rows.append(row)

    return pd.DataFrame(rows)
