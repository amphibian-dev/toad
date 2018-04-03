from scipy import stats
import numpy as np
import pandas as pd


def KS(score, target, bucket = 10):
    df = pd.DataFrame({
        'score': score,
        'bad': target,
    })

    df['good'] = 1 - df['bad']
    df['bucket'] = pd.qcut(df['score'], bucket, duplicates = 'drop')
    grouped = df.groupby('bucket', as_index = False)

    agg1 = pd.DataFrame()
    agg1['min'] = grouped.min()['score']
    agg1['max'] = grouped.max()['score']
    agg1['bads'] = grouped.sum()['bad']
    agg1['goods'] = grouped.sum()['good']
    agg1['total'] = agg1['bads'] + agg1['goods']

    agg2 = (agg1.sort_values(by = 'min')).reset_index(drop = True)
    agg2['bad_rate'] = (agg2['bads'] / agg2['total']).apply('{0:.2%}'.format)

    agg2['ks'] = np.round((agg2['bads'] / agg2['bads'].sum()).cumsum() - (agg2['goods'] / agg2['goods'].sum()).cumsum(), 4) * 100

    return agg2

def gini(target):
    """get gini index of a feature
    """
    target = pd.Series(target)

    return 1 - ((target.value_counts() / target.size) ** 2).sum()

def gini_cond(dataframe, feature = "feature", target = "target"):
    """get conditional gini index of a feature
    """
    size = dataframe[feature].size

    value = 0
    for v, c in dataframe[feature].value_counts().iteritems():
        target_series = dataframe[dataframe[feature] == v][target]
        value += c / size * gini(target_series)

    return value


def entropy(target):
    """get infomation entropy of a feature
    """
    target = pd.Series(target)
    prob = target.value_counts() / target.size
    entropy = stats.entropy(prob)
    return entropy

def entropy_cond(dataframe, feature = "feature", target = "target"):
    """get conditional entropy of a feature
    """
    size = dataframe[feature].size

    value = 0
    for v, c in dataframe[feature].value_counts().iteritems():
        target_series = dataframe[dataframe[feature] == v][target]
        value += c/size * entropy(target_series)

    return value

def WOE(y_prob, n_prob):
    """get WOE of a group

    Args:
        y_prob: the probability of grouped y in total y
        n_prob: the probability of grouped n in total n
    """
    return np.log(y_prob / n_prob)




def IV(dataframe, feature = "feature", target = "target"):
    """get IV of a feature
    """
    t_counts = dataframe[target].value_counts()

    value = 0
    for v, c in dataframe[feature].value_counts().iteritems():
        f_counts = dataframe[dataframe[feature] == v][target].value_counts()

        y_prob = f_counts.get(1, default = 1) / t_counts[1]
        n_prob = f_counts.get(0, default = 1) / t_counts[0]

        value += (y_prob - n_prob) * WOE(y_prob, n_prob)

    return value


def quality(dataframe, target = 'target'):
    """get quality of features in data

    Returns:
        dataframe
    """
    rows = []
    for column in dataframe:
        c = dataframe[column].nunique()

        if c / dataframe[column].size > 0.5:
            print(column + ': --')
            continue

        iv = IV(dataframe, feature = column, target = target)
        g = gini_cond(dataframe, feature = column, target = target)
        e = entropy_cond(dataframe, feature = column, target = target)


        row = pd.Series(
            index = ['iv', 'gini', 'entropy', 'unique'],
            data = [iv, g, e, c],
        )

        row.name = column
        rows.append(row)

    return pd.DataFrame(rows)
