import pytest
import numpy as np
import pandas as pd

from .func import (
    np_unique,
    fillna,
    clip,
    diff_time_frame,
    bin_to_number,
    generate_target,
    generate_str,
    get_dummies,
    feature_splits,
)

np.random.seed(1)
feature = np.random.rand(500)
target = np.random.randint(2, size = 500)



def test_fillna():
    res = fillna(np.array([1, 2, 3, np.nan, 4, 5]))
    assert res[3] == -1


def test_np_unique():
    res = np_unique(np.array([np.nan, np.nan, np.nan]))
    assert len(res) == 1


def test_clip():
    res1 = clip(feature, quantile = (.05, .95))
    res2 = clip(feature, quantile = 0.05)
    assert np.testing.assert_array_equal(res1, res2) is None


def test_feature_splits():
    value = feature_splits(feature, target)
    assert len(value) == 243


def test_diff_time_frame():
    time_data = [
        {
            'base': '2018-01',
            'time1': '2018-04',
            'time2': '2018-04-02',
        },
        {
            'base': '2018-01',
            'time1': '2018-05',
            'time2': '2018-04-05',
        },
        {
            'base': '2018-02',
            'time1': '2018-04',
            'time2': '2018-04-10',
        },
    ]

    frame = pd.DataFrame(time_data)
    res = diff_time_frame(frame['base'], frame[['time1', 'time2']], format='%Y-%m-%d')
    assert res.iloc[0, 1] == 91

def test_bin_to_number():
    s = pd.Series([
        '1',
        '1-100',
        '-',
        '100-200',
        np.nan,
        '200-300',
        '300',
        '100-200',
        '>500',
    ])

    res = s.apply(bin_to_number())
    assert res[3] == 150

def test_bin_to_number_for_frame():
    df = pd.DataFrame([
        {
            'area_1': '100-200',
            'area_2': '150~200',
        },
        {
            'area_1': '300-400',
            'area_2': '200~250',
        },
        {
            'area_1': '200-300',
            'area_2': '450~500',
        },
        {
            'area_1': '100-200',
            'area_2': '250~300',
        },
    ])

    res = df.applymap(bin_to_number())
    assert res.loc[1, 'area_2'] == 225

def test_generate_target():
    t = generate_target(len(feature), rate = 0.3, weight = feature)
    rate = t.sum() / len(t)
    assert rate == 0.3

@pytest.fixture
def test_generate_str():
    s = generate_str(size = 8)
    assert s == 'EPL5MTQK'

def test_get_dummies_binary():
    ab = np.array(list('ABCDEFG'))
    df = pd.DataFrame({
        'binary': ab[np.random.choice(2, 500)],
        'multiple': ab[np.random.choice(5, 500)],
    })
    data = get_dummies(df, binary_drop = True)
    
    assert 'binary_A' not in data.columns
