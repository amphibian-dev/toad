import numpy as np
import pandas as pd

from .impute import impute


ab = np.array(list('ABCDEFG'))
int_feat = np.random.randint(10, size = 500)
float_feat = np.random.rand(500)
str_feat = ab[np.random.choice(7, 500)]
uni_feat = np.ones(500)
# empty_feat = np.full(500, np.nan)

target = np.random.randint(2, size = 500)

df = pd.DataFrame({
    'A': int_feat,
    'B': str_feat,
    'C': uni_feat,
    'D': float_feat,
    # 'E': empty_feat,
})

mask = np.random.choice([True, False], size = 500 * 4, p = [0.95, 0.05]).reshape(500, 4)
df = df.where(mask, np.nan)


def test_impute_with_number():
    res = impute(df.drop(columns = 'B'))

    assert res.isna().sum().sum() == 0


def test_impute_with_str():
    res = impute(df)

    assert res.isna().sum().sum() == 0








