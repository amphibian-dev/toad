import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .scorecard import ScoreCard, WOETransformer, Combiner

np.random.seed(1)

# Create a testing dataframe and a scorecard model.

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size = 500)
target = np.random.randint(2, size = 500)
str_feat = ab[np.random.choice(7, 500)]

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': ab[np.random.choice(2, 500)],
    'D': np.ones(500),
})

card_config = {
    'A': {
        '[-inf ~ 3)': 100,
        '[3 ~ 5)': 200,
        '[5 ~ 8)': 300,
        '[8 ~ inf)': 400,
        'nan': 500,
    },
    'B': {
        ','.join(list('ABCD')): 200,
        ','.join(list('EF')): 400,
        'else': 500,
    },
    'C': {
        'A': 200,
        'B': 100,
    },
}

combiner = Combiner()
bins = combiner.fit_transform(df, target, n_bins = 5)
woe_transer = WOETransformer()
woe = woe_transer.fit_transform(bins, target)

# create a score card
card = ScoreCard(
    combiner = combiner,
    transer = woe_transer,
)
card.fit(woe, target)


FUZZ_THRESHOLD = 1e-6
TEST_SCORE = pytest.approx(453.5702462572068, FUZZ_THRESHOLD)
TEST_PROBA = pytest.approx(0.4673322872985267, FUZZ_THRESHOLD)


def test_representation():
    repr(card)


def test_load():
    card = ScoreCard().load(card_config)
    score = card.predict(df)
    assert score[200] == 600


def test_load_after_init_combiner():
    card = ScoreCard(
        combiner = combiner,
        transer = woe_transer,
    )
    card.load(card_config)
    score = card.predict(df)
    assert score[200] == 600


def test_proba_to_score():
    model = LogisticRegression()
    model.fit(woe, target)

    proba = model.predict_proba(woe)[:, 1]
    score = card.proba_to_score(proba)
    assert score[404] == TEST_SCORE


def test_score_to_prob():
    score = card.predict(df)
    proba = card.score_to_proba(score)
    assert proba[404] == TEST_PROBA


def test_predict():
    score = card.predict(df)
    assert score[404] == TEST_SCORE


def test_predict_proba():
    proba = card.predict_proba(df)
    assert proba[404, 1] == TEST_PROBA


def test_card_feature_effect():
    """
    verify the `base effect of each feature` is consistent with assumption
    FEATURE_EFFECT is manually calculated with following logic:
    FEATURE_EFFECT = np.median(card.woe_to_score(df),axis = 0)
    """
    FEATURE_EFFECT = pytest.approx(np.array([142.26368948220417, 152.82747912111066, 148.82665746001695, 0.]), FUZZ_THRESHOLD)
    assert card.base_effect.values == FEATURE_EFFECT


def test_predict_sub_score():
    score, sub = card.predict(df, return_sub=True)
    assert sub.loc[250, 'B'] == pytest.approx(162.09822360428146, FUZZ_THRESHOLD)


def test_woe_to_score():
    score = card.woe_to_score(woe)
    score = np.sum(score, axis=1)
    assert score[404] == TEST_SCORE


def test_bin_to_score():
    score = card.bin_to_score(bins)
    assert score[404] == TEST_SCORE


def test_export_map():
    card_map = card.export()
    assert card_map['B']['D'] == 159.26


def test_card_map():
    config = card.export()
    card_from_map = ScoreCard().load(config)
    score = card_from_map.predict(df)
    assert score[404] == 453.57


def test_card_map_with_else():
    card_from_map = ScoreCard().load(card_config)
    score = card_from_map.predict(df)
    assert score[80] == 1000


def test_generate_testing_frame():
    card = ScoreCard().load(card_config)
    frame = card.testing_frame()
    assert frame.loc[4, 'B'] == 'E'


def test_export_frame():
    card = ScoreCard().load(card_config)
    frame = card.export(to_frame=True)
    rows = frame[(frame['name'] == 'B') & (frame['value'] == 'else')].reset_index()
    assert rows.loc[0, 'score'] == 500


def test_card_combiner_number_not_match():
    c = combiner.export()
    c['A'] = [0, 3, 6, 8]
    com = Combiner().load(c)
    bins = com.transform(df)
    woe_transer = WOETransformer()
    woe = woe_transer.fit_transform(bins, target)

    card = ScoreCard(
        combiner=com,
        transer=woe_transer,
    )

    with pytest.raises(Exception) as e:
        # will raise an exception when fitting a card
        card.fit(woe, target)

    assert '\'A\' is not matched' in str(e.value)


def test_card_combiner_str_not_match():
    c = combiner.export()
    c['C'] = [['A'], ['B'], ['C']]
    com = Combiner().load(c)
    bins = com.transform(df)
    woe_transer = WOETransformer()
    woe = woe_transer.fit_transform(bins, target)

    card = ScoreCard(
        combiner=com,
        transer=woe_transer,
    )

    with pytest.raises(Exception) as e:
        # will raise an exception when fitting a card
        card.fit(woe, target)

    assert '\'C\' is not matched' in str(e.value)


def test_card_with_less_X():
    x = woe.drop(columns='A')
    card = ScoreCard(
        combiner=combiner,
        transer=woe_transer,
    )

    card.fit(x, target)
    assert card.predict(df)[200] == pytest.approx(457.5903160102142, FUZZ_THRESHOLD)


def test_card_predict_with_unknown_feature():
    np.random.seed(9)
    unknown_df = df.copy()
    unknown_df.loc[200, 'C'] = 'U'
    assert card.predict(unknown_df)[200] == pytest.approx(456.41288777297257, FUZZ_THRESHOLD)


def test_card_predict_with_unknown_feature_default_max():
    np.random.seed(9)
    unknown_df = df.copy()
    unknown_df.loc[200, 'C'] = 'U'
    score, sub = card.predict(unknown_df, default = 'max', return_sub = True)

    assert sub.loc[200, 'C'] == card['C']['scores'].max()
    assert score[200] == pytest.approx(462.2871531373114, FUZZ_THRESHOLD)


def test_card_predict_with_unknown_feature_default_with_value():
    np.random.seed(9)
    unknown_df = df.copy()
    unknown_df.loc[200, 'C'] = 'U'
    score, sub = card.predict(unknown_df, default = 42, return_sub = True)
    
    assert sub.loc[200, 'C'] == 42
    assert score[200] == pytest.approx(355.46049567729443, FUZZ_THRESHOLD)


def test_get_reason_vector():
    """
    verify the score reason of df is consistent with assumption
    DF_REASON is manually calculated with following logic:
    if score is lower than base_odds, select top k feature with lowest subscores where their corresponding  subscores are lower than the base effect of features.
    if score is higher than base_odds, select top k feature with highest subscores where their corresponding  subscores are higher than the base effect of features.

    e.g. xx.iloc[404]
    sub_scores:  151    159 143 0
    base_effect: 142    153 149 0
    diff_effect:  +9     +6  -6 0

    total_score: 453(151+159+143+0) > base_odds(35)
        which is larger than base, hence, we try to find top `keep` features who contributed most to positivity
    find_largest_top_3:  A(+9) B(+6) D(+0)
    """
    reason = card.get_reason(df)
    assert reason.iloc[404]['top1'].tolist() == ['C', pytest.approx(142.9523920956781, FUZZ_THRESHOLD), 'B']


@pytest.mark.timeout(0.007)
def test_predict_dict():
    """ a test for scalar inference time cost """
    proba = card.predict(df.iloc[404].to_dict())
    assert proba == TEST_SCORE

