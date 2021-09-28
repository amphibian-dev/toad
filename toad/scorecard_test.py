import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .scorecard import ScoreCard, WOETransformer, Combiner

np.random.seed(1)

# Create a testing dataframe and a scorecard model.

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size=500)
target = np.random.randint(2, size=500)
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
bins = combiner.fit_transform(df, target, n_bins=5)
woe_transer = WOETransformer()
woe = woe_transer.fit_transform(bins, target)

# create a score card
card = ScoreCard(
    combiner=combiner,
    transer=woe_transer,
)
card.fit(woe, target)

# create a list of wide dict and a scorecard for scalar-loop inference
samples_in_wide_dict = [{
    'some_platform': 'android', 'some_check': 0, 'some_num': 8, 'some_max_num': '1516800',
    'some_min_num': '499200', 'some_capacity': 2600, 'some_c_status': 0, 'some_brand': 'brand180',
    'some_status': -1, 'some_abc_status': -1, 'some_type': 'type0', 'some_check_2': 0, 'some_check_3': 0,
    'some_check_4': 0, 'some_click': None, 'some_check_5': 0, 'province': '四川',
    'cate': 'cate_v', 'pre1': '192', 'width': 1080, 'height': 1920, 'num1': 3, 'num2': 54,
    'cate_2': 'cate2_v', 'weeks': 0, 'days': 141, 'pre1_2': '192',
    'cate_3': 'cate3_g,cate3_h', 'cate_4': 'others'
}]
card_wide_config = {'some_capacity': {'[-inf ~ 1000)': 10.16, '[1000 ~ 1390)': -22.21, '[1390 ~ inf)': 6.3},
                    'some_c_status': {'[-inf ~ 0)': -11.43, '[0 ~ 1)': 9.28, '[1 ~ inf)': -4.35},
                    'some_num': {'[-inf ~ 3)': 1.45, '[3 ~ 5)': 7.88, '[5 ~ inf)': 2.9},
                    'some_max_num': {'1804800,!notset!,2000000,1954000,2301000,1708800,nan,1989000,2001000,'
                                     '1844000,1882000,1843200,1785600,1508000,1820000,1863000,2045000,1401600,'
                                     '1958400,2500000,1700000': 0.5,
                                     '1709000,1697000,1209600,1690000,1300000,1401000,1794000,1800000,1950000,'
                                     '1612800,2002000,1805000, '
                                     '1766400,1497600,2016000,1200000,1001000,1391000,1547000,1516800,0,1593600,'
                                     '1900800,1495000,1190400, '
                                     '1459200,1363200,1267200,1500000,1440000,1305600,1352000,2188800,1807000,'
                                     '1512000,2208000,2457600, '
                                     '2600000,2300000,2210000,2340000,1898000,1555200,1586000,1608000,1651200,'
                                     '1904200,1536000,1404000,1344000': 7.11,
                                     '2400000,1416000,1996800,2465600,2265600': 16.32},
                    'some_min_num': {'500000,!notset!,900000,554000,830000': 17.15,
                                     '541000,nan,793000,509000,633600,614400,400000,338000,884000': 7.52,
                                     '200000,960000,533000,156000,442000,299000,800000,652800,774000,480000,247000,'
                                     '768000,300000,2465600,221000,286000,403200,307200,403000,384000,0,787200, '
                                     '1200000,449000,208000,468000,1000000,126000,850000,2300000,249000,303400,'
                                     '408000,455000,460800,624000,497250,604500,600000,1100000,546000,533333,1950000,'
                                     '1600000': -20.4},
                    'some_brand': {
                        'brand1,1,brand2,brand3,2,brand4,1,brand5,brand6,1,brand7,8,brand8,brand9,2,brand10,'
                        'brand11,4,brand12,2,brand13,2, '
                        'brand14,6,brand15,3,brand16,4,brand17,1,brand18,2,brand19,5,brand20,brand21,brand22,2,'
                        'brand23,1,brand24,3,brand25,3, '
                        'brand26,brand27,3,brand28,6,brand29,brand30,brand31,3,brand32,2,brand33,7,brand34,5,brand35,'
                        'brand36,8,brand37,6,brand38,6': 11.05,
                        'brand39,brand40,brand41,5,brand42,4,brand43,4,brand44,brand45,brand46,4,brand47,'
                        'brand48,1,brand49,brand50,1,brand51, '
                        'brand52,brand53,brand54,1,brand55,brand56,1,brand57,11,brand58,brand59,brand60,brand61,'
                        'brand62,11,brand63,brand64, '
                        'brand65,brand66,brand67,brand68,brand69,1,brand70,brand71,brand72,'
                        'brand73,brand74,brand75,brand76, '
                        'brand77,brand78,2,brand79,brand80,brand81,brand82,brand83,4,brand84,brand85,2,brand86,brand87,'
                        'brand88,4,brand89,7,brand90,6,brand91, '
                        'brand92,3,brand93,brand94,brand95,brand96,brand97,brand98,brand99,brand100,brand101,3,'
                        'brand102, '
                        'brand103,brand104,brand105,brand106,9,brand107,brand108,brand109,1, '
                        'brand110,3,brand111,brand112,brand113,brand114,brand115,brand116,3,brand117,brand118,'
                        'brand119,3,brand120,11,brand121,1,brand122,brand123,brand124,4, '
                        'brand125,5_G,brand126,12,brand127,brand128,2,brand129,brand130,2,brand131,6,brand132,'
                        'brand133,brand134,brand135,brand136, '
                        'brand137,brand138,brand139,brand140,4,brand141,brand142,brand143,brand144, '
                        'brand145,1,brand146,brand147,brand148,brand149,brand150,8,brand151,brand152,brand153,'
                        'brand154,brand155,brand156,brand157, '
                        'brand158,brand159,brand160,brand161,brand162,brand163,brand164, '
                        'brand165,brand166,brand167,brand168,brand169,brand170,brand171,brand172,brand173,brand174,'
                        'brand175,brand176, '
                        'brand177,brand178,brand179,brand180,brand181,5,brand182,4, '
                        'brand183,7,brand184,brand185,brand186,brand187,brand188,brand189,brand190,brand191,brand192,'
                        'brand193,brand194,brand195,brand196, '
                        'brand197,brand198,brand199,brand200,brand201': -17.97,
                        'brand202,brand203,brand204,brand205,brand206,brand207,brand208,brand209': -68.06},
                    'some_platform': {'p1': 9.08, 'p2': 1.76},
                    'province': {'省份1,省份2,nan,省份3,省份4,省份5,省份6,省份7,省份8,省份9,省份10': 17.21,
                                 '四川,省份11,省份12,省份13': 4.62,
                                 '维也纳州,庆和省,突尼斯,省份14,省份15': -8.29},
                    'some_type': {'type1,type2,type3,nan,type0': 8.24,
                                  'type4,type5,unknown': -5.89, 'type6,type7,type8,type9,type10,type11,type12,type13,'
                                                                'type14,type15,type16,type17,type18,type19': -19.47},
                    'some_check_2': {'[-inf ~ 1)': 8.46, '[1 ~ inf)': -21.87},
                    'some_click': {'!notset!': 29.97, 'nan': 1.19, 'cka,777': -45.73},
                    'cate': {'cate1,cate2': 22.04, 'NAN,cate3': 3.28,
                             'cate_v,cate4': -4.09},
                    'pre1': {'192,NAN': 4.26, 'pre1_a,other': 2.0, '10,172': -2.94},
                    'height': {'[-inf ~ 2250)': -9.74, '[2250 ~ 3840)': 25.77, '[3840 ~ inf)': -39.88},
                    'num1': {'[-inf ~ 16)': -6.08,
                             '[16 ~ 1005)': 21.89, '[1005 ~ inf)': -38.07},
                    'num2': {'[-inf ~ 125)': 0.81, '[125 ~ 127)': -49.99, '[127 ~ inf)': 18.22},
                    'cate_2': {'cate2_a': 9.77, 'cate2_b,cate2_v': 6.3,
                               'cate2_c,NAN': -36.43},
                    'weeks': {'[-inf ~ 1.0)': 4.0, '[1.0 ~ 5.0)': -3.43, '[5.0 ~ inf)': -8.76, 'nan': 9.01},
                    'days': {'[-inf ~ 2.0)': -10.06, '[2.0 ~ 107.0)': -0.66, '[107.0 ~ inf)': 32.27, 'nan': -15.78},
                    'pre1_2': {'pre1_a,192,NAN': 9.23, 'pre1_b,10': -11.83, '172': -31.65},
                    'cate_3': {'cate3_a,cate3_b,cate3_c': -3.17, 'np': 0.37,
                               'cate3_d,cate3_e,cate3_f,NAN,cate3_g,cate3_h,0,0,bg': 12.66},
                    'cate_4': {'cate4_a,cate4_b,cate4_c,cate4_d': 2.16, 'others,NAN': 2.21, 'cate4_e': 2.33}}


card_wide = ScoreCard().load(card_wide_config)
card_wide.base_effect_of_features = pd.Series(1, index=card_wide.features_)

FUZZ_THRESHOLD = 1e-4
TEST_SCORE = pytest.approx(453.58, FUZZ_THRESHOLD)


def test_representation():
    repr(card)


def test_load():
    card = ScoreCard().load(card_config)
    score = card.predict(df)
    assert score[200] == 600


def test_load_after_init_combiner():
    card = ScoreCard(
        combiner=combiner,
        transer=woe_transer,
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
    assert proba[404] == 0.4673929989138551


def test_predict():
    score = card.predict(df)
    assert score[404] == TEST_SCORE


def test_predict_proba():
    proba = card.predict_proba(df)
    assert proba[404, 1] == 0.4673929989138551


def test_card_feature_effect():
    """
    verify the `base effect of each feature` is consistent with assumption
    FEATURE_EFFECT is manually calculated with following logic:
    FEATURE_EFFECT = np.median(card.woe_to_score(df),axis = 0)
    """
    FEATURE_EFFECT = pytest.approx(np.array([142.26722434, 152.81922244, 148.82801326, 0.]), FUZZ_THRESHOLD)
    assert card.base_effect_of_features.values == FEATURE_EFFECT


def test_predict_sub_score():
    score, sub = card.predict(df, return_sub=True)
    assert sub.loc[250, 'B'] == pytest.approx(162.0781460573475, FUZZ_THRESHOLD)


def test_woe_to_score():
    score = card.woe_to_score(woe)
    score = np.sum(score, axis=1)
    assert score[404] == TEST_SCORE


def test_bin_to_score():
    score = card.bin_to_score(bins)
    assert score[404] == TEST_SCORE


def test_export_map():
    card_map = card.export()
    assert card_map['B']['D'] == 159.24


def test_card_map():
    config = card.export()
    card_from_map = ScoreCard().load(config)
    score = card_from_map.predict(df)
    assert score[404] == TEST_SCORE


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
    assert card.predict(x)[200] == pytest.approx(411.968588097131, FUZZ_THRESHOLD)


sub_df_for_vector = df.iloc[[404, 410]]


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
    reason = card.get_reason(sub_df_for_vector)
    # list of tuple
    # the list has length `keep`, which means the top `keep` features who contributed most
    # the tuple means tuple of <feature_name, sub_score, raw_value>
    assert reason.iloc[0]['reason'] == [('A', '+151.4', 3), ('B', '+159.2', 'D'), ('D', '+0.0', 1.0)]


rows_for_scalar = df.iloc[[404]].to_dict(orient='records')  # use list for scalar-loop


def test_get_reason_scalar():
    reason = card.get_reason(rows_for_scalar)
    # list of list of tuple
    # the outer-most list has length batch_size
    # the list-in-middle has length `keep`, which means the top `keep` features who contributed most
    # the inner tuple means tuple of <feature_name, sub_score, raw_value>
    assert reason == [[('A', '+151.4', 3), ('B', '+159.2', 'D'), ('D', '+0.0', 1.0)]]


@pytest.mark.timeout(0.061)
def test_predict_vector_wide():
    """ a test for vector inference time cost """
    # prepare wide dataframe for vector inference
    df_wide = pd.DataFrame(samples_in_wide_dict)
    proba = card_wide.predict(df_wide)


@pytest.mark.timeout(0.007)
def test_predict_scalar_wide():
    """ a test for scalar inference time cost """
    proba = card_wide.predict(samples_in_wide_dict[0])


@pytest.mark.timeout(0.060)
def test_get_reason_vector_wide():
    """ a test for vector inference time cost """
    # prepare wide dataframe for vector inference
    df_wide = pd.DataFrame(samples_in_wide_dict)
    reason = card_wide.get_reason(df_wide)


@pytest.mark.timeout(0.005)
def test_get_reason_scalar_wide():
    """ a test for scalar inference time cost """
    reason = card_wide.get_reason(samples_in_wide_dict)
    assert True


def test_empty_predict():  # TODO
    rows = []
    ...

    X = pd.DataFrame(data=[])
    ...
