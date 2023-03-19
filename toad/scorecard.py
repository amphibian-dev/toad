import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .transform import WOETransformer, Combiner
from .utils import to_ndarray, bin_by_splits, save_json, read_json
from .utils.mixin import RulesMixin, BinsMixin


NUMBER_EMPTY = -9999999
NUMBER_INF = 1e10
FACTOR_EMPTY = 'MISSING'
FACTOR_UNKNOWN = 'UNKNOWN'



class ScoreCard(BaseEstimator, RulesMixin, BinsMixin):
    def __init__(self, pdo = 60, rate = 2, base_odds = 35, base_score = 750,
        card = None, combiner = {}, transer = None, **kwargs):
        """

        Args:
            combiner (toad.Combiner)
            transer (toad.WOETransformer)
        """
        self.pdo = pdo
        self.rate = rate
        self.base_odds = base_odds
        self.base_score = base_score

        self.factor = pdo / np.log(rate)
        self.offset = base_score - self.factor * np.log(base_odds)

        self._combiner = combiner
        self.transer = transer
        self.model = LogisticRegression(**kwargs)

        self._feature_names = None
        # keep track of median-effect of each feature during fit(), as `self.base_effect`
        # for reason-calculation later during predict()
        self.base_effect = None

        self.card = card
        if card is not None:
            # self.generate_card(card = card)
            import warnings
            warnings.warn(
                """`ScoreCard(card = {.....})` will be deprecated soon,
                    use `ScoreCard().load({.....})` instead!
                """,
                DeprecationWarning,
            )

            self.load(card)
        
    @property
    def coef_(self):
        """ coef of LR model
        """
        return self.model.coef_[0]
    
    @property
    def intercept_(self):
        return self.model.intercept_[0]
    
    @property
    def n_features_(self):
        return (self.coef_ != 0).sum()
    
    @property
    def features_(self):
        if not self._feature_names:
            self._feature_names = list(self.rules.keys())
        
        return self._feature_names
    
    @property
    def combiner(self):
        if not self._combiner:
            # generate a new combiner if not exists
            rules = {}
            for key in self.rules:
                rules[key] = self.rules[key]['bins']
            
            self._combiner = Combiner().load(rules)
        
        return self._combiner


    def fit(self, X, y):
        """
        Args:
            X (2D DataFrame)
            Y (array-like)
        """
        self._feature_names = X.columns.tolist()

        for f in self.features_:
            if f not in self.transer:
                raise Exception('column \'{f}\' is not in transer'.format(f = f))

        self.model.fit(X, y)
        self.rules = self._generate_rules()

        # keep sub_score-median of each feature, as `base_effect` for reason-calculation
        sub_score = self.woe_to_score(X)
        self.base_effect = pd.Series(
            np.median(sub_score, axis=0),
            index = self.features_
        )

        return self

    def predict(self, X, **kwargs):
        """predict score
        Args:
            X (2D-DataFrame|dict): X to predict
            return_sub (Bool): if need to return sub score of each feature
            default (str|number): default sub score for unknown feature, `min`(default), `max`

        Returns:
            array-like: predicted score
            DataFrame|dict: sub score for each feature
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.features_]
        
        bins = self.combiner.transform(X)
        res = self.bin_to_score(bins, **kwargs)
        return res

    def get_reason(self, X, base_effect = None, threshold_score = None, keep = 3):
        """
        calculate top-effect-of-features as reasons

        Args:
            X (2D DataFrame): X to find reason
            base_effect (Series): base effect score of each feature 
            threshold_score (float): threshold to find top k most important features,
                show the highest top k features when prediction score > threshold
                and show the lowest top k when prediction score <= threshold
                default is the sum of `base_effect` score
            keep(int): top k most important reasons to keep, default `3`
        
        Returns:
            DataFrame: top k most important reasons for each feature
        """

        # use the memory during `fit()` by default
        if base_effect is None:
            base_effect = self.base_effect
        
        # use zero-vector if scorecard doesn't have `base_effect`
        if base_effect is None:  
            base_effect = pd.Series(0, index = self.features_)
        
        # set default threshold score
        if threshold_score is None:
            threshold_score = np.sum(base_effect.values)

        # get score and sub scores
        score, sub = self.predict(X, return_sub = True)

        bias = sub - base_effect
        # find direction for each row, `-1` means keep high bias, `1` keeps low bias
        direction = 1 - 2 * (score > threshold_score).reshape(-1, 1).astype(np.uint8)

        # sort by bias and keep top k columns
        idx = np.argsort(direction * bias.values, axis = -1)[:,:keep]
        # get effect data by sorted index
        effect_bias = np.take_along_axis(sub.values, idx, axis = -1)
        effect_values = np.take_along_axis(X[self.features_].values, idx, axis = -1)
        effect_feats = np.take(self.features_, idx)

        # merge effect data into a DataFrame
        effect_matrix = np.dstack((effect_feats.T, effect_bias.T, effect_values.T))
        cols = pd.MultiIndex.from_product(
            [[f"top{i}" for i in range(1, keep+1)], ['feats', 'bias', 'value']]
        )
        reason = pd.DataFrame(
            np.hstack(effect_matrix),
            columns = cols,
        )

        return reason


    def bin_to_score(self, bins, return_sub = False, default = 'min'):
        """predict score from bins
        """
        score = 0
        res = bins.copy()
        for col, rule in self.rules.items():
            s_map = rule['scores']
            b = bins[col]

            # set default value for empty group
            default_value = default
            if default == 'min':
                default_value = np.min(s_map)
            elif default == 'max':
                default_value = np.max(s_map)
            elif isinstance(default, str):
                raise ValueError(f'default `{default}` is not valid, only support `min`, `max` or number')
            
            # append default value to the end of score map
            s_map = np.append(s_map, default_value)

            # # set default group to min score
            # if np.isscalar(b):
            #     b = np.argmin(s_map) if b == self.EMPTY_BIN else b
            # else:
            #     b[b == self.EMPTY_BIN] = np.argmin(s_map)

            # replace score
            res[col] = s_map[b]
            score += s_map[b]

        if return_sub:
            return score, res
        else:
            return score


    def predict_proba(self, X):
        """predict probability

        Args:
            X (2D array-like): X to predict
        
        Returns:
            2d array: probability of all classes
        """
        proba = self.score_to_proba(self.predict(X))
        return np.stack((1 - proba, proba), axis=1)
    

    def _generate_rules(self):
        if not self._check_rules(self.combiner, self.transer):
            raise Exception('generate failed')
        
        rules = {}

        for idx, key in enumerate(self.features_):
            weight = self.coef_[idx]

            if weight == 0:
                continue
            
            woe = self.transer[key]['woe']
            
            rules[key] = {
                'bins': self.combiner[key],
                'woes': woe,
                'weight': weight,
                'scores': self.woe_to_score(woe, weight = weight),
            }
        
        return rules


    def _check_rules(self, combiner, transer):
        for col in self.features_:
            if col not in combiner:
                raise Exception('column \'{col}\' is not in combiner'.format(col = col))
            
            if col not in transer:
                raise Exception('column \'{col}\' is not in transer'.format(col = col))

            l_c = len(combiner[col])
            l_t = len(transer[col]['woe'])

            if l_c == 0:
                continue

            if np.issubdtype(combiner[col].dtype, np.number):
                if l_c != l_t - 1:
                    raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col = col, l_t = l_t, l_c = l_c + 1))
            else:
                if l_c != l_t:
                    raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col = col, l_t = l_t, l_c = l_c))

        return True


    def proba_to_score(self, prob):
        """covert probability to score
        
        odds = (1 - prob) / prob
        score = factor * log(odds) * offset
        """
        return self.factor * (np.log(1 - prob) - np.log(prob)) + self.offset
    

    def score_to_proba(self, score):
        """covert score to probability

        Returns:
            array-like|float: the probability of `1`
        """
        return 1 / (np.e ** ((score - self.offset) / self.factor) + 1)


    def woe_to_score(self, woe, weight = None):
        """calculate score by woe
        """
        woe = to_ndarray(woe)

        if weight is None:
            weight = self.coef_

        b = self.offset - self.factor * self.intercept_
        s = -self.factor * weight * woe

        # drop score whose weight is 0
        mask = 1
        if isinstance(weight, np.ndarray):
            mask = (weight != 0).astype(int)

        return (s + b / self.n_features_) * mask


    def _parse_rule(self, rule, **kwargs):
        bins = self.parse_bins(list(rule.keys()))
        scores = np.array(list(rule.values()))

        return {
            'bins': bins,
            'scores': scores,
        }
    
    def _format_rule(self, rule, decimal = 2, **kwargs):
        bins = self.format_bins(rule['bins'])
        scores = np.around(rule['scores'], decimals = decimal).tolist()
        
        return dict(zip(bins, scores))


    def after_load(self, rules):
        """after load card
        """
        # reset combiner
        self._combiner = {}

    def after_export(self, card, to_frame = False, to_json = None, to_csv = None, **kwargs):
        """generate a scorecard object

        Args:
            to_frame (bool): return DataFrame of card
            to_json (str|IOBase): io to write json file
            to_csv (filepath|IOBase): file to write csv

        Returns:
            dict
        """
        if to_json is not None:
            save_json(card, to_json)

        if to_frame or to_csv is not None:
            rows = list()
            for name in card:
                for value, score in card[name].items():
                    rows.append({
                        'name': name,
                        'value': value,
                        'score': score,
                    })

            card = pd.DataFrame(rows)


        if to_csv is not None:
            return card.to_csv(to_csv)

        return card



    def _generate_testing_frame(self, maps, size = 'max', mishap = True, gap = 1e-2):
        """
        Args:
            maps (dict): map of values or splits to generate frame
            size (int|str): size of frame. 'max' (default), 'lcm'
            mishap (bool): is need to add mishap patch to test frame
            gap (float): size of gap for testing border

        Returns:
            DataFrame
        """
        number_patch = np.array([NUMBER_EMPTY, NUMBER_INF])
        factor_patch = np.array([FACTOR_EMPTY, FACTOR_UNKNOWN])

        values = []
        cols = []
        for k, v in maps.items():
            is_number = isinstance(v[0], (int, float))
            if is_number:
                v = np.array(v)
                items = np.concatenate((v, v - gap))
                patch = number_patch
            else:
                v = np.array(v, dtype = object)
                # remove else group
                mask = np.argwhere(v == self.ELSE_GROUP)
                if mask.size > 0:
                    v = np.delete(v, mask)

                items = np.concatenate(v)
                patch = factor_patch

            if mishap:
                # add patch to items
                items = np.concatenate((items, patch))

            cols.append(k)
            values.append(np.unique(items))

        # calculate length of values in each columns
        lens = [len(x) for x in values]

        # get size
        if isinstance(size, str):
            if size == 'lcm':
                size = np.lcm.reduce(lens)
            else:
                size = np.max(lens)

        stacks = dict()
        for i in range(len(cols)):
            l = lens[i]
            # generate indexes of value in column
            ix = np.arange(size) % l
            stacks[cols[i]] = values[i][ix]

        return pd.DataFrame(stacks)

    def testing_frame(self, **kwargs):
        """get testing frame with score

        Returns:
            DataFrame: testing frame with score
        """
        maps = self.combiner.export()

        frame = self._generate_testing_frame(maps, **kwargs)
        frame['score'] = self.predict(frame)

        return frame
