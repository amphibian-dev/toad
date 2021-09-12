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
        # keep track of median-effect of each feature during fit(), as `self.base_effect_of_features`
        # for reason-calculation later during predict()
        self.base_effect_of_features = None

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

        # keep sub_score-median of each feature, as `base_effect_of_features` for reason-calculation
        sub_score = self.woe_to_score(X)
        self.base_effect_of_features = np.median(sub_score, axis=0)

        return self

    def predict(self, X, return_sub=False, return_reason=False, keep=3):
        """predict score
        Args:
            X (2D array-like): X to predict
            return_reason (bool): if need to return reason, default 'False'
            return_sub (bool): if need to return sub score, default `False`
            keep(int): top k most important reasons to keep, default `3`

        Returns:
            Components:
            A. array-like: predicted score
            B. DataFrame(optional): sub score for each feature
            C. DataFrame(optional): score and top k most important reasons for each feature

            for cases:
            1. return_sub=False, return_reason=False
                just return A
            2. return_sub=True, return_reason=False
                return a tuple of <A, B>
            3. return_sub=False, return_reason=True
                return a tuple of <A, C> CAUTION: SAME FORMAT AS CASE 2
            4. return_sub=True, return_reason=True
                return a tuple of <A, B, C>
        """
        bins = self.combiner.transform(X[self.features_])
        if return_reason is False:
            return self.bin_to_score(bins, return_sub=return_sub)

        pred, sub_score = self.bin_to_score(bins, return_sub=True)
        score_reason = (
            pd.concat([
                X.rename(mapper=lambda c: f'raw_val_{c}', axis=1),  # raw value of features
                sub_score,  # score of features
                pd.Series(pred, name='score')  # predicted total score
            ], axis=1)
                .assign(reason=lambda df: df.apply(self._get_reason_column, axis=1, keep=keep))
                .loc[:, ['score', 'reason']]  # keep only the predicted-total-score and reason columns
        )
        return pred, sub_score, score_reason

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


    def bin_to_score(self, bins, return_sub = False):
        """predict score from bins
        """
        res = bins.copy()
        for col in self.rules:
            s_map = self.rules[col]['scores']
            b = bins[col].values
            # set default group to min score
            b[b == self.EMPTY_BIN] = np.argmin(s_map)
            # replace score
            res[col] = s_map[b]

        score = np.sum(res.values, axis = 1)

        if return_sub:
            return score, res
        else:
            return score


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

    def _get_reason_column(self, row, keep=3):
        """ calculate top `keep` reasons of the prediction
        Args:
            row (pd.Series): row to predict
            keep (int): top k most important features to keep
        Returns:
            reason (list of list) : top k most important reason(feature name, sub_score, raw feature value)
        """
        is_raw_val = row.index.str.startswith('raw_val_')
        s_score = row[~is_raw_val].drop('score')  # sub_score columns, except the total-score-column
        pd_s_score = pd.DataFrame({
            'sub_score': s_score,
            'score_diff': s_score.values - self.base_effect_of_features,
        })

        # if total score is lower than base_odds, select top k feature who contribute most negativity
        # vice versa
        df_reason = (pd_s_score
                     .sort_values(by='score_diff', ascending=(row.score <= self.base_odds))['sub_score']
                     .head(keep)
                     .apply('{:+.1f}'.format)
                     .to_frame(name='score')
                     # append raw value for reference. note: len('raw_val_') == 8
                     .join(row[is_raw_val].rename('value').rename(index=lambda name: name[8:]), how='left')
                     .reset_index()
                     )

        reason = df_reason.values.tolist()
        return reason

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
            v = np.array(v)
            if np.issubdtype(v.dtype, np.number):
                items = np.concatenate((v, v - gap))
                patch = number_patch
            else:
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
