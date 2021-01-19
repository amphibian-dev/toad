import pytest
import numpy as np
from .mixin import RulesMixin, BinsMixin

np.random.seed(1)

class RulesObject(RulesMixin):
    def _parse_rule(self, rule):
        return {
            'rule': rule
        }


    def _format_rule(self, rule):
        return 'rule -> %s' % rule['rule']


rules = {'A': 'rule_A'}

def test_rule_parse():
    r = RulesObject().load(rules)
    assert r.rules['A']['rule'] == 'rule_A'

def test_rule_format():
    r = RulesObject().load(rules)
    assert r.export()['A'] == 'rule -> rule_A'

def test_save_update():
    r = RulesObject().load(rules)
    r.update({'A': 'update_A'})
    assert r.rules['A']['rule'] == 'update_A'

def test_format_bins():
    obj = BinsMixin()
    formated = obj.format_bins(np.array([2,4,6]))
    expect = ['[-inf ~ 2)', '[2 ~ 4)', '[4 ~ 6)', '[6 ~ inf)']
    assert all([a == b for a, b in zip(formated, expect)])

def test_format_bins_with_index():
    obj = BinsMixin()
    formated = obj.format_bins(np.array([2,4,6]), index = True)
    assert '01.[2 ~ 4)' in formated

def test_format_bins_with_ellipsis():
    obj = BinsMixin()
    formated = obj.format_bins(np.array([['A', 'B', 'C'], ['D', 'E']], dtype = object), ellipsis = 3)
    assert formated[0] == 'A,B..' and formated[1] == 'D,E'
