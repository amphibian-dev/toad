from copy import deepcopy
from .decorator import save_to_json, load_from_json

DEFAULT_NAME = '_feature_default_name_'


class RulesMixin:
    _rules = {}

    def _parse_rule(self, rule):
        return rule
    
    def _format_rule(self, rule):
        return rule
    
    def default_rule(self):
        return self._rules[DEFAULT_NAME]
    
    @property
    def _default_name(self):
        return DEFAULT_NAME

    @property
    def rules(self):
        return self._rules
    
    @rules.setter
    def rules(self, value):
        self._rules = value
    

    @load_from_json(is_class = True, require_first = True)
    def load(self, rules, update = False, **kwargs):
        rules = deepcopy(rules)
        
        if not isinstance(rules, dict):
            rules = {
                DEFAULT_NAME: rules,
            }
        
        for key in rules:
            rules[key] = self._parse_rule(rules[key])
        
        if update:
            self._rules.update(rules)
        else:
            self._rules = rules
        
        return self
    
    @save_to_json(is_class = True)
    def export(self, **kwargs):
        res = {}
        for key in self._rules:
            res[key] = self._format_rule(self._rules[key], **kwargs)
        
        return res
    
    def update(self, *args, **kwargs):
        return self.load(*args, update = True, **kwargs)
    

    def __len__(self):
        return len(self._rules.keys())
