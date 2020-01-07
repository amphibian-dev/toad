from .decorator import save_to_json, load_from_json

DEFAULT_NAME = '_feature_default_name_'


class SaveMixin:
    _rules = {}

    def _parse_rule(self, rule):
        return rule
    
    def _format_rule(self, rule):
        return rule

    @load_from_json(is_class = True, require_first = True)
    def load(self, rules, update = False, **kwargs):
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