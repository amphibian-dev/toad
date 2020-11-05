import re
import numpy as np
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
        if len(self._rules) == 1:
            # return the only rule as default
            return next(iter(self._rules.values()))
        
        if self._default_name not in self._rules:
            raise Exception('can not get default rule')

        return self._rules[self._default_name]
    
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
        """load rules from dict or json file

        Args:
            rules (dict): dictionary of rules
            from_json (str|IOBase): json file of rules
            update (bool): if need to use updating instead of replacing rules
        """
        rules = deepcopy(rules)

        if not isinstance(rules, dict):
            rules = {
                self._default_name: rules,
            }
        
        for key in rules:
            rules[key] = self._parse_rule(rules[key], **kwargs)
        
        if update:
            self._rules.update(rules)
        else:
            self._rules = rules
        
        return self
    
    @save_to_json(is_class = True)
    def export(self, **kwargs):
        """export rules to dict or a json file

        Args:
            to_json (str|IOBase): json file to save rules
        
        Returns:
            dict: dictionary of rules
        """
        res = {}
        for key in self._rules:
            res[key] = self._format_rule(self._rules[key], **kwargs)
        
        if hasattr(self, 'after_export'):
            res = self.after_export(res, **kwargs)
        
        return res
    
    def update(self, *args, **kwargs):
        """update rules

        Args:
            rules (dict): dictionary of rules
            from_json (str|IOBase): json file of rules
        """
        return self.load(*args, update = True, **kwargs)
    

    def __len__(self):
        return len(self._rules.keys())
    
    def __contains__(self, key):
        return key in self._rules
    
    def __getitem__(self, key):
        return self._rules[key]
    
    def __setitem__(self, key, value):
        self._rules[key] = value

    def __iter__(self):
        return iter(self._rules)




RE_NUM = r'-?\d+(.\d+)?'
RE_SEP = r'[~-]'
RE_BEGIN = r'(-inf|{num})'.format(num = RE_NUM)
RE_END = r'(inf|{num})'.format(num = RE_NUM)
RE_RANGE = r'\[{begin}\s*{sep}\s*{end}\)'.format(
    begin = RE_BEGIN,
    end = RE_END,
    sep = RE_SEP,
)





class BinsMixin:
    EMPTY_BIN = -1
    ELSE_GROUP = 'else'
    NUMBER_EXP = re.compile(RE_RANGE)

    @classmethod
    def parse_bins(self, bins):
        """parse labeled bins to array
        """
        if self._is_numeric(bins):
            return self._numeric_parser(bins)
        
        l = list()

        for item in bins:
            if item == self.ELSE_GROUP:
                l.append(item)
            else:
                l.append(item.split(','))

        return np.array(l, dtype = object)


    @classmethod
    def format_bins(self, bins, index = False, ellipsis = None):
        """format bins to label

        Args:
            bins (ndarray): bins to format
            index (bool): if need index prefix
            ellipsis (int): max length threshold that labels will not be ellipsis, `None` for skipping ellipsis
        
        Returns:
            ndarray: array of labels
        """
        l = list()

        if np.issubdtype(bins.dtype, np.number):
            has_empty = len(bins) > 0 and np.isnan(bins[-1])
            
            if has_empty:
                bins = bins[:-1]
            
            sp_l = [-np.inf] + bins.tolist() + [np.inf]
            for i in range(len(sp_l) - 1):
                l.append('['+str(sp_l[i])+' ~ '+str(sp_l[i+1])+')')
            
            if has_empty:
                l.append('nan')
        else:
            for keys in bins:
                if isinstance(keys, str) and keys == self.ELSE_GROUP:
                    l.append(keys)
                else:
                    label = ','.join(keys)
                    if ellipsis is not None:
                        label = label[:ellipsis] + '..' if len(label) > ellipsis else label
                    l.append(label)

        if index:
            l = ["{:02}.{}".format(ix, lab) for ix, lab in enumerate(l)]

        return np.array(l)
    

    @classmethod
    def _is_numeric(self, bins):
        m = self.NUMBER_EXP.match(bins[0])

        return m is not None
    
    @classmethod
    def _numeric_parser(self, bins):
        l = list()

        for item in bins:

            if item == 'nan':
                l.append(np.nan)
                continue
            
            m = self.NUMBER_EXP.match(item)
            split = m.group(3)

            if split == 'inf':
                # split = np.inf
                continue
            
            split = float(split)

            l.append(split)
        
        return np.array(l)