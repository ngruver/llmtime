import numpy as np
import numbers
import random
from collections import defaultdict
from collections.abc import Iterable

import itertools,operator,functools

class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)

class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnlyDict")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__

class NoGetItLambdaDict(dict):
    """ Regular dict, but refuses to __getitem__ pretending
        the element is not there and throws a KeyError
        if the value is a non string iterable or a lambda """
    def __init__(self,d={}):
        super().__init__()
        for k,v in d.items():
            if isinstance(v,dict):
                self[k] = NoGetItLambdaDict(v)
            else:
                self[k] = v
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value) and value.__name__ == "<lambda>":
            raise LookupError("You shouldn't try to retrieve lambda {} from this dict".format(value))
        if isinstance(value,Iterable) and not isinstance(value,(str,bytes,dict,tuple)):
            raise LookupError("You shouldn't try to retrieve iterable {} from this dict".format(value))
        return value
        
    # pop = __readonly__
    # popitem = __readonly__

def sample_config(config_spec):
    """ Generates configs from the config spec.
        It will apply lambdas that depend on the config and sample from any
        iterables, make sure that no elements in the generated config are meant to 
        be iterable or lambdas, strings are allowed."""
    cfg_all = config_spec
    more_work=True
    i=0
    while more_work:
        cfg_all, more_work = _sample_config(cfg_all,NoGetItLambdaDict(cfg_all))
        i+=1
        if i>10: 
            raise RecursionError("config dependency unresolvable with {}".format(cfg_all))
    out = defaultdict(dict)
    out.update(cfg_all)
    return out

def _sample_config(config_spec,cfg_all):
    cfg = {}
    more_work = False
    for k,v in config_spec.items():
        if isinstance(v,dict):
            new_dict,extra_work = _sample_config(v,cfg_all)
            cfg[k] = new_dict
            more_work |= extra_work
        elif isinstance(v,Iterable) and not isinstance(v,(str,bytes,dict,tuple)):
            cfg[k] = random.choice(v)
        elif callable(v) and v.__name__ == "<lambda>":
            try:cfg[k] = v(cfg_all)
            except (KeyError, LookupError,Exception):
                cfg[k] = v # is used isntead of the variable it returns
                more_work = True
        else: cfg[k] = v
    return cfg, more_work

def flatten(d, parent_key='', sep='/'):
    """An invertible dictionary flattening operation that does not clobber objs"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and v: # non-empty dict
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten(d,sep='/'):
    """Take a dictionary with keys {'k1/k2/k3':v} to {'k1':{'k2':{'k3':v}}}
        as outputted by flatten """
    out_dict={}
    for k,v in d.items():
        if isinstance(k,str):
            keys = k.split(sep)
            dict_to_modify = out_dict
            for partial_key in keys[:-1]:
                try: dict_to_modify = dict_to_modify[partial_key]
                except KeyError:
                    dict_to_modify[partial_key] = {}
                    dict_to_modify = dict_to_modify[partial_key]
                # Base level reached
            if keys[-1] in dict_to_modify:
                dict_to_modify[keys[-1]].update(v)
            else:
                dict_to_modify[keys[-1]] = v
        else: out_dict[k]=v
    return out_dict

class grid_iter(object):
    """ Defines a length which corresponds to one full pass through the grid
        defined by grid variables in config_spec, but the iterator will continue iterating
        past that by repeating over the grid variables"""
    def __init__(self,config_spec,num_elements=-1,shuffle=True):
        self.cfg_flat = flatten(config_spec)
        is_grid_iterable = lambda v: (isinstance(v,Iterable) and not isinstance(v,(str,bytes,dict,tuple)))
        iterables = sorted({k:v for k,v in self.cfg_flat.items() if is_grid_iterable(v)}.items())
        if iterables: self.iter_keys,self.iter_vals = zip(*iterables)
        else: self.iter_keys,self.iter_vals = [],[[]]
        self.vals = list(itertools.product(*self.iter_vals))
        if shuffle:
            with FixedNumpySeed(0): random.shuffle(self.vals)
        self.num_elements = num_elements if num_elements>=0 else (-1*num_elements)*len(self)

    def __iter__(self):
        self.i=0
        self.vals_iter = iter(self.vals)
        return self
    def __next__(self):
        self.i+=1
        if self.i > self.num_elements: raise StopIteration
        if not self.vals: v = []
        else:
            try: v = next(self.vals_iter)
            except StopIteration:
                self.vals_iter = iter(self.vals)
                v = next(self.vals_iter)
        chosen_iter_params = dict(zip(self.iter_keys,v))
        self.cfg_flat.update(chosen_iter_params)
        return sample_config(unflatten(self.cfg_flat))
    def __len__(self):
        product = functools.partial(functools.reduce, operator.mul)
        return product(len(v) for v in self.iter_vals) if self.vals else 1

def flatten_dict(d):
    """ Flattens a dictionary, ignoring outer keys. Only
        numbers and strings allowed, others will be converted
        to a string. """
    out = {}
    for k,v in d.items():
        if isinstance(v,dict):
            out.update(flatten_dict(v))
        elif isinstance(v,(numbers.Number,str,bytes)):
            out[k] = v
        else:
            out[k] = str(v)
    return out