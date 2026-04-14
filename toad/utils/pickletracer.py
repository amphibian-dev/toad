import cloudpickle
from pickle import Unpickler
from cloudpickle import CloudPickler

_global_tracer = None

def get_current_tracer():
    global _global_tracer
    # if _global_tracer is None:
    #     raise ValueError("tracer is not initialized")
    return _global_tracer


class Unpickler(Unpickler):
    """trace object dependences during unpickle"""
    def find_class(self, module, name):
        tracer = get_current_tracer()
        tracer.add(module)
        return super().find_class(module, name)


class Pickler(CloudPickler):
    """trace object dependences during pickle"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import types
        self._reduce_module = CloudPickler.dispatch_table[types.ModuleType]
        self.dispatch_table[types.ModuleType] = self.reduce_module
    

    def reduce_module(self, obj):
        tracer = get_current_tracer()
        tracer.add(obj.__name__)
        return self._reduce_module(obj)
    

    def __setattr__(self, name, value):
        if name == 'persistent_id':
            # fix torch module
            def wrapper_func(obj):
                from torch.nn import Module
                if isinstance(obj, Module):
                    return None
                
                return value(obj)
            
            return super().__setattr__(name, wrapper_func)
        
        return super().__setattr__(name, value)


class Tracer:
    def __init__(self):
        import re

        self._modules = set()
        self._ignore_modules = {"builtins"}
        self._temp_dispatch_table = {}

        # match python site packages path
        self._regex = re.compile(r".*python[\d\.]+\/site-packages/[\w-]+")
    
    def add(self, module):
        root = module.split(".")[0]
        
        if root in self._ignore_modules:
            return
        
        self._modules.add(root)
    
    def trace(self, obj):
        """trace `obj` by picke and unpicke
        """
        import io
        dummy = io.BytesIO()

        with self:
            Pickler(dummy).dump(obj)
            dummy.seek(0)
            Unpickler(dummy).load()

        return self.get_deps()


    def get_deps(self):
        import sys
        
        deps = {
            "pip": [],
            "files": [],
        }

        for name in self._modules:
            if name not in sys.modules:
                # TODO: should raise error
                continue
            
            module = sys.modules[name]
            # package module
            if self._regex.match(module.__spec__.origin):
                # TODO: spilt pip and conde pkg
                deps["pip"].append(module)
                continue
            
            # local file module
            deps["files"].append(module)

        return deps
    

    def __enter__(self):
        global _global_tracer
        if _global_tracer is not None:
            raise ValueError("a tracer is already exists")
        
        # save the Cloudpickler global dispatch table
        self._temp_dispatch_table = CloudPickler.dispatch_table.copy()
        # setup the global tracer
        _global_tracer = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_tracer

        # restore the dispatch table to Cloudpickler
        CloudPickler.dispatch_table = self._temp_dispatch_table
        # clean the global tracer
        _global_tracer = None
        
    


def dump(obj, file, *args, **kwargs):
    return Pickler(file).dump(obj)


def load(file, *args, **kwargs):
    return Unpickler(file).load()

