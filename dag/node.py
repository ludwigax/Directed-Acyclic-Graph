import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import Tuple, Dict, List, Union, Optional, Any, Callable
from dataclasses import dataclass
import functools

import inspect
from .dbg import Debug, get_debug_state


class Node(ABC):
    def __init__(self, name: str = None, **kwargs):
        self._auto_name = self.__class__.__name__ + str(id(self))[-4:]
        self.name = name or self._auto_name

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.name):
            raise ValueError(f"Invalid name '{self.name}'. Name must start with a letter or underscore and can only contain letters, digits, and underscores.")

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
    
    @abstractmethod
    def __repr__(self):
        extra_repr = self.extra_repr()
        if extra_repr:
            return f"{self.__class__.__name__}({self.name}, {extra_repr})"
        return f"{self.__class__.__name__}({self.name})"
    
    def extra_repr(self) -> str:
        return ""


class _PlaceholderNodeType:
    __slots__ = ("_name", )

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"<{self._name}>"

NullNode = _PlaceholderNodeType("NullNode")
VirtualNode = _PlaceholderNodeType("VirtualNode")


@dataclass
class Edge:
    name: str
    src: Union['Node', _PlaceholderNodeType] = NullNode
    tgt: Union['Node', _PlaceholderNodeType] = NullNode
    src_key: Optional[str] = None
    tgt_key: Optional[str] = None
    is_cached: bool = False
    _cache: Optional[Any] = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, src={self.src}, tgt={self.tgt}, cache={type(self._cache)}, is_cached={self.is_cached})"

    @property
    def null(self):
        return self.src is NullNode or self.tgt is NullNode
    
    @property
    def virtual(self):
        return self.src is VirtualNode or self.tgt is VirtualNode
    
    @property
    def cache(self):
        return self._cache
    
def returns_keys(**kwargs):
    def decorator(func):
        func.__returns_keys__ = kwargs
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class Module(Node, Debug):
    def __init__(
        self, 
        name: str = None, 
        parent: Optional['Module'] = None,
        indirect = None,
        outdirect = None,
        **kwargs
    ):
        Node.__init__(self, name, **kwargs)
        Debug.__init__(self)
        self.parent = parent

        self.indirect = indirect
        self.outdirect = outdirect

        self._prev: Dict[str, Edge] = {}
        self._next: Dict[str, List[Edge]] = defaultdict(list)

        self.use_default_return = True

    def __repr__(self):
        return super().__repr__()
    
    def extra_repr(self):
        return f"parent={self.parent}, prev={self._prev}, next={self._next}"

    def __call__(self, *args, **kwargs):
        # Check if debugging is enabled
        if get_debug_state():
            self._start_timer()
            
        # Check for cached results
        if all(edge.is_cached for edges in self._next.values() if edges for edge in edges):
            results = {}
            for k, edges in self._next.items():
                if not edges:
                    warnings.warn(f"Empty edge list found for key {k}", UserWarning)
                    continue
                results[k] = edges[0].cache
                
            # Log time for cached execution if debugging is enabled
            if get_debug_state():
                elapsed = self._stop_timer()
                print(f"[DEBUG] Module {self.name} returned cached result in {elapsed:.6f}s")
                
            return results

        # Prepare inputs
        self.prepare()
        
        # Execute forward computation
        forward_results = self.forward(**{k: edge.cache for k, edge in self._prev.items()})
        
        # Process results
        if self.use_default_return and not isinstance(forward_results, dict):
            results = {"_return": forward_results}
        else:
            results = forward_results

        # Update output edges
        for _next_k, res in results.items():
            if _next_k in self._next:
                for _next_edge in self._next[_next_k]:
                    _next_edge._cache = res
                    _next_edge.is_cached = True
            else:
                warnings.warn(f"Output key {_next_k} not found in next edges", UserWarning)
        
        # Log execution time if debugging is enabled
        if get_debug_state():
            elapsed = self._stop_timer()
            print(f"[DEBUG] Module {self.name} executed in {elapsed:.6f}s")
            
        return results
    
    @property
    def num_entries(self):
        return self.indirect or len(self._prev)
    
    @property
    def undefined_entries(self):
        return 0 if self.indirect is None else sum(edge.null for edge in self._prev.values())

    def prepare(self):
        for _prev_k, prev_edge in self._prev.items():
            if prev_edge.null:
                raise ValueError(f"Undefined input: {prev_edge.name}")
            if prev_edge.virtual:
                map_key = next(k for k, v in self.parent._prev_name_map.items() if v == f"{self.name}.{_prev_k}")
                prev_edge._cache = self.parent._prev[map_key].cache
                prev_edge.is_cached = True
            else:
                prev_edge.src()

    @abstractmethod
    def forward(self, **kwargs) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class FunctionModule(Module):
    def __init__(
        self,
        func: callable,
        name: str = None,
        parent: Optional['Module'] = None,
        **kwargs
    ):
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")

        self.func = func
        self.func_signature = inspect.signature(func)

        parameters = list(self.func_signature.parameters.keys())
        
        if parameters and parameters[0] == 'self':
            parameters = parameters[1:]

        filtered_parameters = []
        for param in parameters:
            param_kind = self.func_signature.parameters[param].kind
            if param_kind != inspect.Parameter.VAR_POSITIONAL and param_kind != inspect.Parameter.VAR_KEYWORD:
                filtered_parameters.append(param)
        
        indirect = len(filtered_parameters)

        output_keys = ["_return"]
        
        if hasattr(func, '__returns_keys__'):
            output_keys = list(func.__returns_keys__.keys())
            self.use_default_return = False

        super().__init__(name=name, parent=parent, indirect=indirect, outdirect=len(output_keys), **kwargs)

        for param in filtered_parameters:
            self._prev[param] = Edge(name=param)
        for output in output_keys:
            self._next[output] = [Edge(name=output)]

    def extra_repr(self):
        return f"func={self.func.__name__}, parent={self.parent}, prev={self._prev}, next={self._next}"

    def forward(self, **kwargs) -> Any:
        filtered_args = {
            name: kwargs[name]
            for name in self.func_signature.parameters
            if name in kwargs
        }
        return self.func(**filtered_args)


class InspectModule(Module):
    def __init__(self, name: str = None, parent: Optional['Module'] = None, **kwargs):
        super().__init__(name, parent, **kwargs)
        
        forward_method = self.forward
        self.forward_signature = inspect.signature(forward_method)
        parameters = list(self.forward_signature.parameters.keys())
        
        if parameters and parameters[0] == 'self':
            parameters = parameters[1:]

        filtered_parameters = []
        for param in parameters:
            param_kind = self.forward_signature.parameters[param].kind
            if param_kind != inspect.Parameter.VAR_POSITIONAL and param_kind != inspect.Parameter.VAR_KEYWORD:
                filtered_parameters.append(param)
        
        self.indirect = len(filtered_parameters)

        for param in filtered_parameters:
            self._prev[param] = Edge(name=param)
        
        output_keys = ["_return"]
        
        if hasattr(forward_method, '__returns_keys__'):
            output_keys = list(forward_method.__returns_keys__.keys())
            self.use_default_return = False
        
        for key in output_keys:
            self._next[key] = [Edge(name=key)]

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ModuleGroup(Module):
    def __init__(self, name: str = None, parent: Optional['Module'] = None, modules: List[Module] = [], **kwargs):
        super().__init__(name, parent, **kwargs)
        self._modules: Dict[str, Module] = {}
        self._next_name_map = {}
        self._prev_name_map = {}

        if modules:
            self.initialize(modules)

    def initialize(self, modules: List[Module] = []):
        if not modules:
            return
        
        for module in modules:
            if not isinstance(module, Module):
                raise TypeError(f"Expected Module instance, got {type(module)}")
            module.parent = self

            if module.name in self._modules: # TODO change the logic of repeat name
                raise ValueError(f"Module with name {module.name} already exists in the group.")
                # warnings.warn(f"Module with name {module.name} already exists in the group.", UserWarning)
            self._modules[module.name] = module
    
        prev_key_repeat = defaultdict(set)
        next_key_repeat = defaultdict(set)

        for _, module in self._modules.items():
            for key, edge in module._prev.items():
                if edge.null:
                    if key in prev_key_repeat:
                        idx = max([x for x in prev_key_repeat[key] if isinstance(x, int)], default=1)
                        group_key = f"{key}_{idx + 1}"
                        prev_key_repeat[key].add(idx + 1)
                    else:
                        group_key = key
                        prev_key_repeat[key].add(1)

                    self._prev_name_map[group_key] = f"{module.name}.{key}"

                    if group_key not in self._prev:
                        self._prev[group_key] = Edge(name=group_key, tgt=self)
                    else:
                        warnings.warn(f"Duplicate _prev_key {group_key} found in module {module.name}.", UserWarning)

                    module._prev[key] = Edge(
                        name=key,
                        src=VirtualNode,
                        tgt=module,
                        tgt_key=key
                    )

            for key, edges in module._next.items():
                for idx, edge in enumerate(edges):
                    if edge.null:
                        if key in next_key_repeat:
                            idx = max([x for x in next_key_repeat[key] if isinstance(x, int)], default=1)
                            group_key = f"{key}_{idx + 1}"
                            next_key_repeat[key].add(idx + 1)
                        else:
                            group_key = key
                            next_key_repeat[key].add(1)

                        self._next_name_map[group_key] = f"{module.name}.{key}"
                        if group_key not in self._next:
                            self._next[group_key] = [Edge(name=group_key, src=self, src_key=group_key)]
                        else:
                            warnings.warn(f"Duplicate _next_key {group_key} found in module {module.name}.", UserWarning)

                        module._next[key][idx].tgt = VirtualNode
                        break
        
        self.indirect = len(self._prev)
        self.outdirect = len(self._next)

    def forward(self, **kwargs) -> Any:
        results = {}

        module_set = set()
        for map_key, module_key in self._next_name_map.items():
            module_name = module_key.split(".")[0]
            module_set.add(module_name)
        
        for module_name in module_set:
            self._modules[module_name]()

        for key, edges in self._next.items():
            results[key] = self._modules[self._next_name_map[key].split(".")[0]]._next[key][0].cache
        return results

def connect(
        src: Union['Module', _PlaceholderNodeType], tgt: Union['Module', _PlaceholderNodeType],
        src_key: Optional[int] = None, tgt_key: Optional[int] = None, name: Optional[str] = None
    ):
    name = name or f"{src_key} -> {tgt_key}"
    new_edge = Edge(name, src=src, src_key=src_key, tgt=tgt, tgt_key=tgt_key)

    if hasattr(src, "_next") and src_key is not None:
        insert_idx = -1
        for idx, edge in enumerate(src._next[src_key]):
            if edge.null:
                insert_idx = idx
                break

        if insert_idx == -1:
            src._next[src_key].append(new_edge)
        else:
            edge = src._next[src_key][insert_idx]
            edge.src = NullNode
            edge.src_key = None

            src._next[src_key][insert_idx] = new_edge

    if hasattr(tgt, "_prev") and tgt_key is not None:
        edge = tgt._prev[tgt_key]
        edge.tgt = NullNode
        edge.tgt_key = None

        tgt._prev[tgt_key] = new_edge


def get_module_stats(module: Module, recursive: bool = False) -> Dict[str, Any]:
    stats = {module.name: module.get_stats()}
    
    if recursive and isinstance(module, ModuleGroup):
        for name, submodule in module._modules.items():
            substats = get_module_stats(submodule, recursive=True)
            stats.update(substats)
    
    return stats

def reset_module_stats(module: Module, recursive: bool = False) -> None:
    module.reset_stats()
    module.reset_stats()
    
    if recursive and isinstance(module, ModuleGroup):
        for submodule in module._modules.values():
            reset_module_stats(submodule, recursive=True)



# custom module for example
class Constant(InspectModule):
    def input(self, data: Any):
        self._data = data

    # default return with no typing decoration
    def forward(self) -> Any:
        return self._data


class InspectConstant(InspectModule):
    def input(self, data: Any):
        self._data = data

    # use typing decoration to specify the return type
    @returns_keys(data=Any)
    def forward(self):
        return {"data": self._data}


class Addition(InspectModule):
    @returns_keys(result=int)
    def forward(self, a: int, b: int): # simple addition module
        return {"result": a + b}


class Multiplication(InspectModule):
    @returns_keys(result=int)
    def forward(self, a: int, b: int): # simple multiplication module
        return {"result": a * b}