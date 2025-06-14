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
    is_active: bool = True
    is_cached: bool = False
    _cache: Optional[Any] = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, src={self.src}, tgt={self.tgt}, cache={type(self._cache)}, is_cached={self.is_cached}, is_active={self.is_active})"

    @property
    def null(self):
        return self.src is NullNode or self.tgt is NullNode
    
    @property
    def virtual(self):
        return self.src is VirtualNode or self.tgt is VirtualNode
    
    @property
    def cache(self):
        return self._cache

    def __call__(self, force: bool = False):
        if self.is_active or force:
            return self.cache
        else:
            return None

    
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
        self._default_values: Dict[str, Any] = {}

        self.use_default_return = True

    def __repr__(self):
        return super().__repr__()
    
    def extra_repr(self):
        return f"parent={self.parent}, prev={self._prev}, next={self._next}"

    @staticmethod
    def infer_input_parameters(func: Callable) -> Tuple[List[str], Dict[str, Any]]:
        """
        Infer input parameters from a callable function.
        
        Args:
            func: The callable function to inspect
            
        Returns:
            Tuple of (parameter_names, default_values)
            - parameter_names: List of parameter names (excluding self, *args, **kwargs)
            - default_values: Dict mapping parameter names to their default values
        """
        func_signature = inspect.signature(func)
        parameters = list(func_signature.parameters.keys())
        
        # Remove 'self' parameter if present
        if parameters and parameters[0] == 'self':
            parameters = parameters[1:]

        # Filter out *args and **kwargs parameters, collect defaults
        filtered_parameters = []
        default_values = {}
        
        for param in parameters:
            param_obj = func_signature.parameters[param]
            param_kind = param_obj.kind
            
            if (param_kind != inspect.Parameter.VAR_POSITIONAL and 
                param_kind != inspect.Parameter.VAR_KEYWORD):
                filtered_parameters.append(param)
                
                # Check if parameter has a default value
                if param_obj.default != inspect.Parameter.empty:
                    default_values[param] = param_obj.default
        
        return filtered_parameters, default_values

    @staticmethod
    def infer_output_keys(func: Callable) -> Tuple[List[str], bool]:
        """
        Infer output keys from a callable function.
        
        Args:
            func: The callable function to inspect
            
        Returns:
            Tuple of (output_keys, use_default_return)
            - output_keys: List of output key names
            - use_default_return: Whether to use default return behavior
        """
        output_keys = ["_return"]
        use_default_return = True
        
        if hasattr(func, '__returns_keys__'):
            output_keys = list(func.__returns_keys__.keys())
            use_default_return = False
            
        return output_keys, use_default_return

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
                results[k] = edges[0].__call__()
                
            # Log time for cached execution if debugging is enabled
            if get_debug_state():
                elapsed = self._stop_timer()
                print(f"[DEBUG] Module {self.name} returned cached result in {elapsed:.6f}s")
                
            return results

        # Prepare inputs
        self.prepare()
        
        # Execute forward computation
        forward_kwargs = {}
        for k, edge in self._prev.items():
            if edge.is_cached and edge.is_active:
                forward_kwargs[k] = edge.__call__()
            elif k in self._default_values:
                forward_kwargs[k] = self._default_values[k]
        
        forward_results = self.forward(**forward_kwargs)
        
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
                # Check if parameter has default value or edge is inactive
                if _prev_k in self._default_values or not prev_edge.is_active:
                    continue  # Skip error for parameters with defaults or inactive edges
                else:
                    raise ValueError(f"Undefined input: {prev_edge.name}")
            if prev_edge.virtual:
                # Find the mapping using new tuple format
                for group_key, (module_name, module_key) in self.parent._prev_name_map.items():
                    if (module_name, module_key) == (self.name, _prev_k):
                        prev_edge._cache = self.parent._prev[group_key].cache
                        prev_edge.is_cached = True
                        break
                else:
                    raise RuntimeError(f"Virtual edge mapping not found for {self.name}.{_prev_k}")
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

        # Use static methods for inference
        filtered_parameters, default_values = self.infer_input_parameters(func)
        output_keys, use_default_return = self.infer_output_keys(func)
        
        indirect = len(filtered_parameters)
        outdirect = len(output_keys)

        super().__init__(name=name, parent=parent, indirect=indirect, outdirect=outdirect, **kwargs)
        
        # Set use_default_return based on inference
        self.use_default_return = use_default_return

        # Set default values
        self._default_values = default_values

        # Initialize input edges
        for param in filtered_parameters:
            self._prev[param] = Edge(name=param)
        
        # Initialize output edges
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
        
        # Use static methods for inference
        filtered_parameters, default_values = self.infer_input_parameters(self.forward)
        output_keys, use_default_return = self.infer_output_keys(self.forward)
        
        self.indirect = len(filtered_parameters)
        self.outdirect = len(output_keys)
        
        # Set use_default_return based on inference
        self.use_default_return = use_default_return

        # Set default values
        self._default_values = default_values

        # Initialize input edges
        for param in filtered_parameters:
            self._prev[param] = Edge(name=param)
        
        # Initialize output edges
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
    
        def parse_key(key: str) -> Tuple[str, int]:
            if '.' in key:
                parts = key.split('.')
                try:
                    return parts[0], int(parts[1])
                except (ValueError, IndexError):
                    return key, 1
            return key, 1
        
        def create_key_generators(keys: List[str]) -> Dict[str, Callable[[], str]]:
            root_groups = defaultdict(list)
            for key in keys:
                root, num = parse_key(key)
                root_groups[root].append((key, num))
            
            key_generators = {}
            for root, key_list in root_groups.items():
                if len(key_list) == 1:
                    key_generators[root] = lambda r=root: r
                else:
                    def make_generator(root_name):
                        counter = 1
                        def generator():
                            nonlocal counter
                            if counter == 1:
                                result = root_name
                            else:
                                result = f"{root_name}.{counter}"
                            counter += 1
                            return result
                        return generator
                    key_generators[root] = make_generator(root)
            
            return key_generators

        # Collect all keys that need group-level mapping
        prev_keys_to_process = []
        next_keys_to_process = []

        for _, module in self._modules.items():
            # Validate module name doesn't contain separator
            if '.' in module.name:
                raise ValueError(f"Module name '{module.name}' cannot contain '.' character as it's used as separator in mapping")
            
            for key, edge in module._prev.items():
                if edge.null:
                    prev_keys_to_process.append(key)

            for key, edges in module._next.items():
                for edge in edges:
                    if edge.null:
                        next_keys_to_process.append(key)
                        break  # Only count once per key

        # Create key generators to avoid conflicts
        prev_key_generators = create_key_generators(prev_keys_to_process)
        next_key_generators = create_key_generators(next_keys_to_process)

        # Apply the reassigned keys
        for _, module in self._modules.items():
            for key, edge in module._prev.items():
                if edge.null:
                    root, _ = parse_key(key)
                    group_key = prev_key_generators[root]()

                    # Create mapping using tuple for robustness
                    self._prev_name_map[group_key] = (module.name, key)

                    # Create group input edge
                    if group_key in self._prev:
                        raise RuntimeError(f"Internal error: duplicate group key '{group_key}' should not occur with correct naming logic")
                    
                    self._prev[group_key] = Edge(name=group_key, tgt=self)

                    # Set module edge to virtual
                    module._prev[key] = Edge(
                        name=key,
                        src=VirtualNode,
                        tgt=module,
                        tgt_key=key
                    )

                    # Handle default values if they exist
                    if key in module._default_values:
                        self._default_values[group_key] = module._default_values[key]

            for key, edges in module._next.items():
                for idx, edge in enumerate(edges):
                    if edge.null:
                        root, _ = parse_key(key)
                        group_key = next_key_generators[root]()

                        # Create mapping using tuple for robustness
                        self._next_name_map[group_key] = (module.name, key)
                        
                        # Create group output edge
                        if group_key in self._next:
                            raise RuntimeError(f"Internal error: duplicate group key '{group_key}' should not occur with correct naming logic")
                            
                        self._next[group_key] = [Edge(name=group_key, src=self, src_key=group_key)]

                        # Set module edge to virtual
                        module._next[key][idx] = Edge(
                            name=key,
                            src=module,
                            tgt=VirtualNode,
                            src_key=key,
                        )
                        break
        
        self.indirect = len(self._prev)
        self.outdirect = len(self._next)

    def forward(self, **kwargs) -> Any:
        results = {}

        module_set = set()
        for map_key, (module_name, module_key) in self._next_name_map.items():
            module_set.add(module_name)
        
        for module_name in module_set:
            self._modules[module_name]()

        for key, edges in self._next.items():
            module_name, module_key = self._next_name_map[key]
            results[key] = self._modules[module_name]._next[module_key][0].cache
        return results

def connect(
    src: Union['Module', _PlaceholderNodeType] = NullNode, 
    src_key: Optional[str] = None, 
    tgt: Union['Module', _PlaceholderNodeType] = NullNode, 
    tgt_key: Optional[str] = None,
    name: Optional[str] = None
):
    if src is NullNode and tgt is NullNode:
        raise ValueError("Both source and target are NullNodes")

    # Validate src_key exists in source module
    if hasattr(src, "_next") and src_key is not None:
        if src_key not in src._next:
            raise ValueError(f"Source key '{src_key}' not found in module '{src.name}'")
    
    # Validate tgt_key exists in target module
    if hasattr(tgt, "_prev") and tgt_key is not None:
        if tgt_key not in tgt._prev:
            raise ValueError(f"Target key '{tgt_key}' not found in module '{tgt.name}'")
    
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


def clear_cache(module: Module):
    if not module._prev is None:
        for edge in module._prev.values():
            edge.is_cached = False
            edge._cache = None

    if not module._next is None:
        for edges in module._next.values():
            for edge in edges:
                edge.is_cached = False
                edge._cache = None
    
    if isinstance(module, ModuleGroup):
        for child in module._modules.values():
            clear_cache(child)


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