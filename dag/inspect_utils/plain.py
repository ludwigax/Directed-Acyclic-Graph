"""
Utility functions for visualizing and inspecting DAG modules
"""
from typing import Any, Dict, List, Union
from ..node import Module, FunctionModule, InspectModule, ModuleGroup, Edge, NullNode, VirtualNode


def inspect_module(module: Module, show_details: bool = True, indent: int = 0) -> str:
    """
    Generate a comprehensive visualization of a module's interface
    
    Args:
        module: The module to inspect
        show_details: Whether to show detailed information
        indent: Indentation level for nested display
        
    Returns:
        Formatted string representation of the module
    """
    lines = []
    prefix = "  " * indent
    
    # Module header
    module_type = module.__class__.__name__
    lines.append(f"{prefix}📦 {module_type}: {module.name}")
    lines.append(f"{prefix}{'=' * (len(module_type) + len(module.name) + 4)}")
    
    # Basic info
    if show_details:
        lines.append(f"{prefix}🔧 Parent: {module.parent.name if module.parent else 'None'}")
        lines.append(f"{prefix}📊 Input Count: {module.indirect or len(module._prev)}")
        lines.append(f"{prefix}📤 Output Count: {module.outdirect or len(module._next)}")
        lines.append("")
    
    # Input parameters
    if module._prev:
        lines.append(f"{prefix}📥 INPUTS:")
        lines.append(f"{prefix}{'-' * 40}")
        for param_name, edge in module._prev.items():
            status = _get_edge_status(edge)
            default_info = ""
            if param_name in getattr(module, '_default_values', {}):
                default_val = module._default_values[param_name]
                default_info = f" (default: {repr(default_val)})"
            
            lines.append(f"{prefix}  • {param_name}{default_info}")
            if show_details:
                lines.append(f"{prefix}    Status: {status}")
                if edge.src != NullNode and edge.src != VirtualNode:
                    lines.append(f"{prefix}    Source: {edge.src}")
        lines.append("")
    else:
        lines.append(f"{prefix}📥 INPUTS: None")
        lines.append("")
    
    # Output parameters
    if module._next:
        lines.append(f"{prefix}📤 OUTPUTS:")
        lines.append(f"{prefix}{'-' * 40}")
        for output_name, edges in module._next.items():
            lines.append(f"{prefix}  • {output_name}")
            if show_details and edges:
                for i, edge in enumerate(edges):
                    status = _get_edge_status(edge)
                    lines.append(f"{prefix}    [{i}] Status: {status}")
                    if edge.tgt != NullNode and edge.tgt != VirtualNode:
                        lines.append(f"{prefix}    [{i}] Target: {edge.tgt}")
        lines.append("")
    else:
        lines.append(f"{prefix}📤 OUTPUTS: None")
        lines.append("")
    
    # Special handling for different module types
    if isinstance(module, FunctionModule):
        lines.extend(_inspect_function_module(module, prefix))
    elif isinstance(module, InspectModule) and not isinstance(module, ModuleGroup):
        lines.extend(_inspect_inspect_module(module, prefix))
    elif isinstance(module, ModuleGroup):
        lines.extend(_inspect_module_group(module, prefix, show_details))
    
    return "\n".join(lines)


def _get_edge_status(edge: Edge) -> str:
    """Get human-readable status of an edge"""
    if edge.src == NullNode or edge.tgt == NullNode:
        return "🔴 Null (Unconnected)"
    elif edge.src == VirtualNode or edge.tgt == VirtualNode:
        return "🟡 Virtual (Group Interface)"
    elif edge.is_cached:
        cache_type = type(edge._cache).__name__ if edge._cache is not None else "None"
        return f"🟢 Cached ({cache_type})"
    elif edge.is_active:
        return "🔵 Active (Ready)"
    else:
        return "⚫ Inactive"


def _inspect_function_module(module: FunctionModule, prefix: str) -> List[str]:
    """Inspect FunctionModule specific details"""
    lines = []
    lines.append(f"{prefix}🔍 FUNCTION DETAILS:")
    lines.append(f"{prefix}{'-' * 40}")
    lines.append(f"{prefix}  Function: {module.func.__name__}")
    if hasattr(module.func, '__doc__') and module.func.__doc__:
        doc = module.func.__doc__.strip().split('\n')[0]  # First line only
        lines.append(f"{prefix}  Description: {doc}")
    lines.append(f"{prefix}  Signature: {module.func_signature}")
    lines.append(f"{prefix}  Use Default Return: {module.use_default_return}")
    lines.append("")
    return lines


def _inspect_inspect_module(module: InspectModule, prefix: str) -> List[str]:
    """Inspect InspectModule specific details"""
    lines = []
    lines.append(f"{prefix}🔍 INSPECT MODULE DETAILS:")
    lines.append(f"{prefix}{'-' * 40}")
    if hasattr(module, 'forward') and hasattr(module.forward, '__doc__') and module.forward.__doc__:
        doc = module.forward.__doc__.strip().split('\n')[0]  # First line only
        lines.append(f"{prefix}  Forward Description: {doc}")
    lines.append(f"{prefix}  Use Default Return: {module.use_default_return}")
    lines.append("")
    return lines


def _inspect_module_group(module: ModuleGroup, prefix: str, show_details: bool) -> List[str]:
    """Inspect ModuleGroup specific details"""
    lines = []
    lines.append(f"{prefix}🏗️  MODULE GROUP DETAILS:")
    lines.append(f"{prefix}{'-' * 40}")
    lines.append(f"{prefix}  Child Modules: {len(module._modules)}")
    
    # Input mappings
    if module._prev_name_map:
        lines.append(f"{prefix}  📥 Input Mappings:")
        for group_key, mapping in module._prev_name_map.items():
            if isinstance(mapping, tuple) and len(mapping) == 2:
                if hasattr(mapping[0], 'name'):  # Module reference
                    module_name = mapping[0].name
                else:  # String name
                    module_name = mapping[0]
                lines.append(f"{prefix}    {group_key} → {module_name}.{mapping[1]}")
            else:
                lines.append(f"{prefix}    {group_key} → {mapping}")
    
    # Output mappings
    if module._next_name_map:
        lines.append(f"{prefix}  📤 Output Mappings:")
        for group_key, mapping in module._next_name_map.items():
            if isinstance(mapping, tuple) and len(mapping) == 2:
                if hasattr(mapping[0], 'name'):  # Module reference
                    module_name = mapping[0].name
                else:  # String name
                    module_name = mapping[0]
                lines.append(f"{prefix}    {group_key} → {module_name}.{mapping[1]}")
            else:
                lines.append(f"{prefix}    {group_key} → {mapping}")
    
    # Child modules summary
    if show_details and module._modules:
        lines.append(f"{prefix}  🧩 Child Modules:")
        for child_name, child_module in module._modules.items():
            child_type = child_module.__class__.__name__
            input_count = len(child_module._prev)
            output_count = len(child_module._next)
            lines.append(f"{prefix}    • {child_name} ({child_type}) - In:{input_count}, Out:{output_count}")
    
    lines.append("")
    return lines


def inspect_module_tree(module: Module, max_depth: int = 2, current_depth: int = 0) -> str:
    """
    Generate a tree view of module hierarchy
    
    Args:
        module: Root module to inspect
        max_depth: Maximum depth to traverse
        current_depth: Current traversal depth
        
    Returns:
        Tree-formatted string representation
    """
    lines = []
    
    # Current module info
    lines.append(inspect_module(module, show_details=(current_depth == 0), indent=current_depth))
    
    # Recurse into child modules if it's a ModuleGroup and we haven't reached max depth
    if isinstance(module, ModuleGroup) and current_depth < max_depth and module._modules:
        lines.append("  " * current_depth + "📁 CHILD MODULES:")
        lines.append("  " * current_depth + "=" * 50)
        for child_name, child_module in module._modules.items():
            lines.append(inspect_module_tree(child_module, max_depth, current_depth + 1))
    
    return "\n".join(lines)


def print_module(module: Module, detailed: bool = True, tree_view: bool = False, max_depth: int = 2):
    """
    Print module information to console
    
    Args:
        module: Module to inspect
        detailed: Whether to show detailed information
        tree_view: Whether to show tree view for ModuleGroups
        max_depth: Maximum depth for tree view
    """
    if tree_view and isinstance(module, ModuleGroup):
        print(inspect_module_tree(module, max_depth))
    else:
        print(inspect_module(module, detailed))
