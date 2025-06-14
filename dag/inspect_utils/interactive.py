"""
Interactive DAG visualization using Flask + Plotly
"""
import networkx as nx
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import warnings

from ..node import Module, ModuleGroup, Edge, NullNode, VirtualNode

@dataclass
class INode:
    id: str
    module: Optional[Module]
    name: str
    type: str    # 考虑 module, module_group, virtual, null，用不同颜色圆形
    clickable: bool
    inputs: List[str]    # inputs, outputs和default_values都保存的data，用于备用。
    outputs: List[str] 
    default_values: Dict[str, Any]


@dataclass
class IParamNode:
    id: str
    name: str
    type: str = "param"   # 考虑为方形
    clickable: bool = False   # 默认无法点击
    is_default: bool = False   # 该参数是否包含默认参数，如果包含，则需要显示默认参数（default_value）
    default_value: Any = None

@dataclass
class IEdge:
    id: str
    edge: Optional[Edge]
    source: str
    target: str
    source_key: Optional[str] = None # 留作备用
    target_key: Optional[str] = None # 留作备用
    edge_type: str = "normal"

class GraphState:
    """Graph building state container"""
    def __init__(self):
        self.node_counter = 0
        self.edge_counter = 0
        self.nodes: List[Union[INode, IParamNode]] = []
        self.edges: List[IEdge] = []
        self.visited: set = set()
        self.module_to_node: Dict[Module, INode] = {}

    def add_node(self, module: Module) -> INode:
        """Add a module node to the graph"""
        # Check if node already exists
        if module in self.module_to_node:
            return self.module_to_node[module]
        
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        node = INode(
            id=node_id,
            module=module,
            name=module.name,
            type=module.__class__.__name__,
            clickable=self._is_clickable(module),
            inputs=list(module._prev.keys()),
            outputs=list(module._next.keys()),
            default_values=getattr(module, '_default_values', {})
        )
        self.nodes.append(node)
        self.module_to_node[module] = node
        return node

    def add_virtual_node(self, name: str, node_type: str = "virtual") -> INode:
        """Add a virtual node (for ModuleGroup inputs/outputs)"""
        node_id = f"virtual_{self.node_counter}"
        self.node_counter += 1
        
        node = INode(
            id=node_id,
            module=None,
            name=name,
            type=node_type,
            clickable=False,
            inputs=[],
            outputs=[],
            default_values={}
        )
        self.nodes.append(node)
        return node

    def add_param_node(self, name: str, is_default: bool = False, default_value: Any = None) -> IParamNode:
        """Add a parameter node to the graph"""
        node_id = f"param_{self.node_counter}"
        self.node_counter += 1
        
        param_node = IParamNode(
            id=node_id,
            name=name,
            type="param",
            clickable=False,
            is_default=is_default,
            default_value=default_value
        )
        self.nodes.append(param_node)
        return param_node
    
    def _is_clickable(self, module: Module) -> bool:
        """Determine if a module node should be clickable"""
        if module in [NullNode, VirtualNode] or module is None:
            return False
        
        # Module is clickable if it has connections or is a ModuleGroup
        has_connections = (len(module._prev) > 0 or len(module._next) > 0)
        is_group = isinstance(module, ModuleGroup)
        
        return has_connections or is_group
    
    def add_edge(self, source_node: str, target_node: str, source_key: str = None, target_key: str = None, edge_type: str = "normal", original_edge: Edge = None) -> IEdge:
        """Add an edge to the graph"""
        edge_id = f"edge_{self.edge_counter}"
        self.edge_counter += 1
        
        iedge = IEdge(
            id=edge_id,
            edge=original_edge,
            source=source_node,
            target=target_node,
            source_key=source_key,
            target_key=target_key,
            edge_type=edge_type,
        )
        self.edges.append(iedge)
        return iedge

    def get_node_by_id(self, node_id: str) -> Optional[INode]:
        """Get node by node ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

def _universal_graph_iteration(current_module: Module, state: GraphState, from_edge: Edge = None):
    """
    Recursive universal graph iteration - flatten the entire DAG structure
    Handles null_node, virtual_node, and regular connections
    
    Args:
        current_module: Current module to process
        state: Graph building state
        from_edge: The edge we came from (to avoid going back)
    """
    if id(current_module) in state.visited:
        return
        
    state.visited.add(id(current_module))
    
    # Add current module as node
    current_node = state.add_node(current_module)
    
    # Process all outgoing edges (_next)
    for output_key, edge_list in current_module._next.items():
        for edge in edge_list:
            # Skip the edge we came from
            if edge == from_edge:
                continue
                
            if edge.null:
                # Null connection - create null node placeholder
                null_node = state.add_virtual_node(f"Null Target", "null_node")
                state.add_edge(
                    current_node.id, null_node.id,
                    output_key, None,
                    "null", edge
                )
            elif edge.virtual:
                # Virtual connection - create virtual node  
                virtual_node = state.add_virtual_node(f"Virtual Target", "virtual_node")
                state.add_edge(
                    current_node.id, virtual_node.id,
                    output_key, edge.tgt_key,
                    "virtual", edge
                )
            else:
                # Real connection - recursively process target module
                target_module = edge.tgt
                if target_module not in [NullNode, VirtualNode]:
                    # First add the target node (may already exist)
                    target_node = state.add_node(target_module)
                    state.add_edge(
                        current_node.id, target_node.id,
                        output_key, edge.tgt_key,
                        "normal", edge
                    )
                    # Recursively process target module
                    _universal_graph_iteration(target_module, state, edge)
    
    # Process all incoming edges (_prev) 
    for input_key, edge in current_module._prev.items():
        # Skip the edge we came from
        if edge == from_edge:
            continue
            
        if edge.null:
            # Null connection - create null node placeholder
            null_node = state.add_virtual_node(f"Null Source", "null_node")
            state.add_edge(
                null_node.id, current_node.id,
                None, input_key,
                "null", edge
            )
        elif edge.virtual:
            # Virtual connection - create virtual node
            virtual_node = state.add_virtual_node(f"Virtual Source", "virtual_node")
            state.add_edge(
                virtual_node.id, current_node.id,
                edge.src_key, input_key,
                "virtual", edge
            )
        else:
            # Real connection - recursively process source module
            source_module = edge.src
            if source_module not in [NullNode, VirtualNode]:
                # First add the source node (may already exist)
                source_node = state.add_node(source_module)
                state.add_edge(
                    source_node.id, current_node.id,
                    edge.src_key, input_key,
                    "normal", edge
                )
                # Recursively process source module
                _universal_graph_iteration(source_module, state, edge)

def build_graph(endpoint: Module) -> Tuple[List[Union[INode, IParamNode]], List[IEdge]]:
    """
    Build main graph view - universal graph iteration
    Flattens the entire DAG structure
    """
    state = GraphState()
    _universal_graph_iteration(endpoint, state)
    return state.nodes, state.edges

def build_details(module: Module) -> Tuple[List[Union[INode, IParamNode]], List[IEdge]]:
    """
    Build detailed view for a specific module (for clicking)
    Uses auxiliary methods for different module types
    """
    state = GraphState()
    
    if isinstance(module, ModuleGroup):
        # ModuleGroup内部迭代：显示内部结构+key映射，不递归嵌套的ModuleGroup
        _iterate_module_group_internal(module, state)
    else:
        # 普通module迭代：显示该模块的输入输出key
        _iterate_regular_module_keys(module, state)
    
    return state.nodes, state.edges

def _iterate_module_group_internal(module_group: ModuleGroup, state: GraphState):
    """
    辅助方法1: ModuleGroup内部迭代
    Step 1: 使用_universal_graph_iteration构造内部节点图
    Step 2: 找到virtual nodes并转换为参数节点
    """
    # Step 1: 使用_universal_graph_iteration构造内部节点图
    # 对ModuleGroup的每个子module运行universal iteration
    for child_name, child_module in module_group._modules.items():
        if id(child_module) not in state.visited:
            _universal_graph_iteration(child_module, state)
    
    # Step 2: 找到virtual nodes并转换为参数节点
    # 收集所有virtual nodes (来自virtual edges)
    virtual_nodes_to_convert = []
    
    for iedge in state.edges[:]:  # 使用切片避免迭代时修改
        if iedge.edge and (iedge.edge.virtual):
            # 找到virtual edge对应的virtual nodes
            if iedge.edge_type == "virtual":
                # 检查source或target是否为virtual node
                source_node = state.get_node_by_id(iedge.source)
                target_node = state.get_node_by_id(iedge.target)
                
                if source_node and source_node.type == "virtual_node":
                    virtual_nodes_to_convert.append((source_node, "input", iedge))
                if target_node and target_node.type == "virtual_node":
                    virtual_nodes_to_convert.append((target_node, "output", iedge))
    
    # 转换virtual nodes为参数节点
    for virtual_node, param_type, related_edge in virtual_nodes_to_convert:
        if param_type == "input":
            # 输入参数：virtual node -> real module
            # 通过_prev_name_map查找对应的group input key
            for group_input_key, (child_name, child_key) in module_group._prev_name_map.items():
                
                if related_edge.target_key == child_key and child_name == related_edge.edge.tgt.name:
                    # 创建新的参数节点来替换virtual node
                    has_default = hasattr(module_group, '_default_values') and group_input_key in module_group._default_values
                    default_value = getattr(module_group, '_default_values', {}).get(group_input_key, None) if has_default else None
                    
                    param_node = IParamNode(
                        id=virtual_node.id,  # 保持相同的ID
                        name=f"Input Param: {group_input_key}",
                        type="param",
                        clickable=False,
                        is_default=has_default,
                        default_value=default_value
                    )
                    
                    # 在nodes列表中替换virtual node
                    for i, node in enumerate(state.nodes):
                        if node.id == virtual_node.id:
                            state.nodes[i] = param_node
                            break
                    
                    # 更新edge信息
                    related_edge.source_key = group_input_key
                    related_edge.edge_type = "param_mapping"
                    break
        
        elif param_type == "output":
            # 输出参数：real module -> virtual node  
            # 通过_next_name_map查找对应的group output key
            for group_output_key, (child_name, child_key) in module_group._next_name_map.items():
                if related_edge.source_key == child_key and child_name == related_edge.edge.src.name:
                    # 创建新的参数节点来替换virtual node
                    param_node = IParamNode(
                        id=virtual_node.id,  # 保持相同的ID
                        name=f"Output Param: {group_output_key}",
                        type="param",
                        clickable=False,
                        is_default=False,  # 输出参数通常没有默认值
                        default_value=None
                    )
                    
                    # 在nodes列表中替换virtual node
                    for i, node in enumerate(state.nodes):
                        if node.id == virtual_node.id:
                            state.nodes[i] = param_node
                            break
                    
                    # 更新edge信息
                    related_edge.target_key = group_output_key
                    related_edge.edge_type = "param_mapping"
                    break

def _iterate_regular_module_keys(module: Module, state: GraphState):
    """
    辅助方法2: 普通module迭代  
    Step 1: 添加裸节点（该module本身）
    Step 2: 为每个输入输出创建参数节点并连接
    """
    # Step 1: 添加裸节点（该module本身）
    main_node = state.add_node(module)
    
    # Step 2: 为每个输入创建参数节点
    for input_key in module._prev.keys():
        # Get edge info for additional details
        edge = module._prev[input_key]
        has_default = input_key in getattr(module, '_default_values', {})
        default_value = getattr(module, '_default_values', {}).get(input_key, None) if has_default else None
        
        # Create input parameter node (square shape)
        input_param_node = state.add_param_node(
            f"Input: {input_key}",
            is_default=has_default,
            default_value=default_value
        )
        
        # Connect parameter node to main module
        state.add_edge(
            input_param_node.id, main_node.id,
            input_key, input_key,
            "param_connection"
        )
    
    # Step 3: 为每个输出创建参数节点
    for output_key in module._next.keys():
        edge_list = module._next[output_key]
        
        # Create output parameter node (square shape)
        output_param_node = state.add_param_node(
            f"Output: {output_key}",
            is_default=False  # 输出参数通常没有默认值
        )
        
        # Connect main module to parameter node
        state.add_edge(
            main_node.id, output_param_node.id,
            output_key, output_key,
            "param_connection"
        )

def get_node_id(module: Module, nodes: List[Union[INode, IParamNode]]) -> Optional[str]:
    """Get node ID by module reference"""
    for node in nodes:
        if isinstance(node, INode) and node.module == module:
            return node.id
    return None

def get_edge_id(edge: Edge, edges: List[IEdge]) -> Optional[str]:
    """Get edge ID by edge reference"""
    for iedge in edges:
        if iedge.edge == edge:
            return iedge.id
    return None