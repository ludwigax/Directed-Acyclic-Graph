"""
Flask application for interactive DAG visualization

Design Logic:
- Main Graph: Only contains INode (modules, virtual nodes, null nodes) - all circles
- Detail View: 
  * Regular Module: 1 main INode + multiple IParamNode (parameters) - circles + squares
  * ModuleGroup: Multiple child INode + IParamNode (when virtual->param conversion is implemented)

Layout Algorithms:
- Main Graph: Spring Layout (force-directed) - provides natural spacing based on connections
- Detail View: 
  * Single Module: Shell Layout - main module centered, parameters in outer ring
  * ModuleGroup: Spring Layout - distributes child modules based on their connections
  * Fallback: Circular Layout - simple uniform distribution
"""
from flask import Flask, render_template, request, jsonify, Response
import plotly.graph_objects as go
import plotly.utils
import json
import math
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
import warnings
import traceback
import networkx as nx
import sys
import io
import time
from contextlib import redirect_stdout, redirect_stderr

from .interactive import build_graph, build_details, INode, IParamNode, IEdge
from ..node import Module, ModuleGroup, Edge, NullNode, VirtualNode, clear_cache


class VisualizationStyle:
    """Centralized style configuration for DAG visualization"""
    
    # Edge colors
    EDGE_COLORS = {
        'normal': '#888',               # Default gray
        'internal': '#4CAF50',          # Green for internal connections within groups
        'virtual': '#9C27B0',           # Purple for virtual connections
        'null': '#F44336',              # Red for null connections
        'param_connection': '#FF9800',  # Orange for parameter connections
        'param_mapping': '#2196F3',     # Blue for parameter mapping
        'executed': '#FF6B6B'           # Red for executed edges
    }
    
    # Edge widths
    EDGE_WIDTHS = {
        'normal': 2,
        'executed': 3,
        'default': 2
    }
    
    # Arrow widths
    ARROW_WIDTHS = {
        'normal': 1.5,
        'executed': 2,
        'default': 1.5
    }
    
    # Node colors
    NODE_COLORS = {
        'module_group': '#FFD700',      # Gold for ModuleGroup
        'module': '#87CEEB',            # Sky blue for regular modules
        'virtual_node': '#DDA0DD',      # Plum for virtual nodes
        'null_node': '#FFB6C1',         # Light pink for null nodes
        'input_parameter': '#FF9800',   # Orange for input parameters
        'output_parameter': '#4CAF50',  # Green for output parameters
        'non_clickable': '#D3D3D3'      # Light gray for non-clickable
    }
    
    # Node sizes
    NODE_SIZES = {
        'main_graph': 80,               # Main graph node size
        'detail_view': 60               # Detail view node size
    }
    
    # Node symbols
    NODE_SYMBOLS = {
        'module': 'circle',
        'parameter': 'square'
    }
    
    # Text styles
    TEXT_STYLES = {
        'node_font_size_main': 12,
        'node_font_size_detail': 11,
        'node_font_family': 'Arial Black',
        'node_font_color': 'black',
        'annotation_font_size': 12,
        'annotation_font_color': 'gray',
        'title_font_size': 16
    }
    
    # Layout settings
    LAYOUT_SETTINGS = {
        'canvas_width': 1700,
        'canvas_height': 800,
        'margin': {'b': 20, 'l': 5, 'r': 5, 't': 40},
        'node_border_width': 3,
        'node_border_color': 'black'
    }
    
    # Layout algorithms scale factors
    LAYOUT_SCALES = {
        'hierarchical': 4,
        'multipartite': 4,
        'spring': 4,
        'shell': 3,
        'circular': 3
    }
    
    # Layout spacing settings
    LAYOUT_SPACING = {
        'layer_separation': 3.0,      # 增加层间距离
        'node_separation': 2.0,       # 增加同层节点间距
        'min_node_distance': 110,     # 节点间最小距离（像素）
        'overlap_threshold': 0.8,     # 重叠检测阈值
    }
    
    # Arrow settings
    ARROW_SETTINGS = {
        'head_type': 2,
        'size': 1.5,              # Arrow head size
        'width': 1.5,             # Arrow line width (pixels)
        'pixel_length': 15,       # Fixed pixel length for short arrows
        'pixel_length_executed': 20  # Larger arrows for executed edges
    }
    
    # Edge curve settings for avoiding overlaps
    CURVE_SETTINGS = {
        'enable_curves': True,
        'curve_strength': 0.3,    # How much to curve the edges
        'overlap_threshold': 0.1, # Distance threshold to consider edges overlapping
        'curve_points': 20,       # Number of points in curved edge
    }
    
    @classmethod
    def get_edge_color(cls, edge_type: str) -> str:
        """Get edge color by type"""
        return cls.EDGE_COLORS.get(edge_type, cls.EDGE_COLORS['normal'])
    
    @classmethod
    def get_edge_width(cls, edge_type: str) -> int:
        """Get edge width by type"""
        if edge_type == 'executed':
            return cls.EDGE_WIDTHS['executed']
        return cls.EDGE_WIDTHS.get(edge_type, cls.EDGE_WIDTHS['default'])
    
    @classmethod
    def get_arrow_width(cls, edge_type: str) -> float:
        """Get arrow width by type"""
        if edge_type == 'executed':
            return cls.ARROW_WIDTHS['executed']
        return cls.ARROW_WIDTHS.get(edge_type, cls.ARROW_WIDTHS['default'])
    
    @classmethod
    def get_node_color(cls, node_type: str, is_clickable: bool = False, param_direction: str = None, module: Module = None) -> str:
        """Get node color based on node type"""
        # Parameter nodes use different colors for input/output
        if node_type == 'param':
            if param_direction == 'input':
                return cls.NODE_COLORS['input_parameter']
            elif param_direction == 'output':
                return cls.NODE_COLORS['output_parameter']
            else:
                # Fallback to input color if direction not specified
                return cls.NODE_COLORS['input_parameter']
        
        # ModuleGroup always uses gold color
        if module and isinstance(module, ModuleGroup) or node_type == 'ModuleGroup':
            return cls.NODE_COLORS['module_group']
        
        # Virtual and null nodes have specific colors
        if node_type == 'virtual_node':
            return cls.NODE_COLORS['virtual_node']
        elif node_type == 'null_node':
            return cls.NODE_COLORS['null_node']
        
        # Regular modules use blue if clickable, gray if not
        if is_clickable:
            return cls.NODE_COLORS['module']
        else:
            return cls.NODE_COLORS['non_clickable']
    
    @classmethod
    def get_node_size(cls, is_detail_view: bool = False) -> int:
        """Get node size based on view type"""
        return cls.NODE_SIZES['detail_view'] if is_detail_view else cls.NODE_SIZES['main_graph']
    
    @classmethod
    def get_node_symbol(cls, node_type: str) -> str:
        """Get node symbol based on type"""
        if node_type == 'param':
            return cls.NODE_SYMBOLS['parameter']
        return cls.NODE_SYMBOLS['module']


class DAGVisualizationApp:
    """Flask application for interactive DAG visualization"""
    
    def __init__(self, app_name: str = "DAG Visualizer"):
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.secret_key = 'dag_visualizer_secret_key'
        
        self.current_module: Optional[Module] = None
        self.module_registry: Dict[str, Module] = {}  # Store modules by ID for click handling
        self.navigation_stack: List[Dict[str, Any]] = []  # Stack for navigation history
        self.execution_logs: List[str] = []
        self.style = VisualizationStyle()
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dag_visualizer.html')
        
        @self.app.route('/api/visualize', methods=['POST'])
        def visualize():
            """Main visualization endpoint"""
            try:
                if not self.current_module:
                    return jsonify({
                        'success': False,
                        'error': 'No module set for visualization'
                    })
                
                # Clear navigation stack for new visualization
                self.navigation_stack = []
                
                # Build graph data
                nodes, edges = build_graph(self.current_module)
                
                # Check for cycles
                has_cycles = False
                cycles = []
                try:
                    # Create a directed graph for cycle detection
                    G = nx.DiGraph()
                    for edge in edges:
                        if edge.edge and not edge.edge.null and not edge.edge.virtual:
                            G.add_edge(edge.source, edge.target)
                    
                    cycles = list(nx.simple_cycles(G))
                    has_cycles = len(cycles) > 0
                except:
                    pass  # Ignore cycle detection errors
                
                # Convert to Plotly format
                plotly_data = self._convert_to_plotly(nodes, edges)
                
                return jsonify({
                    'success': True,
                    'plotly_data': plotly_data,
                    'has_cycles': has_cycles,
                    'cycles': cycles
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/node_detail', methods=['POST'])
        def node_detail():
            """Get detailed view of a specific node"""
            try:
                data = request.get_json()
                node_id = data.get('node_id')
                
                if not node_id or node_id not in self.module_registry:
                    return jsonify({
                        'success': False,
                        'error': f'Module not found for node ID: {node_id}'
                    })
                
                module = self.module_registry[node_id]
                
                # Save current state to navigation stack
                current_state = {
                    'type': 'detail',
                    'module': module,
                    'module_name': module.name
                }
                self.navigation_stack.append(current_state)
                
                # Build detail data
                nodes, edges = build_details(module)
                
                # Convert to Plotly format
                plotly_data = self._convert_detail_to_plotly(nodes, edges, module)
                
                return jsonify({
                    'success': True,
                    'plotly_data': plotly_data,
                    'module_name': module.name
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/back', methods=['POST'])
        def back():
            """Go back to previous level"""
            try:
                if not self.navigation_stack:
                    # Go back to main overview
                    nodes, edges = build_graph(self.current_module)
                    plotly_data = self._convert_to_plotly(nodes, edges)
                    
                    return jsonify({
                        'success': True,
                        'plotly_data': plotly_data,
                        'is_detail_view': False,
                        'can_go_back': False
                    })
                
                # Remove current state
                self.navigation_stack.pop()
                
                if not self.navigation_stack:
                    # Back to main overview
                    nodes, edges = build_graph(self.current_module)
                    plotly_data = self._convert_to_plotly(nodes, edges)
                    
                    return jsonify({
                        'success': True,
                        'plotly_data': plotly_data,
                        'is_detail_view': False,
                        'can_go_back': False
                    })
                else:
                    # Back to previous detail view
                    previous_state = self.navigation_stack[-1]
                    module = previous_state['module']
                    
                    nodes, edges = build_details(module)
                    plotly_data = self._convert_detail_to_plotly(nodes, edges, module)
                    
                    return jsonify({
                        'success': True,
                        'plotly_data': plotly_data,
                        'is_detail_view': True,
                        'module_name': module.name,
                        'can_go_back': len(self.navigation_stack) > 1
                    })
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/execute', methods=['POST'])
        def execute():
            try:
                if not self.current_module:
                    return jsonify({
                        'success': False,
                        'error': 'No module set for execution'
                    })
                
                # Determine which module to execute
                if self.navigation_stack:
                    # Execute the module in current detail view
                    current_state = self.navigation_stack[-1]
                    module_to_execute = current_state['module']
                else:
                    # Execute the main module
                    module_to_execute = self.current_module
                
                # Capture output
                output_buffer = io.StringIO()
                error_buffer = io.StringIO()
                
                start_time = time.time()
                execution_status = "success"
                results = None
                
                try:
                    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                        results = module_to_execute()
                except Exception as e:
                    execution_status = "error"
                    error_buffer.write(str(e))
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Collect output logs
                output_logs = []
                stdout_content = output_buffer.getvalue()
                stderr_content = error_buffer.getvalue()
                
                if stdout_content:
                    output_logs.extend(stdout_content.strip().split('\n'))
                if stderr_content:
                    output_logs.extend(['ERROR: ' + line for line in stderr_content.strip().split('\n')])
                
                # Store execution logs
                self.execution_logs.extend(output_logs)
                
                # Get updated visualization with executed edges
                if self.navigation_stack:
                    nodes, edges = build_details(module_to_execute)
                    plotly_data = self._convert_detail_to_plotly(nodes, edges, module_to_execute)
                else:
                    nodes, edges = build_graph(self.current_module)
                    plotly_data = self._convert_to_plotly(nodes, edges)
                
                return jsonify({
                    'success': True,
                    'results': results,
                    'execution_info': {
                        'module_name': module_to_execute.name,
                        'execution_time': f"{execution_time:.6f}",
                        'status': execution_status
                    },
                    'output_logs': output_logs,
                    'plotly_data': plotly_data
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/edge_status', methods=['POST'])
        def edge_status():
            try:
                if not self.current_module:
                    return jsonify({
                        'success': False,
                        'error': 'No module set'
                    })
                
                # Determine which module to get edge status from
                if self.navigation_stack:
                    current_state = self.navigation_stack[-1]
                    target_module = current_state['module']
                    nodes, edges = build_details(target_module)
                    plotly_data = self._convert_detail_to_plotly(nodes, edges, target_module)
                else:
                    target_module = self.current_module
                    nodes, edges = build_graph(self.current_module)
                    plotly_data = self._convert_to_plotly(nodes, edges)
                
                # Collect edge status information
                edge_status_list = []
                for iedge in edges:
                    if iedge.edge:
                        edge_info = {
                            'name': iedge.edge.name,
                            'is_cached': iedge.edge.is_cached,
                            'is_active': iedge.edge.is_active,
                            'edge_type': iedge.edge_type,
                            'cache_type': type(iedge.edge._cache).__name__ if iedge.edge._cache is not None else None
                        }
                        edge_status_list.append(edge_info)
                
                return jsonify({
                    'success': True,
                    'edge_status': edge_status_list,
                    'plotly_data': plotly_data
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/clear_cache', methods=['POST'])
        def clear_module_cache():
            """Clear all edge caches and refresh visualization"""
            try:
                if not self.current_module:
                    return jsonify({
                        'success': False,
                        'error': 'No module set for cache clearing'
                    })
                
                # Determine which module's cache to clear
                if self.navigation_stack:
                    current_state = self.navigation_stack[-1]
                    target_module = current_state['module']
                else:
                    target_module = self.current_module
                
                # Clear the cache
                clear_cache(target_module)
                
                # Get updated visualization
                if self.navigation_stack:
                    nodes, edges = build_details(target_module)
                    plotly_data = self._convert_detail_to_plotly(nodes, edges, target_module)
                    is_detail_view = True
                    module_name = target_module.name
                else:
                    nodes, edges = build_graph(self.current_module)
                    plotly_data = self._convert_to_plotly(nodes, edges)
                    is_detail_view = False
                    module_name = self.current_module.name
                
                return jsonify({
                    'success': True,
                    'message': f'Cleared cache for {target_module.name}',
                    'plotly_data': plotly_data,
                    'is_detail_view': is_detail_view,
                    'module_name': module_name if is_detail_view else None
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
    
    def _format_node_text(self, text: str, max_length: int = 12) -> str:
        """Format node text with line breaks for better display"""
        if len(text) <= max_length:
            return text
        
        # Split long text into multiple lines
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "<br>".join(lines)

    def _get_layout_positions(self, nodes: List[Union[INode, IParamNode]], edges: List[IEdge]) -> Dict:
        """Get node positions using appropriate layout algorithm"""
        G = nx.DiGraph()
        
        # Add nodes to graph
        for node in nodes:
            if isinstance(node, INode):
                G.add_node(node.id, node_type='module', name=node.name)
            elif isinstance(node, IParamNode):
                G.add_node(node.id, node_type='param', name=node.name)
        
        # Add edges to graph
        for edge in edges:
            G.add_edge(edge.source, edge.target, edge_type=edge.edge_type)

        if len(G.nodes()) <= 1:
            node_id = list(G.nodes())[0] if G.nodes() else None
            return {node_id: (0, 0)} if node_id else {}
        
        try:
            # 根据节点数量动态调整间距参数
            node_count = len(G.nodes())
            
            # 基础间距参数
            base_nodesep = 1.5  # 同层节点间距
            base_ranksep = 2.0  # 层间距离
            
            # 根据节点数量调整间距
            if node_count > 20:
                nodesep = base_nodesep * 1.5
                ranksep = base_ranksep
            elif node_count > 10:
                nodesep = base_nodesep * 1.2
                ranksep = base_ranksep
            else:
                nodesep = base_nodesep
                ranksep = base_ranksep
            
            # 增强的GraphViz参数
            graphviz_args = [
                '-Grankdir=LR',                    # 从左到右布局
                f'-Gnodesep={nodesep}',           # 同层节点间距（动态调整）
                f'-Granksep={ranksep}',           # 层间距离（动态调整）
                '-Gsplines=true',                  # 启用边的样条线
                '-Goverlap=false',                 # 防止节点重叠
                '-Gconcentrate=false',             # 不合并边
                '-Gpack=true',                     # 紧凑布局
                '-Gpackmode=clust',                # 聚类打包模式
                '-Gsep=+20',                       # 额外的分离距离
                '-Gesep=+10',                      # 边的分离距离
                '-Gmclimit=2.0',                   # 最小交叉限制
                '-Gnslimit=2.0',                   # 节点分离限制
                '-Gremincross=true',               # 减少边交叉
                '-Gsplines=ortho'                  # 使用正交样条线减少重叠
            ]
            
            return nx.nx_agraph.graphviz_layout(G, prog='dot', args=' '.join(graphviz_args))
        except:
            pass

        try:
            if nx.is_directed_acyclic_graph(G):
                # Assign layers based on longest path from sources
                layers = {}
                for node in nx.topological_sort(G):
                    if G.in_degree(node) == 0:
                        layers[node] = 0
                    else:
                        layers[node] = max(layers[pred] for pred in G.predecessors(node)) + 1
                for node, layer in layers.items():
                    G.nodes[node]['layer'] = layer

                pos = nx.multipartite_layout(G, subset_key='layer', scale=self.style.LAYOUT_SCALES['multipartite'])
                return self._adjust_layer_spacing(pos, layers)
        except:
            pass

        return nx.spring_layout(G, k=3, iterations=100, scale=self.style.LAYOUT_SCALES['spring'])
    
    def _adjust_layer_spacing(self, positions: Dict, layers: Dict) -> Dict:
        """Adjust spacing between layers in multipartite layout"""
        if not layers:
            return positions
        
        # Get layer information
        max_layer = max(layers.values())
        layer_separation = self.style.LAYOUT_SPACING['layer_separation']
        
        # Adjust x-coordinates based on layer
        adjusted_positions = {}
        for node, pos in positions.items():
            layer = layers[node]
            # Spread layers evenly with controlled separation
            new_x = layer * layer_separation
            adjusted_positions[node] = (new_x, pos[1])
        
        return adjusted_positions

    def _create_node_trace(self, nodes: List[Union[INode, IParamNode]], node_positions: Dict, is_detail_view: bool = False) -> go.Scatter:
        """Create node trace with centralized styling"""
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_customdata = []
        node_symbols = []
        
        for node in nodes:
            pos = node_positions[node.id]
            node_x.append(pos[0])
            node_y.append(pos[1])
            
            # Determine node type and properties
            if isinstance(node, INode):
                node_type = node.type
                is_clickable = node.clickable
            
                # Node text
                inputs = len(node.inputs)
                outputs = len(node.outputs)
                text = f"{node.name}<br>{node_type}<br>In:{inputs}, Out:{outputs}"
                node_text.append(text)
                
                # Node color based on type and clickability
                node_colors.append(self.style.get_node_color(node_type, is_clickable, module=node.module))
                
                # Node symbol
                node_symbols.append(self.style.get_node_symbol('module'))
                
                # Custom data for click handling
                node_customdata.append({
                    'node_id': node.id,
                    'clickable': is_clickable,
                    'type': node_type
                })
                
            elif isinstance(node, IParamNode):
                # Parameter node - determine direction from name
                has_default = node.is_default
                default_info = f" (default: {node.default_value})" if has_default else ""
                text = f"{node.name}{default_info}<br>Parameter"
                node_text.append(text)
                
                # Determine parameter direction from name
                param_direction = 'input'  # default
                if node.name.startswith('Output'):
                    param_direction = 'output'
                elif node.name.startswith('Input'):
                    param_direction = 'input'
                
                # Node color based on direction
                node_colors.append(self.style.get_node_color('param', param_direction=param_direction))
                
                # Node symbol
                node_symbols.append(self.style.get_node_symbol('param'))
                
                # Custom data
                node_customdata.append({
                    'node_id': node.id,
                    'clickable': False,
                    'type': 'param',
                    'direction': param_direction
                })
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[self._format_node_text(node.name) for node in nodes],
            textposition="middle center",
            textfont=dict(
                size=self.style.TEXT_STYLES['node_font_size_detail'] if is_detail_view else self.style.TEXT_STYLES['node_font_size_main'],
                color=self.style.TEXT_STYLES['node_font_color'],
                family=self.style.TEXT_STYLES['node_font_family']
            ),
            hovertext=node_text,
            marker=dict(
                size=self.style.get_node_size(is_detail_view),
                color=node_colors,
                symbol=node_symbols,
                line=dict(
                    width=self.style.LAYOUT_SETTINGS['node_border_width'],
                    color=self.style.LAYOUT_SETTINGS['node_border_color']
                )
            ),
            customdata=node_customdata,
            name='Modules   ' if not is_detail_view else 'Nodes'
        )
        
        return node_trace
    
    def _create_edge_traces(self, edges: List[IEdge], node_positions: Dict, is_detail_view: bool = False) -> tuple:
        """Create edge traces and annotations with centralized styling"""
        edge_traces = []
        edge_annotations = []
        
        # Group edges by type and check for execution status
        edges_by_type = {}
        for edge in edges:
            edge_type = edge.edge_type
            
            # Check if edge is executed (cached)
            if edge.edge and edge.edge.is_cached:
                edge_type = 'executed'
            
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)
        
        # Detect potential edge overlaps for curve calculation
        edge_paths = []
        for edge_type, type_edges in edges_by_type.items():
            for edge in type_edges:
                src_pos = node_positions[edge.source]
                tgt_pos = node_positions[edge.target]
                edge_paths.append({
                    'edge': edge,
                    'edge_type': edge_type,
                    'src_pos': src_pos,
                    'tgt_pos': tgt_pos
                })
        
        # Create traces for each edge type
        for edge_type, type_edges in edges_by_type.items():
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(
                    width=self.style.get_edge_width(edge_type),
                    color=self.style.get_edge_color(edge_type)
                ),
                hoverinfo='none',
                mode='lines',
                name=f'{edge_type.title()} Connections',
                showlegend=len(edges_by_type) > 1
            )
            
            # Add edge coordinates and create arrows
            for edge in type_edges:
                src_pos = node_positions[edge.source]
                tgt_pos = node_positions[edge.target]
                
                # Generate curved path if curves are enabled
                if self.style.CURVE_SETTINGS['enable_curves']:
                    edge_x, edge_y = self._generate_curved_edge(
                        src_pos, tgt_pos, edge, edge_paths
                    )
                else:
                    edge_x = [src_pos[0], tgt_pos[0], None]
                    edge_y = [src_pos[1], tgt_pos[1], None]
                
                edge_trace['x'] += tuple(edge_x)
                edge_trace['y'] += tuple(edge_y)
                
                # Create arrow annotation at the midpoint of the edge
                if self.style.CURVE_SETTINGS['enable_curves'] and len(edge_x) > 3:
                    # For curved edges, find the midpoint in the curve
                    # Remove None values to get valid points
                    valid_x = [x for x in edge_x if x is not None]
                    valid_y = [y for y in edge_y if y is not None]
                    
                    if len(valid_x) > 1:
                        mid_idx = len(valid_x) // 2
                        arrow_x = valid_x[mid_idx]
                        arrow_y = valid_y[mid_idx]
                    else:
                        # Fallback to straight line midpoint
                        arrow_x = (src_pos[0] + tgt_pos[0]) / 2
                        arrow_y = (src_pos[1] + tgt_pos[1]) / 2
                else:
                    # For straight edges, use simple midpoint
                    arrow_x = (src_pos[0] + tgt_pos[0]) / 2
                    arrow_y = (src_pos[1] + tgt_pos[1]) / 2
                
                # Create arrow annotation
                # 计算从源节点到目标节点的方向向量
                dx = tgt_pos[0] - src_pos[0]
                dy = tgt_pos[1] - src_pos[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # 计算从源节点到目标节点的方向单位向量
                    unit_dx = dx / length
                    unit_dy = dy / length
                    
                    # 动态调整箭头大小基于边的重要性
                    arrow_size = self.style.ARROW_SETTINGS['size']
                    arrow_width = self.style.ARROW_SETTINGS['width']
                    arrow_pixel_length = self.style.ARROW_SETTINGS['pixel_length']
                    
                    # 对执行状态的边使用更大的箭头
                    if edge_type == 'executed':
                        arrow_size *= 1.3
                        arrow_width *= 1.5
                        arrow_pixel_length = self.style.ARROW_SETTINGS['pixel_length_executed']
                    
                    # 关键修复：箭头整体中心在边的中点，方向从src指向tgt
                    # 箭头尾部相对于中心的像素偏移（向源节点方向后退）
                    tail_offset_x = -unit_dx * (arrow_pixel_length / 2)
                    tail_offset_y = -unit_dy * (arrow_pixel_length / 2)
                    
                    edge_annotations.append(dict(
                        x=arrow_x,                # 边的中点作为箭头中心
                        y=arrow_y,
                        ax=tail_offset_x,         # 尾部相对于中心的像素偏移（向后）
                        ay=-tail_offset_y,
                        xref='x', yref='y',
                        axref='pixel', ayref='pixel',  # 使用像素偏移
                        showarrow=True,
                        arrowhead=self.style.ARROW_SETTINGS['head_type'],
                        arrowsize=arrow_size,
                        arrowwidth=arrow_width,
                        arrowcolor=self.style.get_edge_color(edge_type),
                        text='',
                    ))
            
            edge_traces.append(edge_trace)
        
        return edge_traces, edge_annotations
    
    def _generate_curved_edge(self, src_pos: tuple, tgt_pos: tuple, current_edge: IEdge, all_edge_paths: List[Dict]) -> tuple:
        """Generate curved edge path to avoid overlaps"""
        src_x, src_y = src_pos
        tgt_x, tgt_y = tgt_pos
        
        # Check for overlapping edges
        overlap_count = 0
        overlap_direction = 1  # 1 for curve up, -1 for curve down
        
        for path_info in all_edge_paths:
            if path_info['edge'] == current_edge:
                continue
            
            other_src = path_info['src_pos']
            other_tgt = path_info['tgt_pos']
            
            # Check if edges are roughly parallel and close
            if self._edges_overlap(src_pos, tgt_pos, other_src, other_tgt):
                overlap_count += 1
                # Alternate curve direction for overlapping edges
                if overlap_count % 2 == 0:
                    overlap_direction *= -1
        
        if overlap_count == 0:
            # No overlap, return straight line
            return [src_x, tgt_x, None], [src_y, tgt_y, None]
        
        # Generate curved path using quadratic Bezier curve
        curve_strength = self.style.CURVE_SETTINGS['curve_strength'] * overlap_count * overlap_direction
        curve_points = self.style.CURVE_SETTINGS['curve_points']
        
        # Calculate control point for the curve
        mid_x = (src_x + tgt_x) / 2
        mid_y = (src_y + tgt_y) / 2
        
        # Perpendicular direction for curve offset
        dx = tgt_x - src_x
        dy = tgt_y - src_y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Perpendicular vector
            perp_x = -dy / length * curve_strength
            perp_y = dx / length * curve_strength
            
            control_x = mid_x + perp_x
            control_y = mid_y + perp_y
        else:
            control_x = mid_x
            control_y = mid_y + curve_strength
        
        # Generate points along the Bezier curve
        curve_x = []
        curve_y = []
        
        for i in range(curve_points + 1):
            t = i / curve_points
            # Quadratic Bezier formula: (1-t)²P0 + 2(1-t)tP1 + t²P2
            x = (1-t)**2 * src_x + 2*(1-t)*t * control_x + t**2 * tgt_x
            y = (1-t)**2 * src_y + 2*(1-t)*t * control_y + t**2 * tgt_y
            curve_x.append(x)
            curve_y.append(y)
        
        curve_x.append(None)  # Add separator for next edge
        curve_y.append(None)
        
        return curve_x, curve_y
    
    def _edges_overlap(self, src1: tuple, tgt1: tuple, src2: tuple, tgt2: tuple) -> bool:
        """Check if two edges overlap or are too close"""
        # Simple overlap detection based on distance between edge midpoints
        mid1_x = (src1[0] + tgt1[0]) / 2
        mid1_y = (src1[1] + tgt1[1]) / 2
        mid2_x = (src2[0] + tgt2[0]) / 2
        mid2_y = (src2[1] + tgt2[1]) / 2
        
        distance = math.sqrt((mid1_x - mid2_x)**2 + (mid1_y - mid2_y)**2)
        threshold = self.style.CURVE_SETTINGS['overlap_threshold']
        
        # Also check if edges are roughly parallel
        vec1_x = tgt1[0] - src1[0]
        vec1_y = tgt1[1] - src1[1]
        vec2_x = tgt2[0] - src2[0]
        vec2_y = tgt2[1] - src2[1]
        
        # Normalize vectors
        len1 = math.sqrt(vec1_x**2 + vec1_y**2)
        len2 = math.sqrt(vec2_x**2 + vec2_y**2)
        
        if len1 > 0 and len2 > 0:
            vec1_x /= len1
            vec1_y /= len1
            vec2_x /= len2
            vec2_y /= len2
            
            # Dot product to check if vectors are parallel (close to 1 or -1)
            dot_product = abs(vec1_x * vec2_x + vec1_y * vec2_y)
            is_parallel = dot_product > 0.8  # Threshold for considering vectors parallel
            
            return distance < threshold and is_parallel
        
        return distance < threshold

    def _create_layout(self, title: str, edge_annotations: List, show_legend: bool = False) -> go.Layout:
        """Create layout with centralized styling"""
        layout_annotations = [
            dict(
                text="Gold: ModuleGroup, Blue: Module, Orange: Input Param, Green: Output Param, Purple: Virtual, Pink: Null, Red: Executed",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(
                    color=self.style.TEXT_STYLES['annotation_font_color'],
                    size=self.style.TEXT_STYLES['annotation_font_size']
                )
            )
        ]
        layout_annotations.extend(edge_annotations)
        
        return go.Layout(
            title=dict(
                text=title,
                font=dict(size=self.style.TEXT_STYLES['title_font_size'])
            ),
            showlegend=show_legend,
            hovermode='closest',
            margin=self.style.LAYOUT_SETTINGS['margin'],
            annotations=layout_annotations,
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                scaleanchor="y",  # 关键修复：让x轴和y轴保持等比例
                scaleratio=1      # 1:1的比例
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            clickmode='event+select',
            width=self.style.LAYOUT_SETTINGS['canvas_width'],
            height=self.style.LAYOUT_SETTINGS['canvas_height']
        )

    def _convert_to_plotly(self, nodes: List[Union[INode, IParamNode]], edges: List[IEdge]) -> Dict[str, Any]:
        """Convert graph data to Plotly format - main graph only contains INode"""
        # Register modules for potential clicking
        for node in nodes:
            if isinstance(node, INode) and node.module:
                self.module_registry[node.id] = node.module
        
        # Get layout positions using centralized method
        node_positions = self._get_layout_positions(nodes, edges)
        
        # Create edge traces using centralized method
        edge_traces, edge_annotations = self._create_edge_traces(edges, node_positions, is_detail_view=False)
        
        # Create node trace using centralized styling
        node_trace = self._create_node_trace(nodes, node_positions, is_detail_view=False)
        
        # Create layout using centralized method
        layout = self._create_layout('DAG Module Visualization', edge_annotations, show_legend=len(edge_traces) > 1)
        
        # Convert to JSON-serializable format using Plotly's encoder
        return json.loads(json.dumps({
            'data': edge_traces + [node_trace],
            'layout': layout
        }, cls=plotly.utils.PlotlyJSONEncoder))
    
    def _convert_detail_to_plotly(self, nodes: List[Union[INode, IParamNode]], edges: List[IEdge], module: Module) -> Dict[str, Any]:
        """Convert detail view data to Plotly format"""
        
        # Categorize nodes based on current interactive.py logic
        module_nodes = []  # INode instances (main module or child modules)
        param_nodes = []   # IParamNode instances
        
        for node in nodes:
            if isinstance(node, INode):
                module_nodes.append(node)
                # Register modules for potential clicking
                if node.module:
                    self.module_registry[node.id] = node.module
            elif isinstance(node, IParamNode):
                param_nodes.append(node)
        
        # Get layout positions using centralized method
        node_positions = self._get_layout_positions(nodes, edges)
        
        # Create edge traces using centralized method
        edge_traces, edge_annotations = self._create_edge_traces(edges, node_positions, is_detail_view=True)
        
        # Create node trace using centralized styling
        node_trace = self._create_node_trace(nodes, node_positions, is_detail_view=True)
        
        # Create layout using centralized method
        title = f'Module Detail: {module.name} ({module.__class__.__name__})'
        layout = self._create_layout(title, edge_annotations, show_legend=len(edge_traces) > 1)
        
        # Convert to JSON-serializable format using Plotly's encoder
        return json.loads(json.dumps({
            'data': edge_traces + [node_trace],
            'layout': layout
        }, cls=plotly.utils.PlotlyJSONEncoder))
    
    def set_module(self, module: Module):
        """Set the module to visualize"""
        self.current_module = module
    
    def run(self, host='localhost', port=5000, debug=True):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)


def create_app(module: Module = None) -> DAGVisualizationApp:
    """Create and configure the DAG visualization app"""
    app = DAGVisualizationApp()
    if module:
        app.set_module(module)
    return app 