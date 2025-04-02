from typing import Dict, Set, Tuple, List, Optional, Union, Any
from .utils import ensure_utf8_encoding

ensure_utf8_encoding()

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

from .node import Node, Module, ModuleGroup, Edge, NullNode, VirtualNode

def check_graphviz_dependency():
    if not HAS_GRAPHVIZ:
        print("错误: Graphviz Python包未安装。请使用 'pip install graphviz' 安装。")
        return False
    
    # 检查系统Graphviz可执行文件是否存在
    import shutil
    if shutil.which('dot') is None:
        print("错误: Graphviz软件未安装在系统中。")
        print("请安装Graphviz: ")
        print("  - Windows: 从 https://graphviz.org/download/ 下载并安装")
        print("  - MacOS: 使用 'brew install graphviz'")
        print("  - Linux: 使用 'apt-get install graphviz' 或对应的包管理器")
        print("安装后，确保将Graphviz添加到系统PATH环境变量中。")
        return False
    
    return True

def group_visualize(group: ModuleGroup, dot: Digraph, processed_nodes: Set[int], 
                   depth: int = 0, max_depth: int = 3, parent_group: Optional[ModuleGroup] = None):
    """
    可视化ModuleGroup的内部结构
    
    Args:
        group: 要可视化的ModuleGroup
        dot: 主Digraph对象
        processed_nodes: 已处理节点的ID集合
        depth: 当前深度
        max_depth: 最大展开深度
        parent_group: 父ModuleGroup（如果有）
    """
    if id(group) in processed_nodes:
        return
        
    processed_nodes.add(id(group))
    
    # 创建子图
    with dot.subgraph(name=f'cluster_{group.name}') as c:
        c.attr(label=group.name, style='filled', color='lightgrey')
        
        # 处理内部模块
        for module_name, module in group._modules.items():
            if id(module) in processed_nodes:
                continue
                
            processed_nodes.add(id(module))
            
            # 根据模块类型设置颜色
            color = '#FFFFFF'  # 默认白色
            if isinstance(module, ModuleGroup):
                color = '#F0E68C'  # ModuleGroup使用淡黄色
            
            # 添加模块节点
            c.node(module_name, 
                   label=f"{module_name}\n{module.__class__.__name__}", 
                   fillcolor=color)
            
            # 处理输入连接
            for key, edge in module._prev.items():
                if edge.null:
                    # 未连接的输入
                    input_name = f"{module_name}_in_{key}"
                    c.node(input_name, 
                           label=f"{key} (null)", 
                           shape='diamond', 
                           fillcolor='#FFD0D0')
                    c.edge(input_name, module_name, style='dashed')
                elif edge.virtual:
                    # 虚连接
                    if parent_group and key in parent_group._prev_name_map:
                        map_key = parent_group._prev_name_map[key]
                        if map_key.startswith(f"{group.name}."):
                            # 内部映射
                            inner_key = map_key.split('.')[1]
                            c.edge(f"{module_name}_in_{inner_key}", module_name, 
                                   style='dashed', color='blue')
                else:
                    # 正常连接
                    src = edge.src
                    if src is not NullNode and src is not VirtualNode:
                        src_key = edge.src_key if edge.src_key else "output"
                        c.edge(f"{src.name}_out_{src_key}", module_name, 
                               label=f"{src_key} → {key}")
            
            # 处理输出连接
            for key, edges in module._next.items():
                if edges:
                    output_name = f"{module_name}_out_{key}"
                    c.node(output_name, 
                           label=key, 
                           shape='diamond', 
                           fillcolor='#90EE90')
                    c.edge(module_name, output_name)
            
            # 递归处理ModuleGroup
            if isinstance(module, ModuleGroup) and depth < max_depth:
                group_visualize(module, c, processed_nodes, depth + 1, max_depth, group)

def dag_visualize(endpoint: Module, output_file: str = None, format: str = 'png', 
                  view: bool = True, expand_groups: bool = False, max_depth: int = 3):
    """
    可视化整个DAG结构
    
    Args:
        endpoint: 终端节点
        output_file: 输出文件名
        format: 输出格式
        view: 是否立即显示
        expand_groups: 是否展开ModuleGroup
        max_depth: 最大展开深度
    """
    if not check_graphviz_dependency():
        print("可视化已中止。请安装必需的依赖。")
        return None
    
    try:
        dot = Digraph(comment='DAG Visualization')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='filled', fontname='Arial')
        
        processed_nodes = set()
        
        def add_node(node: Module, parent_group: Optional[ModuleGroup] = None):
            if id(node) in processed_nodes:
                return
                
            processed_nodes.add(id(node))
            
            # 设置节点颜色
            color = '#FFFFFF'
            if isinstance(node, ModuleGroup):
                color = '#F0E68C'
            elif node == endpoint:
                color = '#90EE90'
            
            # 添加节点
            node_label = f"{node.name}\n{node.__class__.__name__}"
            dot.node(node.name, label=node_label, fillcolor=color)
            
            # 处理输入连接
            for key, edge in node._prev.items():
                if edge.null:
                    # 未连接的输入
                    input_name = f"{node.name}_in_{key}"
                    dot.node(input_name, 
                            label=f"{key} (null)", 
                            shape='diamond', 
                            fillcolor='#FFD0D0')
                    dot.edge(input_name, node.name, style='dashed')
                elif edge.virtual and parent_group:
                    # 虚连接
                    map_key = next((k for k, v in parent_group._prev_name_map.items() 
                                if v == f"{node.name}.{key}"), None)
                    if map_key:
                        virtual_src = parent_group._prev[map_key].src
                        if virtual_src is not NullNode and virtual_src is not VirtualNode:
                            add_node(virtual_src)
                            src_key = parent_group._prev[map_key].src_key
                            dot.edge(virtual_src.name, node.name, 
                                    label=f"{src_key} → {key}",
                                    color='blue')
                else:
                    # 正常连接
                    src = edge.src
                    if src is not NullNode and src is not VirtualNode:
                        add_node(src)
                        src_key = edge.src_key if edge.src_key else "output"
                        dot.edge(src.name, node.name, 
                                label=f"{src_key} → {key}")
            
            # 处理ModuleGroup
            if isinstance(node, ModuleGroup):
                if expand_groups:
                    # 展开ModuleGroup
                    group_visualize(node, dot, processed_nodes, 0, max_depth, parent_group)
                else:
                    # 作为单个节点处理
                    for module_name, module in node._modules.items():
                        if id(module) not in processed_nodes:
                            add_node(module, node)
        
        # 从终端节点开始构建图
        add_node(endpoint)
        
        # 检查ModuleGroup终端节点的输出情况
        if isinstance(endpoint, ModuleGroup):
            # 为每个输出添加一个虚拟节点，以更好地显示ModuleGroup的输出
            for out_key, out_edges in endpoint._next.items():
                # 检查输出映射
                if out_key in endpoint._next_name_map:
                    module_path = endpoint._next_name_map[out_key]
                    parts = module_path.split('.')
                    if len(parts) == 2:
                        module_name, module_key = parts
                        if module_name in endpoint._modules:
                            module = endpoint._modules[module_name]
                            # 创建输出节点
                            output_node_name = f"{endpoint.name}_output_{out_key}"
                            dot.node(output_node_name, 
                                    label=f"Output: {out_key}", 
                                    shape='oval', 
                                    fillcolor='#98FB98')  # 浅绿色
                            # 从内部模块连接到输出节点
                            dot.edge(module.name, output_node_name, 
                                    label=f"{module_key} → output",
                                    style='dashed', 
                                    color='green')
        
        # 保存并显示
        if output_file:
            try:
                dot.render(output_file, format=format, cleanup=True, view=view)
                print(f"可视化已保存到 {output_file}.{format}")
            except Exception as e:
                print(f"可视化错误: {e}")
                # 尝试保存为无视图模式
                try:
                    dot.render(output_file, format=format, cleanup=True, view=False)
                    print(f"可视化已保存到 {output_file}.{format} (无法自动打开)")
                except Exception as inner_e:
                    print(f"无法保存可视化: {inner_e}")
        else:
            try:
                dot.render('dag_visualization', format=format, cleanup=True, view=view)
                print(f"可视化已保存到 dag_visualization.{format}")
            except Exception as e:
                print(f"可视化错误: {e}")
        
        return dot
    
    except Exception as e:
        print(f"生成可视化图时发生错误: {e}")
        return None


def visualize_module_group(group: ModuleGroup, output_file: str = None, format: str = 'png', view: bool = True):
    """
    专门用于可视化ModuleGroup及其内部结构
    
    Args:
        group: 要可视化的ModuleGroup
        output_file: 输出文件名（不包含扩展名）
        format: 输出文件格式，如'png', 'pdf', 'svg'等
        view: 是否立即显示图像
    """
    if not check_graphviz_dependency():
        print("可视化已中止。请安装必需的依赖。")
        return None
    
    try:
        dot = Digraph(comment='ModuleGroup Visualization')
        dot.attr(rankdir='TB')  # 从上到下布局
        dot.attr('node', shape='box', style='filled', fontname='Arial')
        
        # 处理过的节点集合
        processed_nodes = set()
        
        # 创建一个子图来表示ModuleGroup
        with dot.subgraph(name=f'cluster_{group.name}') as c:
            c.attr(label=f"{group.name} (ModuleGroup)", style='filled', color='lightgrey', fontname='Arial', fontsize='16')
            
            # 添加所有内部模块
            for name, module in group._modules.items():
                if id(module) in processed_nodes:
                    continue
                    
                processed_nodes.add(id(module))
                module_color = '#FFFFFF'  # 默认白色
                
                # 为不同类型的模块使用不同颜色
                if isinstance(module, ModuleGroup):
                    module_color = '#F0E68C'  # 嵌套的ModuleGroup使用淡黄色
                
                # 添加模块节点
                c.node(name, 
                       label=f"{name}\n{module.__class__.__name__}", 
                       fillcolor=module_color)
                
                # 添加模块的输入接口节点
                for key, edge in module._prev.items():
                    input_node_name = f"{name}_in_{key}"
                    c.node(input_node_name, 
                           label=key, 
                           shape='diamond', 
                           fillcolor='#ADD8E6')  # 浅蓝色
                    c.edge(input_node_name, name, style='solid')
                    
                    # 如果有连接，检查是否是内部模块间的连接
                    if not edge.virtual and edge.src is not NullNode and edge.src is not VirtualNode:
                        if hasattr(edge.src, 'name') and edge.src.name in group._modules:
                            # 内部连接
                            src_module = edge.src
                            src_output_node = f"{src_module.name}_out_{edge.src_key}"
                            c.node(src_output_node, 
                                   label=edge.src_key, 
                                   shape='diamond', 
                                   fillcolor='#90EE90')  # 浅绿色
                            c.edge(src_output_node, input_node_name, 
                                   style='solid', 
                                   color='red')
                
                # 添加模块的输出接口节点
                for key, edges in module._next.items():
                    if edges:  # 确保有输出边
                        output_node_name = f"{name}_out_{key}"
                        c.node(output_node_name, 
                               label=key, 
                               shape='diamond', 
                               fillcolor='#90EE90')  # 浅绿色
                        c.edge(name, output_node_name, style='solid')
            
            # 添加模块组的输入输出接口
            # 输入接口
            for key in group._prev:
                port_name = f"{group.name}_input_{key}"
                dot.node(port_name, 
                         label=f"Input: {key}", 
                         shape='ellipse', 
                         fillcolor='#ADD8E6')  # 浅蓝色
                
                # 映射到内部模块
                if key in group._prev_name_map:
                    module_key = group._prev_name_map[key].split('.')
                    if len(module_key) == 2:
                        module_name, inner_key = module_key
                        if module_name in group._modules:
                            inner_node_name = f"{module_name}_in_{inner_key}"
                            dot.edge(port_name, inner_node_name, 
                                    style='dashed', 
                                    color='blue')
            
            # 输出接口
            for key in group._next:
                port_name = f"{group.name}_output_{key}"
                dot.node(port_name, 
                         label=f"Output: {key}", 
                         shape='ellipse',
                         fillcolor='#90EE90')  # 浅绿色
                
                # 映射到内部模块
                if key in group._next_name_map:
                    module_key = group._next_name_map[key].split('.')
                    if len(module_key) == 2:
                        module_name, inner_key = module_key
                        if module_name in group._modules:
                            inner_node_name = f"{module_name}_out_{inner_key}"
                            dot.edge(inner_node_name, port_name, 
                                    style='dashed', 
                                    color='green')
            
        # 额外的信息节点，显示映射关系
        info_node_name = f"{group.name}_info"
        info_text = f"输入映射:\n{group._prev_name_map}\n\n输出映射:\n{group._next_name_map}"
        dot.node(info_node_name, 
                 label=info_text, 
                 shape='note', 
                 fillcolor='#F5F5DC')  # 米色
        
        # 保存并显示
        if output_file:
            try:
                dot.render(output_file, format=format, cleanup=True, view=view)
                print(f"可视化已保存到 {output_file}.{format}")
            except Exception as e:
                print(f"可视化错误: {e}")
                # 尝试保存为无视图模式
                try:
                    dot.render(output_file, format=format, cleanup=True, view=False)
                    print(f"可视化已保存到 {output_file}.{format} (无法自动打开)")
                except Exception as inner_e:
                    print(f"无法保存可视化: {inner_e}")
        else:
            try:
                dot.render('modulegroup_visualization', format=format, cleanup=True, view=view)
                print(f"可视化已保存到 modulegroup_visualization.{format}")
            except Exception as e:
                print(f"可视化错误: {e}")
        
        return dot
    
    except Exception as e:
        print(f"生成可视化图时发生错误: {e}")
        return None


def text_visualize_dag(endpoint: Module):
    """
    当Graphviz不可用时，提供一个简单的文本版可视化功能
    
    Args:
        endpoint: 计算图的终端节点
    """
    print(f"\n======== 文本版DAG可视化 (从 {endpoint.name} 开始) ========")
    
    # 使用缩进来表示层次
    def print_node(node: Module, indent=0, visited=None):
        if visited is None:
            visited = set()
            
        if id(node) in visited:
            print(" " * indent + f"[已访问过] {node.name} ({node.__class__.__name__})")
            return
            
        visited.add(id(node))
        
        # 打印当前节点
        print(" " * indent + f"{node.name} ({node.__class__.__name__})")
        
        # 打印输入节点
        for key, edge in node._prev.items():
            if edge.null:
                print(" " * (indent+2) + f"输入 {key}: [未连接]")
            elif edge.virtual:
                print(" " * (indent+2) + f"输入 {key}: [虚拟连接] 通过父模块")
            else:
                src = edge.src
                if src is not NullNode and src is not VirtualNode:
                    print(" " * (indent+2) + f"输入 {key}: 来自 {src.name}.{edge.src_key}")
                    print_node(src, indent+4, visited)
        
        # 如果是ModuleGroup，打印内部模块
        if isinstance(node, ModuleGroup):
            print(" " * (indent+2) + "内部模块:")
            for module_name, module in node._modules.items():
                print(" " * (indent+4) + f"{module_name} ({module.__class__.__name__})")
            
            print(" " * (indent+2) + "输入映射:")
            for external_key, internal_path in node._prev_name_map.items():
                print(" " * (indent+4) + f"{external_key} -> {internal_path}")
                
            print(" " * (indent+2) + "输出映射:")
            for external_key, internal_path in node._next_name_map.items():
                print(" " * (indent+4) + f"{external_key} <- {internal_path}")
    
    # 从终端节点开始打印
    print_node(endpoint)
    print("=" * 60)

