import json
import os
import sys
import importlib
import copy
from typing import Dict, List, Tuple, Any, Union, Optional, Type

from .node import Module, Node, connect, ModuleGroup, NullNode, VirtualNode

def ensure_utf8_encoding():
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    print("编码已设置为UTF-8")


def build_dag_from_dict(dag_dict: Dict[str, Any], module_registry: Optional[Dict[str, Type[Module]]] = None) -> Dict[str, Module]:
    """
    从字典构建DAG图
    
    Args:
        dag_dict: 包含节点和连接信息的字典，格式为:
            {
                "modules": {
                    "module_name": {
                        "type": "ModuleClassName",
                        "params": {key: value, ...},  # 模块初始化参数
                        "input": value  # 对于Constant等需要输入的模块
                    },
                    ...
                },
                "connections": [
                    ["src_module", "tgt_module", "src_key", "tgt_key"],
                    ...
                ]
            }
        module_registry: 包含可用模块类的注册表，格式为 {"ModuleName": ModuleClass}。
                      如果为None，则自动从node模块导入所有可用的模块类
    
    Returns:
        包含所有创建的模块的字典 {module_name: module_instance}
    """
    if module_registry is None:
        module_registry = {}
        node_module = importlib.import_module("node")
        for attr_name in dir(node_module):
            attr = getattr(node_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Module) and attr != Module:
                module_registry[attr_name] = attr
    
    # 创建所有模块 (除了ModuleGroup)
    modules = {}
    for module_name, module_info in dag_dict.get("modules", {}).items():
        module_type = module_info.get("type")
        if module_type not in module_registry:
            raise ValueError(f"未知模块类型: {module_type}")
        
        # 跳过ModuleGroup，放在后面创建
        if module_type == "ModuleGroup":
            continue
            
        module_class = module_registry[module_type]
        module_params = module_info.get("params", {})
        
        if "name" not in module_params:
            module_params["name"] = module_name
        
        module = module_class(**module_params)
        modules[module_name] = module
        
        if "input" in module_info and hasattr(module, "input"):
            module.input(module_info["input"])
    
    # 第一步：连接普通模块
    non_group_connections = []
    for conn in dag_dict.get("connections", []):
        if len(conn) != 4:
            raise ValueError(f"连接格式错误: {conn}，应为 [src_module, tgt_module, src_key, tgt_key]")
        
        src_name, tgt_name, src_key, tgt_key = conn
        
        # 跳过涉及ModuleGroup的连接，先处理普通模块间的连接
        if src_name in modules and tgt_name in modules:
            src_module = modules[src_name]
            tgt_module = modules[tgt_name]
            connect(src_module, tgt_module, src_key=src_key, tgt_key=tgt_key)
        else:
            non_group_connections.append(conn)
    
    # 第二步：创建ModuleGroup
    for group_name, group_info in dag_dict.get("groups", {}).items():
        if group_name in modules:
            raise ValueError(f"模块名冲突: {group_name} 已存在")
        
        group_members = []
        for member_name in group_info.get("members", []):
            if member_name not in modules:
                raise ValueError(f"组成员不存在: {member_name}")
            group_members.append(modules[member_name])
        
        # 创建模块组
        group_params = group_info.get("params", {})
        if "name" not in group_params:
            group_params["name"] = group_name
        
        # 注意：先把组内的模块进行连接（已经在第一步完成）
        # 然后创建ModuleGroup
        group = ModuleGroup(modules=group_members, **group_params)
        modules[group_name] = group
    
    # 第三步：处理剩余的连接（涉及ModuleGroup的连接）
    for conn in non_group_connections:
        src_name, tgt_name, src_key, tgt_key = conn
        
        if src_name not in modules:
            raise ValueError(f"源模块不存在: {src_name}")
        if tgt_name not in modules:
            raise ValueError(f"目标模块不存在: {tgt_name}")
        
        src_module = modules[src_name]
        tgt_module = modules[tgt_name]
        connect(src_module, tgt_module, src_key=src_key, tgt_key=tgt_key)
    
    return modules


def build_template_dag(template_dict: Dict[str, Any], input_modules: Dict[str, Module] = None, module_registry: Optional[Dict[str, Type[Module]]] = None) -> Dict[str, Module]:
    """
    从模板字典构建DAG，支持引用外部模块
    
    Args:
        template_dict: 包含节点和连接信息的模板字典
        input_modules: 外部引入的模块，可以在连接中引用
        module_registry: 包含可用模块类的注册表
    
    Returns:
        包含所有创建的模块的字典
    """
    input_modules = input_modules or {}
    
    # 构建基本DAG
    modules = build_dag_from_dict(template_dict, module_registry)
    
    # 合并外部引入的模块
    all_modules = {**input_modules, **modules}
    
    # 处理跨模板的连接
    for conn in template_dict.get("external_connections", []):
        if len(conn) != 4:
            raise ValueError(f"外部连接格式错误: {conn}，应为 [src_module, tgt_module, src_key, tgt_key]")
        
        src_name, tgt_name, src_key, tgt_key = conn
        
        if src_name not in all_modules:
            raise ValueError(f"源模块不存在: {src_name}")
        if tgt_name not in all_modules:
            raise ValueError(f"目标模块不存在: {tgt_name}")
        
        src_module = all_modules[src_name]
        tgt_module = all_modules[tgt_name]
        connect(src_module, tgt_module, src_key=src_key, tgt_key=tgt_key)
    
    return all_modules


def create_module_instance(module_spec: Dict[str, Any], module_registry: Optional[Dict[str, Type[Module]]] = None) -> Module:
    """
    创建单个模块实例
    
    Args:
        module_spec: 模块规格，包含type和params
        module_registry: 包含可用模块类的注册表
        
    Returns:
        创建的模块实例
    """
    if module_registry is None:
        module_registry = {}
        node_module = importlib.import_module("node")
        for attr_name in dir(node_module):
            attr = getattr(node_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Module) and attr != Module:
                module_registry[attr_name] = attr
    
    module_type = module_spec.get("type")
    if module_type not in module_registry:
        raise ValueError(f"未知模块类型: {module_type}")
    
    module_class = module_registry[module_type]
    module_params = module_spec.get("params", {})
    
    module = module_class(**module_params)
    
    if "input" in module_spec and hasattr(module, "input"):
        module.input(module_spec["input"])
    
    return module


def load_dag_from_templates(template_files: List[str], endpoint_file: str, module_registry: Optional[Dict[str, Type[Module]]] = None) -> Dict[str, Module]:
    """
    从多个模板文件加载DAG
    
    Args:
        template_files: 模板JSON文件列表
        endpoint_file: 终端模板文件，定义最终连接
        module_registry: 包含可用模块类的注册表
        
    Returns:
        包含所有创建的模块的字典
    """
    all_modules = {}
    
    # 首先加载所有模板
    for template_file in template_files:
        with open(template_file, 'r', encoding='utf-8') as f:
            template_dict = json.load(f)
        
        # 构建模板内部的DAG
        template_modules = build_dag_from_dict(template_dict, module_registry)
        
        # 检查命名冲突
        for name in template_modules:
            if name in all_modules:
                # 如果有冲突，创建新的实例
                print(f"警告: 模块名 {name} 已存在，将创建新实例")
                module_info = next((info for mod_name, info in template_dict.get("modules", {}).items() if mod_name == name), None)
                if module_info:
                    # 创建一个新的唯一名称
                    new_name = f"{name}_{id(module_info)}"
                    # 创建新实例
                    all_modules[new_name] = create_module_instance(
                        {"type": module_info["type"], "params": {**module_info.get("params", {}), "name": new_name}, 
                         "input": module_info.get("input")}, 
                        module_registry
                    )
            else:
                all_modules[name] = template_modules[name]
    
    # 最后加载终端模板
    with open(endpoint_file, 'r', encoding='utf-8') as f:
        endpoint_dict = json.load(f)
    
    # 构建终端DAG，并合并所有模块
    endpoint_modules = build_template_dag(endpoint_dict, all_modules, module_registry)
    
    return {**all_modules, **endpoint_modules}


# 从JSON文件加载DAG
def load_dag_from_json(json_file: str, module_registry: Optional[Dict[str, Type[Module]]] = None) -> Dict[str, Module]:
    """
    从JSON文件加载DAG
    
    Args:
        json_file: JSON文件路径
        module_registry: 包含可用模块类的注册表
    
    Returns:
        包含所有创建的模块的字典
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        dag_dict = json.load(f)
    
    return build_dag_from_dict(dag_dict, module_registry)


# 保存DAG到JSON文件，支持分离为模板
def save_dag_to_json(modules: Dict[str, Module], json_file: str, create_templates: bool = False, templates_dir: str = "templates"):
    """
    将DAG保存到JSON文件
    
    Args:
        modules: 包含模块的字典 {module_name: module_instance}
        json_file: 输出的JSON文件路径
        create_templates: 是否将ModuleGroup分离为单独的模板
        templates_dir: 模板目录
    """
    if create_templates:
        # 创建模板目录
        os.makedirs(templates_dir, exist_ok=True)
        
        # 记录所有ModuleGroup
        groups = {}
        for module_name, module in modules.items():
            if isinstance(module, ModuleGroup):
                groups[module_name] = module
        
        # 主DAG字典
        main_dag_dict = {
            "modules": {},
            "external_connections": []
        }
        
        # 为每个ModuleGroup创建单独的模板
        for group_name, group in groups.items():
            template_dict = {
                "modules": {},
                "connections": []
            }
            
            # 添加内部模块
            for inner_name, inner_module in group._modules.items():
                module_type = inner_module.__class__.__name__
                template_dict["modules"][inner_name] = {
                    "type": module_type,
                    "params": {"name": inner_module.name}
                }
            
            # 添加内部连接
            for inner_name, inner_module in group._modules.items():
                for key, edge in inner_module._prev.items():
                    if not edge.virtual and edge.src is not NullNode and edge.src is not VirtualNode:
                        src_module_name = next((name for name, m in group._modules.items() if m == edge.src), None)
                        if src_module_name:
                            template_dict["connections"].append([
                                src_module_name,
                                inner_name,
                                edge.src_key,
                                key
                            ])
            
            # 保存模板
            template_file = os.path.join(templates_dir, f"{group_name}.json")
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_dict, f, ensure_ascii=False, indent=2)
            
            # 在主DAG中引用此模板
            main_dag_dict["modules"][group_name] = {
                "type": "ModuleGroup",
                "params": {"name": group.name},
                "template": template_file
            }
        
        # 添加其他非ModuleGroup模块
        for module_name, module in modules.items():
            if not isinstance(module, ModuleGroup):
                module_type = module.__class__.__name__
                main_dag_dict["modules"][module_name] = {
                    "type": module_type,
                    "params": {"name": module.name}
                }
                if hasattr(module, "_data"):
                    main_dag_dict["modules"][module_name]["input"] = module._data
        
        # 添加外部连接
        for module_name, module in modules.items():
            for key, edges in module._prev.items():
                if hasattr(edges, 'src') and edges.src is not NullNode and edges.src is not VirtualNode:
                    src_module_name = next((name for name, m in modules.items() if m == edges.src), None)
                    if src_module_name:
                        # 检查是否跨模板连接
                        is_cross_template = (isinstance(module, ModuleGroup) or 
                                            isinstance(edges.src, ModuleGroup))
                        
                        if is_cross_template:
                            main_dag_dict["external_connections"].append([
                                src_module_name,
                                module_name,
                                edges.src_key,
                                key
                            ])
        
        # 保存主DAG
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(main_dag_dict, f, ensure_ascii=False, indent=2)
    else:
        # 原始的单文件保存方式
        dag_dict = {
            "modules": {},
            "connections": []
        }
        
        # 添加模块信息
        for module_name, module in modules.items():
            module_type = module.__class__.__name__
            dag_dict["modules"][module_name] = {
                "type": module_type,
                "params": {"name": module.name}
            }
            if hasattr(module, "_data"):
                dag_dict["modules"][module_name]["input"] = module._data
        
        # 添加连接信息
        for module_name, module in modules.items():
            for key, edges in module._prev.items():
                if hasattr(edges, 'src') and edges.src is not NullNode and edges.src is not VirtualNode:
                    src_module_name = next((name for name, m in modules.items() if m == edges.src), None)
                    if src_module_name:
                        dag_dict["connections"].append([
                            src_module_name,
                            module_name,
                            edges.src_key,
                            key
                        ])
        
        # 保存到JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dag_dict, f, ensure_ascii=False, indent=2)


# 示例用法：创建一个简单的计算图
def create_example_dag_dict():
    """创建一个示例DAG字典，计算 (1+2)*3"""
    return {
        "modules": {
            "const1": {
                "type": "Constant",
                "params": {"name": "const1"},
                "input": 1
            },
            "const2": {
                "type": "Constant",
                "params": {"name": "const2"},
                "input": 2
            },
            "const3": {
                "type": "Constant",
                "params": {"name": "const3"},
                "input": 3
            },
            "add": {
                "type": "Addition",
                "params": {"name": "add"}
            },
            "mul": {
                "type": "Multiplication",
                "params": {"name": "mul"}
            }
        },
        "connections": [
            ["const1", "add", "_return", "a"],
            ["const2", "add", "_return", "b"],
            ["add", "mul", "result", "a"],
            ["const3", "mul", "_return", "b"]
        ]
    }

# 示例：使用ModuleGroup
def create_example_group_dag_dict():
    """创建一个带有ModuleGroup的示例DAG字典"""
    return {
        "modules": {
            "const1": {
                "type": "Constant",
                "params": {"name": "const1"},
                "input": 1
            },
            "const2": {
                "type": "Constant",
                "params": {"name": "const2"},
                "input": 2
            },
            "const3": {
                "type": "Constant",
                "params": {"name": "const3"},
                "input": 3
            },
            "add": {
                "type": "Addition",
                "params": {"name": "add"}
            },
            "mul": {
                "type": "Multiplication",
                "params": {"name": "mul"}
            }
        },
        "groups": {
            "calculator": {
                "params": {"name": "calculator"},
                "members": ["add", "mul"]
            }
        },
        "connections": [
            ["add", "mul", "result", "a"],  # 这个内部连接必须首先处理
            ["const1", "calculator", "_return", "a"],
            ["const2", "calculator", "_return", "b"],
            ["const3", "calculator", "_return", "b_2"]
        ]
    }

# 示例：创建模板式的DAG
def create_example_template_dags():
    """创建用于模板式处理的示例DAG字典"""
    # 计算器模板 - 单独的加法器和乘法器
    calculator_template = {
        "modules": {
            "add": {
                "type": "Addition",
                "params": {"name": "add"}
            },
            "mul": {
                "type": "Multiplication",
                "params": {"name": "mul"}
            }
        },
        "connections": [
            ["add", "mul", "result", "a"]
        ],
        "groups": {
            "calculator": {
                "params": {"name": "calculator"},
                "members": ["add", "mul"]
            }
        }
    }
    
    # 常量模板 - 提供输入值
    constants_template = {
        "modules": {
            "const1": {
                "type": "Constant",
                "params": {"name": "const1"},
                "input": 1
            },
            "const2": {
                "type": "Constant",
                "params": {"name": "const2"},
                "input": 2
            },
            "const3": {
                "type": "Constant",
                "params": {"name": "const3"},
                "input": 3
            }
        }
    }
    
    # 终端模板 - 连接所有组件
    endpoint_template = {
        "external_connections": [
            ["const1", "calculator", "_return", "a"],
            ["const2", "calculator", "_return", "b"],
            ["const3", "calculator", "_return", "b_2"]
        ]
    }
    
    return {
        "calculator": calculator_template,
        "constants": constants_template,
        "endpoint": endpoint_template
    }

# 示例函数：运行从字典创建的DAG
def run_dag_from_dict(dag_dict: Dict[str, Any], endpoint_name: str):
    """
    从字典创建并运行DAG
    
    Args:
        dag_dict: DAG字典
        endpoint_name: 终端节点的名称
    
    Returns:
        计算结果
    """
    ensure_utf8_encoding()
    modules = build_dag_from_dict(dag_dict)
    
    if endpoint_name not in modules:
        raise ValueError(f"终端节点不存在: {endpoint_name}")
    
    endpoint = modules[endpoint_name]
    result = endpoint()
    
    print(f"计算结果: {result}")
    return result, endpoint

# 示例函数: 从模板创建和运行DAG
def run_dag_from_templates(templates: Dict[str, Dict[str, Any]], endpoint_name: str):
    """
    从模板创建并运行DAG
    
    Args:
        templates: 模板字典集
        endpoint_name: 终端节点的名称
    
    Returns:
        计算结果和终端节点
    """
    ensure_utf8_encoding()
    
    # 构建所有模块
    modules = {}
    
    # 首先处理非终端模板
    for template_name, template_dict in templates.items():
        if template_name != "endpoint":
            template_modules = build_dag_from_dict(template_dict)
            modules.update(template_modules)
    
    # 然后处理终端模板及其连接
    endpoint_modules = build_template_dag(templates["endpoint"], modules)
    modules.update(endpoint_modules)
    
    if endpoint_name not in modules:
        raise ValueError(f"终端节点不存在: {endpoint_name}")
    
    endpoint = modules[endpoint_name]
    result = endpoint()
    
    print(f"计算结果: {result}")
    return result, endpoint
