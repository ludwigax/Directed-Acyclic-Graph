# DAG Package 使用示例

本文档提供了DAG包的使用示例，展示如何创建和使用有向无环图进行计算。

## 示例1: 简单DAG计算

以下示例展示了如何使用DAG创建一个简单的计算图来计算 `(1+2)*3`：

```python
from dag.node import *
from dag.view import dag_visualize, visualize_module_group, text_visualize_dag

def example_simple_dag():
    print("\n========== 示例1: 简单DAG计算 (1+2)*3 ==========")
    
    # 创建常数节点
    const1 = Constant(name="const1")
    const2 = Constant(name="const2")
    const3 = Constant(name="const3")
    
    # 设置常数值
    const1.input(1)
    const2.input(2)
    const3.input(3)
    
    # 创建加法和乘法节点
    add_node = Addition(name="add")
    mul_node = Multiplication(name="mul")
    
    # 连接节点
    connect(const1, add_node, src_key="_return", tgt_key="a")
    connect(const2, add_node, src_key="_return", tgt_key="b")
    connect(add_node, mul_node, src_key="result", tgt_key="a")
    connect(const3, mul_node, src_key="_return", tgt_key="b")
    
    # 执行计算
    result = mul_node()
    
    # 输出结果
    print(f"计算结果: {result}")
    assert result['result'] == 9, "计算结果应该是9"
    print("验证成功！")
    
    return mul_node  # 返回终端节点用于可视化
```

## 示例2: 使用ModuleGroup进行抽象

以下示例展示了如何使用ModuleGroup来对计算节点进行抽象，同样计算 `(1+2)*3`：

```python
def example_module_group():
    print("\n========== 示例2: 使用ModuleGroup计算 (1+2)*3 ==========")
    
    # 创建常数节点
    const1 = Constant(name="const1")
    const2 = Constant(name="const2")
    const3 = Constant(name="const3")
    
    # 设置常数值
    const1.input(1)
    const2.input(2)
    const3.input(3)
    
    # 创建内部模块
    add_node = Addition(name="add")
    mul_node = Multiplication(name="mul")
    
    # 首先连接加法和乘法节点
    connect(add_node, mul_node, src_key="result", tgt_key="a")
    
    # 创建模块组
    calculator = ModuleGroup(name="calculator", modules=[add_node, mul_node])
    
    # 输出calculator的边信息，查看自动命名情况
    print("ModuleGroup预连接信息:")
    print(f"Input edges: {calculator._prev}")
    print(f"Output edges: {calculator._next}")
    print(f"输入映射: {calculator._prev_name_map}")
    print(f"输出映射: {calculator._next_name_map}")
    
    # 连接外部节点到模块组
    connect(const1, calculator, src_key="_return", tgt_key="a")
    connect(const2, calculator, src_key="_return", tgt_key="b")
    connect(const3, calculator, src_key="_return", tgt_key="b_2")
    
    # 执行计算
    result = calculator()
    
    # 输出结果
    print(f"计算结果: {result}")
    if 'result' in result:
        assert result['result'] == 9, "计算结果应该是9"
    print("验证成功！")
    
    return calculator  # 返回终端节点用于可视化
```

## 可视化计算图

您可以使用以下代码来可视化上述计算图：

```python
if __name__ == "__main__":
    end_node1 = example_simple_dag()
    end_node2 = example_module_group()
    
    # 可视化示例
    print("\n========== 可视化计算图 ==========")
    print("\n使用Graphviz可视化计算图...")
    visualize_dag(end_node1, output_file='example1_visualization')
    visualize_module_group(end_node2, output_file='example2_visualization')

    print("\n========== 文本化计算图 ==========")
    print("\n使用文本模式显示计算图...")
    text_visualize_dag(end_node1)
    print("\n")
    text_visualize_dag(end_node2)
``` 


# DAG调试功能使用指南

DAG包现在支持调试功能，可以帮助用户监控和分析计算图中各模块的执行时间和调用次数。

## 启用调试功能

您可以通过设置全局变量`IS_DEBUGGING`为`True`来启用调试功能：

```python
from dag import IS_DEBUGGING

# 启用调试
IS_DEBUGGING = True
```

一旦启用，DAG将在每个模块执行时自动记录执行时间和调用次数。

## 获取调试统计信息

您可以使用`get_module_stats`函数获取模块的调试统计信息：

```python
from dag import get_module_stats

# 创建并执行DAG...
# ...

# 获取某个模块的统计信息
stats = get_module_stats(end_node)
print(stats)

# 递归获取ModuleGroup中所有子模块的统计信息
stats = get_module_stats(module_group, recursive=True)
print(stats)
```

统计信息包括：
- `execution_time`: 累计执行时间（秒）
- `call_count`: 调用次数
- `avg_time`: 平均执行时间（秒）

## 重置统计信息

您可以使用`reset_module_stats`函数重置模块的统计信息：

```python
from dag import reset_module_stats

# 重置某个模块的统计信息
reset_module_stats(end_node)

# 递归重置ModuleGroup中所有子模块的统计信息
reset_module_stats(module_group, recursive=True)
```

## 调试输出示例

启用调试后，模块执行时会输出类似以下的信息：

```
[DEBUG] Module add executed in 0.000123s
[DEBUG] Module mul executed in 0.000098s
[DEBUG] Module calculator executed in 0.002345s
```

## 调试功能的实现细节

调试功能通过以下方式实现：

1. `Debug`抽象类提供了计时和统计功能
2. `Module`类继承自`Debug`类，在`__call__`方法中使用计时功能
3. 全局变量`IS_DEBUGGING`控制是否启用调试功能

这种设计确保了调试功能对原有代码的侵入性最小，用户可以根据需要自由启用或禁用调试功能。 