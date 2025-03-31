from node import *
from view import visualize_dag, visualize_module_group, text_visualize_dag

# 示例1: 简单DAG计算
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


# 示例2: 使用ModuleGroup进行抽象
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


# 主函数调用所有示例
if __name__ == "__main__":
    end_node1 = example_simple_dag()
    end_node2 = example_module_group()
    
    # 可视化示例
    print("\n========== 可视化计算图 ==========")
    print("\n使用Graphviz可视化计算图...")
    visualize_dag(end_node1, output_file='example1_visualization')
    visualize_module_group(end_node2, output_file='example2_visualization')


    print("\n========== 文本化计算图 ==========")
    print("\n使用文本模式显示计算图（因为未安装Graphviz）...")
    text_visualize_dag(end_node1)
    print("\n")
    text_visualize_dag(end_node2)
