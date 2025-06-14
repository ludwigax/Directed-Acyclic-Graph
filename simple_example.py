#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAG Visualization Test with Launcher
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dag.node import ModuleGroup, connect, Addition, Multiplication, Constant
from dag import visualize_module

def create_complex_hierarchical_structure():
    """Create a complex hierarchical structure with multiple levels of nesting"""
    print("Creating complex hierarchical example with multiple levels...")
    
    # Level 0: Input data
    input_data = Constant(name="InputData_0")
    input_data_1 = Constant(name="InputData_1")
    input_data_2 = Constant(name="InputData_2")
    input_data_3 = Constant(name="InputData_3")
    input_data.input(10)
    input_data_1.input(20)
    input_data_2.input(30)
    input_data_3.input(40)
    
    # Level 1: Basic processing modules
    preprocessor1 = Addition(name="Preprocessor1")
    preprocessor2 = Multiplication(name="Preprocessor2")
    
    # Level 2: Inner processing group (Level 2.1)
    inner_add1 = Addition(name="InnerAdd1")
    inner_mult1 = Multiplication(name="InnerMult1")
    inner_add2 = Addition(name="InnerAdd2")
    
    # Connect inner modules
    connect(inner_add1, "result", inner_mult1, "a")
    connect(inner_mult1, "result", inner_add2, "a")
    
    # Create inner group (Level 2)
    inner_group = ModuleGroup(name="InnerProcessingGroup", modules=[inner_add1, inner_mult1, inner_add2]) # a, b, b_2, b_3
    from dag.inspect_utils import print_module

    connect(input_data, "_return", preprocessor1, "a")
    connect(input_data_1, "_return", preprocessor1, "b")
    connect(input_data_2, "_return", preprocessor2, "a")
    connect(input_data_3, "_return", preprocessor2, "b")

    connect(preprocessor1, "result", inner_group, "a")
    connect(preprocessor2, "result", inner_group, "b")
    connect(input_data, "_return", inner_group, "b.2")
    connect(input_data_1, "_return", inner_group, "b.3")
    
    return inner_group

def main():
    print("Encoding set to UTF-8")
    print("DAG Visualization Test with Launcher")
    print("="*50)
    
    print("="*50)
    print("LAUNCHING COMPLEX HIERARCHICAL VISUALIZER")
    print("="*50)
    
    # Create complex hierarchical structure
    main_group = create_complex_hierarchical_structure()
    
    # Launch visualizer
    visualize_module(main_group, port=5001)
    
    print("Test completed!")

if __name__ == "__main__":
    main()