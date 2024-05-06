# 通过onnx_parse.py 将onnx模型文件解析，解析出很多算子信息，本篇文档是对这些信息的解读

# Constant 算子
```
Operator Name: /Constant
Operator Type: Constant
value: dims: 1
data_type: 7
raw_data: "\001\000\000\000\000\000\000\000"

Inputs: 
Outputs: /Constant_output_0
```

> 数据类型编号 7 对应的是字符串类型 且 他的维度为1
