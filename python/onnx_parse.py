import numpy as np
import onnx


def tensor_to_string(tensor):
    if hasattr(tensor, 'float_data') and len(tensor.float_data) > 0:
        # Float data available
        param = np.array(tensor.float_data, dtype=np.float32)
    elif hasattr(tensor, 'int_data') and len(tensor.int_data) > 0:
        # Integer data available
        param = np.array(tensor.int_data, dtype=np.int32)
    elif hasattr(tensor, 'raw_data') and len(tensor.raw_data) > 0:
        # Raw data needs to be converted based on the data_type
        dt = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]
        param = np.frombuffer(tensor.raw_data, dtype=dt)
    else:
        # No data found, return an empty string
        return "[]"

    param = param.reshape(tuple(tensor.dims))  # Reshape according to the dimensions
    np.set_printoptions(threshold=np.inf)  # 设置显示整个数组，无省略
    return np.array2string(param, separator=',', precision=6).replace('\n', '')

def attribute_to_string(attribute):
    # This function will handle different types of attribute values
    if attribute.type == onnx.AttributeProto.TENSOR:
        # If the attribute is a tensor, format it properly
        tensor = attribute.t
        dims = tensor.dims
        data_type = tensor.data_type
        raw_data = tensor.raw_data
        # Convert raw_data to human-readable format, assuming it's int64 here
        if data_type == onnx.TensorProto.INT64:
            formatted_data = np.frombuffer(raw_data, dtype=np.int64)
        elif data_type == onnx.TensorProto.FLOAT:
            formatted_data = np.frombuffer(raw_data, dtype=np.float32)
        else:
            formatted_data = list(raw_data)
        return f"dims: {dims}, data_type: {data_type}, raw_data: {formatted_data}"
    else:
        # Fallback for other types of attributes
        return onnx.helper.get_attribute_value(attribute)





# 加载ONNX模型
# model_path = 'part_11_5.onnx'
model_path = "fusion11_5.onnx"
model = onnx.load(model_path)

# 提取并保存模型中所有算子的信息
with open('../main/model_parameters.txt', 'w') as file:
    # 写入图的输入节点信息
    file.write("Model Inputs:\n")
    for input_tensor in model.graph.input:
        file.write(f"Input Name: {input_tensor.name}\n")
        tensor_type = input_tensor.type.tensor_type
        dims = [d.dim_value for d in tensor_type.shape.dim]
        file.write(f"Shape: {dims}\n")
    file.write("\n")
    
    # 写入图的输出节点信息
    file.write("Model Outputs:\n")
    for output_tensor in model.graph.output:
        file.write(f"Output Name: {output_tensor.name}\n")
        tensor_type = output_tensor.type.tensor_type
        dims = [d.dim_value for d in tensor_type.shape.dim]
        file.write(f"Shape: {dims}\n")
    file.write("\n")
    
    for node in model.graph.node:
        # 写入算子名称和类型
        file.write(f"Operator Name: {node.name}\n")
        file.write(f"Operator Type: {node.op_type}\n")
        
        # 写入算子属性
        # Now use this function when writing attributes
        for attr in node.attribute:
            attr_value_str = attribute_to_string(attr)
            if attr_value_str:
                file.write(f"{attr.name}: {attr_value_str}\n")
            
        # 写入输入和输出信息
        file.write("Inputs: " + ", ".join(node.input) + "\n")
        file.write("Outputs: " + ", ".join(node.output) + "\n")
        
        # 提取并写入权重和偏置（如果有）
        for input_name in node.input:
            tensor = None
            for tensor in model.graph.initializer:
                if tensor.name == input_name:
                    param_str = tensor_to_string(tensor)
                    file.write(f"Parameter: {input_name}, Shape: {tuple(tensor.dims)}, Values: {param_str}\n")
        
        file.write("\n")  # 每个算子的信息之间空一行以便区分

# Model information has been saved. Now, create a signal file.
with open('../ready.txt', 'w') as signal_file:
    signal_file.write('ready')

