import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

# 创建一个图(Graph)来存放算子和张量
graph = helper.make_graph(
    nodes=[
        # Conv算子节点
        helper.make_node(
            "Conv",
            name="Conv_1",
            inputs=["input", "Conv_1.weight", "Conv_1.bias"],
            outputs=["Conv_1_output_0"],
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        # LeakyReLU算子节点
        helper.make_node(
            "LeakyRelu",
            name="LeakyRelu_1",
            inputs=["Conv_1_output_0"],
            outputs=["LeakyRelu_1_output_0"],
            alpha=0.2,
        ),
    ],
    name="ConvLeakyReluModel",
    inputs=[
        # 输入张量描述
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 480, 640]),
        helper.make_tensor_value_info("Conv_1.weight", TensorProto.FLOAT, [4, 1, 3, 3]),  
        helper.make_tensor_value_info("Conv_1.bias", TensorProto.FLOAT, [4]),
    ],
    outputs=[
        # 输出张量描述
        helper.make_tensor_value_info("LeakyRelu_1_output_0", TensorProto.FLOAT, [1, 4, 480, 640]),
    ],
    # 初始值设定
    initializer=[
        # Conv权重初始化
        helper.make_tensor(
            "Conv_1.weight",
            TensorProto.FLOAT,
            [4, 1, 3, 3],  
            np.random.randn(4, 1, 3, 3).astype(float).flatten(),  # 使用随机数作为示例
        ),
        # Conv偏置初始化
        helper.make_tensor(
            "Conv_1.bias",
            TensorProto.FLOAT,
            [4],  # 对应输出通道数
            np.random.randn(4).astype(float).flatten(),
        ),
    ],
)

# 创建模型
model = helper.make_model(graph, producer_name="python-script")

# 保存模型到文件
onnx.save(model, "conv_leakyrelu_model.onnx")
