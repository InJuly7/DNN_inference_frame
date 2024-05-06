import onnx
from onnx import numpy_helper
import numpy as np
from ctypes import cdll, c_int, c_float, POINTER, cast
import ctypes

def loadOnnxModel(path):
    model = onnx.load(path)
    return model

def Add(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape_list = []
    output_shape = []
    for i in range(len(model.graph.input)):
        input_shape = [dim.dim_value for dim in model.graph.input[i].type.tensor_type.shape.dim]
        input_shape_list.append(input_shape)
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    if len(input_shape_list) == 2:
        cuda_add = lib.cuda_add_2
        cuda_add.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int]
        NumSize = 1
        for i in input_shape_list[0]:
            NumSize = NumSize * i
        result_array = np.zeros(1, dtype=np.float32)
        cuda_add(result_array, NumSize)
    elif len(input_shape_list) == 1:
        cuda_add = lib.cuda_add_1
        cuda_add.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int, c_float]
        NumSize = 1
        for i in input_shape_list[0]:
            NumSize = NumSize * i
        result_array = np.zeros(1, dtype=np.float32)
        Addinit = onnx.numpy_helper.to_array(model.graph.initializer[0])
        cuda_add(result_array, NumSize, Addinit)
    else :
        print('error')
    return result_array[0]

def LeakyRelu(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    
    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    cuda_leakyrelu = lib.cuda_leaky_1
    cuda_leakyrelu.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int, c_float]
    NumSize = 1
    for i in input_shape:
        NumSize = NumSize * i
    result_array = np.zeros(100, dtype=np.float32)
    Leakyinit = model.graph.node[0].attribute[0].f
    cuda_leakyrelu(result_array, NumSize, Leakyinit)
    return result_array[0]

def Abs(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    cuda_abs = lib.cuda_abs_1
    cuda_abs.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int]
    NumSize = 1
    for i in input_shape:
        NumSize = NumSize * i
    result_array = np.zeros(1, dtype=np.float32)
    cuda_abs(result_array, NumSize)
    return result_array[0]

def Div(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    cuda_div = lib.cuda_div_1
    cuda_div.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int, c_float]
    NumSize = 1
    for i in input_shape:
        NumSize = NumSize * i
    result_array = np.zeros(1, dtype=np.float32)
    Divinit = onnx.numpy_helper.to_array(model.graph.initializer[0])
    cuda_div(result_array, NumSize, Divinit)
    return result_array[0]

def Tanh(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    cuda_tanh = lib.cuda_tanh_1
    cuda_tanh.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int]
    NumSize = 1
    for i in input_shape:
        NumSize = NumSize * i
    result_array = np.zeros(1, dtype=np.float32)
    cuda_tanh(result_array, NumSize)
    return result_array[0]

#适配性待修改
def Slice(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    cuda_slice = lib.cuda_slice_1
    cuda_slice.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), POINTER(c_int), POINTER(c_int)]
    result_array = np.zeros(1, dtype=np.float32)
    starts = onnx.numpy_helper.to_array(model.graph.initializer[0])[0]
    ends = onnx.numpy_helper.to_array(model.graph.initializer[1])[0]
    axes = onnx.numpy_helper.to_array(model.graph.initializer[2])[0]
    steps = onnx.numpy_helper.to_array(model.graph.initializer[3])[0]
    argc = [starts, ends, axes, steps]
    cuda_slice(result_array, (c_int * len(input_shape))(*input_shape), (c_int * len(argc))(*argc))
    return result_array[0]

def Concat(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape_list = []
    for i in range(len(model.graph.input)):
        input_shape = [dim.dim_value for dim in model.graph.input[i].type.tensor_type.shape.dim]
        input_shape_list.append(input_shape)
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./op/SpeedTest.so")
    cuda_concat = lib.cuda_concat_1
    cuda_concat.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), POINTER(POINTER(c_int)), c_int, POINTER(c_int), c_int]
    result_array = np.zeros(1, dtype=np.float32)
    Concatinit = model.graph.node[0].attribute[0].i
    rows = len(input_shape_list)
    cols = [len(row) for row in input_shape_list]
    input_shape_list_c = (POINTER(c_int) * rows)()
    input_shape_list_c[:] = [cast((c_int * len(row))(*row), POINTER(c_int)) for row in input_shape_list]
    cols_c = (c_int * rows)(*cols)
    cuda_concat(result_array, input_shape_list_c, rows, cols_c, Concatinit, (c_int * len(output_shape))(*output_shape))
    return result_array[0]

def Conv(ModelPath):
    model = loadOnnxModel(str(ModelPath))
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

    lib = cdll.LoadLibrary("./SpeedTest.so")
    if (len(model.graph.initializer) == 1):
        cuda_conv2d = lib.cuda_conv2d_1
        group = model.graph.node[0].attribute[1].i
        kernel = model.graph.node[0].attribute[2].ints
        pads = model.graph.node[0].attribute[3].ints
        stride = model.graph.node[0].attribute[4].ints
        weight = onnx.numpy_helper.to_array(model.graph.initializer[0])
        data_ptr = weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cuda_conv2d.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int),
                                POINTER(c_int), POINTER(c_float)]
        result_array = np.zeros(1, dtype=np.float32)
        cuda_conv2d(result_array, group, (c_int * len(input_shape))(*input_shape), (c_int * len(output_shape))(*output_shape), (c_int * len(weight.shape))(*weight.shape),
                    (c_int * len(pads))(*pads), (c_int * len(stride))(*stride), data_ptr)
        return result_array[0]

    elif (len(model.graph.initializer) == 2):
        cuda_conv2d = lib.cuda_conv2d_2
        group = model.graph.node[0].attribute[1].i
        kernel= model.graph.node[0].attribute[2].ints
        pads  = model.graph.node[0].attribute[3].ints
        stride= model.graph.node[0].attribute[4].ints
        weight= onnx.numpy_helper.to_array(model.graph.initializer[0])
        bias  = onnx.numpy_helper.to_array(model.graph.initializer[1])
        data_ptr = weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cuda_conv2d.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), c_int, POINTER(c_int), POINTER(c_int),
                                POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_float), POINTER(c_float)]
        result_array = np.zeros(1, dtype=np.float32)
        cuda_conv2d(result_array, group, (c_int * len(input_shape))(*input_shape),(c_int * len(output_shape))(*output_shape), (c_int * len(weight.shape))(*weight.shape),
                    (c_int * len(pads))(*pads), (c_int * len(stride))(*stride), data_ptr, (c_float * len(bias))(*bias))
        return result_array[0]

def CallBackTime(ModelPath, OpType):
    function_dict = globals()
    function_name = OpType
    if function_name in function_dict:
        function_to_call = function_dict[function_name]
        return function_to_call(ModelPath)


if __name__=='__main__':
    time = CallBackTime('vis_conv_conv.onnx', 'Conv')
    print(time)
