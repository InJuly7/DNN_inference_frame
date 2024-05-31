#include <iostream>
#include <map>

#include "util.h"
#include "cudaop.h"
#include "OpKernel.cuh"

extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
extern std::map<std::string, graphNode> graph; 
extern std::multimap<size_t, std::string> tensorOffsets;
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern size_t totalParaSize;
extern size_t max_pad_temp;


cuda::Node::Node(const std::string &Node_type, const std::string &Node_name)
{
    type = Node_type;
    name = Node_name;
    std::string output_tensor = getOutputTensor(Node_name);
    int input_size = graph[Node_name].inputs.size();
    for (const auto& pair : tensorOffsets)
    {
        if(pair.second == output_tensor)
        {
            outputs_idx.push_back(pair.first);
        }
        else if(input_size != 0)
        {
            for (const auto& str : graph[Node_name].inputs)
            {
                std::string str_out = getOutputTensor(str);
                if(pair.second == str_out)
                {
                    inputs_idx.push_back(pair.first);
                }
            }
        }
    }
}

void cuda::Node::PrintCudaNode()
{
    std::cout << "Type: " << type << "\n";
    std::cout << "Name: " << name << "\n";
    std::cout << "Inputs Indexes: ";
    for (int idx : inputs_idx)
    {
        std::cout << idx << " ";
    }
    std::cout << "\n";
    std::cout << "Outputs Indexes: ";
    for (int idx : outputs_idx)
    {
        std::cout << idx << " ";
    }
    std::cout<< "Para Index "<<para_index<<"\n";
    std::cout << "\n";
}

void cuda::Node::Execute()
{

}

int cuda::Node::SetKernelPara()
{
    return -1;
}

// Conv

void cuda::Conv::Execute()
{
    ConvKernel(d_A, d_C, d_weight, d_pads, d_edag, d_output_shape, d_kshape, 
                        d_strides, d_pad_temp, d_input_shape, group, d_bias);
}

int cuda::Conv::SetKernelPara()
{
    /*
        内存布局
        0                               --> weights
        ... + weight_size               --> bias
        ... + bias_size                 --> pads
        ... + pads.size()               --> edag
        ... + edag.size()               --> output_shape
        ... + output_shape.size()       --> kshape
        ... + kshape.size()             --> strides
        ... + strides.size()            --> input_shape
    */
    auto& node = *operatorMap[name];
    auto conv_ptr = dynamic_cast<op::Conv*>(&node);

    input_shape = tensor_lifetimes[getOutputTensor(graph[name].inputs[0])].tensor_shape;
    output_shape = tensor_lifetimes[getOutputTensor(name)].tensor_shape;

    for (const auto& pair : conv_ptr->parameters)
    {
        const std::string& key = pair.first;
        const Parameter& param = pair.second;
        if (key.find("weight") != std::string::npos)
        {
            kshape = param.shape;
            weight = param.values;
            weight_size = param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3];
        }
        else if (key.find("bias") != std::string::npos)
        {
            bias = param.values;
            bias_size = param.shape[0];
        }
    }
    
    // NCHW 
    // input_shape 1,1,480.640      op_pads 1,1,1,1 up down left right
    // pad  1,482,642
    // edag 1,481,1,640 
    std::vector<int> op_pads = conv_ptr->pads;
    pads.push_back(input_shape[1]);
    pads.push_back(input_shape[2] + op_pads[0] + op_pads[1]);
    pads.push_back(input_shape[3] + op_pads[2] + op_pads[3]);

    edag.push_back(op_pads[0]);
    edag.push_back(input_shape[2] + op_pads[1]);
    edag.push_back(op_pads[2]);
    edag.push_back(input_shape[3] + op_pads[3]);

    group = conv_ptr->group;
    pads_temp_size =  pads[0] * pads[1] * pads[2];
    strides = conv_ptr->strides;
    kernelpara_size = weight_size + pads.size() + edag.size() + output_shape.size() + 
                                                kshape.size() + strides.size() + input_shape.size() + bias_size;
    
    (pads_temp_size > max_pad_temp) ? max_pad_temp = pads_temp_size : max_pad_temp;
    return kernelpara_size;

}

void cuda::Conv::printArgInfo()
{
    if(PRINTKERNELPRARA)
    {
        // 打印所有变量
        std::cout << "input_shape: ";
        for (const auto& dim : input_shape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "output_shape: ";
        for (const auto& dim : output_shape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "pads: ";
        for (const auto& dim : pads) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "edag: ";
        for (const auto& dim : edag) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "kshape: ";
        for (const auto& dim : kshape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "strides: ";
        for (const auto& dim : strides) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "weight_size: " << weight_size << std::endl;
        std::cout << "bias_size: " << bias_size << std::endl;
        std::cout << "kernelpara_size: " << kernelpara_size << std::endl;
    }

}

// Abs

void cuda::Abs::Execute()
{
   AbsKernel(d_A, d_C, numElements);
}

int cuda::Abs::SetKernelPara()
{
    kernelpara_size = 0;
    std::vector<int> input_shape = tensor_lifetimes[getOutputTensor(graph[name].inputs[0])].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    return kernelpara_size;
}

void cuda::Abs::printArgInfo()
{
    std::cout <<"numElements: "<<numElements<<std::endl;
    std::cout <<"kernelpara_size: "<< kernelpara_size << std::endl;
}

// LeakyRelu

void cuda::LeakyRelu::Execute()
{
    LeakyReluKernel(d_A, alpha, d_C, numElements);
}

int cuda::LeakyRelu::SetKernelPara()
{
    kernelpara_size = 0;
    std::vector<int> input_shape = tensor_lifetimes[getOutputTensor(graph[name].inputs[0])].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    auto& node = *operatorMap[name];
    auto leaky_ptr = dynamic_cast<op::LeakyRelu*>(&node);
    alpha = leaky_ptr->alpha;
    return kernelpara_size;
}

void cuda::LeakyRelu::printArgInfo()
{
    std::cout <<"alpha: "<<alpha<<std::endl;
    std::cout <<"numElements: "<<numElements<<std::endl;
    std::cout <<"kernelpara_size: "<< kernelpara_size << std::endl;
}

// Tanh

void cuda::Tanh::Execute()
{
    TanhKernel(d_A, d_C, numElements);
}

int cuda::Tanh::SetKernelPara()
{
    kernelpara_size = 0;
    std::vector<int> input_shape = tensor_lifetimes[getOutputTensor(graph[name].inputs[0])].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    return kernelpara_size;
}

void cuda::Tanh::printArgInfo()
{
    std::cout <<"numElements: "<<numElements<<std::endl;
    std::cout <<"kernelpara_size: "<< kernelpara_size << std::endl;
}

// Add

void cuda::Add::Execute()
{
    AddKernel(d_A, d_B, d_C, add_value, numElements);
}

int cuda::Add::SetKernelPara()
{
    auto& node = *operatorMap[name];
    auto add_ptr = dynamic_cast<op::Add*>(&node);
    add_value = add_ptr->add_value;
    kernelpara_size = 0;
    std::vector<int> input_shape = tensor_lifetimes[getOutputTensor(graph[name].inputs[0])].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    return kernelpara_size;
}

void cuda::Add::printArgInfo()
{
    std::cout <<"numElements: "<<numElements<<std::endl;
    std::cout <<"add_value: "<<add_value<<std::endl;
    std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
}

// Div

void cuda::Div::Execute()
{
    DivKernel(d_A, d_C, div_value, numElements);
}

int cuda::Div::SetKernelPara()
{
    auto& node = *operatorMap[name];
    auto div_ptr = dynamic_cast<op::Div*>(&node);
    div_value = div_ptr->div_value;

    kernelpara_size = 0;
    std::vector<int> input_shape = tensor_lifetimes[getOutputTensor(graph[name].inputs[0])].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    return kernelpara_size;
    
}

void cuda::Div::printArgInfo()
{
    std::cout <<"numElements: "<<numElements<<std::endl;
    std::cout <<"div_value: "<<div_value<<std::endl; 
    std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
}
