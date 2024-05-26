#include <iostream>
#include <map>

#include "util.h"
#include "cudaop.h"

extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
extern std::map<std::string, graphNode> graph; 
extern std::multimap<size_t, std::string> tensorOffsets;


cuda::Node::Node(const std::string &Node_type, const std::string &Node_name)
{
    type = Node_type;
    name = Node_type;
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

}

int cuda::Conv::SetKernelPara()
{
    // /*
    //     内存布局
    // */
    // input_shape = tensor_lifetimes[graph[name].inputs[0]].tensor_shape;
    // output_shape = tensor_lifetimes[getOutputTensor(name)].tensor_shape;

    // for (const auto& pair : parameters)
    // {
    //     const std::string& key = pair.first;
    //     const Parameter& param = pair.second;

    //     if (key.find("weight") != std::string::npos)
    //     {
    //         kshape = param.shape;
    //         weight_size = param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3];
    //     }
    //     else if (key.find("bias") != std::string::npos)
    //     {
    //         bias_size = param.shape[0];
    //     }
    // }
    
    // // NCHW 
    // // input_shape 1,1,480.640      pads 1,1,1,1 up down left right
    // // pad  1,482,642
    // // edag 1,481,1,640 
    
    // pad.push_back(input_shape[1]);
    // pad.push_back(input_shape[2] + pads[0] + pads[1]);
    // pad.push_back(input_shape[3] + pads[2] + pads[3]);

    // edag.push_back(pads[0]);
    // edag.push_back(input_shape[2] + pads[1]);
    // edag.push_back(pads[2]);
    // edag.push_back(input_shape[3] + pads[3]);

    // pad_temp_size =  pad[0] * pad[1] * pad[2];
    // kernelpara_size = weight_size + pad.size() + edag.size() + output_shape.size() + 
    //                                             kshape.size() + strides.size() + input_shape.size() + bias_size;

    // if(PRINTKERNELPRARA)
    // {
    //     // 打印所有变量
    //     std::cout << "input_shape: ";
    //     for (const auto& dim : input_shape) std::cout << dim << " ";
    //     std::cout << std::endl;

    //     std::cout << "output_shape: ";
    //     for (const auto& dim : output_shape) std::cout << dim << " ";
    //     std::cout << std::endl;

    //     std::cout << "pad: ";
    //     for (const auto& dim : pad) std::cout << dim << " ";
    //     std::cout << std::endl;

    //     std::cout << "edag: ";
    //     for (const auto& dim : edag) std::cout << dim << " ";
    //     std::cout << std::endl;

    //     std::cout << "kshape: ";
    //     for (const auto& dim : kshape) std::cout << dim << " ";
    //     std::cout << std::endl;

    //     std::cout << "strides: ";
    //     for (const auto& dim : strides) std::cout << dim << " ";
    //     std::cout << std::endl;

    //     std::cout << "weight_size: " << weight_size << std::endl;
    //     std::cout << "bias_size: " << bias_size << std::endl;
    //     std::cout << "kernelpara_size: " << kernelpara_size << std::endl;
    // }
    return kernelpara_size;

}

// Abs

void cuda::Abs::Execute()
{

}

int cuda::Abs::SetKernelPara()
{
    /*
        内存布局
    */
    kernelpara_size = 4;
    std::vector<int> input_shape = tensor_lifetimes[graph[name].inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    if(PRINTKERNELPRARA)
    {
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"kernelpara_size: "<< kernelpara_size << std::endl;
    }
    return kernelpara_size;

}

// LeakyRelu

void cuda::LeakyRelu::Execute()
{

}

int cuda::LeakyRelu::SetKernelPara()
{
     /*
        内存布局
    */

    kernelpara_size = 8;
    std::vector<int> input_shape = tensor_lifetimes[graph[name].inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    
    if(PRINTKERNELPRARA)
    {
        // 打印所有变量
        std::cout <<"alpha: "<<alpha<<std::endl;
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"kernelpara_size: "<< kernelpara_size << std::endl;
    }
    return kernelpara_size;
}

// Tanh

void cuda::Tanh::Execute()
{

}

int cuda::Tanh::SetKernelPara()
{
    kernelpara_size = 4;
    std::vector<int> input_shape = tensor_lifetimes[graph[name].inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    if(PRINTKERNELPRARA)
    {
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"kernelpara_size: "<< kernelpara_size << std::endl;
    }
    return kernelpara_size;
}

// Add

void cuda::Add::Execute()
{

}

int cuda::Add::SetKernelPara()
{
    if(add_value != 0) kernelpara_size = 8;
    else kernelpara_size = 4;
    std::vector<int> input_shape = tensor_lifetimes[graph[name].inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];

    if(PRINTKERNELPRARA)
    {
        
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"add_value: "<<add_value<<std::endl;
        std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
    }
    return kernelpara_size;
}

// Div

void cuda::Div::Execute()
{

}

int cuda::Div::SetKernelPara()
{
    if(div_value != 1) kernelpara_size = 8;
    else kernelpara_size = 4;
    std::vector<int> input_shape = tensor_lifetimes[graph[name].inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];

    if(PRINTKERNELPRARA)
    {
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"div_value: "<<div_value<<std::endl; 
        std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
    }
    return kernelpara_size;
    
}


