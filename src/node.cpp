#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "./include/node.h"
#include "./include/node.h"
// #include "./include/graph.h"

extern std::map<std::string, std::unique_ptr<ONNX::Node>> ONNX_NODE;

// Node

// void ONNX::Node::PrintInfo()
// {
//     std::cout << "Type: " << type << std::endl;
//     std::cout << "Name: " << name << std::endl;
//     std::cout << "Inputs: ";
//     for (const auto& input : inputs)
//     {
//         std::cout << input << " ";
//     }
//     std::cout << std::endl;
//     std::cout << "Outputs: ";
//     for (const auto& output : outputs)
//     {
//         std::cout << output << " ";
//     }
//     std::cout << std::endl;
// }

// void ONNX::Node::StoreParameter(const std::string& line)
// {

// }

// void ONNX::Node::PrintPara()
// {

// }

// void ONNX::Node::SetAttributesFromFile(std::string line)
// {

// }

ONNX::InputNode::InputNode(std::string node_name, std::vector<int> node_shape)
{
    this->inputs = {};
    this->output = {};
    this->type = "";
    this->name = node_name;
    this->shape = node_shape;
}

void ONNX::InputNode::print_node()
{
    std::cout<< "Node Name: " << this->name << std::endl;
    std::cout<< "Shape: ";
    for(int dim : this->shape)
    {
        std::cout<< dim <<" ";
    }
    std::cout << std::endl;
}

ONNX::OutputNode::OutputNode(std::string node_name, std::vector<int> node_shape)
{
    this->inputs = {};
    this->output = {};
    this->type = "";
    this->name = node_name;
    this->shape = node_shape;
}

void ONNX::OutputNode::print_node()
{
    std::cout<< "Node Name: " << this->name << std::endl;
    std::cout<< "Shape: ";
    for(int dim : this->shape)
    {
        std::cout<< dim <<" ";
    }
    std::cout << std::endl;
}



// Abs

ONNX::Abs::Abs(std::string node_name, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "Abs";
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::Abs::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout<<std::endl;
    std::cout << "Outputs: " << output << std::endl;
}
// Add

ONNX::Add::Add(std::string node_name, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "Add";
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::Add::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout<<std::endl;
    std::cout << "Outputs: " << output << std::endl;
}


// Concat

ONNX::Concat::Concat(std::string node_name, int axis, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "Concat";
    this->axis = axis;
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::Concat::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout << "Axis: " << axis << std::endl;
    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout<<std::endl;
    std::cout << "Outputs: " << output << std::endl;
}


// Conv

ONNX::Conv::Conv(std::string node_name, std::vector<int> dilations, int group, 
                    std::vector<int> kernel_shape, std::vector<int> pads, std::vector<int> strides, std::vector<std::string> node_inputs, 
                    std::string node_output, std::vector<int> weight_shape, std::vector<float> weights, std::vector<int> bias_shape, 
                    std::vector<float> bias)
{
    this->name = node_name;
    this->type = "Conv";
    this->dilations = dilations;
    this->group = group;
    this->kernel_shape = kernel_shape;
    this->pads = pads;
    this->strides = strides;
    this->inputs = node_inputs;
    this->output = node_output;
    this->weight_shape = weight_shape;
    this->weights = weights;
    this->bias_shape = bias_shape;
    this->bias = bias;
}

void ONNX::Conv::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;

    std::cout << "dilations: ";
    for(int dim : this->dilations)
    {
        std::cout<< dim <<" ";
    }
    std::cout << std::endl;
    
    std::cout << "group: " <<group<<std::endl;
    std::cout << "kernel_shape: ";
    for(int dim : this->kernel_shape)
    {
        std::cout<< dim <<" ";
    }
    std::cout << std::endl;

    std::cout<< "pads: ";
    for(int dim : this->pads)
    {
        std::cout<< dim <<" ";
    }
    std::cout << std::endl;

    std::cout<< "strides: ";
    for(int dim : this->strides)
    {
        std::cout<< dim <<" ";
    }
    std::cout << std::endl;

    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout << std::endl;
    std::cout << "Outputs: " << output << std::endl;

    std::cout<< "weights-shape: ";
    for(int dim : this->weight_shape)
    {
        std::cout<< dim <<" ";
    }
    
    std::cout << ", Values: ";
    for(float value : this->weights)
    {
        std::cout<< value <<" ";
    }
    std::cout << std::endl;


    if(!this->bias.empty())
    {
        std::cout<< "bias-shape: ";
        for(int dim : this->bias_shape)
        {
            std::cout<< dim <<" ";
        }
    
        std::cout << ", Values: ";
        for(float value : this->bias)
        {
            std::cout<< value <<" ";
        }
        std::cout << std::endl;
    }
}


// Div

ONNX::Div::Div(std::string node_name, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "Div";
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::Div::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout<<std::endl;
    std::cout << "Outputs: " << output << std::endl;

}

// LeakyRelu

ONNX::LeakyRelu::LeakyRelu(std::string node_name, float alpha, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "LeakyRelu";
    this->alpha = alpha;
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::LeakyRelu::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout << "Alpha: "<< alpha << std::endl;
    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout<<std::endl;
    std::cout << "Outputs: " << output << std::endl;

}


// Slice

ONNX::Slice::Slice(std::string node_name, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "Slice";
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::Slice::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout<< "Node Inputs: ";
    for(std::string str : this->inputs)
    {
        std::cout<< str <<" ";
    }
    std::cout<<std::endl;
    std::cout << "Outputs: " << output << std::endl;
}

// Tanh

ONNX::Tanh::Tanh(std::string node_name, std::vector<std::string> node_inputs, std::string node_output)
{
    this->name = node_name;
    this->type = "Tanh";
    this->inputs = node_inputs;
    this->output = node_output;
}

void ONNX::Tanh::print_node()
{
    std::cout << "Node Name: " << name << std::endl;
    std::cout << "Node Type: " << type << std::endl;
    std::cout<< "Node Inputs: "<< inputs[0] << std::endl;
    std::cout << "Outputs: " << output << std::endl;

}
