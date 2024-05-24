#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "operator.h"
#include "util.h"
#include "OpKernel.cuh" 



extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::map<std::string, graphNode> graph; 
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;

// Node

void op::Node::PrintInfo()
{
    std::cout << "Type: " << type << std::endl;
    std::cout << "Name: " << name << std::endl;
    std::cout << "Inputs: ";
    for (const auto& input : inputs)
    {
        std::cout << input << " ";
    }
    std::cout << std::endl;
    std::cout << "Outputs: ";
    for (const auto& output : outputs)
    {
        std::cout << output << " ";
    }
    std::cout << std::endl;
}

void op::Node::StoreParameter(const std::string& line)
{

}

void op::Node::PrintPara()
{

}

void op::Node::SetAttributesFromFile(std::string line)
{

}

// Conv

op::Conv::Conv(std::string Node_type,std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::Conv::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"Dilations: ";
    for(const auto& d : dilations)
    {
        std::cout<<d<< " ";
    }
    std::cout<<std::endl;

    std::cout<<"Group: "<<group<<std::endl;

    std::cout<<"Kernel Shape: ";
    for(const auto& ks : kernel_shape)
    {
        std::cout<<ks<< " ";
    }
    std::cout<<std::endl;

    std::cout<<"Pads: ";
    for(const auto& p : pads)
    {
        std::cout<<p<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"Strides: ";
    for(const auto& s : strides)
    {
        std::cout<<s<< " ";
    }
    std::cout<<std::endl;
}

void op::Conv::SetAttributesFromFile(std::string line)
{
    // 解析当前行 来设置属性
    if(line.find("dilations") != std::string::npos)
    {
        this->dilations = parseNumbers(line);
    }
    else if(line.find("group") != std::string::npos)
    {
        std::istringstream iss(line.substr(line.find(":")+1));
        std::string group_str;
        iss >> group_str;
        group_str.erase(std::remove_if(group_str.begin(), group_str.end(), ::isspace), group_str.end());
        // string -> int
        group = std::stoi(group_str);
    
    }
    else if(line.find("kernel_shape") != std::string::npos)
    {
        this->kernel_shape = parseNumbers(line);
    }
    else if(line.find("pads") != std::string::npos)
    {
        this->pads = parseNumbers(line);
    }
    else if(line.find("strides") != std::string::npos)
    {
        this->strides = parseNumbers(line);
    }
    else
    {
        std::cout<<"存在卷积设置参数的错误  "<<line<<std::endl;
    }
    
}

void op::Conv::StoreParameter(const std::string& line)
{
    if (line.find("Parameter:") != std::string::npos)
    {
        Parameter param;
        param.setValues(line);
        parameters[param.name] = param;
    }
}

void op::Conv::PrintPara()
{
    std::cout<<"----- "<<name<<" Parameter -----"<<std::endl;
    for(const auto& para : parameters)
    {
        const Parameter& param = para.second;
        std::cout << "Parameter Name: " << param.name << std::endl;
        std::cout << "Shape: ";
        for (size_t i = 0; i < param.shape.size(); ++i)
        {
            std::cout << param.shape[i];
            if (i < param.shape.size() - 1) std::cout << ", ";
        }
        std::cout<<std::endl;

        std::cout << "Values: ";
        for (const auto& value : param.values)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

void Parameter::setValues(const std::string& line)
{   
    std::string name_substr,shape_substr,para_substr; 
    int name_pos = line.find("Parameter: ")+11;
    int shape_pos = line.find("Shape: ")+7;
    int para_pos = line.find("Values: ")+8;
    // paser name
    name_substr = line.substr(name_pos,shape_pos-7-name_pos);
    name = parseString(name_substr);
    // parse shape
    shape_substr = line.substr(shape_pos,para_pos-8-shape_pos);
    shape = parseNumbers(shape_substr);
    // parse parameter
    para_substr = line.substr(para_pos);
    values = parseFloats(para_substr);
}

void op::Conv::Execute()
{
    

}

int op::Conv::SetKernelPara()
{
    /*
        内存布局
    */
    input_shape = tensor_lifetimes[inputs[0]].tensor_shape;
    output_shape = tensor_lifetimes[outputs[0]].tensor_shape;

    for (const auto& pair : parameters)
    {
        const std::string& key = pair.first;
        const Parameter& param = pair.second;

        if (key.find("weight") != std::string::npos)
        {
            kshape = param.shape;
            weight_size = param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3];
        }
        else if (key.find("bias") != std::string::npos)
        {
            bias_size = param.shape[0];
        }
    }
    
    // NCHW 
    // input_shape 1,1,480.640      pads 1,1,1,1 up down left right
    // pad  1,482,642
    // edag 1,481,1,640 
    
    pad.push_back(input_shape[1]);
    pad.push_back(input_shape[2] + pads[0] + pads[1]);
    pad.push_back(input_shape[3] + pads[2] + pads[3]);

    edag.push_back(pads[0]);
    edag.push_back(input_shape[2] + pads[1]);
    edag.push_back(pads[2]);
    edag.push_back(input_shape[3] + pads[3]);

    pad_temp_size =  pad[0] * pad[1] * pad[2];
    kernelpara_size = weight_size + pad.size() + edag.size() + output_shape.size() + 
                                                kshape.size() + strides.size() + input_shape.size() + bias_size;

    if(PRINTKERNELPRARA)
    {
        // 打印所有变量
        std::cout << "input_shape: ";
        for (const auto& dim : input_shape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "output_shape: ";
        for (const auto& dim : output_shape) std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "pad: ";
        for (const auto& dim : pad) std::cout << dim << " ";
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
    return kernelpara_size;
}

// LeakyRelu

op::LeakyRelu::LeakyRelu(std::string Node_type,std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::LeakyRelu::SetAttributesFromFile(std::string line)
{
    if(line.find("alpha") != std::string::npos)
    {
        std::istringstream iss(line.substr(line.find(":")+1));
        std::string alpha_str;
        iss >> alpha_str;
        alpha_str.erase(std::remove_if(alpha_str.begin(), alpha_str.end(), ::isspace), alpha_str.end());
        // string -> int
        alpha = std::stof(alpha_str);
    }
}

void op::LeakyRelu::Execute()
{
    std::cout<<"This is a LeakyRelu operator's Implementation"<<std::endl;
}

void op::LeakyRelu::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"Alpha: "<<alpha<<std::endl;
}

int op::LeakyRelu::SetKernelPara()
{
    /*
        内存布局
    */

    kernelpara_size = 8;
    std::vector<int> input_shape = tensor_lifetimes[inputs[0]].tensor_shape;
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

// Constant

op::Constant::Constant(std::string Node_type,std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::Constant::SetAttributesFromFile(std::string line)
{
    if(line.find("raw_data:") != std::string::npos)
    {
        std::istringstream iss(line.substr(line.find("raw_data:")+9));
        std::string constant_value_str;
        std::string cleanString;
        iss >> constant_value_str;
        for (char c : constant_value_str)
        {   
            if (isdigit(c) || c == ' ' || c == '.' || c == '-')
            {
                cleanString += c;
            }
        }
        cleanString.erase(std::remove_if(cleanString.begin(), cleanString.end(), ::isspace), cleanString.end());
        constant_value = std::stof(cleanString);
    }
}

void op::Constant::Execute()
{
    std::cout<<"This is a Constant operator's Implementation"<<std::endl;
}

void op::Constant::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"Constant_value "<<constant_value<<std::endl;
}

// Slice

op::Slice::Slice(std::string Node_type, std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::Slice::SetAttributesFromFile()
{
    std::string start_str,end_str,axis_str,steps_str;
    start_str = getNodeName(inputs[1]);
    end_str = getNodeName(inputs[2]);
    axis_str = getNodeName(inputs[3]);
    steps_str = getNodeName(inputs[4]);
    auto& Node_op_0 = operatorMap[start_str];
    if (auto constant_Ptr = dynamic_cast<op::Constant*>(Node_op_0.get()))
    {
        start_index = constant_Ptr->constant_value;
    }
    auto& Node_op_1 = operatorMap[end_str];
    if (auto constant_Ptr = dynamic_cast<op::Constant*>(Node_op_1.get()))
    {
        end_index = constant_Ptr->constant_value;
    }
    auto& Node_op_2 = operatorMap[axis_str];
    if (auto constant_Ptr = dynamic_cast<op::Constant*>(Node_op_2.get()))
    {
        axis = constant_Ptr->constant_value;
    }
    auto& Node_op_3 = operatorMap[steps_str];
    if (auto constant_Ptr = dynamic_cast<op::Constant*>(Node_op_3.get()))
    {
        steps = constant_Ptr->constant_value;
    }
}

void op::Slice::Execute()
{
    std::cout<<"This is a Constant operator's Implementation"<<std::endl;
}

void op::Slice::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"start_index "<<start_index<<std::endl;
    std::cout<<"end_index "<<end_index<<std::endl;
    std::cout<<"axis "<<axis<<std::endl;
    std::cout<<"steps "<<steps<<std::endl;
}

// Concat

op::Concat::Concat(std::string Node_type, std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::Concat::SetAttributesFromFile(std::string line)
{
    if(line.find("axis") != std::string::npos)
    {
        std::istringstream iss(line.substr(line.find(":")+1));
        std::string axis_str;
        iss >> axis_str;
        axis_str.erase(std::remove_if(axis_str.begin(), axis_str.end(), ::isspace), axis_str.end());
        // string -> int
        axis = std::stoi(axis_str);
    }
}

void op::Concat::Execute()
{
    std::cout<<"This is a Constant operator's Implementation"<<std::endl;
}

void op::Concat::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"axis "<<axis<<std::endl;
}

// Abs

op::Abs::Abs(std::string Node_type, std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::Abs::SetAttributesFromFile(std::string line)
{

}

void op::Abs::Execute()
{
    std::cout<<"This is a Abs operator's Implementation"<<std::endl;
}

void op::Abs::PrintAttributes()
{

}

int op::Abs::SetKernelPara()
{
    /*
        内存布局
    */
    std::vector<int> input_shape = tensor_lifetimes[inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    kernelpara_size = 4;
    if(PRINTKERNELPRARA)
    {
        std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
        std::cout <<"numElements: "<<numElements<<std::endl;
    }
    return kernelpara_size;
}

// Tanh

op::Tanh::Tanh(std::string Node_type, std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
}

void op::Tanh::SetAttributesFromFile(std::string line)
{

}

void op::Tanh::Execute()
{
    std::cout<<"This is a Tanh operator's Implementation"<<std::endl;
}

void op::Tanh::PrintAttributes()
{

}

int op::Tanh::SetKernelPara()
{
    std::vector<int> input_shape = tensor_lifetimes[inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];
    kernelpara_size = 4;
    if(PRINTKERNELPRARA)
    {
        std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
        std::cout <<"numElements: "<<numElements<<std::endl;
    }
    return kernelpara_size;
}

// Div

op::Div::Div(std::string Node_type, std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
}

void op::Div::SetAttributesFromFile()
{
    std::string div_value_str = {};
    if(inputs.size() == 2)
    {
        
        for(int i = 0; i < 2; i++)
        {
            if(inputs[i].find("Constant") != std::string::npos)
            {
                div_value_str = getNodeName(inputs[i]);
                break;
            }
        }
    }
    if(!div_value_str.empty())
    {
        auto& Node_op_0 = operatorMap[div_value_str];
        auto constant_Ptr = dynamic_cast<op::Constant*>(Node_op_0.get());
        div_value = constant_Ptr->constant_value;
    }
}

void op::Div::Execute()
{
    std::cout<<"This is a Div operator's Implementation"<<std::endl;
}

void op::Div::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"Div_constant "<<div_value<<std::endl;
}

int op::Div::SetKernelPara()
{
    if(div_value != 1) kernelpara_size = 8;
    else kernelpara_size = 4;
    std::vector<int> input_shape = tensor_lifetimes[inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];

    if(PRINTKERNELPRARA)
    {
        std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"div_value: "<<div_value<<std::endl; 
    }
    return kernelpara_size;
}
// Add

op::Add::Add(std::string Node_type, std::string Node_name)
{
    this->name = Node_name;
    this->type = Node_type;
    // std::cout<<"Creating "<<" "<<this->type<<" "<<"operator"<<"  "<<this->name<<std::endl;
}

void op::Add::SetAttributesFromFile()
{
    std::string add_value_str = {};
    if(inputs.size() == 2)
    {
        for(int i = 0; i < 2; i++)
        {
            if(inputs[i].find("Constant") != std::string::npos)
            {
                add_value_str = getNodeName(inputs[i]);
                break;
            }
        }
    }
    if(!add_value_str.empty())
    {
        auto& Node_op_0 = operatorMap[add_value_str];
        auto constant_Ptr = dynamic_cast<op::Constant*>(Node_op_0.get());
        add_value = constant_Ptr->constant_value;
    }
}

void op::Add::Execute()
{

}

void op::Add::PrintAttributes()
{
    std::cout<<"----- "<<name<<" Attribute -----"<<std::endl;
    std::cout<<"Add_constant "<<add_value<<std::endl;
}

int op::Add::SetKernelPara()
{
    if(add_value != 0) kernelpara_size = 8;
    else kernelpara_size = 4;
    std::vector<int> input_shape = tensor_lifetimes[inputs[0]].tensor_shape;
    numElements = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3];

    if(PRINTKERNELPRARA)
    {
        std::cout <<"kernelpara_size: "<<kernelpara_size<<std::endl;
        std::cout <<"numElements: "<<numElements<<std::endl;
        std::cout <<"add_value: "<<add_value<<std::endl; 
    }
    return kernelpara_size;
}
