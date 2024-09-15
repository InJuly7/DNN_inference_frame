#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <regex>
#include <string>
#include "./include/parse.h"
#include "./include/node.h"
#define PRINT_NODE 1
 
extern std::map<std::string, std::unique_ptr<ONNX::Node>> NodeMap;

// 去掉字符串的前后空格
std::string trim(const std::string& str)
{
    size_t first = str.find_first_not_of(" \t\n\r"); 
    if (first == std::string::npos) 
        return ""; 
    size_t last = str.find_last_not_of(" \t\n\r");   
    return str.substr(first, (last - first + 1));
}

// parse_xxxNode 从txt文件构建节点,存储到 NodeMap中
void parse_inputNode(std::string str)
{
    std::regex name_regex(R"(Input Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex shape_regex(R"(Shape:\s*\[([\d,\s]+)\])");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    while(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        std::string node_name = match[1].str();
        search_start = match.suffix().first;
        if(std::regex_search(search_start, str.cend(), match, shape_regex))
        {
            std::string shape_str = match[1].str();
            search_start = match.suffix().first;

            std::vector<int> shape = {};
            std::stringstream ss(shape_str);
            std::string temp;
            while(std::getline(ss, temp, ',' ))
            {
                shape.push_back(std::stoi(trim(temp)));
            }
            NodeMap[node_name] = std::make_unique<ONNX::InputNode>(node_name,shape);
        }
        if(PRINT_NODE)
        {
            auto& node = NodeMap[node_name];
            node->print_node();
        }
        
    }
}

void parse_outputNode(std::string str)
{
    std::regex name_regex(R"(Output Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex shape_regex(R"(Shape:\s*\[([\d,\s]+)\])");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    while(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        std::string node_name = match[1].str();
        search_start = match.suffix().first;
        if(std::regex_search(search_start, str.cend(), match, shape_regex))
        {
            std::string shape_str = match[1].str();
            search_start = match.suffix().first;

            std::vector<int> shape = {};
            std::stringstream ss(shape_str);
            std::string temp;
            while(std::getline(ss, temp, ',' ))
            {
                shape.push_back(std::stoi(trim(temp)));
            }
            NodeMap[node_name] = std::make_unique<ONNX::OutputNode>(node_name,shape);
        }
        if(PRINT_NODE)
        {
            auto& node = NodeMap[node_name];
            node->print_node();
        }
    }
}

void parse_absNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_,\s]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};

    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        node_inputs.push_back(match[1]);
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1];
    }
    NodeMap[node_name] = std::make_unique<ONNX::Abs>(node_name,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }
}   

void parse_addNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_,\s]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        std::string inputs_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(inputs_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            node_inputs.push_back(trim(temp));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1].str();
        std::cout<<node_output<<std::endl;
    }
    
    NodeMap[node_name] = std::make_unique<ONNX::Add>(node_name,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }

}

void parse_concatNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex axis_regex(R"(axis:\s*(\d*)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_,\s]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};
    int axis = 0;

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, axis_regex))
    {
        axis = std::stoi(match[1].str());
        search_start = match.suffix().first;
    } 
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        std::string inputs_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(inputs_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            node_inputs.push_back(trim(temp));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1].str();
    }
    NodeMap[node_name] = std::make_unique<ONNX::Concat>(node_name,axis,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }
}

void parse_constantNode(std::string str)
{   
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex datatype_regex(R"(data_type:\s*(\d))");
    std::regex rawdata_regex(R"(raw_data:\s*\[([-+]?\d*\.?\d+)\])");
    
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    int datatype = 0;
    
    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, datatype_regex))
    {
        datatype = std::stoi(match[1].str());
        search_start = match.suffix().first;
    }

    // value: dims: [1], data_type: 7, raw_data: [1]
    // 一维张量 数据类型    7 : INT64 1 : FLOAT
    // raw_data: [ ]  表示张量的原始数据
    if(std::regex_search(search_start, str.cend(), match, rawdata_regex))
    {
        std::string raw_data = match[1].str();
        search_start = match.suffix().first;
        if(std::regex_search(search_start, str.cend(), match, output_regex))
        {
            node_output = match[1].str();
            search_start = match.suffix().first;
        }
        // float
        if(datatype == 1)
        {
            NodeMap[node_name] = std::make_unique<ONNX::Constant<float>>(node_name,std::stof(raw_data),node_output);
        }
        else if(datatype == 7)
        {
            NodeMap[node_name] = std::make_unique<ONNX::Constant<int>>(node_name,std::stoi(raw_data),node_output);
        }
        
    }
    if(PRINT_NODE)
    {
        auto &node = NodeMap[node_name];
        node->print_node();
    }
        
   
}

void parse_convNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex dilations_regex(R"(dilations:\s*\[([\d,\s]+)\])");
    std::regex group_regex(R"(group:\s*(\d*)\s*\n)");
    std::regex kernelshape_regex(R"(kernel_shape:\s*\[([\d,\s]+)\])");
    std::regex pads_regex(R"(pads:\s*\[([\d,\s]+)\])");
    std::regex strides_regex(R"(strides:\s*\[([\d,\s]+)\])");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_\.,\s]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");

    std::string node_name = "";
    std::vector<int> dilations = {};
    int group;
    std::vector<int> kernel_shape = {};
    std::vector<int> pads = {};
    std::vector<int> strides = {};
    std::vector<std::string> node_inputs = {};
    std::string node_output;

    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, dilations_regex))
    {
        std::string dilations_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(dilations_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            dilations.push_back(std::stoi(trim(temp)));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, group_regex))
    {
        group = std::stoi(match[1].str());
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, kernelshape_regex))
    {
        std::string kernelshape_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(kernelshape_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            kernel_shape.push_back(std::stoi(trim(temp)));
        }

    }
    if(std::regex_search(search_start, str.cend(), match, pads_regex))
    {
        std::string pads_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(pads_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            pads.push_back(std::stoi(trim(temp)));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, strides_regex))
    {
        std::string strdes_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(strdes_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            strides.push_back(std::stoi(trim(temp)));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        std::string inputs_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(inputs_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            node_inputs.push_back(trim(temp));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1].str();
        search_start = match.suffix().first;
    }


    NodeMap[node_name] = std::make_unique<ONNX::Conv>(node_name,dilations,group,kernel_shape,pads,strides,node_inputs,node_output);



    
}   


void parse_divNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_,\s]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        std::string inputs_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(inputs_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            node_inputs.push_back(trim(temp));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1].str();
        std::cout<<node_output<<std::endl;
    }
    
    NodeMap[node_name] = std::make_unique<ONNX::Div>(node_name,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }

}

void parse_leakyreluNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex alpha_regex(R"(alpha:\s*([-+]?\d*\.?\d+)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};
    float alpha = 0;

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, alpha_regex))
    {
        alpha = std::stof(match[1].str());
        search_start = match.suffix().first;
    } 
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        node_inputs.push_back(match[1].str());
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1].str();
    }
    NodeMap[node_name] = std::make_unique<ONNX::LeakyRelu>(node_name,alpha,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }


}

void parse_sliceNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_,\s]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};

    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        std::string inputs_str = match[1].str();
        search_start = match.suffix().first;
        std::stringstream ss(inputs_str);
        std::string temp;
        while(std::getline(ss, temp, ','))
        {
            node_inputs.push_back(trim(temp));
        }
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1].str();
        std::cout<<node_output<<std::endl;
    }
    
    NodeMap[node_name] = std::make_unique<ONNX::Slice>(node_name,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }
}

void parse_tanhNode(std::string str)
{
    std::regex name_regex(R"(Operator Name:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex inputs_regex(R"(Inputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::regex output_regex(R"(Outputs:\s*([a-zA-Z0-9/_]+)\s*\n)");
    std::smatch match;
    std::string::const_iterator search_start = str.cbegin();
    std::string node_name = "";
    std::string node_output = "";
    std::vector<std::string> node_inputs = {};
    if(std::regex_search(search_start, str.cend(), match, name_regex))
    {
        node_name = match[1].str();
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, inputs_regex))
    {
        node_inputs.push_back(match[1]);
        search_start = match.suffix().first;
    }
    if(std::regex_search(search_start, str.cend(), match, output_regex))
    {
        node_output = match[1];
    }
    NodeMap[node_name] = std::make_unique<ONNX::Tanh>(node_name,node_inputs,node_output);
    if(PRINT_NODE)
    {
        auto& node = NodeMap[node_name];
        node->print_node();
    }
}








// 读取模型文件
// 每次读取一段, 每次处理一段, 添加图节点和算子节点
void Read_Model(std::string model_txt)
{
    std::ifstream file(model_txt);
    std::string line;
    std::stringstream paragraph;
  
    while(getline(file, line))
    {
        if(line.empty())
        {
            std::string currentNodeType;
            std::string content = paragraph.str();
            std::regex nodetype_regex(R"(Operator Type:\s*(\w+))"); 
            std::smatch match;
            std::string::const_iterator search_start = content.cbegin();
            // 中间节点 带有算子类型 
            if(std::regex_search(search_start, content.cend(), match, nodetype_regex))
            {   
                currentNodeType = match[1].str();
                if(currentNodeType == "Abs") parse_absNode(content);
                else if(currentNodeType == "Add") parse_addNode(content);
                else if(currentNodeType == "Concat") parse_concatNode(content);
                else if(currentNodeType == "Constant") parse_constantNode(content);
                else if(currentNodeType == "Conv") parse_convNode(content);
                else if(currentNodeType == "Div") parse_divNode(content); 
                else if(currentNodeType == "LeakyRelu") parse_leakyreluNode(content); 
                else if(currentNodeType == "Slice") parse_sliceNode(content);
                else if(currentNodeType == "Tanh") parse_tanhNode(content);
                content = "";
                currentNodeType = "";
            }
            
            // 输入节点 输出节点
            else if(content.find("Model Inputs:") != std::string::npos)
            {
                parse_inputNode(content);
            }
            else if(content.find("Model Outputs:") != std::string::npos)
            {   
                parse_outputNode(content);
            }

            paragraph.str("");
            paragraph.clear();
        }
        else
        {
            paragraph << line << "\n";

        }
    }
}


