#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "util.h"
#include <algorithm>


#define PRINT_OP 0
#define PRINT_GRAPH 1
#define PRINT_TOPO 0

extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::map<std::string, graphNode> graph;
std::unordered_map<std::string, int> visited;
extern std::vector<std::string> topologicalOrder;


std::vector<int> parseNumbers(const std::string& line)
{
    std::vector<int> numbers;
    // 跳过 '[ ()'
    std::string cleanString;
    for (char c : line)
    {   
        if (isdigit(c) || c == ' ' || c == ',' || c == '-')
        {
            cleanString += c;
        }
    }
    std::istringstream iss(cleanString);
    std::string num_str;
    while(std::getline(iss, num_str, ','))
    {
        num_str.erase(std::remove_if(num_str.begin(), num_str.end(), ::isspace), num_str.end());
        // string -> int
        if(!num_str.empty()) numbers.push_back(std::stoi(num_str));
    }
    return numbers;
}

// 去除空格 或者结尾逗号
std::string parseString(std::string src_sub)
{
    std::string dst_sub;
    std::istringstream iss(src_sub);
    std::getline(iss, dst_sub, ',');
    dst_sub.erase(std::remove_if(dst_sub.begin(), dst_sub.end(), ::isspace), dst_sub.end());
    return dst_sub;
}

std::vector<float> parseFloats(const std::string& line)
{
    std::vector<float> paras;
    std::string cleanString;
    for (char c : line)
    {   
        if (isdigit(c) || c == ' ' || c == ',' || c == '-' || c == '.')
        {
            cleanString += c;
        }
    }
    std::istringstream iss(cleanString);
    std::string num_str;
    while (std::getline(iss, num_str, ','))
    {
        num_str.erase(std::remove_if(num_str.begin(), num_str.end(), ::isspace), num_str.end());
        if(!num_str.empty()) paras.push_back(std::stof(num_str));
    }

    return paras;
}

std::string getNodeName(const std::string& outputName)
{
    std::string suffix = "_output_";  
    size_t pos = outputName.find(suffix);
    if (pos != std::string::npos)
    {
        return outputName.substr(0, pos);  
    }
    return outputName;  
}

void ConstInput(op::Node& operatorNode)
{
    if (operatorNode.type == "Slice")
    {
        // 使用 dynamic_cast 将 operatorNode 转换为 op::Slice 类型指针
        if (auto slice_Ptr = dynamic_cast<op::Slice*>(&operatorNode))
        {
            slice_Ptr->SetAttributesFromFile();
        }
        else
        {
            std::cerr << "Failed to cast operatorNode to op::Slice type." << std::endl;
        }
    }
    else if(operatorNode.type == "Add")
    {

    }
    else if(operatorNode.type == "Div")
    {

    }

}

// 一种管理动态内存的智能指针
// 自动负责删除它指向的对象当不再需要它时。不再发生内存泄漏，不再忘记调用删除
std::unique_ptr<op::Node>CreateOperator(const std::string& operatorType,const std::string& operatorName)
{    
    if(operatorType == "Conv")
    {
        return std::make_unique<op::Conv>(operatorType,operatorName);
    }
    else if(operatorType == "LeakyRelu")
    {
        return std::make_unique<op::LeakyRelu>(operatorType,operatorName);
    }
    else if(operatorType == "Constant")
    {
        return std::make_unique<op::Constant>(operatorType,operatorName);
    }
    else if(operatorType == "Slice")
    {
        return std::make_unique<op::Slice>(operatorType,operatorName);
    }
    else if(operatorType == "Concat")
    {
        return std::make_unique<op::Concat>(operatorType,operatorName);
    }
    else if(operatorType == "Add")
    {
        return std::make_unique<op::Add>(operatorType,operatorName);
    }
    else if(operatorType == "Abs")
    {
        return std::make_unique<op::Abs>(operatorType,operatorName);
    }
    else
    {
        std::cout<<"算子库里没有该算子"<<std::endl;
        return nullptr;
    }
    
}

// 读取模型文件
void Read_Model(std::string model_txt)
{
    std::ifstream file(model_txt);
    std::string line;
    std::string currentOperatorType;
    std::string Node_name;
    int lineNum = 0;
    //  创建一个算子向量
    std::vector<std::unique_ptr<op::Node>> operators;
    int para_index = 1;
    // 读取参数文件创建各个算子 当遍历到空行 进行创建算子 之后把各个变量初始化
    while(getline(file, line))
    {
        // 当前在访问的行数
        lineNum++;
        // 创建一个字符串流
        std::istringstream iss(line);
        std::string key;
        size_t colonPos;
        // 该行的第一部分 赋值给key
        iss >> key;
        
        // 返回 npos 表示没找到 某个字符串
        // 存储算子名称
        if((key == "Operator") && (colonPos = line.find("Name:")) != std::string::npos)
        {
            Node_name = line.substr(colonPos+5);
            Node_name.erase(0, Node_name.find_first_not_of(" \t"));
        }
        // 根据Type构建算子
        else if((key == "Operator") && (colonPos = line.find("Type:")) != std::string::npos)
        {
            currentOperatorType = line.substr(colonPos+5);
            currentOperatorType.erase(0, currentOperatorType.find_first_not_of(" \t"));
            operatorMap[Node_name] = CreateOperator(currentOperatorType,Node_name);
        }
        else if((key == "Inputs:")||(key == "Outputs:"))
        {
            std::string item;
            std::istringstream iss(line.substr(line.find(':') + 1));
            auto& CurrentOperator = operatorMap[Node_name];
            while (std::getline(iss, item, ','))
            {
                // 移除空格
                // 将item中的空格移到末尾,返回erase时的起始位置 之后再 erase
                item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
                if(key == "Inputs:")    CurrentOperator->inputs.push_back(item);
                else if(key == "Outputs:")  CurrentOperator->outputs.push_back(item);
            }
        }
        else if(key == "Parameter:")
        {   
            auto& CurrentOperator = operatorMap[Node_name];
            // 每个算子的参数 一定要与其该算子的输入匹配
            if(line.find(CurrentOperator->inputs[para_index]) != std::string::npos)
            {
                // 将inputs[para_index]这个参数存储
                CurrentOperator->StoreParameter(line);
                para_index++;
            }
        }
        else if(line.empty())
        {
            para_index = 1;
            bool hasConstantInput = false;
            auto& CurrentOperator = operatorMap[Node_name];
            // 判断常数输入的特殊情况
            for(const auto& inputName : CurrentOperator->inputs)
            {
                if (inputName.find("Constant") != std::string::npos)
                {
                    hasConstantInput = true;
                    break;
                }
            }
            if(hasConstantInput)
            {
                ConstInput(*CurrentOperator);
            }
            if(PRINT_OP)
            {
                // CurrentOperator->PrintInfo();
                // CurrentOperator->PrintAttributes();
                CurrentOperator->PrintPara();
                std::cout<<std::endl;
            }
            
        }
        // 设置属性 
        else
        {   
            auto& CurrentOperator = operatorMap[Node_name];
            CurrentOperator->SetAttributesFromFile(line);
        }
    }
}

// 创建图节点
void addNode(const std::string& name)
{
    graphNode node;
    node.inputs = {};
    node.dependents = {};
    graph[name] = node;
}

// 构建计算图
void BuildGraph()
{
    std::map<std::string, int> in_degree;

    //构建计算图
    for(auto& current_op : operatorMap)
    {
        op::Node* graph_node = current_op.second.get();
        if(graph_node->type == "Constant") continue; 

        // 非 constant 算子 都加入计算图
        in_degree[graph_node->name] = 0;
        // 可能会在 创建依赖的时候 生成了该节点
        if(graph.find(graph_node->name) == graph.end()) addNode(graph_node->name);
    }
}
     