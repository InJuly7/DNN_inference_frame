#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "util.h"

#define PRINT_OP 0
#define PRINT_GRAPH 0
#define PRINT_TOPO 1
#define PRINT_TENSORLIFETIMES 0


extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
extern std::map<std::string, graphNode> graph;
extern std::vector<std::string> topologicalOrder;
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;

extern int input_vis_C;
extern int input_ir_C;
extern int input_H;
extern int input_W;


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
        if (isdigit(c) || c == ' ' || c == ',' || c == '-' || c == '.' || c == 'e')
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

std::string getOutputTensor(const std::string& NodeName)
{
    std::string  suffix = "_output_0";
    if(graph[NodeName].in_degree == 0)
    {
        return NodeName;
    }
    else if(graph[NodeName].dependents[0] == "output_1")
    {
        return "output_1";
    }
    else if(graph[NodeName].dependents.empty())
    {
        return nullptr;
    }
    else
    {
        return NodeName+suffix;
    }

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
    }
    else if(operatorNode.type == "Add")
    {
        if (auto add_Ptr = dynamic_cast<op::Add*>(&operatorNode))
        {
            add_Ptr->SetAttributesFromFile();
        }
    }
    else if(operatorNode.type == "Div")
    {
        if (auto div_Ptr = dynamic_cast<op::Div*>(&operatorNode))
        {   
            div_Ptr->SetAttributesFromFile();
        }
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
    else if(operatorType == "Tanh")
    {
        return std::make_unique<op::Tanh>(operatorType,operatorName);
    }
    else if(operatorType == "Div")
    {
        return std::make_unique<op::Div>(operatorType,operatorName);
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
                CurrentOperator->PrintInfo();
                CurrentOperator->PrintAttributes();
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
        // 创建图输出节点
        if((graph_node->outputs[0] != graph_node->name+"_output_0"))
        {
            auto output_Node = graph_node->outputs[0];
            if(graph.find(graph_node->outputs[0]) == graph.end())
            {
                addNode(output_Node);
                in_degree[output_Node] = 0;
            }
            graph[output_Node].inputs.push_back(graph_node->name);
            graph[graph_node->name].dependents.push_back(output_Node);
            in_degree[output_Node]++;
        }
        
        // 记录每个计算图中节点的入度
        for(auto& input : graph_node->inputs)
        {
            // 与权重,常数相关的输入
            if(input.find("weight") != std::string::npos || 
                input.find("bias") != std::string::npos ||
                input.find("Constant") != std::string::npos)
            {
                continue;
            }
            
            // 具有依赖关系的输入 "_output_"
            std::string nodeName = getNodeName(input);
            
            // 更改入度
            if(operatorMap.find(nodeName) != operatorMap.end())
            {  
                graph[graph_node->name].inputs.push_back(nodeName);
                if(graph.find(nodeName) == graph.end()) addNode(nodeName);
                graph[nodeName].dependents.push_back(graph_node->name);
                in_degree[graph_node->name]++;
            }

            // 图的输入节点(并不是算子类型) vis ir
            else
            {   
                addNode(nodeName);
                in_degree[nodeName] = 0;
                graph[nodeName].dependents.push_back(graph_node->name);

                graph[graph_node->name].inputs.push_back(nodeName);
                in_degree[graph_node->name]++;
            }
        }
    }

    for (const auto &id : in_degree)
    {
        if (graph.find(id.first) != graph.end())
        {
            graph[id.first].in_degree = id.second;
        }
    }

    if(PRINT_GRAPH)
    {
        PrintGraph();
    }
}

void PrintGraph()
{
    for (const auto &node : graph)
    {
        std::cout << "Node: " << node.first << "\n";
        std::cout << "  Inputs: ";
        for (const std::string &input : node.second.inputs)
        {
            std::cout << input << " ";
        }
        std::cout << "\n  Dependents: ";
        for (const std::string &dependent : node.second.dependents)
        {
            std::cout << dependent << " ";
        }
        std::cout << "\n  In-degree: " << node.second.in_degree << "\n";
    }
}

// 拓扑排序
void DFS(const std::string& node, std::unordered_map<std::string,int>& visited)
{
    // 0 = 未访问, 1 = 访问中, 2 = 已访问
    if (visited[node] == 1)
    {
        std::cout<<"Detected a cycle in the graph"<<std::endl;
    }
    if (visited[node] == 0)
    {
        visited[node] = 1;  // 标记为正在访问
        // 递归访问所有依赖此节点的节点
        for (const std::string& dependent : graph[node].dependents)
        {
            DFS(dependent, visited);
        }
        visited[node] = 2;  // 标记为已访问
        topologicalOrder.push_back(node);  // 在递归返回时加入结果
    }
}

void topologicalSort()
{
    std::unordered_map<std::string, int> visited;
    // 执行DFS仅从入度为0的节点开始
    for (const auto& node : graph)
    {
        if (node.second.in_degree == 0)
        {
            DFS(node.first,visited);
        }
    }
    // 由于DFS结果是逆序的，我们需要反转结果
    std::reverse(topologicalOrder.begin(), topologicalOrder.end());
    if(PRINT_TOPO)
    {
        PrintTopo();
    }
}

void PrintTopo()
{
    std::cout << "Topological Order:" << std::endl;
    for (size_t i = 0; i < topologicalOrder.size(); ++i)
    {
        std::cout << i << ": " << topologicalOrder[i];
        if (i < topologicalOrder.size() - 1)
        {
            std::cout << " -> ";
        }
    }
    std::cout << std::endl;
}

// 构建内存池 Tensor 生命周期表
std::vector<int> calculateOpOutputShape(const std::string& nodeName, const std::vector<std::vector<int>>& inputShapes)
{
    auto node = operatorMap[nodeName].get();
    std::vector<int> outputShape(4);

    if (node->type == "Conv")
    {
        // NCHW
        op::Conv* convNode = dynamic_cast<op::Conv*>(node);
        std::vector<int> weightShape;
        auto inputShape = inputShapes[0];

        outputShape[0] = inputShape[0];
        outputShape[2] = std::floor((inputShape[2] + 2*convNode->pads[0] - convNode->kernel_shape[0]) / convNode->strides[0] + 1);
        outputShape[3] = std::floor((inputShape[3] + 2*convNode->pads[1] - convNode->kernel_shape[1]) / convNode->strides[1] + 1);
        
        // 检索参数权重来确定输入和输出channel
        for (const auto& param : convNode->parameters)
        {
            if (param.first.find("weight") != std::string::npos)
            {
                weightShape = param.second.shape;
                break;
            }
        }

        int outputChannels = weightShape[0]; 
        int inputChannels = weightShape[1];

        outputShape[1] = outputChannels;
        return outputShape;
    }
    else if (node->type == "Concat")
    {
        // Concat操作通常是沿一个特定轴合并张量
        op::Concat* concatNode = dynamic_cast<op::Concat*>(node);
        outputShape = inputShapes[0]; // Start with the shape of the first input tensor
        // NCHW 格式 axis = 1
        for (size_t i = 1; i < inputShapes.size(); ++i)
        {
            outputShape[concatNode->axis] += inputShapes[i][concatNode->axis];
        }
        return outputShape;
    }
    else if (node->type == "LeakyRelu" || node->type == "Add" || node->type == "Abs" || node->type == "Div" || node->type == "Tanh")
    {
        // 这些操作不改变形状
        return inputShapes[0];
    }
    else if (node->type == "Slice")
    {
        op::Slice* sliceNode = dynamic_cast<op::Slice*>(node);
        outputShape = inputShapes[0]; 
        int channel = (sliceNode->end_index - sliceNode->start_index - 1) / sliceNode->steps + 1;
        outputShape[sliceNode->axis] = channel;
        return outputShape;
    }

    // 如果没有匹配的类型，返回输入形状 (vis ir)
    return inputShapes[0];
}

void Initialize_LifeSpan(TensorLifeSpan& lifespan)
{
    lifespan.end_time = 65535;
    lifespan.start_time = -1;
    lifespan.special_flag = false;
    lifespan.tensor_size = 65535;
    lifespan.tensor_shape = {-1,-1,-1,-1};
}

void BuildTensorLifetimes()
{
    int time = 0;
    // tensor_lifetimes 索引为 算子名+"_output_0"
    for (const auto& node_name : topologicalOrder)
    {
        TensorLifeSpan lifespan;
        Initialize_LifeSpan(lifespan);
        // 判断是否是 vis ir 图节点
        if(graph[node_name].in_degree == 0)
        {
            if(node_name == "vis")
            {
                lifespan.start_time = time;
                lifespan.special_flag = false;
                // NCHW
                lifespan.tensor_shape = {1,input_vis_C,input_H,input_W};
                lifespan.tensor_size = lifespan.tensor_shape[0]*lifespan.tensor_shape[1]*lifespan.tensor_shape[2]*lifespan.tensor_shape[3];
            }
            else if(node_name == "ir")
            {
                lifespan.start_time = time;
                lifespan.special_flag = false;
                // NCHW
                lifespan.tensor_shape = {1,input_ir_C,input_H,input_W};
                lifespan.tensor_size = lifespan.tensor_shape[0]*lifespan.tensor_shape[1]*lifespan.tensor_shape[2]*lifespan.tensor_shape[3];
            }
            tensor_lifetimes[node_name] = lifespan;
        }

        else if(graph[node_name].in_degree != 0)
        {   
            
            if (graph[node_name].dependents.empty())
            {    
                continue;
            }
            const auto& node = operatorMap[node_name];
            std::vector<std::vector<int>> inputShapes;
            
            for(const auto& input : node->inputs)
            {
                if(input.find("weight") != std::string::npos || 
                input.find("bias") != std::string::npos ||
                input.find("Constant") != std::string::npos)
                {
                    continue;
                }
                // 对于每一次访问到的tensor 更改end_time
                tensor_lifetimes[input].end_time = time;
                inputShapes.push_back(tensor_lifetimes[input].tensor_shape);
            }
            lifespan.start_time = time;
            lifespan.special_flag = false;
            // 判断当前算子输出的依赖是否有 concat 来设置 special_flag
            for(const auto& dependent : graph[node_name].dependents)
            {
                if(dependent.find("Concat") != std::string::npos)
                {
                    lifespan.special_flag = true;
                    break;
                }
            }
            
            lifespan.tensor_shape = calculateOpOutputShape(node_name,inputShapes);
            lifespan.tensor_size = lifespan.tensor_shape[0]*lifespan.tensor_shape[1]*lifespan.tensor_shape[2]*lifespan.tensor_shape[3];
            tensor_lifetimes[node->outputs[0]] = lifespan;
        }
        time++;
        
    }

    if(PRINT_TENSORLIFETIMES)
    {
        PrintTensorLifetimes();
    }
}

void PrintTensorLifetimes()
{
    std::cout << "Tensor Lifetimes Information:\n";
    for (const auto& pair : tensor_lifetimes)
    {
        const auto& name = pair.first;
        const auto& lifespan = pair.second;
        std::cout << "Tensor Name: " << name << "\n"
                  << "  Start Time: " << lifespan.start_time << "\n"
                  << "  End Time: " << lifespan.end_time << "\n"
                  << "  Special Flag: " << (lifespan.special_flag ? "Yes" : "No") << "\n"
                  << "  Tensor Shape: [";
        for (size_t i = 0; i < lifespan.tensor_shape.size(); ++i)
        {
            std::cout << lifespan.tensor_shape[i];
            if (i < lifespan.tensor_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n"
                  << "  Tensor Size: " << lifespan.tensor_size << "\n\n";
    }
} 

