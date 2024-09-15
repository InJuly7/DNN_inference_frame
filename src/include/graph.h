#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <cmath>

#include "node.h"




struct graphNode
{
    std::vector<std::string> inputs;
    std::vector<std::string> dependents; // 依赖于该节点输出 的节点
    int in_degree;
};

struct TensorLifeSpan {
    int start_time;  // 张量创建的时间，对应算子的执行序号
    int end_time;    // 张量最后一次被使用的时间
    bool special_flag;  // 特殊标志位，例如用于标记concat操作
    std::vector<int> tensor_shape;
    size_t tensor_size;
};

std::vector<int> parseNumbers(const std::string& line);
std::string parseString(std::string src_sub);
std::vector<float> parseFloats(const std::string &line);
std::string getNodeName(const std::string &outputName);
std::string getOutputTensor(const std::string &NodeName);
void ConstInput(op::Node &operatorNode);
std::unique_ptr<op::Node> CreateOperator(const std::string &operatorName, const std::string &operatorType);
void Read_Model(std::string model_txt);


void addNode(const std::string &name);
void BuildGraph();
void DFS(const std::string & node, std::unordered_map<std::string,int>& visited);
void topologicalSort();

void PrintGraph();
void PrintTopo();

void Initialize_LifeSpan(TensorLifeSpan &lifespan);

void BuildTensorLifetimes();
std::vector<int> calculateOpOutputShape(const std::string &nodeName, const std::vector<std::vector<int>> &inputShapes);


void PrintTensorLifetimes();

#endif // UTIL_H