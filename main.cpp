#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <string>


#include "./include/operator.h"
#include "./include/util.h"

#define MODEL_TXT "../model_parameters.txt"


std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
std::map<std::string, graphNode> graph; 
std::vector<std::string> topologicalOrder;
std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;


int main()
{   
    // 读取模型文件 将算子存储到operatorMap
    Read_Model(MODEL_TXT);
    // 构建计算图
    BuildGraph();
    // 拓扑排序 确定算子的执行顺序
    topologicalSort();
    // 构建内存池
    BuildTensorLifetimes();

    // 测试算子
    

    return 0;
}


