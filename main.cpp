#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <list>

#include "./include/operator.h"
#include "./include/util.h"
#include "./include/memorypool.h"
#include "./include/cudaop.h"
#include "./include/engine.h"

#define MODEL_TXT "../model_parameters.txt"


std::map<std::string, std::unique_ptr<op::Node>> operatorMap;
std::map<std::string, graphNode> graph; 
std::vector<std::string> topologicalOrder;
std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
std::list<MemoryBlock> memoryPool;
std::list<MemoryBlock> paraPool;
std::multimap<size_t, std::string> tensorOffsets; // 使用multimap来允许相同偏移量的多个Tensor
std::map<std::string, size_t> paraOffsets;
std::map<std::string, std::unique_ptr<cuda::Node>> cudaMap;
size_t totalMemorySize = 0;
size_t totalParaSize = 0;
size_t max_pad_temp = 0;

int input_vis_C = 3;
int input_ir_C = 1;
int input_H = 15;
int input_W = 15;

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
    MemoryPoolImplementation();
    // 构建推理引擎
    BuildCudaOperator();
    engine();

    return 0;
}


