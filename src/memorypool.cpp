#include <iostream>
#include <list>
#include <vector>
#include <string>


#include "util.h"
#include "memorypool.h"



extern std::list<MemoryBlock> memoryPool;
extern std::vector<std::string> topologicalOrder;
extern std::map<std::string, graphNode> graph; 
extern std::unordered_map<std::string, TensorLifeSpan> tensor_lifetimes;
extern std::map<std::string, std::unique_ptr<op::Node>> operatorMap;


void MemoryPoolImplementation()
{
    std::vector<std::string> inputTensors;
    std::string outputTensor;
    int current_time = 0;
    for (const auto& operatorName : topologicalOrder)
    {
        // 对于 vis ir 图节点 input_tensor = NULL output_tensor = operatorName 
        if(graph[operatorName].in_degree == 0)
        {
            inputTensors = {};
            outputTensor = operatorName;
        }
        else
        {
            for(int i = 0; i < graph[operatorName].in_degree;i++)
            {
                if(graph[operatorName].inputs[i] == "vis" || graph[operatorName].inputs[i] == "ir")
                {
                    inputTensors.push_back(graph[operatorName].inputs[i]);
                }
                else
                {
                    inputTensors.push_back(graph[operatorName].inputs[i] + "_output_0");
                }
                
            }
            outputTensor = operatorName + "_output_0";
        }
        processOperator(operatorName, inputTensors, outputTensor,current_time);
        current_time++;
        // printMemoryPool();
        inputTensors = {};
        outputTensor = {};
    }

}

void processOperator(const std::string& operator_name, const std::vector<std::string>& inputTensors, const std::string& outputTensor,int current_time)
{
    // std::cout<<operator_name<<std::endl;
    // for(const auto& inputTensor : inputTensors)
    // {
    //     std::cout<<inputTensor<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<outputTensor<<std::endl;

    // 图输入节点
    if(graph[operator_name].in_degree == 0)
    {
        allocateMemory(tensor_lifetimes[operator_name].tensor_size,tensor_lifetimes[operator_name].end_time);
        return;
    }

    // 不需要执行的 算子节点 concat 
    else if(operatorMap[operator_name]->type == "Concat")
    {
        return ;
    }

    // 需要执行的算子节点 每次执行完都需要考虑释放块的问题
    else(canTensorBeOverwritten(operator_name,inputTensors,current_time))
    {



    }
    allocateMemory()
}


bool canTensorBeOverwritten(const std::string& operator_name, const std::vector<std::string>& inputTensors, int current_time)
{
    // 强依赖算子 算子类型为conv
    if(operatorMap[operator_name]->type == "Conv")
    {
        return false;
    }
    // 弱依赖算子 add Leakyrelu Tanh div abs 
    else
    {
        // 单/多输入情形
        for (const auto& inputTensor : inputTensors)
        {
            if(tensor_lifetimes[inputTensor].end_time > current_time)
            return false
        }
        // 弱依赖算子的输入Tensor为单依赖情形
        return true;
    }
}




// 分配内存块
void allocateMemory(size_t size, std::string tensor_name,int releaseTime)
{
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (!it->isAllocated && it->size >= size)
        {  // 找到第一个足够大的空闲块
            if (it->size > size)
            {  // 如果块比需要的大，拆分它
                memoryPool.insert(it, MemoryBlock(size, true, releaseTime));
                it->size -= size;
            }
            else
            {  // 正好符合大小，直接分配
                it->isAllocated = true;
                it->releaseTime = releaseTime;
            }
            return;
        }
    }

    // 没有足够的空闲块，创建新块
    memoryPool.emplace_back(size, true, releaseTime);
}

// 释放内存块
void freeMemory(int releaseTime)
{
    for (auto it = memoryPool.begin(); it != memoryPool.end(); ++it)
    {
        if (it->isAllocated && it->releaseTime <= releaseTime)
        {
            it->isAllocated = false;
            it->releaseTime = -1;
            // 尝试与前后块合并
            if (it != memoryPool.begin())
            {
                auto prev = std::prev(it);
                if (!prev->isAllocated)
                {
                    prev->size += it->size;
                    memoryPool.erase(it);
                    it = prev;
                }
            }
            auto next = std::next(it);
            if (next != memoryPool.end() && !next->isAllocated)
            {
                it->size += next->size;
                memoryPool.erase(next);
            }
        }
    }
}


void printMemoryPool()
{
    std::cout << "Memory Pool Status:\n";
    std::cout << "---------------------------------------\n";
    std::cout << "| Allocated | Size (Bytes) | Release Time |\n";
    std::cout << "---------------------------------------\n";
    
    for (const auto& block : memoryPool) {
        std::cout << "| " << (block.isAllocated ? "Yes" : "No ")
                  << "       | " << block.size
                  << "          | ";
        if (block.releaseTime == -1) {
            std::cout << "Free        ";
        } else {
            std::cout << block.releaseTime;
        }
        std::cout << "         |\n";
    }
    
    std::cout << "---------------------------------------\n";
}